"""
Migration Runner for SQLite Database Schema Versioning.
Handles migration discovery, execution order, transactional safety, and schema_version tracking.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when a migration fails to apply or validate."""
    pass


class MigrationRunner:
    """
    Discovers and applies sequential schema migrations to a SQLite database.
    Tracks state using the 'schema_version' table.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.migrations_dir = Path(__file__).parent

    def _ensure_schema_version_table(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT NOT NULL
            )
        ''')
        conn.commit()

    def discover_migrations(self) -> List[Tuple[int, str, Path]]:
        r"""
        Discovers migration files matching '^m?(\d+)_(.+)\.py$' in the migrations directory.
        Returns a sorted list of (version, description, file_path).
        """
        migrations: List[Tuple[int, str, Path]] = []
        pattern = re.compile(r"^m?(\d+)_(.+)\.py$")

        if not self.migrations_dir.exists():
            return migrations

        for file in self.migrations_dir.iterdir():
            if not file.is_file():
                continue
            match = pattern.match(file.name)
            if match:
                version = int(match.group(1))
                desc = match.group(2).replace("_", " ")
                migrations.append((version, desc, file))

        migrations.sort(key=lambda m: m[0])
        return migrations

    def get_applied_versions(self, conn: sqlite3.Connection) -> List[int]:
        """Returns all versions currently recorded in schema_version table."""
        self._ensure_schema_version_table(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM schema_version ORDER BY version ASC")
        return [row[0] for row in cursor.fetchall()]

    def get_current_version(self) -> int:
        """Returns the highest applied migration version or 0 if none applied."""
        if not self.db_path.exists():
            return 0
        with sqlite3.connect(self.db_path) as conn:
            applied = self.get_applied_versions(conn)
            return max(applied) if applied else 0

    def apply_all(self, conn: Optional[sqlite3.Connection] = None) -> List[int]:
        """
        Applies all pending migrations in sequential order.
        Returns the list of newly applied migration versions.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        should_close = False

        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 10000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            should_close = True

        try:
            self._ensure_schema_version_table(conn)
            applied_versions = set(self.get_applied_versions(conn))
            all_migrations = self.discover_migrations()
            newly_applied: List[int] = []

            for version, desc, path in all_migrations:
                if version in applied_versions:
                    continue

                logger.info(f"Applying migration {version:03d} ({desc}) from {path.name}")
                module = self._load_migration_module(path)

                # Execute migration inside an atomic transaction
                try:
                    with conn:
                        module.apply(conn)
                        # Record migration in schema_version
                        migration_desc = getattr(module, "DESCRIPTION", desc)
                        conn.execute(
                            "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                            (version, migration_desc)
                        )
                    newly_applied.append(version)
                    logger.info(f"Successfully applied migration {version:03d}")
                except Exception as e:
                    conn.rollback()
                    raise MigrationError(f"Migration {version:03d} ({path.name}) failed: {e}") from e

            return newly_applied
        finally:
            if should_close and conn:
                conn.close()

    def _load_migration_module(self, path: Path) -> Any:
        module_name = f"migration_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise MigrationError(f"Could not create module spec for migration file {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "apply") or not callable(module.apply):
            raise MigrationError(f"Migration {path.name} does not export a callable apply(conn) function")
        return module
