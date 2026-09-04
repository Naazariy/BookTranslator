"""
Database Schema Initialization and Migration Routing for BookTranslator V2.
"""

import sqlite3
from pathlib import Path

from src.knowledge_base.migrations.migration_runner import MigrationRunner


def init_db(db_path: Path) -> None:
    """
    Initializes SQLite database schema for entities, glossary, chunk checkpoints,
    entity profiles, mentions, and translation segments.
    Enforces WAL journal mode, optimal synchronization PRAGMAs, and executes
    all database migrations through MigrationRunner.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, timeout=10.0) as conn:
        # High-performance PRAGMAs
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA foreign_keys = ON;")
        
        # Route schema creation and upgrades through MigrationRunner
        runner = MigrationRunner(db_path)
        runner.apply_all(conn=conn)


class DatabaseSchema:
    """Wrapper class for database schema initialization."""
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def initialize_schema(self) -> None:
        init_db(self.db_path)
