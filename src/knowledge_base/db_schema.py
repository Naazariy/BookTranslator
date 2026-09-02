import sqlite3
from pathlib import Path


def init_db(db_path: Path) -> None:
    """
    Initializes SQLite database schema for entities, glossary, and translation chunk checkpoints.
    Enforces WAL journal mode, optimal synchronization PRAGMAs, and compound indexes for high-throughput queries.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # High-performance PRAGMAs
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute("PRAGMA busy_timeout = 10000;")
        cursor.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        cursor.execute("PRAGMA temp_store = MEMORY;")
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # Entities Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                aliases TEXT,
                description TEXT,
                canonical_translation TEXT,
                frequency INTEGER DEFAULT 1
            )
        ''')
        
        # Glossary Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS glossary (
                source_term TEXT PRIMARY KEY,
                target_term TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                case_sensitive INTEGER DEFAULT 1
            )
        ''')
        
        # Chunks Table for Checkpointing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_checkpoints (
                id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL,
                chapter_id TEXT NOT NULL,
                status TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                draft_translation TEXT,
                final_translation TEXT,
                error_message TEXT,
                serialized_data TEXT NOT NULL
            )
        ''')
        
        # Compound & Lookup Indices for zero-scan fast querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chunk_book_status 
            ON chunk_checkpoints(book_id, status);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_entities_name 
            ON entities(name);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_entities_name_type 
            ON entities(name, entity_type);
        ''')
        
        conn.commit()


class DatabaseSchema:
    """Wrapper class for database schema initialization."""
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def initialize_schema(self) -> None:
        init_db(self.db_path)
