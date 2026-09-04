"""
Migration 001: Initial schema baseline.
Sets up entities, glossary, glossary_metadata, and chunk_checkpoints tables.
"""

import sqlite3

VERSION = 1
DESCRIPTION = "Baseline schema: entities, glossary, glossary_metadata, chunk_checkpoints"


def apply(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

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

    # Glossary Metadata Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS glossary_metadata (
            source_term TEXT PRIMARY KEY,
            reviewed INTEGER DEFAULT 0,
            grammatical_gender TEXT
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

    # Compound & Lookup Indices
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


def rollback(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS chunk_checkpoints")
    cursor.execute("DROP TABLE IF EXISTS glossary_metadata")
    cursor.execute("DROP TABLE IF EXISTS glossary")
    cursor.execute("DROP TABLE IF EXISTS entities")
