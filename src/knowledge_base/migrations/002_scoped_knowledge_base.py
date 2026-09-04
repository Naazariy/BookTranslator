"""
Migration 002: Scoped Knowledge Base & Segment Indexing.
Sets up entity_profiles, entity_mentions, and translation_segments tables.
"""

import sqlite3

VERSION = 2
DESCRIPTION = "Scoped knowledge base: entity_profiles, entity_mentions, translation_segments"


def apply(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    # Entity Profiles Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_profiles (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL DEFAULT 'character',
            scope TEXT NOT NULL DEFAULT 'BOOK',
            scope_id TEXT,
            source_name TEXT NOT NULL,
            canonical_target TEXT NOT NULL,
            aliases TEXT,
            allowed_target_forms TEXT,
            forbidden_target_forms TEXT,
            grammatical_gender TEXT,
            translation_policy TEXT DEFAULT 'preserve_canonical',
            confidence REAL DEFAULT 1.0,
            locked INTEGER DEFAULT 0,
            description TEXT,
            case_sensitive INTEGER DEFAULT 1,
            frequency INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Entity Mentions Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_mentions (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            sentence_id TEXT,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            surface_form TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            metadata TEXT,
            FOREIGN KEY(entity_id) REFERENCES entity_profiles(id) ON DELETE CASCADE
        )
    ''')

    # Translation Segments Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS translation_segments (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            chapter_id TEXT NOT NULL,
            paragraph_id TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            source_text TEXT NOT NULL,
            normalized_text TEXT,
            draft_translation TEXT,
            refined_translation TEXT,
            final_translation TEXT,
            status TEXT NOT NULL,
            repair_attempts INTEGER DEFAULT 0,
            error_message TEXT,
            serialized_data TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Indices
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_entity_profiles_source 
        ON entity_profiles(source_name);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_entity_profiles_scope 
        ON entity_profiles(scope, scope_id);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_entity_profiles_lookup 
        ON entity_profiles(source_name, scope, scope_id);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_entity_mentions_segment 
        ON entity_mentions(segment_id);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity 
        ON entity_mentions(entity_id);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_segments_book_status 
        ON translation_segments(book_id, status);
    ''')


def rollback(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS translation_segments")
    cursor.execute("DROP TABLE IF EXISTS entity_mentions")
    cursor.execute("DROP TABLE IF EXISTS entity_profiles")
