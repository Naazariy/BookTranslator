"""
Migration 003: Quality Reports Audit Trail.
Creates quality_reports table and performance indexes for QA audits,
remediation history, and validator diagnostics.
"""

import sqlite3

VERSION = 3
DESCRIPTION = "Quality reports audit trail: quality_reports table and indexes"


def apply(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    # Quality Reports Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quality_reports (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            validator_name TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            details_json TEXT,
            repair_attempts INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Indices for high-speed audit aggregation and lookups
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_quality_reports_book 
        ON quality_reports(book_id);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_quality_reports_segment 
        ON quality_reports(segment_id);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_quality_reports_book_status 
        ON quality_reports(book_id, status);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_quality_reports_severity 
        ON quality_reports(severity);
    ''')


def rollback(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS quality_reports")
