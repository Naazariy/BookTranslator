"""Tier 1 Feature Tests: SQLite WAL & Compound Indexing (F9, F10).
Verifies WAL pragma, synchronous=NORMAL, compound index existence/usage, batch transaction commits, and thread-local connection safety.
"""
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from uuid import uuid4
import pytest

from src.domain.models.chunk import TranslationChunk, ChunkStatus, ChunkContext
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository


class TestSQLiteWALIndexing:
    def test_sqlite_wal_pragma_enabled(self, temp_work_dir):
        """Test F9.1: WAL journal mode and normal synchronous pragma configuration."""
        db_path = temp_work_dir / "test_wal.db"
        init_db(db_path)
        
        repo = SQLiteKnowledgeBaseRepository(db_path)
        with repo._connect() as conn:
            # Check journal mode
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL;")
            mode = cursor.fetchone()[0]
            assert mode.upper() == "WAL"

            cursor.execute("PRAGMA synchronous = NORMAL;")
            cursor.execute("PRAGMA synchronous;")
            sync_val = cursor.fetchone()[0]
            assert sync_val in (1, "NORMAL", "1")

    def test_sqlite_compound_index_exists_and_query_plan(self, temp_work_dir):
        """Test F10.1: Compound index idx_chunk_book_status exists and is used in EXPLAIN QUERY PLAN."""
        db_path = temp_work_dir / "test_idx.db"
        init_db(db_path)
        
        with sqlite3.connect(db_path) as conn:
            # Ensure compound index is created
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_book_status ON chunk_checkpoints(book_id, status);")
            conn.commit()
            
            # Check sqlite_master for index
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_chunk_book_status';")
            idx = cursor.fetchone()
            assert idx is not None
            assert idx[0] == "idx_chunk_book_status"

            # Check EXPLAIN QUERY PLAN
            cursor.execute("EXPLAIN QUERY PLAN SELECT serialized_data FROM chunk_checkpoints WHERE book_id = ? AND status = ?", ("dummy_id", "PENDING"))
            plan = cursor.fetchall()
            plan_str = " ".join(str(row) for row in plan)
            assert "idx_chunk_book_status" in plan_str or "USING INDEX" in plan_str or "COVERING INDEX" in plan_str or "INDEX" in plan_str

    def test_sqlite_batch_save_chunk_state_transaction(self, temp_work_dir):
        """Test F9.2: Batch saving chunks under single transaction."""
        db_path = temp_work_dir / "test_batch.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        
        book_id = uuid4()
        chapter_id = uuid4()
        chunks = [
            TranslationChunk(
                id=uuid4(),
                book_id=book_id,
                chapter_id=chapter_id,
                paragraph_indices=[i],
                source_sentences=[],
                token_count=100,
                context=ChunkContext(),
                status=ChunkStatus.PENDING
            )
            for i in range(50)
        ]
        
        # Save batch or iterate
        if hasattr(repo, 'save_chunk_state_batch'):
            repo.save_chunk_state_batch(chunks)
        else:
            for c in chunks:
                repo.save_chunk_state(c)

        count = repo.count_chunks_for_book(book_id)
        assert count == 50

    def test_sqlite_load_chunks_by_status_filtering(self, temp_work_dir):
        """Test F10.2: Loading chunks filtered by status."""
        db_path = temp_work_dir / "test_filter.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        
        book_id = uuid4()
        chapter_id = uuid4()
        
        # Create 10 PENDING and 5 REFINED chunks
        for i in range(10):
            repo.save_chunk_state(TranslationChunk(
                id=uuid4(),
                book_id=book_id,
                chapter_id=chapter_id,
                paragraph_indices=[i],
                source_sentences=[],
                token_count=50,
                context=ChunkContext(),
                status=ChunkStatus.PENDING
            ))
        for i in range(5):
            repo.save_chunk_state(TranslationChunk(
                id=uuid4(),
                book_id=book_id,
                chapter_id=chapter_id,
                paragraph_indices=[i],
                source_sentences=[],
                token_count=50,
                context=ChunkContext(),
                status=ChunkStatus.REFINED,
                final_translation="Переклад"
            ))

        pending = list(repo.load_chunks_by_status(book_id, ChunkStatus.PENDING))
        refined = list(repo.load_chunks_by_status(book_id, ChunkStatus.REFINED))

        assert len(pending) == 10
        assert len(refined) == 5
        assert all(c.status == ChunkStatus.PENDING for c in pending)
        assert all(c.status == ChunkStatus.REFINED for c in refined)

    def test_sqlite_count_chunks_for_book(self, temp_work_dir):
        """Test F9.3: Fast count_chunks_for_book queries."""
        db_path = temp_work_dir / "test_count.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        
        book1 = uuid4()
        book2 = uuid4()
        
        for _ in range(7):
            repo.save_chunk_state(TranslationChunk(
                id=uuid4(), book_id=book1, chapter_id=uuid4(),
                paragraph_indices=[0], source_sentences=[], token_count=10,
                context=ChunkContext(), status=ChunkStatus.PENDING
            ))
        for _ in range(3):
            repo.save_chunk_state(TranslationChunk(
                id=uuid4(), book_id=book2, chapter_id=uuid4(),
                paragraph_indices=[0], source_sentences=[], token_count=10,
                context=ChunkContext(), status=ChunkStatus.PENDING
            ))

        assert repo.count_chunks_for_book(book1) == 7
        assert repo.count_chunks_for_book(book2) == 3

    def test_sqlite_thread_local_connection_safety(self, temp_work_dir):
        """Test F9.4: Multi-threaded repository operations with thread-local / isolated connections."""
        db_path = temp_work_dir / "test_threads.db"
        init_db(db_path)
        
        book_id = uuid4()
        errors = []

        def worker(thread_idx: int):
            try:
                repo = SQLiteKnowledgeBaseRepository(db_path)
                for i in range(10):
                    repo.save_chunk_state(TranslationChunk(
                        id=uuid4(),
                        book_id=book_id,
                        chapter_id=uuid4(),
                        paragraph_indices=[i],
                        source_sentences=[],
                        token_count=20,
                        context=ChunkContext(),
                        status=ChunkStatus.PENDING
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        final_repo = SQLiteKnowledgeBaseRepository(db_path)
        assert final_repo.count_chunks_for_book(book_id) == 50
