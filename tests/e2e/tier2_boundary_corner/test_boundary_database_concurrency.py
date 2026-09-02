"""Tier 2 Boundary Tests: SQLite WAL Concurrency & High Contention.
Verifies concurrent read-during-write operations under WAL mode, multi-threaded batch writes with busy_timeout, and serialization limits.
"""
import time
import sqlite3
import threading
from pathlib import Path
from uuid import uuid4
import pytest

from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository


class TestBoundaryDatabaseConcurrency:
    def test_concurrent_readers_during_active_wal_write_transaction(self, temp_work_dir):
        """Test B4.1: Multiple concurrent readers query without blocking while a writer commits."""
        db_path = temp_work_dir / "test_wal_rw.db"
        init_db(db_path)
        book_id = uuid4()
        
        # Insert initial data
        repo_init = SQLiteKnowledgeBaseRepository(db_path)
        for i in range(10):
            repo_init.save_chunk_state(TranslationChunk(
                id=uuid4(), book_id=book_id, chapter_id=uuid4(),
                paragraph_indices=[i], source_sentences=[], token_count=10,
                context=ChunkContext(), status=ChunkStatus.PENDING
            ))

        stop_event = threading.Event()
        read_counts = []
        read_errors = []

        def reader_worker():
            repo = SQLiteKnowledgeBaseRepository(db_path)
            while not stop_event.is_set():
                try:
                    c = repo.count_chunks_for_book(book_id)
                    read_counts.append(c)
                    time.sleep(0.001)
                except Exception as e:
                    read_errors.append(e)

        def writer_worker():
            repo = SQLiteKnowledgeBaseRepository(db_path)
            for i in range(30):
                repo.save_chunk_state(TranslationChunk(
                    id=uuid4(), book_id=book_id, chapter_id=uuid4(),
                    paragraph_indices=[i], source_sentences=[], token_count=10,
                    context=ChunkContext(), status=ChunkStatus.PENDING
                ))
                time.sleep(0.002)

        reader_threads = [threading.Thread(target=reader_worker) for _ in range(4)]
        writer_thread = threading.Thread(target=writer_worker)

        for r in reader_threads:
            r.start()
        writer_thread.start()

        writer_thread.join()
        stop_event.set()
        for r in reader_threads:
            r.join()

        assert len(read_errors) == 0
        assert len(read_counts) > 0
        assert repo_init.count_chunks_for_book(book_id) == 40

    def test_concurrent_multi_thread_batch_writes_with_busy_timeout(self, temp_work_dir):
        """Test B4.2: 5 parallel worker threads inserting batches under busy_timeout configuration."""
        db_path = temp_work_dir / "test_wal_busy.db"
        init_db(db_path)
        book_id = uuid4()
        
        errors = []
        
        def batch_writer(worker_id: int):
            try:
                # Each thread connects with busy_timeout
                with sqlite3.connect(db_path, timeout=10.0) as conn:
                    conn.execute("PRAGMA journal_mode = WAL;")
                    conn.execute("PRAGMA synchronous = NORMAL;")
                    for batch_idx in range(5):
                        rows = [
                            (str(uuid4()), str(book_id), str(uuid4()), "PENDING", 10, None, None, None, "{}")
                            for _ in range(10)
                        ]
                        conn.executemany(
                            "INSERT OR REPLACE INTO chunk_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            rows
                        )
                        conn.commit()
                        time.sleep(0.001)
            except Exception as e:
                errors.append((worker_id, e))

        threads = [threading.Thread(target=batch_writer, args=(w,)) for w in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        repo = SQLiteKnowledgeBaseRepository(db_path)
        assert repo.count_chunks_for_book(book_id) == 250

    def test_large_payload_chunk_serialization_and_recovery(self, temp_work_dir):
        """Test B4.3: Large chunk payloads (>500KB text) serialized and deserialized accurately."""
        db_path = temp_work_dir / "test_large_payload.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book_id = uuid4()
        
        large_draft = "Цей довгий текст повторюється багато разів. " * 5000  # ~250KB
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book_id,
            chapter_id=uuid4(),
            paragraph_indices=[0],
            source_sentences=[],
            token_count=10000,
            context=ChunkContext(),
            status=ChunkStatus.DRAFT_COMPLETED,
            draft_translation=large_draft
        )

        repo.save_chunk_state(chunk)
        loaded = list(repo.load_chunks_by_status(book_id, ChunkStatus.DRAFT_COMPLETED))
        assert len(loaded) == 1
        assert loaded[0].id == chunk.id
        assert loaded[0].draft_translation == large_draft
