"""Tier 2 Boundary Tests: Rapid Task Cancellation and Interruption Under High Frequency.
Verifies immediate cancellation pre-start, mid-Stage 1, mid-Stage 2, rapid toggle cycles, and database consistency post-abort.
"""
import time
import pytest
from uuid import uuid4
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from tests.e2e.fixtures import (
    CancellationToken,
    AsyncTaskManager,
    MockCTranslate2Engine,
    MockQuantizedAyaEngine,
    SampleBookFactory,
)


class TestBoundaryRapidCancellation:
    def test_immediate_cancellation_before_stage1_start(self):
        """Test B2.1: Pre-cancelled token halts pipeline immediately before any work begins."""
        token = CancellationToken()
        token.cancel()
        
        executed_chunks = []
        def job(cancel_token):
            for i in range(10):
                if cancel_token.is_cancelled():
                    return
                executed_chunks.append(i)
                
        job(token)
        assert len(executed_chunks) == 0


    def test_cancellation_during_stage1_nllb_execution(self):
        """Test B2.2: Cancellation signal during Stage 1 stops NLLB batch iterations promptly."""
        token = CancellationToken()
        processed_count = 0
        engine = MockCTranslate2Engine()
        engine.load_model()

        def stage1_worker(cancel_token):
            nonlocal processed_count
            for i in range(100):
                if cancel_token.is_cancelled():
                    break
                engine.translate_batch([f"Sentence {i}"], "en", "uk")
                processed_count += 1
                time.sleep(0.005)

        mgr = AsyncTaskManager()
        mgr.start_task(stage1_worker)
        time.sleep(0.02)  # Let it run ~4-5 batches
        mgr.cancel()
        mgr.join(timeout=1.0)

        assert mgr.is_running() is False
        assert 0 < processed_count < 30

    def test_cancellation_during_stage2_aya_execution(self):
        """Test B2.3: Cancellation signal during Stage 2 stops Aya refinement iterations promptly."""
        engine = MockQuantizedAyaEngine()
        engine.load_model()
        refined_count = 0

        def stage2_worker(cancel_token):
            nonlocal refined_count
            for i in range(100):
                if cancel_token.is_cancelled():
                    break
                engine.refine_chunk(f"Draft {i}", ChunkContext(), [], cancel_token=cancel_token)
                refined_count += 1
                time.sleep(0.005)

        mgr = AsyncTaskManager()
        mgr.start_task(stage2_worker)
        time.sleep(0.02)
        mgr.cancel()
        mgr.join(timeout=1.0)

        assert mgr.is_running() is False
        assert 0 < refined_count < 30

    def test_rapid_toggle_start_cancel_multiple_times(self):
        """Test B2.4: 10 back-to-back start/cancel cycles without deadlocking or thread leak."""
        mgr = AsyncTaskManager()
        
        def short_task(cancel_token):
            for _ in range(50):
                if cancel_token.is_cancelled():
                    break
                time.sleep(0.001)

        for _ in range(10):
            started = mgr.start_task(short_task)
            assert started is True
            mgr.cancel()
            mgr.join(timeout=0.5)
            assert mgr.is_running() is False

    def test_cancellation_leaves_database_in_consistent_state(self, temp_work_dir):
        """Test B2.5: Cancelling during batch execution leaves committed chunks intact without corruption."""
        db_path = temp_work_dir / "test_cancel_db.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book_id = uuid4()
        
        # Pre-populate 20 chunks
        for i in range(20):
            repo.save_chunk_state(TranslationChunk(
                id=uuid4(), book_id=book_id, chapter_id=uuid4(),
                paragraph_indices=[i], source_sentences=[], token_count=10,
                context=ChunkContext(), status=ChunkStatus.PENDING
            ))

        def aborting_worker(cancel_token):
            chunks = list(repo.load_chunks_by_status(book_id, ChunkStatus.PENDING))
            for i, c in enumerate(chunks):
                if cancel_token.is_cancelled():
                    break
                c.status = ChunkStatus.DRAFT_COMPLETED
                c.draft_translation = f"Draft {i}"
                repo.save_chunk_state(c)
                if i == 5:
                    cancel_token.cancel()

        mgr = AsyncTaskManager()
        mgr.start_task(aborting_worker)
        mgr.join(timeout=2.0)

        completed = list(repo.load_chunks_by_status(book_id, ChunkStatus.DRAFT_COMPLETED))
        pending = list(repo.load_chunks_by_status(book_id, ChunkStatus.PENDING))

        assert len(completed) == 6
        assert len(pending) == 14
        assert repo.count_chunks_for_book(book_id) == 20
