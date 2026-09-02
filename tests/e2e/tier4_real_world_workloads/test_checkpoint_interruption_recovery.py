"""Tier 4 Real-World Workload Tests: Checkpoint Interruption and State Recovery.
Simulates crash / power-kill mid-process and verifies resuming translation without duplicate work.
"""
from uuid import uuid4
import pytest

from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.translation.pipeline import TwoStageTranslationPipeline
from tests.e2e.fixtures import (
    SampleBookFactory,
    MockCTranslate2Engine,
    MockQuantizedAyaEngine,
)


class TestCheckpointInterruptionRecovery:
    def test_recovery_from_stage1_interruption_resumes_pending_only(self, temp_work_dir):
        """Test R2.1: Interruption during Stage 1 leaves drafts intact; resume translates pending only."""
        db_path = temp_work_dir / "recovery_stage1.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=1)
        
        # Save all chunks as PENDING
        for c in chunks:
            repo.save_chunk_state(c)

        # Simulate half the chunks translated before crash
        half = len(chunks) // 2
        for i in range(half):
            chunks[i].draft_translation = f"Saved Draft {i}"
            chunks[i].status = ChunkStatus.DRAFT_COMPLETED
            repo.save_chunk_state(chunks[i])

        # Track NLLB translation invocations during recovery
        translated_inputs = []
        class TrackingNLLB(MockCTranslate2Engine):
            def translate_batch(self, texts, *args, **kwargs):
                translated_inputs.extend(texts)
                return super().translate_batch(texts, *args, **kwargs)

        nllb = TrackingNLLB()
        aya = MockQuantizedAyaEngine()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        # Execute resume
        pipeline.execute_nllb_stage(book.id)

        # Verify only remaining pending chunks were translated
        remaining_pending = len(chunks) - half
        assert len(translated_inputs) == remaining_pending
        
        all_drafts = list(repo.load_chunks_by_status(book.id, ChunkStatus.DRAFT_COMPLETED))
        assert len(all_drafts) == len(chunks)

    def test_recovery_from_stage2_interruption_preserves_drafts(self, temp_work_dir):
        """Test R2.2: Interruption during Stage 2 preserves completed REFINED chunks without re-editing."""
        db_path = temp_work_dir / "recovery_stage2.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=1)

        # All chunks passed Stage 1
        for i, c in enumerate(chunks):
            c.draft_translation = f"Draft {i}"
            c.status = ChunkStatus.DRAFT_COMPLETED
            repo.save_chunk_state(c)

        # Simulate 2 chunks refined before crash
        chunks[0].final_translation = "Finished Refined 0"
        chunks[0].status = ChunkStatus.REFINED
        repo.save_chunk_state(chunks[0])

        chunks[1].final_translation = "Finished Refined 1"
        chunks[1].status = ChunkStatus.REFINED
        repo.save_chunk_state(chunks[1])

        refine_calls = []
        class TrackingAya(MockQuantizedAyaEngine):
            def refine_chunk(self, *args, **kwargs):
                draft = kwargs.get('draft_translation', args[0] if len(args) > 0 else "")
                refine_calls.append(draft)
                return super().refine_chunk(*args, **kwargs)

        nllb = MockCTranslate2Engine()
        aya = TrackingAya()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        pipeline.execute_aya_stage(book.id)

        # Only the remaining unrefined chunks were processed
        assert len(refine_calls) == len(chunks) - 2
        all_refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        assert len(all_refined) == len(chunks)
        assert all_refined[0].final_translation == "Finished Refined 0" or any(r.final_translation == "Finished Refined 0" for r in all_refined)

    def test_zero_recomputation_of_already_refined_chunks(self, temp_work_dir):
        """Test R2.3: Re-running pipeline on fully refined book does 0 re-computations in either stage."""
        db_path = temp_work_dir / "zero_recomputation.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=1)

        # All chunks already REFINED
        for i, c in enumerate(chunks):
            c.draft_translation = f"Draft {i}"
            c.final_translation = f"Refined {i}"
            c.status = ChunkStatus.REFINED
            repo.save_chunk_state(c)

        nllb_calls = []
        class TrackingNLLB(MockCTranslate2Engine):
            def translate_batch(self, texts, *args, **kwargs):
                nllb_calls.extend(texts)
                return super().translate_batch(texts, *args, **kwargs)

        aya_calls = []
        class TrackingAya(MockQuantizedAyaEngine):
            def refine_chunk(self, *args, **kwargs):
                draft = kwargs.get('draft_translation', args[0] if len(args) > 0 else "")
                aya_calls.append(draft)
                return super().refine_chunk(*args, **kwargs)


        pipeline = TwoStageTranslationPipeline(TrackingNLLB(), TrackingAya(), repo)

        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        assert len(nllb_calls) == 0
        assert len(aya_calls) == 0
