"""Tier 4 Real-World Workload Tests: Full End-to-End Book Translation Simulation.
Executes the full pipeline workflow from parsing and chunking to NLLB translation, Aya refinement, DOM reconciliation, and file output.
"""
from pathlib import Path
from uuid import uuid4
import pytest

from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.domain.models.knowledge import Entity, EntityType, GlossaryItem
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.translation.pipeline import TwoStageTranslationPipeline
from src.writers.txt_writer import TxtWriter
from tests.e2e.fixtures import (
    SampleBookFactory,
    MockCTranslate2Engine,
    MockQuantizedAyaEngine,
    reconcile_document_dom,
)


class TestFullBookTranslationPipeline:
    def test_end_to_end_ukrainian_literary_prose_translation(self, temp_work_dir):
        """Test R1.1: Complete end-to-end pipeline execution on Ukrainian literary prose."""
        db_path = temp_work_dir / "full_pipeline.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        # 1. Create Book
        book = SampleBookFactory.create_sample_book(title="The Night Adventure")
        
        # 2. Chunking
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=2)
        for chunk in chunks:
            repo.save_chunk_state(chunk)

        assert repo.count_chunks_for_book(book.id) == len(chunks)

        # 3. Two-Stage Translation
        nllb = MockCTranslate2Engine()
        aya = MockQuantizedAyaEngine()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        pipeline.execute_nllb_stage(book.id)
        
        # Check Stage 1 completed
        drafts = list(repo.load_chunks_by_status(book.id, ChunkStatus.DRAFT_COMPLETED))
        assert len(drafts) == len(chunks)
        assert all(d.draft_translation is not None for d in drafts)

        # Stage 2 Execution
        pipeline.execute_aya_stage(book.id)

        # Check Stage 2 completed
        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        assert len(refined) == len(chunks)
        assert all(r.final_translation is not None for r in refined)

        # 4. DOM Reconciliation
        reconciled_book = reconcile_document_dom(book, refined)

        # 5. Output Writer
        out_path = temp_work_dir / "The_Night_Adventure_ukr.txt"
        writer = TxtWriter()
        writer.write(reconciled_book, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0
        content = out_path.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_multi_chapter_translation_with_glossary_and_ner(self, temp_work_dir):
        """Test R1.2: Multi-chapter translation enforcing glossary across chapter boundaries."""
        db_path = temp_work_dir / "glossary_pipeline.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        # Add Glossary
        with repo._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO glossary VALUES (?, ?, ?, ?)",
                ("Dr. Watson", "Доктор Ватсон", "character", 1)
            )
            conn.execute(
                "INSERT OR REPLACE INTO glossary VALUES (?, ?, ?, ?)",
                ("Mr. Holmes", "Містер Холмс", "character", 1)
            )
            conn.commit()

        book = SampleBookFactory.create_sample_book(title="Memoirs")
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=2)
        for chunk in chunks:
            repo.save_chunk_state(chunk)

        nllb = MockCTranslate2Engine()
        aya = MockQuantizedAyaEngine()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        combined_text = " ".join(c.final_translation for c in refined if c.final_translation)

        assert "Доктор Ватсон" in combined_text or "Ватсон" in combined_text

    def test_pipeline_memory_cleanup_and_gc_between_stages(self, temp_work_dir):
        """Test R1.3: Verifies NLLB model is completely unloaded before Aya model is loaded."""
        db_path = temp_work_dir / "memory_pipeline.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=2)
        for chunk in chunks:
            repo.save_chunk_state(chunk)

        class LifecycleTrackingNLLB(MockCTranslate2Engine):
            def load_model(self):
                super().load_model()
                self.load_count = getattr(self, 'load_count', 0) + 1
            def unload_model(self):
                super().unload_model()
                self.unload_count = getattr(self, 'unload_count', 0) + 1

        class LifecycleTrackingAya(MockQuantizedAyaEngine):
            def load_model(self):
                super().load_model()
                self.load_count = getattr(self, 'load_count', 0) + 1
            def unload_model(self):
                super().unload_model()
                self.unload_count = getattr(self, 'unload_count', 0) + 1

        nllb = LifecycleTrackingNLLB()
        aya = LifecycleTrackingAya()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        pipeline.execute_nllb_stage(book.id)
        assert nllb.is_loaded is False
        assert nllb.load_count == 1
        assert nllb.unload_count == 1

        pipeline.execute_aya_stage(book.id)
        assert aya.is_loaded is False
        assert aya.load_count == 1
        assert aya.unload_count == 1
