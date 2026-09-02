"""Tier 3 Cross-Feature Tests: Quantized Aya Engine + SQLite Glossary DB + Cancellation.
Verifies live entity/glossary retrieval from SQLite, prompt injection during Aya refinement, stopping criteria, and state transitions.
"""
from uuid import uuid4
import pytest

from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.domain.models.knowledge import Entity, EntityType, GlossaryItem
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from tests.e2e.fixtures import (
    MockQuantizedAyaEngine,
    CancellationToken,
    SampleBookFactory,
)


class TestAyaRefinementWithGlossaryDB:
    def test_aya_refinement_queries_sqlite_glossary_and_injects_terms(self, temp_work_dir):
        """Test X3.1: Aya queries terms from SQLite glossary and applies them to draft translations."""
        db_path = temp_work_dir / "test_glossary_db.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        
        # Add Glossary Entities
        repo.add_entity(Entity(
            id=uuid4(),
            name="Sherlock Holmes",
            entity_type=EntityType.CHARACTER,
            aliases=["Holmes"],
            canonical_translation="Шерлок Холмс"
        ))
        
        # Manually populate glossary table in SQLite
        with repo._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO glossary VALUES (?, ?, ?, ?)",
                ("Sherlock Holmes", "Шерлок Холмс", "character", 1)
            )
            conn.execute(
                "INSERT OR REPLACE INTO glossary VALUES (?, ?, ?, ?)",
                ("Baker Street", "Бейкер-стріт", "location", 1)
            )
            conn.commit()

        glossary = repo.get_glossary()
        assert len(glossary) == 2

        engine = MockQuantizedAyaEngine()
        engine.load_model()

        try:
            draft = "Він зустрів Sherlock Holmes на Baker Street."
            ctx = ChunkContext()
            refined = engine.refine_chunk(draft, ctx, glossary)
            
            assert "Шерлок Холмс" in refined
            assert "Бейкер-стріт" in refined
        finally:
            engine.unload_model()

    def test_aya_stopping_criteria_halts_generation_on_cancel_token(self):
        """Test X3.2: Stopping criteria halts refinement if user cancels mid-stream."""
        engine = MockQuantizedAyaEngine()
        engine.load_model()
        token = CancellationToken()

        # Simulate cancellation during generation
        token.cancel()
        with pytest.raises(RuntimeError) as exc_info:
            engine.refine_chunk("Draft text", ChunkContext(), [], cancel_token=token)
        assert "cancelled" in str(exc_info.value).lower()

    def test_aya_draft_to_refined_state_transition_in_sqlite(self, temp_work_dir):
        """Test X3.3: Stage 2 transitions chunks from DRAFT_COMPLETED to REFINED."""
        db_path = temp_work_dir / "test_state_trans.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book_id = uuid4()
        
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book_id,
            chapter_id=uuid4(),
            paragraph_indices=[0],
            source_sentences=[],
            token_count=20,
            context=ChunkContext(),
            status=ChunkStatus.DRAFT_COMPLETED,
            draft_translation="Попередня чернетка"
        )
        repo.save_chunk_state(chunk)

        # Refine and update status
        engine = MockQuantizedAyaEngine()
        engine.load_model()
        try:
            drafts = list(repo.load_chunks_by_status(book_id, ChunkStatus.DRAFT_COMPLETED))
            assert len(drafts) == 1
            
            c = drafts[0]
            c.final_translation = engine.refine_chunk(c.draft_translation, c.context, [])
            c.status = ChunkStatus.REFINED
            repo.save_chunk_state(c)

            # Check DB state
            assert len(list(repo.load_chunks_by_status(book_id, ChunkStatus.DRAFT_COMPLETED))) == 0
            refined = list(repo.load_chunks_by_status(book_id, ChunkStatus.REFINED))
            assert len(refined) == 1
            assert refined[0].final_translation is not None
        finally:
            engine.unload_model()
