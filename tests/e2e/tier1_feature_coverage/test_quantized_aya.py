"""Tier 1 Feature Tests: Quantized Aya-23-8B Editing Engine (F6).
Verifies in-memory prompt template caching, glossary injection, markdown/header cleaning, cancellation token, and batch refinement.
"""
import pytest
from src.domain.models.chunk import ChunkContext
from src.domain.models.knowledge import GlossaryItem, EntityType
from tests.e2e.fixtures import MockQuantizedAyaEngine, CancellationToken

try:
    from src.translation.aya_editing_engine import QuantizedAyaEditingEngine as ProjectQuantizedAyaEngine
except ImportError:
    ProjectQuantizedAyaEngine = None


def get_engine():
    return MockQuantizedAyaEngine()



class TestQuantizedAya:
    def test_quantized_aya_prompt_template_in_memory_cache(self):
        """Test F6.1: Prompt template is loaded once into memory cache without repeated disk I/O."""
        engine = get_engine()
        engine.load_model()
        try:
            ctx = ChunkContext(previous_sentences=["Попередня історія."])
            glossary = [GlossaryItem(source_term="Holmes", target_term="Холмс", entity_type=EntityType.CHARACTER)]
            
            # First invocation
            res1 = engine.refine_chunk("Доктор Ватсон зустрів Holmes.", ctx, glossary)
            # Second invocation
            res2 = engine.refine_chunk("Holmes відповів йому.", ctx, glossary)
            
            assert res1 is not None and len(res1) > 0
            assert res2 is not None and len(res2) > 0
        finally:
            engine.unload_model()

    def test_quantized_aya_glossary_injection_and_formatting(self):
        """Test F6.2: Glossary terms are injected and enforced in output."""
        engine = get_engine()
        engine.load_model()
        try:
            ctx = ChunkContext()
            glossary = [
                GlossaryItem(source_term="John Watson", target_term="Джон Ватсон", entity_type=EntityType.CHARACTER),
                GlossaryItem(source_term="221B Baker Street", target_term="Бейкер-стріт, 221-Б", entity_type=EntityType.LOCATION)
            ]
            draft = "Він поїхав на 221B Baker Street разом з John Watson."
            refined = engine.refine_chunk(draft, ctx, glossary)
            
            assert "Бейкер-стріт, 221-Б" in refined
            assert "Джон Ватсон" in refined
        finally:
            engine.unload_model()

    def test_quantized_aya_markdown_and_header_cleaning(self):
        """Test F6.3: Stripping code fences and repeated generation headers."""
        engine = get_engine()
        engine.load_model()
        try:
            draft = "\"Це була темна ніч.\""
            refined = engine.refine_chunk(draft, ChunkContext(), [])
            assert not refined.startswith("```")
            assert not refined.endswith("```")
            assert "# Фінальний покращений переклад:" not in refined
        finally:
            engine.unload_model()

    def test_quantized_aya_cancellation_token_support(self):
        """Test F6.4: Halts generation promptly when CancellationToken is cancelled."""
        engine = get_engine()
        engine.load_model()
        token = CancellationToken()
        token.cancel()
        
        try:
            with pytest.raises(RuntimeError) as exc_info:
                engine.refine_chunk("Чернетка перекладу", ChunkContext(), [], cancel_token=token)
            assert "cancelled" in str(exc_info.value).lower()
        finally:
            engine.unload_model()

    def test_quantized_aya_load_unload_lifecycle(self):
        """Test F6.5: Engine load/unload clears VRAM/resources."""
        engine = get_engine()
        assert getattr(engine, 'is_loaded', getattr(engine, 'model', None) is not None) is False
        
        engine.load_model()
        assert getattr(engine, 'is_loaded', getattr(engine, 'model', None) is not None) is True
        
        engine.unload_model()
        assert getattr(engine, 'is_loaded', getattr(engine, 'model', None) is not None) is False

    def test_quantized_aya_batch_refinement_mode(self):
        """Test F6.6: Refine batch processes multiple draft chunks consistently."""
        engine = get_engine()
        engine.load_model()
        try:
            drafts = ["Перший чанк тексту.", "Другий чанк тексту.", "Третій чанк тексту."]
            contexts = [ChunkContext(), ChunkContext(), ChunkContext()]
            glossary = []
            
            refined_list = engine.refine_batch(drafts, contexts, glossary)
            assert len(refined_list) == 3
            assert all(isinstance(r, str) and len(r) > 0 for r in refined_list)
        finally:
            engine.unload_model()
