"""
Comprehensive Unit Tests for Translation Module (Worker 2 ML Pipelines).
Tests DynamicTokenBucketBatcher, CancellationTokenStoppingCriteria, CTranslate2NLLBEngine,
QuantizedAyaEditingEngine, and TwoStageTranslationPipeline.
"""
import pytest
import threading
from uuid import uuid4, UUID
from unittest.mock import MagicMock, patch
from typing import List

from src.domain.models.document import Sentence
from src.domain.models.chunk import TranslationChunk, ChunkStatus, ChunkContext
from src.domain.models.knowledge import GlossaryItem
from src.translation.batching import DynamicTokenBucketBatcher
from src.translation.stopping_criteria import CancellationTokenStoppingCriteria
from src.translation.nllb_engine import CTranslate2NLLBEngine, get_flores_code, FLORES_200_LANG_MAP
from src.translation.aya_editing_engine import QuantizedAyaEditingEngine
from src.translation.pipeline import TwoStageTranslationPipeline, ProgressEvent


class MockCancellationToken:
    def __init__(self, is_cancelled: bool = False):
        self._cancelled = is_cancelled

    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self):
        self._cancelled = True


# ============================================================================
# 1. DynamicTokenBucketBatcher Tests
# ============================================================================

def test_dynamic_batcher_empty_items():
    batcher = DynamicTokenBucketBatcher(max_batch_tokens=2048)
    assert batcher.create_buckets([]) == []
    assert batcher.create_batches([]) == []
    assert batcher.batch_and_translate([], lambda x: x) == {}


def test_dynamic_batcher_grouping_by_length():
    batcher = DynamicTokenBucketBatcher(max_batch_tokens=100)
    # Texts with known word counts
    texts = [
        "short",                               # ~3 tokens
        "another short sentence",              # ~5 tokens
        "this is a much longer sentence containing more tokens to test bucket division", # ~16 tokens
        "medium length sentence here",         # ~6 tokens
        "hi",                                  # ~3 tokens
    ]
    buckets = batcher.create_batches(texts, max_tokens=25)
    # Check that all sentences are included
    flattened = [s for b in buckets for s in b]
    assert len(flattened) == len(texts)
    assert set(flattened) == set(texts)


def test_dynamic_batcher_batch_and_translate_order_preservation():
    batcher = DynamicTokenBucketBatcher(max_batch_tokens=100)
    indexed_items = [
        ((1, 0), "Apple"),
        ((1, 1), "Banana"),
        ((2, 0), "A very long sentence about delicious tropical fresh juicy oranges and pineapples."),
        ((2, 1), "Cat"),
    ]

    def mock_translate_fn(texts: List[str]) -> List[str]:
        return [f"Translated({t})" for t in texts]

    results = batcher.batch_and_translate(indexed_items, mock_translate_fn)

    assert len(results) == 4
    assert results[(1, 0)] == "Translated(Apple)"
    assert results[(1, 1)] == "Translated(Banana)"
    assert "Translated(A very long" in results[(2, 0)]
    assert results[(2, 1)] == "Translated(Cat)"


def test_dynamic_batcher_cancellation():
    batcher = DynamicTokenBucketBatcher(max_batch_tokens=10)
    # Many items so multiple batches are created
    indexed_items = [((i, 0), f"Sentence number {i} is here.") for i in range(50)]
    cancel_token = MockCancellationToken(is_cancelled=True)

    results = batcher.batch_and_translate(indexed_items, lambda x: x, cancel_token=cancel_token)
    # Should halt immediately
    assert len(results) == 0


# ============================================================================
# 2. CancellationTokenStoppingCriteria Tests
# ============================================================================

def test_stopping_criteria_with_cancellation_token():
    token = MockCancellationToken(is_cancelled=False)
    criteria = CancellationTokenStoppingCriteria(token)
    assert criteria(None, None) is False

    token.cancel()
    assert criteria(None, None) is True


def test_stopping_criteria_with_threading_event():
    event = threading.Event()
    criteria = CancellationTokenStoppingCriteria(event)
    assert criteria(None, None) is False

    event.set()
    assert criteria(None, None) is True


def test_stopping_criteria_none_token():
    criteria = CancellationTokenStoppingCriteria(None)
    assert criteria(None, None) is False


# ============================================================================
# 3. CTranslate2NLLBEngine Tests
# ============================================================================

def test_flores_mapping():
    assert get_flores_code("en") == "eng_Latn"
    assert get_flores_code("uk") == "ukr_Cyrl"
    assert get_flores_code("de") == "deu_Latn"
    assert get_flores_code("fr") == "fra_Latn"
    assert get_flores_code("pl") == "pol_Latn"
    assert get_flores_code("eng_Latn") == "eng_Latn"
    assert get_flores_code("") == "ukr_Cyrl"


def test_nllb_engine_init_and_device():
    engine = CTranslate2NLLBEngine(device="cpu", compute_type="int8")
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"


def test_nllb_engine_empty_batch():
    engine = CTranslate2NLLBEngine(device="cpu")
    assert engine.translate_batch([], "en", "uk") == []


def test_nllb_engine_unload():
    engine = CTranslate2NLLBEngine(device="cpu")
    engine.translator = MagicMock()
    engine.tokenizer = MagicMock()
    engine.unload_model()
    assert engine.translator is None
    assert engine.tokenizer is None


# ============================================================================
# 4. QuantizedAyaEditingEngine Tests
# ============================================================================

def test_aya_prompt_caching(tmp_path):
    prompt_file = tmp_path / "custom_prompt.md"
    prompt_file.write_text("Custom prompt: {draft_text}", encoding="utf-8")

    engine = QuantizedAyaEditingEngine(prompt_path=prompt_file, device="cpu", load_in_4bit=False)
    assert engine.cached_prompt_template == "Custom prompt: {draft_text}"

    # Test reload
    prompt_file.write_text("Updated prompt: {draft_text}", encoding="utf-8")
    engine.reload_prompt()
    assert engine.cached_prompt_template == "Updated prompt: {draft_text}"


def test_aya_sanitize_output():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)

    # Test markdown fence stripping
    fenced_text = "```markdown\nЦе відредагований текст.\n```"
    assert engine._sanitize_output(fenced_text) == "Це відредагований текст."

    # Test header prefix stripping
    header_text = "# Фінальний покращений переклад:\nГарний вечір."
    assert engine._sanitize_output(header_text) == "Гарний вечір."


def test_aya_refine_chunk_cancelled():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    token = MockCancellationToken(is_cancelled=True)
    draft = "Чорновий варіант"
    context = ChunkContext(previous_sentences=["Попереднє речення"])
    glossary = [GlossaryItem(source_term="AI", target_term="ШІ")]

    result = engine.refine_chunk(draft, context, glossary, cancel_token=token)
    # Should immediately return the draft text unchanged
    assert result == draft


def test_aya_parse_json_response_valid_json():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    raw = '{"1": "Перше речення.", "2": "Друге речення."}'
    parsed = engine._parse_json_response(raw, expected_count=2, fallback_drafts={1: "Draft 1", 2: "Draft 2"})
    assert parsed == {1: "Перше речення.", 2: "Друге речення."}


def test_aya_parse_json_response_with_markdown_fences():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    raw = "```json\n{\n  \"1\": \"Відредагований текст 1.\",\n  \"2\": \"Відредагований текст 2.\"\n}\n```"
    parsed = engine._parse_json_response(raw, expected_count=2, fallback_drafts={1: "Draft 1", 2: "Draft 2"})
    assert parsed == {1: "Відредагований текст 1.", 2: "Відредагований текст 2."}


def test_aya_parse_json_response_broken_syntax_and_missing_keys():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    # Partial broken JSON: key 1 is valid, key 2 is missing, key 3 is broken
    raw = '{"1": "Тільки перше", "other": "невалідний ключ"}'
    fallbacks = {1: "Draft 1", 2: "Draft 2", 3: "Draft 3"}
    parsed = engine._parse_json_response(raw, expected_count=3, fallback_drafts=fallbacks)
    assert parsed[1] == "Тільки перше"
    assert parsed[2] == "Draft 2"
    assert parsed[3] == "Draft 3"


def test_aya_parse_json_response_completely_invalid_returns_fallbacks():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    raw = "Це не JSON взагалі, а просто випадковий текст помилки моделі."
    fallbacks = {1: "Draft 1", 2: "Draft 2"}
    parsed = engine._parse_json_response(raw, expected_count=2, fallback_drafts=fallbacks)
    assert parsed == {1: "Draft 1", 2: "Draft 2"}


def test_aya_parse_json_response_single_quotes_and_numbered_list():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    # Single quotes
    raw_single = "{'1': 'Речення 1.', '2': 'Речення 2.'}"
    parsed_single = engine._parse_json_response(raw_single, expected_count=2, fallback_drafts={1: "D1", 2: "D2"})
    assert parsed_single == {1: "Речення 1.", 2: "Речення 2."}

    # Numbered list format
    raw_numbered = '1. "Перше речення"\n2. "Друге речення"'
    parsed_num = engine._parse_json_response(raw_numbered, expected_count=2, fallback_drafts={1: "D1", 2: "D2"})
    assert parsed_num == {1: "Перше речення", 2: "Друге речення"}


def test_aya_parse_json_response_with_unescaped_internal_quotes():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    raw = '{\n  "1": "Він сказав: "Стій!" і пішов далі.",\n  "2": "Вона не відповіла."\n}'
    parsed = engine._parse_json_response(raw, expected_count=2, fallback_drafts={1: "D1", 2: "D2"})
    assert parsed[1] == 'Він сказав: "Стій!" і пішов далі.'
    assert parsed[2] == "Вона не відповіла."


def test_aya_parse_json_response_multiline_and_zero_indexing():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    # Multiline string with literal newline
    raw_multiline = '{\n  "1": "Рядок 1.\nРядок 2.",\n  "2": "Речення 2."\n}'
    parsed_multi = engine._parse_json_response(raw_multiline, expected_count=2, fallback_drafts={1: "D1", 2: "D2"})
    assert parsed_multi[1] == "Рядок 1.\nРядок 2."
    assert parsed_multi[2] == "Речення 2."

    # 0-indexed dictionary
    raw_zero = '{"0": "Перше", "1": "Друге"}'
    parsed_zero = engine._parse_json_response(raw_zero, expected_count=2, fallback_drafts={1: "D1", 2: "D2"})
    assert parsed_zero == {1: "Перше", 2: "Друге"}


def test_aya_parse_json_response_arrays_and_wrapped_objects():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    # Direct JSON array
    raw_arr = '["Перше речення.", "Друге речення."]'
    parsed_arr = engine._parse_json_response(raw_arr, expected_count=2, fallback_drafts={1: "D1", 2: "D2"})
    assert parsed_arr == {1: "Перше речення.", 2: "Друге речення."}

    # Wrapped in root object (e.g. {"translations": [...]})
    raw_wrapped_arr = '{"translations": ["Перше речення.", "Друге речення."]}'
    parsed_w_arr = engine._parse_json_response(raw_wrapped_arr, expected_count=2, fallback_drafts={1: "D1", 2: "D2"})
    assert parsed_w_arr == {1: "Перше речення.", 2: "Друге речення."}

    # Wrapped in root object (e.g. {"sentences": {"1": ...}})
    raw_wrapped_dict = '{"sentences": {"1": "Перше.", "2": "Друге."}}'
    parsed_w_dict = engine._parse_json_response(raw_wrapped_dict, expected_count=2, fallback_drafts={1: "D1", 2: "D2"})
    assert parsed_w_dict == {1: "Перше.", 2: "Друге."}

    # Dict of dicts (e.g. {"1": {"uk": "Перше.", "en": "First."}})
    raw_dict_of_dicts = '{"1": {"uk": "Перше.", "en": "First."}, "2": {"translation": "Друге."}}'
    parsed_nested = engine._parse_json_response(raw_dict_of_dicts, expected_count=2, fallback_drafts={1: "D1", 2: "D2"})
    assert parsed_nested == {1: "Перше.", 2: "Друге."}

    # Single sentence with "translation" key
    raw_single_key = '{"translation": "Єдине відредаговане речення."}'
    parsed_single = engine._parse_json_response(raw_single_key, expected_count=1, fallback_drafts={1: "Draft"})
    assert parsed_single == {1: "Єдине відредаговане речення."}


def test_translation_interface_type_hints():
    import typing
    from src.domain.interfaces.translation import IEditingEngine
    hints = typing.get_type_hints(IEditingEngine.refine_chunk_structured)
    assert "return" in hints
    assert hints["return"] == typing.Dict[int, str]


def test_aya_refine_chunk_structured_cancelled():
    engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)
    token = MockCancellationToken(is_cancelled=True)
    sentences = [
        Sentence(id=uuid4(), original_text="Hello.", translated_text="Привіт.", order_index=0),
        Sentence(id=uuid4(), original_text="World.", translated_text="Світ.", order_index=1),
    ]
    res = engine.refine_chunk_structured(sentences, cancel_token=token)
    assert res == {1: "Привіт.", 2: "Світ."}


# ============================================================================
# 5. TwoStageTranslationPipeline Tests
# ============================================================================

class MockKBRepository:
    def __init__(self, chunks: List[TranslationChunk]):
        self.chunks_by_id = {c.id: c for c in chunks}
        self.save_calls: List[TranslationChunk] = []
        self.batch_save_calls: List[List[TranslationChunk]] = []

    def load_chunks_by_status(self, book_id: UUID, status: ChunkStatus):
        return [c for c in self.chunks_by_id.values() if c.book_id == book_id and c.status == status]

    def save_chunk_state(self, chunk: TranslationChunk):
        self.save_calls.append(chunk)
        self.chunks_by_id[chunk.id] = chunk

    def save_chunk_state_batch(self, chunks: List[TranslationChunk]):
        self.batch_save_calls.append(list(chunks))
        for c in chunks:
            self.chunks_by_id[c.id] = c

    def get_glossary(self):
        return [GlossaryItem(source_term="test", target_term="тест")]


def test_pipeline_two_stage_execution():
    book_id = uuid4()
    chunks = []
    for i in range(5):
        sents = [Sentence(id=uuid4(), original_text=f"Sentence {j} of chunk {i}", order_index=j) for j in range(3)]
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book_id,
            chapter_id=uuid4(),
            paragraph_indices=[i],
            source_sentences=sents,
            token_count=30,
            context=ChunkContext(),
            status=ChunkStatus.PENDING
        )
        chunks.append(chunk)

    kb_repo = MockKBRepository(chunks)

    mock_nllb = MagicMock()
    mock_nllb.translate_batch.side_effect = lambda texts, **kwargs: [f"[UK] {t}" for t in texts]

    mock_aya = MagicMock()
    mock_aya.refine_chunk.side_effect = lambda draft_translation="", *args, **kwargs: f"[REFINED] {draft_translation}"

    events = []
    pipeline = TwoStageTranslationPipeline(mock_nllb, mock_aya, kb_repo)

    # Execute Stage 1
    pipeline.execute_nllb_stage(book_id, progress_cb=events.append)

    # Check that chunks are transitioned to DRAFT_COMPLETED
    draft_chunks = kb_repo.load_chunks_by_status(book_id, ChunkStatus.DRAFT_COMPLETED)
    assert len(draft_chunks) == 5
    for c in draft_chunks:
        assert c.draft_translation is not None
        assert "[UK]" in c.draft_translation

    # Execute Stage 2
    pipeline.execute_aya_stage(book_id, progress_cb=events.append)

    # Check that chunks are transitioned to REFINED
    refined_chunks = kb_repo.load_chunks_by_status(book_id, ChunkStatus.REFINED)
    assert len(refined_chunks) == 5
    for c in refined_chunks:
        assert c.final_translation is not None
        assert "[REFINED]" in c.final_translation

    # Verify progress events
    assert len(events) == 10  # 5 in stage 1, 5 in stage 2
    assert all(isinstance(e, ProgressEvent) for e in events)


def test_pipeline_cancellation_stage1():
    book_id = uuid4()
    chunks = [
        TranslationChunk(
            id=uuid4(),
            book_id=book_id,
            chapter_id=uuid4(),
            paragraph_indices=[0],
            source_sentences=[Sentence(id=uuid4(), original_text="Hello world.", order_index=0)],
            token_count=10,
            context=ChunkContext(),
            status=ChunkStatus.PENDING
        )
    ]
    kb_repo = MockKBRepository(chunks)
    mock_nllb = MagicMock()
    mock_aya = MagicMock()

    pipeline = TwoStageTranslationPipeline(mock_nllb, mock_aya, kb_repo)
    cancel_token = MockCancellationToken(is_cancelled=True)

    pipeline.execute_nllb_stage(book_id, cancel_token=cancel_token)
    # Since cancelled immediately, chunk should remain pending or not translated
    assert len(kb_repo.load_chunks_by_status(book_id, ChunkStatus.REFINED)) == 0
