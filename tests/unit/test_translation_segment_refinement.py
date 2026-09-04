"""
Unit tests for TranslationSegment and Paragraph-level Aya Refinement (Milestone 3).
Tests structured JSON I/O, visible failure enforcement, and pipeline integration.
"""
import pytest
import json
from uuid import uuid4, UUID
from unittest.mock import MagicMock, patch

from src.domain.models.segment import (
    TranslationSegment,
    SegmentStatus,
    IllegalStateTransitionError,
)
from src.domain.models.document import Paragraph, Sentence
from src.translation.aya_editing_engine import (
    QuantizedAyaEditingEngine,
    SegmentRefinementResult,
)
from src.context.builder import ContextBuilder, PromptContext
from src.launcher.concurrency import CancellationToken


# ============================================================================
# 1. TranslationSegment Model & Lifecycle Tests
# ============================================================================

def test_translation_segment_draft_compatibility():
    seg_id = uuid4()
    # Test draft_text provided in constructor
    seg1 = TranslationSegment(
        id=seg_id,
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=seg_id,
        source_text="Source paragraph text.",
        draft_text="Чорновий переклад.",
    )
    assert seg1.draft_text == "Чорновий переклад."
    assert seg1.draft_translation == "Чорновий переклад."

    # Test setter
    seg1.draft_text = "Оновлений чорновик."
    assert seg1.draft_text == "Оновлений чорновик."
    assert seg1.draft_translation == "Оновлений чорновик."

    # Test translated_text precedence
    assert seg1.translated_text == "Оновлений чорновик."
    seg1.refined_translation = "Відредагований текст."
    assert seg1.translated_text == "Відредагований текст."
    seg1.final_translation = "Фінальний текст."
    assert seg1.translated_text == "Фінальний текст."


def test_translation_segment_lifecycle_transitions():
    seg = TranslationSegment(
        id=uuid4(),
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="Source text.",
        status=SegmentStatus.PENDING,
    )

    # Valid path: PENDING -> DRAFT_COMPLETED -> EDITED -> ACCEPTED
    seg.transition_to(SegmentStatus.DRAFT_COMPLETED)
    assert seg.status == SegmentStatus.DRAFT_COMPLETED

    seg.transition_to(SegmentStatus.EDITED)
    assert seg.status == SegmentStatus.EDITED

    seg.transition_to(SegmentStatus.ACCEPTED)
    assert seg.status == SegmentStatus.ACCEPTED

    # Invalid transition from ACCEPTED
    with pytest.raises(IllegalStateTransitionError):
        seg.transition_to(SegmentStatus.PENDING)


def test_translation_segment_from_paragraph_factory():
    s1 = Sentence(
        original_text="The hero entered the hall.",
        normalized_source_text="The hero entered the hall.",
        translated_text="Герой увійшов до зали.",
    )
    s2 = Sentence(
        original_text="He was 6 feet tall.",
        normalized_source_text="He was 1.8 meters tall.",
        translated_text="Він був 1,8 метра на зріст.",
    )
    para = Paragraph(sentences=[s1, s2])
    book_id = uuid4()
    chap_id = uuid4()

    seg = TranslationSegment.from_paragraph(para, book_id=book_id, chapter_id=chap_id, order_index=1)
    assert seg.paragraph_id == para.id
    assert seg.id == para.id
    assert seg.book_id == book_id
    assert seg.chapter_id == chap_id
    assert seg.order_index == 1
    assert "The hero entered the hall." in seg.source_text
    assert "6 feet tall" in seg.source_text
    assert "1.8 meters tall" in seg.normalized_text
    assert seg.sentence_ids == [s1.id, s2.id]
    assert seg.draft_translation == "Герой увійшов до зали. Він був 1,8 метра на зріст."
    assert seg.status == SegmentStatus.DRAFT_COMPLETED


# ============================================================================
# 2. Structured Aya Refinement & Multi-Strategy JSON Parsing Tests
# ============================================================================

def test_aya_refine_segments_structured_valid_json():
    engine = QuantizedAyaEditingEngine(model_path=None, gguf_path=None)
    seg_id = uuid4()
    seg = TranslationSegment(
        id=seg_id,
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=seg_id,
        source_text="He ran into the forest. She followed him closely.",
        draft_translation="Він побіг до лісу. Вона слідувала за ним близько.",
        status=SegmentStatus.DRAFT_COMPLETED,
    )

    expected_translation = "Він побіг углиб лісу, а вона квапилася слідом за ним."
    json_output = json.dumps({
        "segments": [
            {
                "id": str(seg_id),
                "translation": expected_translation
            }
        ]
    })

    with patch.object(engine, "_load_cached_prompt", return_value="{target_segments_block}"):
        with patch.object(engine, "load_model"):
            engine.model = MagicMock()  # Mock model loaded
            with patch.object(engine, "_parse_segments_json_response", wraps=engine._parse_segments_json_response) as mock_parser:
                # Mock inference output
                engine.model = None
                engine.gguf_llm = MagicMock()
                engine.gguf_llm.create_chat_completion.return_value = [
                    {"choices": [{"delta": {"content": json_output}}]}
                ]

                results = engine.refine_segments_structured([seg])

                assert seg.status == SegmentStatus.EDITED
                assert seg.refined_translation == expected_translation
                assert seg.error_message is None
                assert results[seg.id].success is True
                assert results[seg.id].refined_text == expected_translation


def test_aya_refine_segments_structured_markdown_fence():
    engine = QuantizedAyaEditingEngine(model_path=None, gguf_path=None)
    seg_id = uuid4()
    seg = TranslationSegment(
        id=seg_id,
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=seg_id,
        source_text="Source paragraph.",
        draft_translation="Чорновик.",
        status=SegmentStatus.DRAFT_COMPLETED,
    )

    fenced_output = f"""```json
{{
  "segments": [
    {{
      "id": "{seg_id}",
      "translation": "Чистий відредагований текст без огорож."
    }}
  ]
}}
```"""

    engine.gguf_llm = MagicMock()
    engine.gguf_llm.create_chat_completion.return_value = [
        {"choices": [{"delta": {"content": fenced_output}}]}
    ]

    results = engine.refine_segments_structured([seg])
    assert seg.status == SegmentStatus.EDITED
    assert seg.refined_translation == "Чистий відредагований текст без огорож."
    assert results[seg.id].success is True


def test_aya_refine_segments_structured_unescaped_internal_dialogue_quotes():
    engine = QuantizedAyaEditingEngine(model_path=None, gguf_path=None)
    seg_id = uuid4()
    seg = TranslationSegment(
        id=seg_id,
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=seg_id,
        source_text="He said: 'Hello!', and left.",
        draft_translation="Він сказав: 'Привіт!', і пішов.",
        status=SegmentStatus.DRAFT_COMPLETED,
    )

    # Notice internal unescaped quotes in translation
    malformed_quote_output = f'{{"segments": [{{"id": "{seg_id}", "translation": "— Зачекай! — гукнула вона, але він не зупинився."}}]}}'

    engine.gguf_llm = MagicMock()
    engine.gguf_llm.create_chat_completion.return_value = [
        {"choices": [{"delta": {"content": malformed_quote_output}}]}
    ]

    results = engine.refine_segments_structured([seg])
    assert seg.status == SegmentStatus.EDITED
    assert "— Зачекай! — гукнула вона" in seg.refined_translation
    assert results[seg.id].success is True


# ============================================================================
# 3. Mandatory Integrity Requirement F14: Visible Failure Diagnostics
# ============================================================================

def test_aya_refine_segments_structured_visible_failure_on_malformed_output():
    """
    CRITICAL INTEGRITY TEST:
    A malformed or unparseable output must explicitly transition segment to FAILED.
    Under NO circumstances may it silently copy draft to refined_translation or mark success!
    """
    engine = QuantizedAyaEditingEngine(model_path=None, gguf_path=None)
    seg_id = uuid4()
    seg = TranslationSegment(
        id=seg_id,
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=seg_id,
        source_text="Critical sentence to be protected.",
        draft_translation="Оригінальний чорновик NLLB.",
        status=SegmentStatus.DRAFT_COMPLETED,
    )

    # Garbage LLM output
    garbage_output = "I am an AI assistant. I cannot translate this text because of copyright reasons."

    engine.gguf_llm = MagicMock()
    engine.gguf_llm.create_chat_completion.return_value = [
        {"choices": [{"delta": {"content": garbage_output}}]}
    ]

    results = engine.refine_segments_structured([seg])

    # Assert visible failure invariants
    assert seg.status == SegmentStatus.FAILED
    assert seg.refined_translation is None  # Must NOT be set to draft
    assert seg.error_message is not None
    assert "Stage 2 LLM parsing failure" in seg.error_message
    assert results[seg.id].success is False
    assert results[seg.id].fallback_draft == "Оригінальний чорновик NLLB."


def test_aya_refine_segments_structured_missing_segment_id():
    """
    When LLM returns JSON with an unexpected segment ID, target segment becomes FAILED.
    """
    engine = QuantizedAyaEditingEngine(model_path=None, gguf_path=None)
    target_seg_id = uuid4()
    seg = TranslationSegment(
        id=target_seg_id,
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=target_seg_id,
        source_text="Expected source.",
        draft_translation="Expected draft.",
        status=SegmentStatus.DRAFT_COMPLETED,
    )

    different_id = uuid4()
    wrong_id_output = json.dumps({
        "segments": [
            {
                "id": str(different_id),
                "translation": "Some translation for an unexpected ID."
            }
        ]
    })

    engine.gguf_llm = MagicMock()
    engine.gguf_llm.create_chat_completion.return_value = [
        {"choices": [{"delta": {"content": wrong_id_output}}]}
    ]

    results = engine.refine_segments_structured([seg])

    assert seg.status == SegmentStatus.FAILED
    assert seg.refined_translation is None
    assert results[seg.id].success is False
    assert "No valid translation found" in seg.error_message


def test_aya_refine_segments_structured_cancellation():
    """
    Cancelling execution halts generation cleanly and sets FAILED status with diagnostic.
    """
    engine = QuantizedAyaEditingEngine(model_path=None, gguf_path=None)
    seg_id = uuid4()
    seg = TranslationSegment(
        id=seg_id,
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=seg_id,
        source_text="Text to be cancelled.",
        draft_translation="Draft before cancel.",
        status=SegmentStatus.DRAFT_COMPLETED,
    )

    cancel_tok = CancellationToken()
    cancel_tok.cancel()

    results = engine.refine_segments_structured([seg], cancel_token=cancel_tok)

    assert seg.status == SegmentStatus.FAILED
    assert seg.refined_translation is None
    assert "cancelled" in seg.error_message.lower()
    assert results[seg.id].success is False


# ============================================================================
# 4. Pipeline Paragraph-Level Execution Flow
# ============================================================================

def test_pipeline_segment_stages_e2e_flow():
    from src.translation.pipeline import TwoStageTranslationPipeline

    mock_nllb = MagicMock()
    mock_nllb.translate_batch.side_effect = lambda texts, **kwargs: [f"UK: {t}" for t in texts]

    mock_aya = MagicMock()
    mock_repo = MagicMock()

    pipeline = TwoStageTranslationPipeline(
        nllb_engine=mock_nllb,
        aya_engine=mock_aya,
        kb_repo=mock_repo
    )

    book_id = uuid4()
    chap_id = uuid4()

    s1 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=chap_id,
        paragraph_id=uuid4(),
        source_text="Sentence one. Sentence two.",
        status=SegmentStatus.PENDING,
    )

    # Stage 1: execute_segment_nllb_stage
    res_stage1 = pipeline.execute_segment_nllb_stage(book_id, segments=[s1])
    assert s1.status == SegmentStatus.DRAFT_COMPLETED
    assert s1.draft_translation == "UK: Sentence one. UK: Sentence two."
    assert s1.draft_text == "UK: Sentence one. UK: Sentence two."

    # Stage 2: execute_segment_aya_stage
    def mock_refine(segments, prompt_context=None, **kwargs):
        for s in segments:
            s.refined_translation = "Літературний український переклад двох речень в одному."
            s.transition_to(SegmentStatus.EDITED)
        return {s.id: SegmentRefinementResult("...", s.id, True) for s in segments}

    mock_aya.refine_segments_structured.side_effect = mock_refine

    res_stage2 = pipeline.execute_segment_aya_stage(book_id, segments=[s1])
    assert s1.status == SegmentStatus.EDITED
    assert s1.refined_translation == "Літературний український переклад двох речень в одному."
