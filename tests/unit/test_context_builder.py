"""
Unit tests for ContextBuilder and PromptContext (Milestone 3).
"""
import pytest
from uuid import uuid4
from unittest.mock import MagicMock
from pathlib import Path

from src.context.builder import ContextBuilder, PromptContext
from src.context.token_budget import TokenBudget
from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.domain.models.knowledge import EntityProfile, ScopeLevel, GlossaryItem


def test_context_builder_formats_entities_with_gender_and_forbidden():
    """
    Asserts format_entity properly includes gender, locked flags, and forbidden variants.
    """
    cb = ContextBuilder()

    entity = EntityProfile(
        id=uuid4(),
        source_name="Cherry",
        canonical_target="Черрі",
        aliases=["Cherry-pie"],
        allowed_target_forms=["Черрі", "Чері"],
        forbidden_target_forms=["Вишня", "Вішня"],
        grammatical_gender="жіночий",
        scope=ScopeLevel.BOOK,
        locked=True,
        confidence=0.95,
    )

    formatted = cb.format_entity(entity)
    assert "- Cherry => Черрі" in formatted
    assert "(рід: жіночий)" in formatted
    assert "[ОБОВ'ЯЗКОВО]" in formatted
    assert "ЗАБОРОНЕНО: Вишня, Вішня" in formatted
    assert "Дозволені форми: Чері" in formatted


def test_context_builder_formats_unreviewed_entity():
    """
    Asserts format_entity handles unreviewed GlossaryItems without forcing a stub mapping.
    """
    cb = ContextBuilder()

    item = GlossaryItem(
        source_term="Eldoria",
        target_term="Eldoria",
        reviewed=False,
    )

    formatted = cb.format_entity(item)
    assert "рекомендовано однаковий узгоджений переклад та транслітерацію" in formatted


def test_context_builder_retrieves_localized_entities_from_kb():
    """
    Asserts ContextBuilder retrieves localized entities for segment via kb_repo.
    """
    mock_repo = MagicMock()
    seg_id = uuid4()
    book_id = uuid4()

    mock_entity = EntityProfile(
        id=uuid4(),
        source_name="Arthur",
        canonical_target="Артур",
        grammatical_gender="чоловічий",
        scope=ScopeLevel.BOOK,
        locked=True,
    )
    mock_repo.get_entities_for_segment.return_value = [mock_entity]

    cb = ContextBuilder(kb_repo=mock_repo)

    segment = TranslationSegment(
        id=seg_id,
        book_id=book_id,
        chapter_id=uuid4(),
        paragraph_id=seg_id,
        source_text="Arthur walked into the dark cavern.",
        draft_translation="Артур зайшов у темну печеру.",
    )

    ctx = cb.build_context(segment)

    mock_repo.get_entities_for_segment.assert_called_once_with(
        segment_id=seg_id,
        book_id=str(book_id),
    )
    assert len(ctx.active_entities) == 1
    assert "Arthur => Артур" in ctx.entities_text
    assert "(рід: чоловічий)" in ctx.entities_text


def test_context_builder_extracts_last_two_approved_paragraphs():
    """
    Asserts extract_approved_history takes only up to 2 approved Ukrainian translations.
    """
    cb = ContextBuilder()
    book_id = uuid4()
    chap_id = uuid4()

    # 4 segments: 1 accepted, 1 pending, 1 failed, 1 accepted
    s1 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=chap_id,
        paragraph_id=uuid4(),
        source_text="First paragraph.",
        refined_translation="Перший відредагований параграф.",
        status=SegmentStatus.ACCEPTED,
    )
    s2 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=chap_id,
        paragraph_id=uuid4(),
        source_text="Second paragraph.",
        draft_translation="Другий чорновик.",
        status=SegmentStatus.PENDING,
    )
    s3 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=chap_id,
        paragraph_id=uuid4(),
        source_text="Third paragraph.",
        draft_translation="Третій чорновик.",
        status=SegmentStatus.FAILED,
    )
    s4 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=chap_id,
        paragraph_id=uuid4(),
        source_text="Fourth paragraph.",
        refined_translation="Четвертий відредагований параграф.",
        status=SegmentStatus.ACCEPTED,
    )

    history = cb.extract_approved_history([s1, s2, s3, s4], max_paragraphs=2)
    assert len(history) == 2
    assert history[0] == "Перший відредагований параграф."
    assert history[1] == "Четвертий відредагований параграф."


def test_context_builder_generates_valid_prompt_v2():
    """
    Asserts generated prompt adheres to Prompt V2 schema and eliminates 1:1 sentence constraints.
    """
    cb = ContextBuilder()
    seg = TranslationSegment(
        id=uuid4(),
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="He ran. She walked.",
        draft_translation="Він біг. Вона йшла.",
        status=SegmentStatus.DRAFT_COMPLETED,
    )

    ctx = cb.build_context(
        segment=seg,
        summary="Chapter overview: escape through forest."
    )

    prompt = ctx.prompt_text
    # Check section headers
    assert "=== ПОПЕРЕДНІЙ КОНТЕКСТ ===" in prompt
    assert "=== ГЛОСАРІЙ ТА СУТНОСТІ ===" in prompt
    assert "=== КОРОТКИЙ ЗМІСТ ГЛАВИ ===" in prompt
    assert "=== ПАРАГРАФ ДЛЯ РЕДАГУВАННЯ ===" in prompt
    assert "=== ВІДРЕДАГОВАНИЙ JSON ===" in prompt

    # Verify JSON schema expectation
    assert '"segments"' in prompt
    assert f"[ID: {seg.id}]" in prompt
    assert "He ran. She walked." in prompt
    assert "Він біг. Вона йшла." in prompt
    assert "Chapter overview: escape through forest." in prompt

    # Verify absence of legacy 1:1 sentence constraint
    assert "Відповідність 1-до-1" not in prompt
    assert "Одне вхідне речення відповідає одному" not in prompt


def test_context_builder_batch_segments():
    """
    Asserts ContextBuilder supports batching multiple TranslationSegments.
    """
    cb = ContextBuilder()
    book_id = uuid4()
    chap_id = uuid4()

    s1 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=chap_id,
        paragraph_id=uuid4(),
        source_text="First source paragraph.",
        draft_translation="Перший чорновик.",
    )
    s2 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=chap_id,
        paragraph_id=uuid4(),
        source_text="Second source paragraph.",
        draft_translation="Другий чорновик.",
    )

    ctx = cb.build_batch_context([s1, s2])
    assert len(ctx.segment_ids) == 2
    assert str(s1.id) in ctx.prompt_text
    assert str(s2.id) in ctx.prompt_text
    assert ctx.token_breakdown["total_used"] > 0
    assert ctx.budget_allocation is not None


def test_context_builder_empty_segments_raises():
    cb = ContextBuilder()
    with pytest.raises(ValueError, match="empty list of segments"):
        cb.build_batch_context([])
