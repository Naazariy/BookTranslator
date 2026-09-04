"""
tests/unit/test_bounded_repair_engine.py

Unit tests for BoundedRepairEngine: targeted corrective prompts,
2-retry bound, state transitions (ACCEPTED vs REVIEW_REQUIRED),
and Sentence.original_text immutability.
"""
from uuid import uuid4
import pytest
from pydantic import ValidationError

from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.domain.models.document import Sentence
from src.domain.models.knowledge import EntityProfile
from src.context.builder import PromptContext
from src.quality.models import QAReport, QAViolation, QASeverity
from src.quality.pipeline import QualityPipeline
from src.quality.repair import BoundedRepairEngine
from src.launcher.concurrency import CancellationToken


def test_synthesize_repair_prompt_contains_diagnostics():
    engine = BoundedRepairEngine()
    seg = TranslationSegment(
        id=uuid4(),
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="Cherry smiled warmly at the boy.",
        draft_translation="Вишня тепло усміхнулася хлопчикові.",
    )
    report = QAReport(segment_id=seg.id)
    report.add_violation(
        QAViolation(
            validator_name="EntityConsistencyValidator",
            rule_code="FORBIDDEN_ENTITY_VARIANT",
            severity=QASeverity.CRITICAL,
            message="Forbidden variant 'Вишня' detected.",
            target_snippet="Вишня тепло усміхнулася",
            forbidden_form="Вишня",
            suggested_fix="Черрі",
        )
    )

    prompt = engine.synthesize_repair_prompt(seg, report, seg.draft_translation)
    assert "Cherry smiled warmly at the boy." in prompt
    assert "Вишня тепло усміхнулася хлопчикові." in prompt
    assert "EntityConsistencyValidator" in prompt
    assert "Вишня" in prompt
    assert "Черрі" in prompt
    assert "ЗАБОРОНЕНО" in prompt
    assert "JSON" in prompt


def test_repair_successful_on_first_attempt():
    qp = QualityPipeline()
    cherry = EntityProfile(
        source_name="Cherry",
        canonical_target="Черрі",
        forbidden_target_forms=["Вишня"],
        locked=True,
    )
    ctx = PromptContext(
        segment_id=uuid4(),
        prompt_text="",
        target_source="Cherry walked into the room.",
        target_draft="",
        previous_context="",
        entities_text="",
        active_entities=[cherry],
    )

    # Initial segment has forbidden variant "Вишня"
    seg = TranslationSegment(
        id=uuid4(),
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="Cherry walked into the room.",
        refined_translation="Вишня зайшла до кімнати.",
        status=SegmentStatus.EDITED,
    )

    initial_report = qp.validate_segment(seg, context=ctx)
    assert not initial_report.is_valid

    # Mock editing engine that corrects "Вишня" -> "Черрі" on attempt 1
    def mock_editing_engine(s, prompt):
        return "Черрі зайшла до кімнати."

    repair_engine = BoundedRepairEngine(editing_engine=mock_editing_engine, quality_pipeline=qp, max_retries=2)
    repaired_seg, final_report = repair_engine.repair_segment(seg, initial_report, context=ctx)

    assert repaired_seg.status == SegmentStatus.ACCEPTED
    assert repaired_seg.final_translation == "Черрі зайшла до кімнати."
    assert repaired_seg.repair_attempts == 1
    assert repaired_seg.error_message is None
    assert final_report.is_valid is True
    assert final_report.status == "ACCEPTED"


def test_repair_successful_on_second_attempt():
    qp = QualityPipeline()
    cherry = EntityProfile(
        source_name="Cherry",
        canonical_target="Черрі",
        forbidden_target_forms=["Вишня", "Вішня"],
        locked=True,
    )
    ctx = PromptContext(
        segment_id=uuid4(),
        prompt_text="",
        target_source="Cherry walked into the room.",
        target_draft="",
        previous_context="",
        entities_text="",
        active_entities=[cherry],
    )

    seg = TranslationSegment(
        id=uuid4(),
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="Cherry walked into the room.",
        refined_translation="Вишня зайшла до кімнати.",
        status=SegmentStatus.EDITED,
    )

    initial_report = qp.validate_segment(seg, context=ctx)
    call_count = 0

    def mock_editing_engine(s, prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt: still produces a forbidden variant "Вішня"
            return "Вішня зайшла до кімнати."
        # Second attempt: properly uses "Черрі"
        return "Черрі зайшла до кімнати."

    repair_engine = BoundedRepairEngine(editing_engine=mock_editing_engine, quality_pipeline=qp, max_retries=2)
    repaired_seg, final_report = repair_engine.repair_segment(seg, initial_report, context=ctx)

    assert call_count == 2
    assert repaired_seg.status == SegmentStatus.ACCEPTED
    assert repaired_seg.final_translation == "Черрі зайшла до кімнати."
    assert repaired_seg.repair_attempts == 2
    assert final_report.is_valid is True


def test_repair_strictly_bounds_retries_and_marks_review_required():
    qp = QualityPipeline()
    cherry = EntityProfile(
        source_name="Cherry",
        canonical_target="Черрі",
        forbidden_target_forms=["Вишня"],
        locked=True,
    )
    ctx = PromptContext(
        segment_id=uuid4(),
        prompt_text="",
        target_source="Cherry walked into the room.",
        target_draft="",
        previous_context="",
        entities_text="",
        active_entities=[cherry],
    )

    seg = TranslationSegment(
        id=uuid4(),
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="Cherry walked into the room.",
        refined_translation="Вишня зайшла до кімнати.",
        status=SegmentStatus.EDITED,
    )

    initial_report = qp.validate_segment(seg, context=ctx)
    call_count = 0

    # Mock editing engine stubbornly keeps returning the forbidden variant
    def stubborn_engine(s, prompt):
        nonlocal call_count
        call_count += 1
        return "Вишня зайшла до кімнати знову."

    repair_engine = BoundedRepairEngine(editing_engine=stubborn_engine, quality_pipeline=qp, max_retries=2)
    repaired_seg, final_report = repair_engine.repair_segment(seg, initial_report, context=ctx)

    # Must NOT exceed 2 retries
    assert call_count == 2
    assert repaired_seg.repair_attempts == 2
    assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
    assert "Quality validation failed" in (repaired_seg.error_message or "")
    assert final_report.is_valid is False
    assert final_report.status == "REVIEW_REQUIRED"


def test_sentence_original_text_immutability_during_repair():
    # Sentence.original_text is frozen=True in document.py
    sent = Sentence(
        id=uuid4(),
        original_text="Original immutable English sentence.",
        order_index=0,
    )

    assert sent.original_text == "Original immutable English sentence."

    # Attempting to mutate original_text must raise ValidationError
    with pytest.raises(ValidationError):
        sent.original_text = "Hacked modified sentence."

    # Verify original_text remains intact
    assert sent.original_text == "Original immutable English sentence."


def test_repair_aborts_on_cancellation():
    qp = QualityPipeline()
    token = CancellationToken()
    token.cancel()

    seg = TranslationSegment(
        id=uuid4(),
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="Some text.",
        refined_translation="",
        status=SegmentStatus.EDITED,
    )
    report = QAReport(segment_id=seg.id, is_valid=False)

    call_count = 0

    def mock_engine(s, p):
        nonlocal call_count
        call_count += 1
        return "Fixed text."

    repair_engine = BoundedRepairEngine(editing_engine=mock_engine, quality_pipeline=qp, max_retries=2)
    repaired_seg, _ = repair_engine.repair_segment(seg, report, cancel_token=token)

    assert call_count == 0
    assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
