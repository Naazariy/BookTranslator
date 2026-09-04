"""
Unit tests for Sentence.original_text immutability, Sentence.normalized_source_text,
7-state SegmentStatus lifecycle, and document writer gating.
"""
from uuid import uuid4
import pytest
from pydantic import ValidationError

from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.segment import (
    SegmentStatus,
    TranslationSegment,
    IllegalStateTransitionError,
    validate_segment_transition,
)
from src.domain.models.chunk import ChunkStatus
from src.preprocessing.unit_converter import UnitConverter
from src.writers.txt_writer import TxtWriter
from src.writers.pdf_writer import PdfWriter


class TestSentenceImmutability:
    """Tests verifying strict immutability of Sentence.original_text."""

    def test_original_text_is_frozen_after_init(self):
        s = Sentence(original_text="The giant stood 80 feet tall.")
        assert s.original_text == "The giant stood 80 feet tall."
        assert s.normalized_source_text is None

        with pytest.raises(ValidationError) as exc_info:
            s.original_text = "The giant stood 24 метри tall."

        assert "frozen" in str(exc_info.value).lower()

    def test_normalized_source_text_is_mutable(self):
        s = Sentence(original_text="The giant stood 80 feet tall.")
        s.normalized_source_text = "The giant stood 24 метри tall."
        assert s.normalized_source_text == "The giant stood 24 метри tall."
        assert s.original_text == "The giant stood 80 feet tall."

    def test_source_for_translation_fallback(self):
        s = Sentence(original_text="Raw English original.")
        assert s.source_for_translation == "Raw English original."

        s.normalized_source_text = "Normalized source text."
        assert s.source_for_translation == "Normalized source text."

    def test_unit_converter_populates_normalized_without_mutating_original(self):
        conv = UnitConverter()
        s = Sentence(original_text="The tower was 80 feet high.")
        conv.convert_sentence(s)

        assert s.normalized_source_text == "The tower was 24 метри high."
        assert s.original_text == "The tower was 80 feet high."


class TestSegmentStatusLifecycle:
    """Tests verifying 7-state SegmentStatus lifecycle and transition constraints."""

    def test_all_seven_lifecycle_states_defined(self):
        expected_states = {
            "PENDING",
            "DRAFT_COMPLETED",
            "EDITED",
            "VALIDATING",
            "ACCEPTED",
            "REVIEW_REQUIRED",
            "FAILED",
        }
        actual_states = {status.value for status in SegmentStatus}
        assert actual_states == expected_states

    def test_chunk_status_backward_compatibility(self):
        assert ChunkStatus.PENDING == "PENDING"
        assert ChunkStatus.DRAFT_COMPLETED == "DRAFT_COMPLETED"
        assert ChunkStatus.EDITED == "EDITED"
        assert ChunkStatus.VALIDATING == "VALIDATING"
        assert ChunkStatus.ACCEPTED == "ACCEPTED"
        assert ChunkStatus.REVIEW_REQUIRED == "REVIEW_REQUIRED"
        assert ChunkStatus.FAILED == "FAILED"
        assert ChunkStatus.REFINED == "REFINED"

    def test_valid_lifecycle_transitions(self):
        seg = TranslationSegment(
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Source paragraph text.",
        )
        assert seg.status == SegmentStatus.PENDING

        seg.transition_to(SegmentStatus.DRAFT_COMPLETED)
        assert seg.status == SegmentStatus.DRAFT_COMPLETED

        seg.transition_to(SegmentStatus.EDITED)
        assert seg.status == SegmentStatus.EDITED

        seg.transition_to(SegmentStatus.VALIDATING)
        assert seg.status == SegmentStatus.VALIDATING

        seg.transition_to(SegmentStatus.ACCEPTED)
        assert seg.status == SegmentStatus.ACCEPTED

    def test_invalid_lifecycle_transition_raises_error(self):
        seg = TranslationSegment(
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Source paragraph text.",
        )
        # Cannot skip directly from PENDING to ACCEPTED
        with pytest.raises(IllegalStateTransitionError):
            seg.transition_to(SegmentStatus.ACCEPTED)

    def test_review_required_and_repair_loop_transitions(self):
        seg = TranslationSegment(
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Source paragraph text.",
            status=SegmentStatus.VALIDATING,
        )
        # Transition to REVIEW_REQUIRED on validation exhaustion
        seg.transition_to(SegmentStatus.REVIEW_REQUIRED)
        assert seg.status == SegmentStatus.REVIEW_REQUIRED

        # Re-validation or manual acceptance from REVIEW_REQUIRED
        assert validate_segment_transition(SegmentStatus.REVIEW_REQUIRED, SegmentStatus.VALIDATING)
        assert validate_segment_transition(SegmentStatus.REVIEW_REQUIRED, SegmentStatus.ACCEPTED)


class TestDocumentWritersGating:
    """Tests verifying document writers consume only ACCEPTED segments by default."""

    @pytest.fixture
    def multi_status_book(self):
        book = Book(title="Gating Test Book", source_language="en", target_language="uk")
        ch = Chapter(title="Chapter 1", order_index=0)
        s_accepted = Sentence(
            original_text="Accepted sentence.",
            translated_text="Прийняте речення.",
            status=SegmentStatus.ACCEPTED,
            order_index=0,
        )
        s_review = Sentence(
            original_text="Review sentence.",
            translated_text="Речення на рецензію.",
            status=SegmentStatus.REVIEW_REQUIRED,
            order_index=1,
        )
        s_pending = Sentence(
            original_text="Pending sentence.",
            translated_text="Очікуюче речення.",
            status=SegmentStatus.PENDING,
            order_index=2,
        )
        s_failed = Sentence(
            original_text="Failed sentence.",
            translated_text="Помилкове речення.",
            status=SegmentStatus.FAILED,
            order_index=3,
        )
        ch.paragraphs.append(Paragraph(sentences=[s_accepted, s_review, s_pending, s_failed]))
        book.chapters.append(ch)
        return book

    def test_txt_writer_gates_to_accepted_by_default(self, tmp_path, multi_status_book):
        out_file = tmp_path / "output_default.txt"
        writer = TxtWriter()
        writer.write(multi_status_book, out_file)

        content = out_file.read_text(encoding="utf-8")
        assert "Прийняте речення." in content
        assert "Речення на рецензію." not in content
        assert "Очікуюче речення." not in content
        assert "Помилкове речення." not in content

    def test_txt_writer_allow_unreviewed_override(self, tmp_path, multi_status_book):
        out_file = tmp_path / "output_unreviewed.txt"
        writer = TxtWriter(allow_unreviewed=True)
        writer.write(multi_status_book, out_file)

        content = out_file.read_text(encoding="utf-8")
        assert "Прийняте речення." in content
        assert "Речення на рецензію." in content
        assert "Очікуюче речення." in content
        assert "Помилкове речення." in content

    def test_pdf_writer_gates_to_accepted_by_default(self, tmp_path, multi_status_book):
        out_file = tmp_path / "output_default.pdf"
        writer = PdfWriter()
        res_path = writer.write(multi_status_book, out_file)
        assert res_path.exists()
        assert res_path.stat().st_size > 0

    def test_pdf_writer_allow_unreviewed_override(self, tmp_path, multi_status_book):
        out_file = tmp_path / "output_unreviewed.pdf"
        writer = PdfWriter(allow_unreviewed=True)
        res_path = writer.write(multi_status_book, out_file)
        assert res_path.exists()
        assert res_path.stat().st_size > 0


class TestStage2DeterministicMode:
    """Tests verifying deterministic Stage 2 generation configuration."""

    def test_settings_deterministic_defaults(self):
        from src.config.settings import settings
        assert settings.stage2_deterministic is True
        assert settings.temperature == 0.0
        assert settings.do_sample is False
        assert settings.top_p == 1.0
        assert settings.repetition_penalty == 1.02

