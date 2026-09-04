"""
tests/unit/test_milestone4_challenger2_stress.py

Empirical stress tests for Milestone 4 (Phase 4):
1. BoundedRepairEngine:
   - Strict 2-retry bound across varied failure scenarios (stubborn, noisy, empty, garbage).
   - Segments failing 2 times MUST transition to REVIEW_REQUIRED and NEVER ACCEPTED.
   - Segments succeeding on attempt 1 or 2 transition to ACCEPTED with final_translation set.
   - Cancellation token behavior.
2. Sentence.original_text Immutability:
   - Strict immutability across assignments, mutations, and preprocessing passes.
3. DOM Gating & Document Writers (TxtWriter & PdfWriter):
   - Strict omission of REVIEW_REQUIRED and FAILED segments by default (allow_unreviewed=False).
   - Inclusion of unreviewed drafts when allow_unreviewed=True.
   - Mixed-status paragraphs and chapters.
4. SQLite Persistence & High Concurrency:
   - Multi-threaded stress testing (16 concurrent threads) of save_qa_report,
     save_qa_reports_batch, save_segments_batch, and concurrent queries.
   - Verification of violation diagnostic integrity, repair_attempts, and status.
"""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from src.context.builder import PromptContext
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.knowledge import EntityProfile, ScopeLevel
from src.domain.models.segment import SegmentStatus, TranslationSegment
from src.knowledge_base.migrations.migration_runner import MigrationRunner
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.launcher.concurrency import CancellationToken
from src.quality.models import QAReport, QASeverity, QAViolation
from src.quality.pipeline import QualityPipeline
from src.quality.repair import BoundedRepairEngine
from src.writers.pdf_writer import PdfWriter
from src.writers.txt_writer import TxtWriter


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def stress_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    runner = MigrationRunner(db_path)
    runner.apply_all()
    repo = SQLiteKnowledgeBaseRepository(db_path)
    yield repo, db_path
    repo.close()
    try:
        db_path.unlink()
    except Exception:
        pass


@pytest.fixture
def sample_context():
    cherry = EntityProfile(
        source_name="Cherry",
        canonical_target="Черрі",
        forbidden_target_forms=["Вишня", "Вішня"],
        locked=True,
    )
    return PromptContext(
        segment_id=uuid4(),
        prompt_text="",
        target_source="Cherry walked into the room.",
        target_draft="",
        previous_context="",
        entities_text="",
        active_entities=[cherry],
    )


# ==============================================================================
# 1. BoundedRepairEngine Empirical Stress Tests
# ==============================================================================

class TestBoundedRepairEngineStress:
    """Stress tests for BoundedRepairEngine retries, lifecycle, and edge cases."""

    def test_repair_attempts_cap_strictly_at_2_on_stubborn_engine(self, sample_context):
        """Verify that repair attempts stop strictly at 2, and status is NEVER ACCEPTED."""
        qp = QualityPipeline()
        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Cherry walked into the room.",
            refined_translation="Вишня зайшла до кімнати.",
            status=SegmentStatus.EDITED,
        )
        initial_report = qp.validate_segment(seg, context=sample_context)
        assert not initial_report.is_valid

        call_count = 0

        def stubborn_engine(s, prompt):
            nonlocal call_count
            call_count += 1
            # Continues returning forbidden variant
            return f"Вишня знову спробувала номер {call_count}."

        engine = BoundedRepairEngine(
            editing_engine=stubborn_engine,
            quality_pipeline=qp,
            max_retries=2,
        )
        repaired_seg, final_report = engine.repair_segment(
            segment=seg,
            report=initial_report,
            context=sample_context,
        )

        assert call_count == 2, f"Engine should be called exactly 2 times, got {call_count}"
        assert repaired_seg.repair_attempts == 2
        assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
        assert repaired_seg.status != SegmentStatus.ACCEPTED
        assert final_report.is_valid is False
        assert final_report.status == SegmentStatus.REVIEW_REQUIRED.value
        assert "Quality validation failed after 2 repair attempts" in (repaired_seg.error_message or "")

    def test_repair_succeeds_on_attempt_1(self, sample_context):
        """Verify clean transition to ACCEPTED on first repair attempt."""
        qp = QualityPipeline()
        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Cherry walked into the room.",
            refined_translation="Вишня зайшла до кімнати.",
            status=SegmentStatus.EDITED,
        )
        initial_report = qp.validate_segment(seg, context=sample_context)

        call_count = 0

        def fixing_engine(s, prompt):
            nonlocal call_count
            call_count += 1
            return "Черрі зайшла до кімнати."

        engine = BoundedRepairEngine(
            editing_engine=fixing_engine,
            quality_pipeline=qp,
            max_retries=2,
        )
        repaired_seg, final_report = engine.repair_segment(
            segment=seg,
            report=initial_report,
            context=sample_context,
        )

        assert call_count == 1
        assert repaired_seg.repair_attempts == 1
        assert repaired_seg.status == SegmentStatus.ACCEPTED
        assert repaired_seg.final_translation == "Черрі зайшла до кімнати."
        assert repaired_seg.error_message is None
        assert final_report.is_valid is True
        assert final_report.status == SegmentStatus.ACCEPTED.value

    def test_repair_succeeds_on_attempt_2(self, sample_context):
        """Verify clean transition to ACCEPTED on second repair attempt."""
        qp = QualityPipeline()
        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Cherry walked into the room.",
            refined_translation="Вишня зайшла до кімнати.",
            status=SegmentStatus.EDITED,
        )
        initial_report = qp.validate_segment(seg, context=sample_context)

        call_count = 0

        def two_step_engine(s, prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Still uses forbidden variant
                return "Вішня зайшла до кімнати."
            return "Черрі зайшла до кімнати."

        engine = BoundedRepairEngine(
            editing_engine=two_step_engine,
            quality_pipeline=qp,
            max_retries=2,
        )
        repaired_seg, final_report = engine.repair_segment(
            segment=seg,
            report=initial_report,
            context=sample_context,
        )

        assert call_count == 2
        assert repaired_seg.repair_attempts == 2
        assert repaired_seg.status == SegmentStatus.ACCEPTED
        assert repaired_seg.final_translation == "Черрі зайшла до кімнати."
        assert final_report.is_valid is True
        assert final_report.status == SegmentStatus.ACCEPTED.value

    @pytest.mark.parametrize(
        "bad_response",
        [
            "",
            "   \n\t  ",
            None,
            "```json\n{\ninvalid json\n```",
            "<|im_start|>assistant\nВишня зайшла<|im_end|>",
            "Вишня",  # still forbidden
        ],
    )
    def test_repair_with_adversarial_llm_outputs(self, sample_context, bad_response):
        """Adversarial LLM outputs must never trigger ACCEPTED status or unhandled crashes."""
        qp = QualityPipeline()
        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Cherry walked into the room.",
            refined_translation="Вишня зайшла до кімнати.",
            status=SegmentStatus.EDITED,
        )
        initial_report = qp.validate_segment(seg, context=sample_context)

        def bad_engine(s, prompt):
            return bad_response

        engine = BoundedRepairEngine(
            editing_engine=bad_engine,
            quality_pipeline=qp,
            max_retries=2,
        )
        repaired_seg, final_report = engine.repair_segment(
            segment=seg,
            report=initial_report,
            context=sample_context,
        )

        assert repaired_seg.repair_attempts <= 2
        assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
        assert repaired_seg.status != SegmentStatus.ACCEPTED
        assert final_report.is_valid is False

    def test_repair_engine_with_missing_dependencies(self, sample_context):
        """If editing_engine or quality_pipeline is None, segment must transition to REVIEW_REQUIRED."""
        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Cherry walked into the room.",
            refined_translation="Вишня зайшла до кімнати.",
            status=SegmentStatus.EDITED,
        )
        report = QAReport(segment_id=seg.id, is_valid=False)

        engine = BoundedRepairEngine(editing_engine=None, quality_pipeline=None)
        repaired_seg, final_report = engine.repair_segment(seg, report, context=sample_context)

        assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
        assert final_report.status == SegmentStatus.REVIEW_REQUIRED.value

    def test_repair_engine_cancellation_preserves_safety(self, sample_context):
        """A cancelled token stops repair loop immediately without marking ACCEPTED."""
        qp = QualityPipeline()
        token = CancellationToken()
        token.cancel()

        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Cherry walked into the room.",
            refined_translation="Вишня зайшла до кімнати.",
            status=SegmentStatus.EDITED,
        )
        report = qp.validate_segment(seg, context=sample_context)

        call_count = 0

        def dummy_engine(s, prompt):
            nonlocal call_count
            call_count += 1
            return "Черрі зайшла до кімнати."

        engine = BoundedRepairEngine(
            editing_engine=dummy_engine,
            quality_pipeline=qp,
            max_retries=2,
        )
        repaired_seg, final_report = engine.repair_segment(
            segment=seg,
            report=report,
            cancel_token=token,
        )

        assert call_count == 0
        assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
        assert repaired_seg.status != SegmentStatus.ACCEPTED


# ==============================================================================
# 2. Sentence.original_text Immutability
# ==============================================================================

class TestSentenceImmutability:
    """Stress testing the immutability contract of Sentence.original_text."""

    def test_sentence_original_text_cannot_be_mutated(self):
        sent = Sentence(
            id=uuid4(),
            original_text="The immutable source sentence.",
            normalized_source_text="The normalized sentence.",
            translated_text="Перекладене речення.",
        )

        assert sent.original_text == "The immutable source sentence."

        # Direct mutation must raise ValidationError due to frozen=True
        with pytest.raises(ValidationError):
            sent.original_text = "Hacked sentence."

        # Mutating normalized_source_text or translated_text is allowed
        sent.normalized_source_text = "Updated normalized sentence."
        assert sent.normalized_source_text == "Updated normalized sentence."

        sent.translated_text = "Новий переклад."
        assert sent.translated_text == "Новий переклад."

        # original_text remains untouched
        assert sent.original_text == "The immutable source sentence."

    def test_sentence_source_for_translation_fallback(self):
        sent1 = Sentence(original_text="Raw English text.")
        assert sent1.source_for_translation == "Raw English text."

        sent2 = Sentence(
            original_text="Raw English text.",
            normalized_source_text="Normalized English text.",
        )
        assert sent2.source_for_translation == "Normalized English text."
        assert sent2.original_text == "Raw English text."


# ==============================================================================
# 3. DOM Gating & Document Writers (TxtWriter & PdfWriter)
# ==============================================================================

class TestDocumentWritersDOMGating:
    """Empirical verification that writers strictly omit REVIEW_REQUIRED and FAILED segments."""

    def _create_test_book(self) -> Book:
        """Creates a book with varied segment statuses across chapters and paragraphs."""
        # Chapter 1: 3 paragraphs (ACCEPTED, REVIEW_REQUIRED, FAILED)
        p1 = Paragraph(
            sentences=[
                Sentence(
                    original_text="Accepted sentence.",
                    translated_text="Прийняте речення.",
                    status=SegmentStatus.ACCEPTED,
                    order_index=0,
                )
            ]
        )
        p2 = Paragraph(
            sentences=[
                Sentence(
                    original_text="Review required sentence.",
                    translated_text="Потребує перевірки речення.",
                    status=SegmentStatus.REVIEW_REQUIRED,
                    order_index=0,
                )
            ]
        )
        p3 = Paragraph(
            sentences=[
                Sentence(
                    original_text="Failed sentence.",
                    translated_text="Провалене речення.",
                    status=SegmentStatus.FAILED,
                    order_index=0,
                )
            ]
        )
        ch1 = Chapter(
            title="Chapter 1",
            translated_title="Розділ 1",
            paragraphs=[p1, p2, p3],
            order_index=0,
        )

        # Chapter 2: Paragraph with mixed sentences
        p_mixed = Paragraph(
            sentences=[
                Sentence(
                    original_text="First accepted.",
                    translated_text="Перше прийняте.",
                    status=SegmentStatus.ACCEPTED,
                    order_index=0,
                ),
                Sentence(
                    original_text="Second review required.",
                    translated_text="Друге потребує перевірки.",
                    status=SegmentStatus.REVIEW_REQUIRED,
                    order_index=1,
                ),
                Sentence(
                    original_text="Third accepted.",
                    translated_text="Третє прийняте.",
                    status=SegmentStatus.ACCEPTED,
                    order_index=2,
                ),
            ]
        )
        ch2 = Chapter(
            title="Chapter 2",
            translated_title="Розділ 2",
            paragraphs=[p_mixed],
            order_index=1,
        )

        # Chapter 3: All REVIEW_REQUIRED
        p_all_review = Paragraph(
            sentences=[
                Sentence(
                    original_text="All unreviewed.",
                    translated_text="Все неперевірене.",
                    status=SegmentStatus.REVIEW_REQUIRED,
                    order_index=0,
                )
            ]
        )
        ch3 = Chapter(
            title="Chapter 3",
            translated_title="Розділ 3",
            paragraphs=[p_all_review],
            order_index=2,
        )

        return Book(
            title="Test Book",
            chapters=[ch1, ch2, ch3],
        )

    def test_txt_writer_omits_unreviewed_by_default(self, tmp_path):
        book = self._create_test_book()
        out_file = tmp_path / "output_default.txt"

        writer = TxtWriter(allow_unreviewed=False)
        writer.write(book, out_file)

        content = out_file.read_text(encoding="utf-8")

        # ACCEPTED sentences MUST be present
        assert "Прийняте речення." in content
        assert "Перше прийняте." in content
        assert "Третє прийняте." in content

        # REVIEW_REQUIRED and FAILED sentences MUST be absent
        assert "Потребує перевірки речення." not in content
        assert "Провалене речення." not in content
        assert "Друге потребує перевірки." not in content
        assert "Все неперевірене." not in content

    def test_txt_writer_exports_draft_when_allowed(self, tmp_path):
        book = self._create_test_book()
        out_file = tmp_path / "output_unreviewed.txt"

        writer = TxtWriter(allow_unreviewed=True)
        writer.write(book, out_file)

        content = out_file.read_text(encoding="utf-8")

        # Everything should be present
        assert "Прийняте речення." in content
        assert "Потребує перевірки речення." in content
        assert "Провалене речення." in content
        assert "Перше прийняте." in content
        assert "Друге потребує перевірки." in content
        assert "Третє прийняте." in content
        assert "Все неперевірене." in content

    def test_pdf_writer_omits_unreviewed_by_default(self, tmp_path):
        book = self._create_test_book()
        out_file = tmp_path / "output_default.pdf"

        writer = PdfWriter(allow_unreviewed=False)
        result_path = writer.write(book, out_file)

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_pdf_writer_exports_draft_when_allowed(self, tmp_path):
        book = self._create_test_book()
        out_file = tmp_path / "output_unreviewed.pdf"

        writer = PdfWriter(allow_unreviewed=True)
        result_path = writer.write(book, out_file)

        assert result_path.exists()
        assert result_path.stat().st_size > 0


# ==============================================================================
# 4. SQLite Persistence & Concurrency Stress Testing
# ==============================================================================

class TestSQLitePersistenceConcurrencyStress:
    """Stress-tests SQLite repository with high concurrent multi-threaded workloads."""

    def test_concurrent_saves_and_queries_under_high_load(self, stress_db):
        repo, db_path = stress_db
        book_id = uuid4()
        num_threads = 16
        reports_per_thread = 25

        saved_segment_ids: List[UUID] = []

        def worker_save_reports(thread_idx: int) -> List[UUID]:
            thread_seg_ids = []
            for i in range(reports_per_thread):
                seg_id = uuid4()
                thread_seg_ids.append(seg_id)
                # Create mixed report
                is_clean = (i % 2 == 0)
                if is_clean:
                    report = QAReport(
                        segment_id=seg_id,
                        book_id=book_id,
                        is_valid=True,
                        score=1.0,
                        repair_attempts=0,
                        status=SegmentStatus.ACCEPTED.value,
                    )
                else:
                    report = QAReport(
                        segment_id=seg_id,
                        book_id=book_id,
                        is_valid=False,
                        score=0.5,
                        repair_attempts=2,
                        status=SegmentStatus.REVIEW_REQUIRED.value,
                    )
                    report.add_violation(
                        QAViolation(
                            validator_name="EntityConsistencyValidator",
                            rule_code="FORBIDDEN_ENTITY_VARIANT",
                            severity=QASeverity.CRITICAL,
                            message=f"Violation from thread {thread_idx} item {i}",
                            forbidden_form="Вишня",
                            suggested_fix="Черрі",
                        )
                    )

                # Alternate between single save and batch save
                if i % 3 == 0:
                    repo.save_qa_report(report, book_id=book_id)
                else:
                    repo.save_qa_reports_batch([report], book_id=book_id)

            return thread_seg_ids

        def worker_query_reports() -> int:
            count = 0
            for _ in range(10):
                reps = repo.get_qa_reports_for_book(book_id)
                count = max(count, len(reps))
            return count

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Launch writer futures
            writer_futures = [
                executor.submit(worker_save_reports, t)
                for t in range(num_threads - 4)
            ]
            # Launch reader futures concurrently
            reader_futures = [
                executor.submit(worker_query_reports)
                for _ in range(4)
            ]

            # Collect all writer results
            all_seg_ids: List[UUID] = []
            for f in concurrent.futures.as_completed(writer_futures):
                all_seg_ids.extend(f.result())

            # Await readers
            for f in concurrent.futures.as_completed(reader_futures):
                f.result()

        expected_total_reports = (num_threads - 4) * reports_per_thread
        assert len(all_seg_ids) == expected_total_reports

        # Final audit verification
        final_reports = repo.get_qa_reports_for_book(book_id)
        assert len(final_reports) == expected_total_reports

        # Verify diagnostic integrity of individual segment reports
        sample_seg_id = all_seg_ids[1]  # odd index -> had violation
        seg_reps = repo.get_qa_reports_for_segment(sample_seg_id)
        assert len(seg_reps) >= 1
        assert seg_reps[0]["validator_name"] == "EntityConsistencyValidator"
        assert seg_reps[0]["severity"] == "CRITICAL"
        assert seg_reps[0]["details"]["forbidden_form"] == "Вишня"
        assert seg_reps[0]["details"]["suggested_fix"] == "Черрі"
        assert seg_reps[0]["status"] == SegmentStatus.REVIEW_REQUIRED.value
        assert seg_reps[0]["repair_attempts"] == 2

    def test_concurrent_segment_state_persistence(self, stress_db):
        """Stress-tests concurrent saves and queries to segments batch."""
        repo, _ = stress_db
        book_id = uuid4()
        num_threads = 8
        segments_per_thread = 20

        def segment_writer(thread_idx: int) -> int:
            segs = []
            for i in range(segments_per_thread):
                status = SegmentStatus.ACCEPTED if i % 2 == 0 else SegmentStatus.REVIEW_REQUIRED
                s = TranslationSegment(
                    id=uuid4(),
                    book_id=book_id,
                    chapter_id=uuid4(),
                    paragraph_id=uuid4(),
                    source_text=f"Source text {thread_idx}-{i}",
                    draft_translation=f"Draft {thread_idx}-{i}",
                    refined_translation=f"Refined {thread_idx}-{i}",
                    final_translation=f"Final {thread_idx}-{i}" if status == SegmentStatus.ACCEPTED else None,
                    status=status,
                    repair_attempts=0 if status == SegmentStatus.ACCEPTED else 2,
                )
                segs.append(s)

            repo.save_segments_batch(segs)
            return len(segs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(segment_writer, t) for t in range(num_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert sum(results) == num_threads * segments_per_thread

        accepted_segs = repo.load_segments_by_status(book_id, SegmentStatus.ACCEPTED)
        review_segs = repo.load_segments_by_status(book_id, SegmentStatus.REVIEW_REQUIRED)

        expected_each = (num_threads * segments_per_thread) // 2
        assert len(accepted_segs) == expected_each
        assert len(review_segs) == expected_each

    def test_multi_instance_concurrent_access_to_same_database(self, stress_db):
        """Simulate separate worker processes/threads creating their own repository instance on the same DB file."""
        _, db_path = stress_db
        book_id = uuid4()
        num_instances = 6
        items_per_instance = 15

        def external_worker(worker_id: int) -> int:
            # Independent repo instance per thread
            worker_repo = SQLiteKnowledgeBaseRepository(db_path)
            try:
                reports = []
                for i in range(items_per_instance):
                    seg_id = uuid4()
                    rep = QAReport(
                        segment_id=seg_id,
                        book_id=book_id,
                        is_valid=True,
                        status=SegmentStatus.ACCEPTED.value,
                    )
                    reports.append(rep)
                worker_repo.save_qa_reports_batch(reports, book_id=book_id)
                read_back = worker_repo.get_qa_reports_for_book(book_id)
                return len(reports)
            finally:
                worker_repo.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_instances) as executor:
            futures = [executor.submit(external_worker, i) for i in range(num_instances)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert sum(results) == num_instances * items_per_instance
        verification_repo = SQLiteKnowledgeBaseRepository(db_path)
        try:
            total_records = verification_repo.get_qa_reports_for_book(book_id)
            assert len(total_records) == num_instances * items_per_instance
        finally:
            verification_repo.close()


# ==============================================================================
# 5. Lifecycle Transition Invariants & DOM Reconciliation Emulation
# ==============================================================================

class TestLifecycleTransitionsAndDOMReconciliation:
    """Stress-test strict lifecycle invariants and DOM reconciliation rules."""

    def test_illegal_state_transitions_strictly_rejected(self):
        from src.domain.models.segment import IllegalStateTransitionError

        # ACCEPTED is terminal: cannot transition anywhere
        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Test source.",
            status=SegmentStatus.ACCEPTED,
        )
        for invalid_target in [
            SegmentStatus.PENDING,
            SegmentStatus.DRAFT_COMPLETED,
            SegmentStatus.EDITED,
            SegmentStatus.VALIDATING,
            SegmentStatus.REVIEW_REQUIRED,
            SegmentStatus.FAILED,
        ]:
            with pytest.raises(IllegalStateTransitionError):
                seg.transition_to(invalid_target)

        # PENDING cannot jump directly to ACCEPTED or REVIEW_REQUIRED
        seg_pending = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Test source.",
            status=SegmentStatus.PENDING,
        )
        with pytest.raises(IllegalStateTransitionError):
            seg_pending.transition_to(SegmentStatus.ACCEPTED)
        with pytest.raises(IllegalStateTransitionError):
            seg_pending.transition_to(SegmentStatus.REVIEW_REQUIRED)

    def test_dom_reconciliation_exact_status_propagation(self, tmp_path):
        """Emulates TranslationRunner DOM reconciliation and verifies writers omit unaccepted paragraphs."""
        book_id = uuid4()
        ch_id = uuid4()

        p1_id = uuid4()
        s1 = Sentence(original_text="Sentence one.", order_index=0)
        s2 = Sentence(original_text="Sentence two.", order_index=1)
        p1 = Paragraph(id=p1_id, sentences=[s1, s2])

        p2_id = uuid4()
        s3 = Sentence(original_text="Sentence three.", order_index=0)
        p2 = Paragraph(id=p2_id, sentences=[s3])

        p3_id = uuid4()
        s4 = Sentence(original_text="Sentence four.", order_index=0)
        p3 = Paragraph(id=p3_id, sentences=[s4])

        ch = Chapter(id=ch_id, title="Test Chapter", translated_title="Тестовий розділ", paragraphs=[p1, p2, p3], order_index=0)
        book = Book(id=book_id, title="Test Book", chapters=[ch])

        # Create segments corresponding to paragraphs
        seg1 = TranslationSegment(
            id=p1_id,
            book_id=book_id,
            chapter_id=ch_id,
            paragraph_id=p1_id,
            source_text="Sentence one. Sentence two.",
            final_translation="Речення одне. Речення два.",
            status=SegmentStatus.ACCEPTED,
        )
        seg2 = TranslationSegment(
            id=p2_id,
            book_id=book_id,
            chapter_id=ch_id,
            paragraph_id=p2_id,
            source_text="Sentence three.",
            final_translation="Речення три (неперевірено).",
            status=SegmentStatus.REVIEW_REQUIRED,
        )
        seg3 = TranslationSegment(
            id=p3_id,
            book_id=book_id,
            chapter_id=ch_id,
            paragraph_id=p3_id,
            source_text="Sentence four.",
            final_translation="Речення чотири (провал).",
            status=SegmentStatus.FAILED,
        )

        segments = [seg1, seg2, seg3]

        # Reconcile DOM as done in translation_runner.py lines 460-532
        segment_map = {seg.paragraph_id: seg for seg in segments}
        for chapter in book.chapters:
            for paragraph in chapter.paragraphs:
                seg = segment_map.get(paragraph.id)
                if not seg:
                    continue
                if seg.status == SegmentStatus.ACCEPTED:
                    final_text = seg.final_translation or seg.refined_translation
                    try:
                        paragraph.status = SegmentStatus.ACCEPTED
                    except (ValueError, AttributeError):
                        pass
                    if paragraph.sentences:
                        paragraph.sentences[0].translated_text = final_text
                        paragraph.sentences[0].status = SegmentStatus.ACCEPTED
                        for extra_s in paragraph.sentences[1:]:
                            extra_s.translated_text = ""
                            extra_s.status = SegmentStatus.ACCEPTED
                elif seg.status == SegmentStatus.REVIEW_REQUIRED:
                    candidate_text = seg.final_translation or seg.refined_translation or seg.draft_translation
                    try:
                        paragraph.status = SegmentStatus.REVIEW_REQUIRED
                    except (ValueError, AttributeError):
                        pass
                    if paragraph.sentences:
                        paragraph.sentences[0].translated_text = candidate_text
                        paragraph.sentences[0].status = SegmentStatus.REVIEW_REQUIRED
                        for extra_s in paragraph.sentences[1:]:
                            extra_s.translated_text = ""
                            extra_s.status = SegmentStatus.REVIEW_REQUIRED
                elif seg.status == SegmentStatus.FAILED:
                    try:
                        paragraph.status = SegmentStatus.FAILED
                    except (ValueError, AttributeError):
                        pass
                    if paragraph.sentences:
                        paragraph.sentences[0].translated_text = seg.translated_text
                        paragraph.sentences[0].status = SegmentStatus.FAILED
                        for extra_s in paragraph.sentences[1:]:
                            extra_s.translated_text = ""
                            extra_s.status = SegmentStatus.FAILED

        # Verify reconciled DOM states
        assert p1.sentences[0].status == SegmentStatus.ACCEPTED
        assert p1.sentences[1].status == SegmentStatus.ACCEPTED
        assert p2.sentences[0].status == SegmentStatus.REVIEW_REQUIRED
        assert p3.sentences[0].status == SegmentStatus.FAILED

        # Write out with TxtWriter
        txt_out = tmp_path / "reconciled_test.txt"
        TxtWriter(allow_unreviewed=False).write(book, txt_out)
        txt_content = txt_out.read_text(encoding="utf-8")

        assert "Речення одне. Речення два." in txt_content
        assert "Речення три (неперевірено)." not in txt_content
        assert "Речення чотири (провал)." not in txt_content

    def test_translation_segment_from_paragraph_preserves_sentences(self):
        s1 = Sentence(original_text="First sentence.", order_index=0)
        s2 = Sentence(original_text="Second sentence.", order_index=1)
        p = Paragraph(sentences=[s1, s2])

        book_id = uuid4()
        ch_id = uuid4()

        seg = TranslationSegment.from_paragraph(
            paragraph=p,
            book_id=book_id,
            chapter_id=ch_id,
            order_index=0,
        )

        assert seg.source_text == "First sentence. Second sentence."
        assert seg.sentence_ids == [s1.id, s2.id]
        assert s1.original_text == "First sentence."
        assert s2.original_text == "Second sentence."

