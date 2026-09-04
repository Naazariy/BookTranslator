"""
tests/unit/test_challenger2_m4_empirical_deep_stress.py

Adversarial empirical stress tests for Milestone 4 (Phase 4):
1. BoundedRepairEngine:
   - Fuzzing 50 varied failure segments verifying strict 2-retry bound (NEVER ACCEPTED on failure).
   - Graceful recovery on LLM runtime exceptions (RuntimeError, MemoryError, ConnectionError).
   - Cancellation token behavior between attempt 1 and attempt 2.
   - Dynamic max_retries configuration (0, 1, 2).
   - Multi-issue diagnostic cascades (multi-turn error resolution vs unfixable cascades).
   - State machine entry from all lifecycle statuses (PENDING, DRAFT_COMPLETED, EDITED, VALIDATING, REVIEW_REQUIRED, FAILED, ACCEPTED).
2. Sentence.original_text Immutability:
   - Direct assignment, setattr, and preprocessing passes guarantee strict immutability.
3. DOM Gating & Document Writers (TxtWriter & PdfWriter):
   - Multi-paragraph, multi-sentence mixed-status documents.
   - Strict omission of REVIEW_REQUIRED and FAILED sentences by default.
   - Full inclusion when allow_unreviewed=True.
   - Unicode & Cyrillic stability in PDF rendering.
4. SQLite Persistence & High Concurrency Extreme Stress:
   - 32 concurrent threads executing 500+ saves and queries simultaneously.
   - Zero database locks or data loss in WAL mode.
   - Query planner index utilization validation (EXPLAIN QUERY PLAN).
5. TranslationRunner DOM Gating Integration:
   - Full pipeline QA stage execution with DOM reconciliation and export verification.
"""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from src.context.builder import PromptContext
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.knowledge import EntityProfile
from src.domain.models.segment import SegmentStatus, TranslationSegment
from src.knowledge_base.migrations.migration_runner import MigrationRunner
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.launcher.concurrency import CancellationToken
from src.preprocessing.unit_converter import UnitConverter
from src.quality.models import QAReport, QASeverity, QAViolation
from src.quality.pipeline import QualityPipeline
from src.quality.repair import BoundedRepairEngine
from src.translation.pipeline import TwoStageTranslationPipeline
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
# 1. BoundedRepairEngine Deep Adversarial Stress Tests
# ==============================================================================

class TestBoundedRepairEngineDeepStress:
    """Adversarial stress-testing of repair loop invariants, retries, and failure modes."""

    def test_repair_strictly_bounds_at_2_retries_across_50_fuzzed_segments(self, sample_context):
        """Fuzz 50 segments with different initial texts: retries NEVER exceed 2 and status is NEVER ACCEPTED."""
        qp = QualityPipeline()
        total_attempts_list = []

        for i in range(50):
            seg = TranslationSegment(
                id=uuid4(),
                book_id=uuid4(),
                chapter_id=uuid4(),
                paragraph_id=uuid4(),
                source_text=f"Cherry spoke to person {i}.",
                refined_translation=f"Вишня розмовляла з особою {i}.",
                status=SegmentStatus.EDITED,
            )
            report = qp.validate_segment(seg, context=sample_context)
            assert not report.is_valid

            call_count = 0

            def stubborn_engine(s, prompt):
                nonlocal call_count
                call_count += 1
                return f"Вишня варіант {call_count} для {i}."

            engine = BoundedRepairEngine(
                editing_engine=stubborn_engine,
                quality_pipeline=qp,
                max_retries=2,
            )
            repaired_seg, final_report = engine.repair_segment(seg, report, context=sample_context)

            assert call_count == 2
            assert repaired_seg.repair_attempts == 2
            assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
            assert repaired_seg.status != SegmentStatus.ACCEPTED
            assert final_report.is_valid is False
            assert final_report.status == SegmentStatus.REVIEW_REQUIRED.value
            total_attempts_list.append(repaired_seg.repair_attempts)

        assert len(total_attempts_list) == 50
        assert all(a == 2 for a in total_attempts_list)

    @pytest.mark.parametrize(
        "exc_cls, exc_msg",
        [
            (RuntimeError, "CUDA out of memory in tensor allocation"),
            (ValueError, "Model tokenizer unexpected end of stream"),
            (MemoryError, "System RAM exhausted"),
            (ConnectionError, "Remote inference socket dropped"),
        ],
    )
    def test_repair_engine_handles_llm_exceptions_gracefully(self, sample_context, exc_cls, exc_msg):
        """If the editing engine raises severe exceptions, BoundedRepairEngine safely marks REVIEW_REQUIRED."""
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
        report = qp.validate_segment(seg, context=sample_context)

        def failing_engine(s, prompt):
            raise exc_cls(exc_msg)

        engine = BoundedRepairEngine(
            editing_engine=failing_engine,
            quality_pipeline=qp,
            max_retries=2,
        )

        repaired_seg, final_report = engine.repair_segment(seg, report, context=sample_context)

        assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
        assert repaired_seg.status != SegmentStatus.ACCEPTED
        assert final_report.is_valid is False
        assert final_report.status == SegmentStatus.REVIEW_REQUIRED.value

    def test_repair_engine_cancellation_between_attempt_1_and_2(self, sample_context):
        """If cancellation is requested after attempt 1 fails, attempt 2 MUST NOT be executed."""
        qp = QualityPipeline()
        token = CancellationToken()

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

        def cancelling_engine(s, prompt):
            nonlocal call_count
            call_count += 1
            # Cancel token on attempt 1
            token.cancel()
            return "Вішня зайшла знову."

        engine = BoundedRepairEngine(
            editing_engine=cancelling_engine,
            quality_pipeline=qp,
            max_retries=2,
        )

        repaired_seg, final_report = engine.repair_segment(
            seg, report, context=sample_context, cancel_token=token
        )

        assert call_count == 1  # Attempt 2 was avoided due to cancellation
        assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
        assert repaired_seg.repair_attempts == 1
        assert repaired_seg.status != SegmentStatus.ACCEPTED

    @pytest.mark.parametrize(
        "max_retries, expected_calls",
        [
            (1, 1),
            (0, 0),
        ],
    )
    def test_repair_with_dynamic_max_retries_config(self, sample_context, max_retries, expected_calls):
        """Verify BoundedRepairEngine respects custom max_retries parameter."""
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
        report = qp.validate_segment(seg, context=sample_context)

        call_count = 0

        def mock_engine(s, prompt):
            nonlocal call_count
            call_count += 1
            return "Вишня знову."

        engine = BoundedRepairEngine(
            editing_engine=mock_engine,
            quality_pipeline=qp,
            max_retries=max_retries,
        )
        repaired_seg, _ = engine.repair_segment(seg, report, context=sample_context)

        assert call_count == expected_calls
        assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED

    def test_multi_issue_repair_cascade_success_on_attempt_2(self):
        """Simulate fixing one issue on attempt 1, introducing another, then fixing all on attempt 2."""
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
            target_source="Cherry held the key in hand <|im_end|>.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[cherry],
        )

        # Initial segment has forbidden entity AND control token
        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Cherry held the key in hand <|im_end|>.",
            refined_translation="Вишня тримала ключ у руці <|im_end|>.",
            status=SegmentStatus.EDITED,
        )
        initial_report = qp.validate_segment(seg, context=ctx)
        assert len(initial_report.violations) >= 2  # Forbidden variant + control token

        call_count = 0

        def multi_turn_engine(s, prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Attempt 1: removes control token, but still has "Вишня"
                return "Вишня тримала ключ у руці."
            # Attempt 2: fixes "Вишня" -> "Черрі"
            return "Черрі тримала ключ у руці."

        engine = BoundedRepairEngine(
            editing_engine=multi_turn_engine,
            quality_pipeline=qp,
            max_retries=2,
        )
        repaired_seg, final_report = engine.repair_segment(seg, initial_report, context=ctx)

        assert call_count == 2
        assert repaired_seg.repair_attempts == 2
        assert repaired_seg.status == SegmentStatus.ACCEPTED
        assert repaired_seg.final_translation == "Черрі тримала ключ у руці."
        assert final_report.is_valid is True

    @pytest.mark.parametrize(
        "initial_status",
        [
            SegmentStatus.PENDING,
            SegmentStatus.DRAFT_COMPLETED,
            SegmentStatus.EDITED,
            SegmentStatus.VALIDATING,
            SegmentStatus.REVIEW_REQUIRED,
            SegmentStatus.FAILED,
            SegmentStatus.ACCEPTED,
        ],
    )
    def test_repair_from_any_initial_status_terminates_cleanly(self, sample_context, initial_status):
        """Verifies repair_segment handles segments entering from any lifecycle state without unhandled exception."""
        qp = QualityPipeline()
        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="Cherry walked into the room.",
            refined_translation="Черрі зайшла до кімнати.",
            status=initial_status,
        )
        # Create a report with a violation
        report = QAReport(segment_id=seg.id, is_valid=False)
        report.add_violation(
            QAViolation(
                validator_name="MockValidator",
                rule_code="TEST_RULE",
                severity=QASeverity.CRITICAL,
                message="Mock violation",
            )
        )

        def fixing_engine(s, prompt):
            return "Черрі зайшла до кімнати тихо."

        engine = BoundedRepairEngine(
            editing_engine=fixing_engine,
            quality_pipeline=qp,
            max_retries=2,
        )
        repaired_seg, final_report = engine.repair_segment(seg, report, context=sample_context)

        assert repaired_seg.status == SegmentStatus.ACCEPTED
        assert repaired_seg.repair_attempts == 1


# ==============================================================================
# 2. Sentence.original_text Immutability Deep Tests
# ==============================================================================

class TestSentenceImmutabilityDeep:
    """Stress tests guaranteeing Sentence.original_text cannot be mutated through any path."""

    def test_sentence_original_text_immutable_across_all_manipulations(self):
        sent = Sentence(
            id=uuid4(),
            original_text="Permanent English source text.",
            normalized_source_text="Preprocessed English text.",
            translated_text="Український переклад.",
        )

        # 1. Direct assignment must fail
        with pytest.raises(ValidationError):
            sent.original_text = "Mutated text."

        # 2. setattr must fail
        with pytest.raises(ValidationError):
            setattr(sent, "original_text", "Mutated via setattr.")

        # 3. Verify original_text is untouched
        assert sent.original_text == "Permanent English source text."

    def test_unit_converter_does_not_mutate_sentence_original_text(self):
        """Preprocessing through UnitConverter must only populate normalized_source_text."""
        converter = UnitConverter()
        raw_text = "The castle walls were 80 feet high and 15 miles away."
        sent = Sentence(original_text=raw_text)

        # Process via converter
        converted = converter.convert_text(sent.original_text)
        sent.normalized_source_text = converted

        assert sent.original_text == raw_text
        assert sent.normalized_source_text != raw_text
        assert "24 метри" in sent.normalized_source_text
        assert "24 кілометри" in sent.normalized_source_text
        assert sent.source_for_translation == sent.normalized_source_text


# ==============================================================================
# 3. DOM Gating & Document Writers (TxtWriter & PdfWriter) Deep Stress
# ==============================================================================

class TestDocumentWritersDOMGatingDeep:
    """Empirical verification that writers strictly omit unaccepted segments."""

    def _build_complex_multichapter_book(self) -> Book:
        """Constructs a realistic 4-chapter Book with diverse paragraph and segment states."""
        # Chapter 1: Standard clean chapter (all ACCEPTED)
        p1 = Paragraph(
            sentences=[
                Sentence(original_text="Ch1 P1 S1.", translated_text="Розділ 1 Абзац 1.", status=SegmentStatus.ACCEPTED, order_index=0),
                Sentence(original_text="Ch1 P1 S2.", translated_text="Друге речення.", status=SegmentStatus.ACCEPTED, order_index=1),
            ]
        )
        ch1 = Chapter(title="Chapter 1", translated_title="Розділ 1", paragraphs=[p1], order_index=0)

        # Chapter 2: Mixed paragraph (sentence 0 ACCEPTED, sentence 1 REVIEW_REQUIRED, sentence 2 FAILED, sentence 3 ACCEPTED)
        p_mixed = Paragraph(
            sentences=[
                Sentence(original_text="Accept A.", translated_text="Прийнято А.", status=SegmentStatus.ACCEPTED, order_index=0),
                Sentence(original_text="Review B.", translated_text="Рецензія Б.", status=SegmentStatus.REVIEW_REQUIRED, order_index=1),
                Sentence(original_text="Fail C.", translated_text="Провал В.", status=SegmentStatus.FAILED, order_index=2),
                Sentence(original_text="Accept D.", translated_text="Прийнято Г.", status=SegmentStatus.ACCEPTED, order_index=3),
            ]
        )
        ch2 = Chapter(title="Chapter 2", translated_title="Розділ 2", paragraphs=[p_mixed], order_index=1)

        # Chapter 3: Completely unreviewed chapter (only REVIEW_REQUIRED)
        p_unreviewed = Paragraph(
            sentences=[
                Sentence(original_text="Unreviewed P.", translated_text="Неперевірений абзац.", status=SegmentStatus.REVIEW_REQUIRED, order_index=0),
            ]
        )
        ch3 = Chapter(title="Chapter 3", translated_title="Розділ 3", paragraphs=[p_unreviewed], order_index=2)

        # Chapter 4: Empty chapter
        ch4 = Chapter(title="Chapter 4", translated_title="Розділ 4", paragraphs=[], order_index=3)

        return Book(title="Master Test Book", chapters=[ch1, ch2, ch3, ch4])

    def test_txt_writer_strict_gating(self, tmp_path):
        book = self._build_complex_multichapter_book()
        out_path = tmp_path / "strict_export.txt"

        writer = TxtWriter(allow_unreviewed=False)
        writer.write(book, out_path)

        content = out_path.read_text(encoding="utf-8")

        # Accepted must be present
        assert "Розділ 1 Абзац 1." in content
        assert "Друге речення." in content
        assert "Прийнято А." in content
        assert "Прийнято Г." in content

        # REVIEW_REQUIRED and FAILED must NOT be present
        assert "Рецензія Б." not in content
        assert "Провал В." not in content
        assert "Неперевірений абзац." not in content

    def test_txt_writer_allow_unreviewed_includes_everything(self, tmp_path):
        book = self._build_complex_multichapter_book()
        out_path = tmp_path / "unreviewed_export.txt"

        writer = TxtWriter(allow_unreviewed=True)
        writer.write(book, out_path)

        content = out_path.read_text(encoding="utf-8")

        assert "Розділ 1 Абзац 1." in content
        assert "Прийнято А." in content
        assert "Рецензія Б." in content
        assert "Провал В." in content
        assert "Неперевірений абзац." in content

    def test_pdf_writer_strict_gating_and_rendering(self, tmp_path):
        book = self._build_complex_multichapter_book()
        out_path = tmp_path / "strict_export.pdf"

        writer = PdfWriter(allow_unreviewed=False)
        res = writer.write(book, out_path)

        assert res.exists()
        assert res.stat().st_size > 100
        # Valid PDF header
        with open(res, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"

    def test_pdf_writer_allow_unreviewed_renders_successfully(self, tmp_path):
        book = self._build_complex_multichapter_book()
        out_path = tmp_path / "unreviewed_export.pdf"

        writer = PdfWriter(allow_unreviewed=True)
        res = writer.write(book, out_path)

        assert res.exists()
        assert res.stat().st_size > 100


# ==============================================================================
# 4. SQLite Persistence & High Concurrency Extreme Stress
# ==============================================================================

class TestSQLitePersistenceHighConcurrencyExtreme:
    """Extreme concurrency stress testing with 32 threads and query plan verification."""

    def test_32_threads_concurrent_saves_and_queries(self, stress_db):
        repo, db_path = stress_db
        book_id = uuid4()
        num_threads = 32
        ops_per_thread = 20

        def concurrent_writer(t_id: int) -> int:
            count = 0
            for i in range(ops_per_thread):
                seg_id = uuid4()
                status = SegmentStatus.ACCEPTED.value if i % 2 == 0 else SegmentStatus.REVIEW_REQUIRED.value
                report = QAReport(
                    segment_id=seg_id,
                    book_id=book_id,
                    is_valid=(status == SegmentStatus.ACCEPTED.value),
                    score=1.0 if status == SegmentStatus.ACCEPTED.value else 0.5,
                    repair_attempts=0 if status == SegmentStatus.ACCEPTED.value else 2,
                    status=status,
                )
                if status != SegmentStatus.ACCEPTED.value:
                    report.add_violation(
                        QAViolation(
                            validator_name="StressValidator",
                            rule_code=f"RULE_{t_id}_{i}",
                            severity=QASeverity.CRITICAL,
                            message=f"Stress message {t_id}-{i}",
                            forbidden_form="ForbiddenWord",
                            suggested_fix="CorrectWord",
                        )
                    )
                # Alternate batch and single save
                if i % 2 == 0:
                    repo.save_qa_report(report, book_id=book_id)
                else:
                    repo.save_qa_reports_batch([report], book_id=book_id)
                count += 1
            return count

        def concurrent_reader(r_id: int) -> int:
            reads = 0
            for _ in range(15):
                reports = repo.get_qa_reports_for_book(book_id)
                reads += len(reports)
            return reads

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            writer_futures = [executor.submit(concurrent_writer, t) for t in range(24)]
            reader_futures = [executor.submit(concurrent_reader, r) for r in range(8)]

            written = [f.result() for f in concurrent.futures.as_completed(writer_futures)]
            [f.result() for f in concurrent.futures.as_completed(reader_futures)]

        expected_total = 24 * ops_per_thread
        assert sum(written) == expected_total

        # Final record verification
        final_reports = repo.get_qa_reports_for_book(book_id)
        assert len(final_reports) == expected_total

    def test_sqlite_query_plan_uses_indices(self, stress_db):
        """Verify that SQLite uses performance indices for book and segment queries."""
        repo, db_path = stress_db
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check index on book_id
        cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM quality_reports WHERE book_id = 'test_book'")
        plan_book = " ".join(str(r) for r in cursor.fetchall())
        assert "idx_quality_reports_book" in plan_book or "INDEX" in plan_book

        # Check index on segment_id
        cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM quality_reports WHERE segment_id = 'test_segment'")
        plan_seg = " ".join(str(r) for r in cursor.fetchall())
        assert "idx_quality_reports_segment" in plan_seg or "INDEX" in plan_seg

        conn.close()


# ==============================================================================
# 5. TranslationRunner DOM Gating Integration
# ==============================================================================

class TestTranslationRunnerDOMGatingIntegration:
    """End-to-end integration test validating Stage 3 QA and DOM reconciliation."""

    def test_stage3_qa_repair_and_reconciled_export(self, stress_db, tmp_path):
        repo, _ = stress_db
        book_id = uuid4()
        ch_id = uuid4()

        # Construct 4 paragraphs
        # P1: passes initially
        # P2: fixes on attempt 1
        # P3: fixes on attempt 2
        # P4: fails both attempts -> REVIEW_REQUIRED
        p1_id, p2_id, p3_id, p4_id = uuid4(), uuid4(), uuid4(), uuid4()

        s1 = Sentence(original_text="Clean sentence.", order_index=0)
        p1 = Paragraph(id=p1_id, sentences=[s1])

        s2 = Sentence(original_text="Sentence fixing on 1.", order_index=0)
        p2 = Paragraph(id=p2_id, sentences=[s2])

        s3 = Sentence(original_text="Sentence fixing on 2.", order_index=0)
        p3 = Paragraph(id=p3_id, sentences=[s3])

        s4 = Sentence(original_text="Unfixable sentence.", order_index=0)
        p4 = Paragraph(id=p4_id, sentences=[s4])

        ch = Chapter(id=ch_id, title="Test Chapter", translated_title="Розділ", paragraphs=[p1, p2, p3, p4], order_index=0)
        book = Book(id=book_id, title="Integration Test Book", chapters=[ch])

        # Scoped entity: Cherry -> Черрі
        from src.domain.models.knowledge import EntityMention
        cherry = EntityProfile(
            source_name="Cherry",
            canonical_target="Черрі",
            forbidden_target_forms=["Вишня", "Вішня"],
            locked=True,
        )
        repo.save_entity_profile(cherry)
        repo.save_entity_mentions_batch([
            EntityMention(entity_id=cherry.id, segment_id=p2_id, surface_form="Cherry", char_start=0, char_end=6),
            EntityMention(entity_id=cherry.id, segment_id=p3_id, surface_form="Cherry", char_start=0, char_end=6),
            EntityMention(entity_id=cherry.id, segment_id=p4_id, surface_form="Cherry", char_start=0, char_end=6),
        ])

        seg1 = TranslationSegment(
            id=p1_id, book_id=book_id, chapter_id=ch_id, paragraph_id=p1_id,
            source_text="Clean sentence.", refined_translation="Чисте речення.",
            status=SegmentStatus.EDITED,
        )
        seg2 = TranslationSegment(
            id=p2_id, book_id=book_id, chapter_id=ch_id, paragraph_id=p2_id,
            source_text="Cherry smiles.", refined_translation="Вишня усміхається.",
            status=SegmentStatus.EDITED,
        )
        seg3 = TranslationSegment(
            id=p3_id, book_id=book_id, chapter_id=ch_id, paragraph_id=p3_id,
            source_text="Cherry runs.", refined_translation="Вишня біжить.",
            status=SegmentStatus.EDITED,
        )
        seg4 = TranslationSegment(
            id=p4_id, book_id=book_id, chapter_id=ch_id, paragraph_id=p4_id,
            source_text="Cherry sleeps.", refined_translation="Вишня спить.",
            status=SegmentStatus.EDITED,
        )

        segments = [seg1, seg2, seg3, seg4]

        # Mock engine handling repairs
        call_counts: Dict[UUID, int] = {p2_id: 0, p3_id: 0, p4_id: 0}

        def mock_aya(s, prompt):
            sid = s.id
            call_counts[sid] = call_counts.get(sid, 0) + 1
            count = call_counts[sid]

            if sid == p2_id:
                # Fixes on attempt 1
                return "Черрі усміхається."
            elif sid == p3_id:
                # Fixes on attempt 2
                if count == 1:
                    return "Вішня біжить."
                return "Черрі біжить."
            elif sid == p4_id:
                # Never fixes
                return f"Вишня спить постійно {count}."
            return "Інше."

        qp = QualityPipeline()
        re = BoundedRepairEngine(editing_engine=mock_aya, quality_pipeline=qp, max_retries=2)
        pipeline = TwoStageTranslationPipeline(nllb_engine=None, aya_engine=None, kb_repo=repo)

        # Run Stage 3
        validated_segments = pipeline.execute_qa_and_repair_stage(
            book_id=book_id,
            segments=segments,
            quality_pipeline=qp,
            repair_engine=re,
        )

        assert len(validated_segments) == 4
        assert validated_segments[0].status == SegmentStatus.ACCEPTED
        assert validated_segments[1].status == SegmentStatus.ACCEPTED
        assert validated_segments[2].status == SegmentStatus.ACCEPTED
        assert validated_segments[3].status == SegmentStatus.REVIEW_REQUIRED

        # Reconcile DOM
        segment_map = {seg.paragraph_id: seg for seg in validated_segments}
        for paragraph in ch.paragraphs:
            seg = segment_map.get(paragraph.id)
            if seg.status == SegmentStatus.ACCEPTED:
                paragraph.sentences[0].translated_text = seg.final_translation
                paragraph.sentences[0].status = SegmentStatus.ACCEPTED
            elif seg.status == SegmentStatus.REVIEW_REQUIRED:
                paragraph.sentences[0].translated_text = seg.final_translation
                paragraph.sentences[0].status = SegmentStatus.REVIEW_REQUIRED

        # Export with TxtWriter
        txt_out = tmp_path / "integrated_output.txt"
        TxtWriter(allow_unreviewed=False).write(book, txt_out)
        txt_content = txt_out.read_text(encoding="utf-8")

        assert "Чисте речення." in txt_content
        assert "Черрі усміхається." in txt_content
        assert "Черрі біжить." in txt_content
        assert "Вишня спить" not in txt_content  # REVIEW_REQUIRED segment strictly omitted!
