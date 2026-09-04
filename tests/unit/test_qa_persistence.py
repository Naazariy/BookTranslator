"""
tests/unit/test_qa_persistence.py

Unit tests for SQLite QA persistence: migration 003, quality_reports table,
audit log saving and querying, and integration with TwoStageTranslationPipeline.
"""
from uuid import uuid4
import sqlite3
import tempfile
from pathlib import Path
import pytest

from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.knowledge_base.migrations.migration_runner import MigrationRunner
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.quality.models import QAReport, QAViolation, QASeverity
from src.translation.pipeline import TwoStageTranslationPipeline


@pytest.fixture
def temp_db():
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


def test_migration_003_creates_table_and_indexes(temp_db):
    repo, db_path = temp_db
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check table existence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quality_reports'")
    assert cursor.fetchone() is not None

    # Check index existence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    index_names = {r[0] for r in cursor.fetchall()}
    assert "idx_quality_reports_book" in index_names
    assert "idx_quality_reports_segment" in index_names
    assert "idx_quality_reports_book_status" in index_names
    assert "idx_quality_reports_severity" in index_names
    conn.close()


def test_save_and_get_qa_reports(temp_db):
    repo, _ = temp_db
    book_id = uuid4()
    seg_id = uuid4()

    report = QAReport(
        segment_id=seg_id,
        book_id=book_id,
        is_valid=False,
        score=0.6,
        repair_attempts=1,
        status=SegmentStatus.REVIEW_REQUIRED.value,
    )
    report.add_violation(
        QAViolation(
            validator_name="EntityConsistencyValidator",
            rule_code="FORBIDDEN_ENTITY_VARIANT",
            severity=QASeverity.CRITICAL,
            message="Forbidden variant detected.",
            target_snippet="Вишня",
            forbidden_form="Вишня",
            suggested_fix="Черрі",
        )
    )

    repo.save_qa_report(report, book_id=book_id)

    # Query for book
    book_reports = repo.get_qa_reports_for_book(book_id)
    assert len(book_reports) >= 1
    rec = book_reports[0]
    assert rec["book_id"] == str(book_id)
    assert rec["segment_id"] == str(seg_id)
    assert rec["validator_name"] == "EntityConsistencyValidator"
    assert rec["severity"] == "CRITICAL"
    assert rec["message"] == "Forbidden variant detected."
    assert rec["details"]["forbidden_form"] == "Вишня"
    assert rec["details"]["suggested_fix"] == "Черрі"
    assert rec["status"] == "REVIEW_REQUIRED"

    # Query for segment
    seg_reports = repo.get_qa_reports_for_segment(seg_id)
    assert len(seg_reports) == 1
    assert seg_reports[0]["segment_id"] == str(seg_id)


def test_save_qa_reports_batch_clean_pass(temp_db):
    repo, _ = temp_db
    book_id = uuid4()
    seg_id = uuid4()

    # Clean pass with no violations
    clean_report = QAReport(
        segment_id=seg_id,
        book_id=book_id,
        is_valid=True,
        score=1.0,
        repair_attempts=0,
        status=SegmentStatus.ACCEPTED.value,
    )

    repo.save_qa_reports_batch([clean_report], book_id=book_id)

    seg_reports = repo.get_qa_reports_for_segment(seg_id)
    assert len(seg_reports) == 1
    assert seg_reports[0]["validator_name"] == "QualityPipeline"
    assert seg_reports[0]["severity"] == "INFO"
    assert seg_reports[0]["status"] == "ACCEPTED"


def test_segment_persistence_methods(temp_db):
    repo, _ = temp_db
    book_id = uuid4()

    seg1 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="First source paragraph.",
        refined_translation="Перший абзац.",
        final_translation="Перший абзац.",
        status=SegmentStatus.ACCEPTED,
    )
    seg2 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="Second source paragraph.",
        status=SegmentStatus.REVIEW_REQUIRED,
    )

    repo.save_segments_batch([seg1, seg2])

    accepted = repo.load_segments_by_status(book_id, SegmentStatus.ACCEPTED)
    assert len(accepted) == 1
    assert accepted[0].id == seg1.id
    assert accepted[0].final_translation == "Перший абзац."

    unreviewed = repo.load_segments_by_status(book_id, SegmentStatus.REVIEW_REQUIRED)
    assert len(unreviewed) == 1
    assert unreviewed[0].id == seg2.id


def test_execute_qa_and_repair_stage_persists_reports(temp_db):
    repo, _ = temp_db
    book_id = uuid4()

    # Segment 1: perfectly valid
    seg1 = TranslationSegment(
        id=uuid4(),
        book_id=book_id,
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text="The sun was shining.",
        refined_translation="Сонце світило яскраво.",
        status=SegmentStatus.EDITED,
    )

    pipeline = TwoStageTranslationPipeline(
        nllb_engine=None,
        aya_engine=None,
        kb_repo=repo,
    )

    validated_segments = pipeline.execute_qa_and_repair_stage(
        book_id=book_id,
        segments=[seg1],
    )

    assert len(validated_segments) == 1
    assert validated_segments[0].status == SegmentStatus.ACCEPTED

    # Verify report was persisted in SQLite
    reports = repo.get_qa_reports_for_book(book_id)
    assert len(reports) >= 1
    assert reports[0]["status"] == "ACCEPTED"
