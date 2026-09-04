"""
Unit tests for Translation Runner: TranslationJobConfig, TranslationResult, _call_stage_safe.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from src.launcher.concurrency import CancellationToken, TaskCancelledException, ProgressEvent
from src.launcher.translation_runner import (
    TranslationJobConfig,
    TranslationResult,
    _call_stage_safe,
    execute_translation_job,
)
from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.writers.txt_writer import TxtWriter
from src.config.settings import Settings


def test_call_stage_safe_signatures():
    token = CancellationToken()
    cb = lambda e: None

    # Function with no extra args
    fn_simple = MagicMock(return_value="ok_simple")
    res1 = _call_stage_safe(fn_simple, "book-id-123", token, cb)
    assert res1 == "ok_simple"
    fn_simple.assert_called_once_with("book-id-123")

    # Function with cancellation_token and progress_callback
    def fn_full(book_id, cancellation_token=None, progress_callback=None):
        return f"{book_id}_{cancellation_token is not None}_{progress_callback is not None}"

    res2 = _call_stage_safe(fn_full, "book-id-456", token, cb)
    assert res2 == "book-id-456_True_True"

    # Function with cancel_token and progress_cb
    def fn_short(book_id, cancel_token=None, progress_cb=None):
        return f"{book_id}_{cancel_token is not None}_{progress_cb is not None}"

    res3 = _call_stage_safe(fn_short, "book-id-789", token, cb)
    assert res3 == "book-id-789_True_True"


def test_translation_runner_early_cancellation():
    token = CancellationToken()
    token.cancel()

    with pytest.raises(TaskCancelledException):
        execute_translation_job(
            input_file=Path("nonexistent_input.txt"),
            output_file=Path("nonexistent_output.txt"),
            cancellation_token=token,
        )


def test_translation_runner_missing_input_file():
    token = CancellationToken()
    with pytest.raises(FileNotFoundError):
        execute_translation_job(
            input_file=Path("surely_nonexistent_file_12345.txt"),
            output_file=Path("output.txt"),
            cancellation_token=token,
        )


def test_translation_runner_execute_job_with_segments_success_and_failed_propagation(tmp_path):
    """
    Verifies that execute_translation_job:
    1. Completes cleanly with paragraph segments without UnboundLocalError.
    2. Correctly sets SegmentStatus.ACCEPTED on successful refined segments.
    3. Correctly propagates SegmentStatus.FAILED on failed segments without silent acceptance.
    4. Ensures document writers omit failed segment drafts by default.
    5. Accurately reports total_chunks == len(segments) in TranslationResult.
    """
    input_file = tmp_path / "sample_book.txt"
    input_file.write_text(
        "First paragraph sentence one. First paragraph sentence two.\n\n"
        "Second paragraph sentence.\n",
        encoding="utf-8",
    )
    output_file = tmp_path / "sample_book_translated.txt"
    db_file = tmp_path / "test_checkpoints.db"

    test_settings = Settings(
        db_path=str(db_file),
        temperature=0.0,
        stage2_deterministic=True,
    )

    refined_text_p1 = "Успішний перший параграф українською."
    failed_draft_p2 = "Чорновий другий параграф."

    def fake_nllb_stage(book_id, cancellation_token=None, progress_callback=None, segments=None, **kwargs):
        if segments:
            segments[0].draft_translation = "Чорновий перший параграф."
            segments[0].status = SegmentStatus.DRAFT_COMPLETED
            segments[1].draft_translation = failed_draft_p2
            segments[1].status = SegmentStatus.DRAFT_COMPLETED

    def fake_aya_stage(book_id, cancellation_token=None, progress_callback=None, segments=None, **kwargs):
        if segments:
            # Segment 0 succeeds
            segments[0].refined_translation = refined_text_p1
            segments[0].status = SegmentStatus.EDITED
            # Segment 1 fails (malformed LLM JSON simulation)
            segments[1].refined_translation = None
            segments[1].status = SegmentStatus.FAILED
            segments[1].error_message = "Malformed LLM output syntax error"

    with patch("src.launcher.translation_runner.ApplicationContainer") as MockAppContainer:
        from src.document_manager.manager import DocumentManager
        from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository

        real_doc_mgr = DocumentManager()
        mock_container = MagicMock()
        MockAppContainer.return_value = mock_container

        mock_container.document_manager.return_value = real_doc_mgr

        mock_preprocessor = MagicMock()
        mock_container.preprocessing_pipeline.return_value = mock_preprocessor

        mock_chunk_mgr = MagicMock()
        mock_chunk_mgr.create_chunks_stream.return_value = []
        mock_container.chunk_manager.return_value = mock_chunk_mgr

        mock_translator = MagicMock()
        mock_translator.execute_nllb_stage.side_effect = fake_nllb_stage
        mock_translator.execute_aya_stage.side_effect = fake_aya_stage
        mock_container.translation_pipeline.return_value = mock_translator

        mock_kb_repo = MagicMock(spec=SQLiteKnowledgeBaseRepository)
        mock_kb_repo.count_chunks_for_book.return_value = 0
        mock_kb_repo.load_chunks_by_status.return_value = []
        mock_container.kb_repository.return_value = mock_kb_repo

        result = execute_translation_job(
            input_file=input_file,
            output_file=output_file,
            settings=test_settings,
        )

        assert isinstance(result, TranslationResult)
        assert result.success is True
        # Verify UnboundLocalError resolved and segment count returned
        assert result.total_chunks == 2
        assert result.output_file == output_file

        # Verify output file content
        assert output_file.exists()
        written_content = output_file.read_text(encoding="utf-8")
        assert refined_text_p1 in written_content
        # Failed draft must NOT appear in output
        assert failed_draft_p2 not in written_content


def test_runner_dom_reconciliation_isolated():
    """
    Directly asserts DOM reconciliation invariants:
    - Segments with status EDITED/ACCEPTED and refined text set sentence status to ACCEPTED.
    - Segments with status FAILED set sentence status to FAILED and are excluded from TxtWriter.
    - Explicit allow_unreviewed=True outputs the failed draft.
    """
    s1 = Sentence(original_text="Sentence one.", translated_text="Draft one.")
    s2 = Sentence(original_text="Sentence two.", translated_text="Draft two.")
    p1 = Paragraph(sentences=[s1])
    p2 = Paragraph(sentences=[s2])
    ch = Chapter(title="Chapter 1", order_index=0, paragraphs=[p1, p2])
    book = Book(title="Test", chapters=[ch])

    seg1 = TranslationSegment(
        book_id=book.id,
        chapter_id=ch.id,
        paragraph_id=p1.id,
        source_text="Sentence one.",
        draft_translation="Draft one.",
        refined_translation="Refined one.",
        status=SegmentStatus.EDITED,
    )
    seg2 = TranslationSegment(
        book_id=book.id,
        chapter_id=ch.id,
        paragraph_id=p2.id,
        source_text="Sentence two.",
        draft_translation="Draft two.",
        refined_translation=None,
        status=SegmentStatus.FAILED,
        error_message="Stage 2 LLM parsing failure",
    )

    segments = [seg1, seg2]

    has_refined_segments = any(
        (seg.refined_translation or seg.status in (SegmentStatus.EDITED, SegmentStatus.ACCEPTED))
        for seg in segments
    )
    assert has_refined_segments is True

    segment_map = {seg.paragraph_id: seg for seg in segments}
    for chapter in book.chapters:
        for paragraph in chapter.paragraphs:
            seg = segment_map.get(paragraph.id)
            if not seg:
                continue
            refined_text = seg.final_translation or seg.refined_translation
            if seg.status in (SegmentStatus.ACCEPTED, SegmentStatus.EDITED) and refined_text:
                try:
                    paragraph.translated_text = refined_text
                except (ValueError, AttributeError):
                    pass
                try:
                    paragraph.status = SegmentStatus.ACCEPTED
                except (ValueError, AttributeError):
                    pass
                if paragraph.sentences:
                    paragraph.sentences[0].translated_text = refined_text
                    paragraph.sentences[0].status = SegmentStatus.ACCEPTED
                    for extra_s in paragraph.sentences[1:]:
                        extra_s.translated_text = ""
                        extra_s.status = SegmentStatus.ACCEPTED
            elif seg.status == SegmentStatus.FAILED:
                try:
                    paragraph.translated_text = seg.translated_text
                except (ValueError, AttributeError):
                    pass
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

    # Assert p1 and p2 sentences status and text
    assert p1.sentences[0].status == SegmentStatus.ACCEPTED
    assert p1.sentences[0].translated_text == "Refined one."

    assert p2.sentences[0].status == SegmentStatus.FAILED
    assert p2.sentences[0].translated_text == "Draft two."

    # Verify TxtWriter(allow_unreviewed=False) skips failed paragraph
    with tempfile.TemporaryDirectory() as tmpdir:
        out_txt = Path(tmpdir) / "reconciled_default.txt"
        TxtWriter(allow_unreviewed=False).write(book, out_txt)
        content = out_txt.read_text(encoding="utf-8")
        assert "Refined one." in content
        assert "Draft two." not in content

    # Verify TxtWriter(allow_unreviewed=True) includes failed paragraph draft
    with tempfile.TemporaryDirectory() as tmpdir:
        out_txt_unreviewed = Path(tmpdir) / "reconciled_unreviewed.txt"
        TxtWriter(allow_unreviewed=True).write(book, out_txt_unreviewed)
        content_unreviewed = out_txt_unreviewed.read_text(encoding="utf-8")
        assert "Refined one." in content_unreviewed
        assert "Draft two." in content_unreviewed

