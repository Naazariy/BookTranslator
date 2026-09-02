"""
Unit tests for Translation Runner: TranslationJobConfig, TranslationResult, _call_stage_safe.
"""
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from src.launcher.concurrency import CancellationToken, TaskCancelledException, ProgressEvent
from src.launcher.translation_runner import (
    TranslationJobConfig,
    TranslationResult,
    _call_stage_safe,
    execute_translation_job,
)


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
