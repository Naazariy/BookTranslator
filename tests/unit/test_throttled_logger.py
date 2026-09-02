"""
Unit tests for ThrottledLogHandler and ThrottledLogBuffer.
"""
import logging
import queue
import pytest
from unittest.mock import MagicMock
from src.launcher.throttled_logger import ThrottledLogHandler, ThrottledLogBuffer


class MockTextbox:
    """Mock Tkinter/CTk Textbox widget for headless unit testing."""
    def __init__(self):
        self.content = []
        self.state = "disabled"
        self.exists = True

    def winfo_exists(self):
        return self.exists

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]

    def insert(self, index, text):
        assert self.state == "normal"
        self.content.append(text)

    def delete(self, start, end):
        assert self.state == "normal"
        # Simulate deleting lines
        # start is "1.0", end is f"{excess + 1}.0" or "end"
        if start == "1.0":
            if str(end).lower() == "end":
                self.content = []
                return
            try:
                line_idx = int(float(end.replace(".0", ""))) - 1
                all_text = "".join(self.content)
                lines = all_text.splitlines(keepends=True)
                remaining = lines[line_idx:]
                self.content = remaining
            except Exception:
                pass

    def yview(self, pos):
        pass

    def get_text(self):
        return "".join(self.content)


def test_throttled_log_handler():
    log_q = queue.Queue()
    handler = ThrottledLogHandler(log_q)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger = logging.getLogger("test_handler")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("Test message 1")
    logger.warning("Test message 2")

    assert not log_q.empty()
    m1 = log_q.get_nowait()
    m2 = log_q.get_nowait()

    assert m1 == "INFO: Test message 1"
    assert m2 == "WARNING: Test message 2"


def test_throttled_log_buffer_batching_and_ring_buffer():
    mock_tb = MockTextbox()
    buffer = ThrottledLogBuffer(mock_tb, flush_interval_ms=50, max_lines=5)

    test_logger = logging.getLogger("test_buffer_logger")
    test_logger.setLevel(logging.INFO)

    # Log 8 messages (exceeding max_lines=5)
    for i in range(8):
        test_logger.info(f"Line {i}")

    # Synchronous flush
    flushed_count = buffer.flush_synchronously()
    assert flushed_count == 8
    assert buffer._line_count == 5

    # Clear buffer
    buffer.clear()
    assert buffer._line_count == 0
    assert mock_tb.get_text() == ""

    # Shutdown
    buffer.shutdown()
    assert not buffer._is_active
