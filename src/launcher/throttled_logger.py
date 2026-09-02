"""
Throttled GUI log handler and circular buffer controller for CustomTkinter Textbox.
Part of Milestone 1 (UI & Concurrency Optimization) for BookTranslator.
"""
from __future__ import annotations

import logging
import queue
from typing import Any, List, Optional

try:
    import customtkinter as ctk
except ImportError:
    ctk = None  # type: ignore


class ThrottledLogHandler(logging.Handler):
    """
    Logging handler that queues formatted records into a thread-safe Queue.
    Zero GUI calls are made in emit(), preventing main-thread event loop starvation.
    """

    def __init__(self, log_queue: queue.Queue[str]) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted = self.format(record)
            self.log_queue.put(formatted)
        except Exception:
            self.handleError(record)


class ThrottledLogBuffer:
    """
    Manages high-frequency log updates to a CTkTextbox with 50ms batching
    and circular ring-buffer truncation (max 1,000 lines) to bound UI memory.
    """

    def __init__(
        self,
        textbox: Any,
        flush_interval_ms: int = 50,
        max_lines: int = 1000,
        formatter: Optional[logging.Formatter] = None,
    ) -> None:
        self.textbox = textbox
        self.flush_interval_ms = flush_interval_ms
        self.max_lines = max_lines
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._line_count = 0
        self._after_id: Optional[str] = None
        self._is_active = True

        # Setup and attach logging handler
        self.handler = ThrottledLogHandler(self.log_queue)
        if formatter is None:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        self.handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(self.handler)
        if root_logger.level == logging.NOTSET:
            root_logger.setLevel(logging.INFO)

        # Start batch flushing loop if textbox is attached
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if not self._is_active or self.textbox is None:
            return

        try:
            if hasattr(self.textbox, "winfo_exists") and not self.textbox.winfo_exists():
                return
            if hasattr(self.textbox, "after"):
                self._after_id = self.textbox.after(self.flush_interval_ms, self._flush)
        except Exception:
            pass

    def _flush(self) -> None:
        """Flush accumulated log lines in a single UI operation."""
        if not self._is_active or self.textbox is None:
            return

        try:
            if hasattr(self.textbox, "winfo_exists") and not self.textbox.winfo_exists():
                return
        except Exception:
            return

        batch: List[str] = []
        while True:
            try:
                batch.append(self.log_queue.get_nowait())
            except queue.Empty:
                break

        if batch:
            text_block = "\n".join(batch) + "\n"
            num_new_lines = len(batch)

            try:
                end_token = ctk.END if ctk is not None else "end"
                self.textbox.configure(state="normal")
                self.textbox.insert(end_token, text_block)
                self._line_count += num_new_lines

                # Enforce circular buffer truncation if line limit is exceeded
                if self._line_count > self.max_lines:
                    excess_lines = self._line_count - self.max_lines
                    self.textbox.delete("1.0", f"{excess_lines + 1}.0")
                    self._line_count = self.max_lines

                self.textbox.configure(state="disabled")
                self.textbox.yview(end_token)
            except Exception as e:
                logging.getLogger(__name__).debug(f"Error appending log batch to textbox: {e}")

        self._schedule_flush()

    def flush_synchronously(self) -> int:
        """
        Drains and appends all queued log lines synchronously.
        Useful for benchmarks, unit tests, or clean flushes before shutdown.
        """
        batch: List[str] = []
        while True:
            try:
                batch.append(self.log_queue.get_nowait())
            except queue.Empty:
                break

        if batch and self.textbox is not None:
            text_block = "\n".join(batch) + "\n"
            num_new_lines = len(batch)

            try:
                end_token = ctk.END if ctk is not None else "end"
                self.textbox.configure(state="normal")
                self.textbox.insert(end_token, text_block)
                self._line_count += num_new_lines

                if self._line_count > self.max_lines:
                    excess_lines = self._line_count - self.max_lines
                    self.textbox.delete("1.0", f"{excess_lines + 1}.0")
                    self._line_count = self.max_lines

                self.textbox.configure(state="disabled")
                if hasattr(self.textbox, "yview"):
                    self.textbox.yview(end_token)
            except Exception as e:
                logging.getLogger(__name__).debug(f"Error in synchronous log flush: {e}")

        return len(batch)

    def clear(self) -> None:
        """Clear the console textbox and reset line counter."""
        if self.textbox is not None:
            try:
                end_token = ctk.END if ctk is not None else "end"
                self.textbox.configure(state="normal")
                self.textbox.delete("1.0", end_token)
                self.textbox.configure(state="disabled")
                self._line_count = 0
            except Exception:
                pass

    def shutdown(self) -> None:
        """Detach logger and stop pending timer."""
        self._is_active = False
        if self._after_id is not None and self.textbox is not None:
            try:
                if hasattr(self.textbox, "after_cancel"):
                    self.textbox.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        root_logger = logging.getLogger()
        if self.handler in root_logger.handlers:
            root_logger.removeHandler(self.handler)
