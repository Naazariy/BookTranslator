"""
Thread-safe concurrency utilities, cancellation tokens, and UI event queue dispatching.
Part of Milestone 1 (UI & Concurrency Optimization) for BookTranslator.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class _CallableBool(int):
    """
    An integer subclass that behaves like a boolean (1=True, 0=False)
    and is also callable () -> bool.
    This guarantees full compatibility whether callers use `token.is_cancelled`
    as a property/attribute or `token.is_cancelled()` as a method.
    """

    def __call__(self) -> bool:
        return bool(self)

    def __bool__(self) -> bool:
        return super().__int__() != 0

    def __repr__(self) -> str:
        return "True" if self else "False"


class TaskState(Enum):
    """Lifecycle states for background tasks managed by AsyncTaskManager."""
    IDLE = auto()
    RUNNING = auto()
    CANCELLING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskCancelledException(Exception):
    """Raised when a background translation task is cancelled cooperatively."""
    pass


class CancellationToken:
    """
    Thread-safe cooperative cancellation token shared between UI and worker threads.
    Supports both `token.is_cancelled` (property/bool evaluation) and `token.is_cancelled()` (callable).
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Signal cancellation request to all listening threads/workers."""
        self._event.set()
        with self._lock:
            for cb in self._callbacks:
                try:
                    cb()
                except Exception as ex:
                    logger.debug(f"Exception in cancellation callback: {ex}")

    def reset(self) -> None:
        """Reset the cancellation token for a new task execution."""
        self._event.clear()

    @property
    def is_cancelled(self) -> _CallableBool:
        """
        Check if cancellation has been requested.
        Can be used as a property: `if token.is_cancelled:`
        or called as a method: `if token.is_cancelled():`
        """
        return _CallableBool(1 if self._event.is_set() else 0)

    def is_set(self) -> bool:
        """Check if cancellation event is set."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise TaskCancelledException immediately if cancellation was requested."""
        if self._event.is_set():
            raise TaskCancelledException("Task was cancelled by user.")

    def register_callback(self, cb: Callable[[], None]) -> None:
        """Register a callback to be executed when cancellation is triggered."""
        with self._lock:
            if self._event.is_set():
                try:
                    cb()
                except Exception as ex:
                    logger.debug(f"Exception in cancellation callback: {ex}")
            else:
                self._callbacks.append(cb)


@dataclass
class ProgressEvent:
    """
    Structured progress update dispatched from background tasks to the UI.
    Fully compatible with positional and keyword arguments across all project components.
    """
    stage: Any = 1
    current: int = 0
    total: int = 0
    eta_seconds: float = 0.0
    status_message: str = ""
    stage_name: str = ""
    percentage: float = 0.0
    speed_chunks_per_sec: float = 0.0
    message: str = ""

    def __post_init__(self) -> None:
        # Calculate percentage if not provided
        if self.percentage == 0.0 and self.total > 0:
            object.__setattr__(self, "percentage", max(0.0, min(1.0, self.current / self.total)))

        # Synchronize status_message and message
        if not self.status_message and self.message:
            object.__setattr__(self, "status_message", self.message)
        elif not self.message and self.status_message:
            object.__setattr__(self, "message", self.status_message)

        # Populate human-readable stage_name if empty
        if not self.stage_name:
            st = str(self.stage).upper()
            if st in ("1", "STAGE_1", "STAGE_1_NLLB"):
                object.__setattr__(self, "stage_name", "Етап 1: Машинний переклад (NLLB)")
            elif st in ("2", "STAGE_2", "STAGE_2_AYA"):
                object.__setattr__(self, "stage_name", "Етап 2: Літературне редагування (Aya)")
            elif st in ("PARSING", "0"):
                object.__setattr__(self, "stage_name", "Парсинг та сегментація документа")
            elif st in ("REBUILDING", "3"):
                object.__setattr__(self, "stage_name", "Збирання вихідного документа")
            else:
                object.__setattr__(self, "stage_name", str(self.stage))


class UIEventQueue:
    """
    Thread-safe event queue for dispatching callbacks to the Tkinter UI thread.
    Zero Tkinter calls are executed in worker threads. The Tkinter main thread
    polls and drains this queue at 50Hz (every 20ms) with a capped batch size.
    """

    def __init__(self, poll_interval_ms: int = 20, max_events_per_tick: int = 50) -> None:
        self._queue: queue.Queue[Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]] = queue.Queue()
        self.poll_interval_ms = poll_interval_ms
        self.max_events_per_tick = max_events_per_tick
        self._after_id: Optional[str] = None
        self._widget: Any = None
        self._lock = threading.Lock()

    def start_polling(self, widget: Any) -> None:
        """Attach to a Tkinter widget and begin the polling loop on the UI thread."""
        with self._lock:
            self._widget = widget
            self._schedule_poll()

    def stop_polling(self) -> None:
        """Stop polling and cancel any pending Tk timer callback."""
        with self._lock:
            if self._widget is not None and self._after_id is not None:
                try:
                    self._widget.after_cancel(self._after_id)
                except Exception:
                    pass
                self._after_id = None
            self._widget = None

    def post(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """
        Thread-safe post. Can be called from ANY thread (e.g. background worker threads).
        Does NOT invoke any Tkinter functions directly.
        """
        self._queue.put((callback, args, kwargs))

    def _schedule_poll(self) -> None:
        if self._widget is not None:
            try:
                # Check widget existence if supported
                if hasattr(self._widget, "winfo_exists") and not self._widget.winfo_exists():
                    self._widget = None
                    return
                self._after_id = self._widget.after(self.poll_interval_ms, self._process_queue)
            except Exception as e:
                logger.debug(f"Failed to schedule UI queue poll: {e}")
                self._widget = None

    def _process_queue(self) -> None:
        """Executed strictly on the Tkinter main thread."""
        if self._widget is None:
            return

        processed = 0
        while processed < self.max_events_per_tick:
            try:
                callback, args, kwargs = self._queue.get_nowait()
            except queue.Empty:
                break

            try:
                callback(*args, **kwargs)
            except Exception as e:
                cb_name = getattr(callback, "__name__", str(callback))
                logger.error(f"Error in UI queue callback {cb_name}: {e}\n{traceback.format_exc()}")
            finally:
                processed += 1

        self._schedule_poll()

    def drain_all_synchronously(self) -> int:
        """
        Drains all currently queued callbacks synchronously.
        Useful for unit tests or main-thread sync flushes.
        """
        count = 0
        while True:
            try:
                callback, args, kwargs = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args, **kwargs)
            except Exception as e:
                cb_name = getattr(callback, "__name__", str(callback))
                logger.error(f"Error executing synchronous UI callback {cb_name}: {e}")
            finally:
                count += 1
        return count


class AsyncTaskManager:
    """
    Manages the lifecycle, cancellation, progress updates, and exceptions of background workers.
    Ensures safe, non-blocking asynchronous execution and thread-safe marshaling to the UI.
    """

    def __init__(self, ui_queue: Optional[UIEventQueue] = None) -> None:
        self.ui_queue = ui_queue if ui_queue is not None else UIEventQueue()
        self.cancellation_token = CancellationToken()
        self._worker_thread: Optional[threading.Thread] = None
        self._state = TaskState.IDLE
        self._error: Optional[Exception] = None
        self._lock = threading.RLock()

    @property
    def cancel_token(self) -> CancellationToken:
        """Alias property for cancellation_token."""
        return self.cancellation_token

    @property
    def state(self) -> TaskState:
        with self._lock:
            return self._state

    def is_running(self) -> bool:
        """Returns True if a task is currently actively executing or in the process of cancelling."""
        with self._lock:
            return self._state in (TaskState.RUNNING, TaskState.CANCELLING)

    def get_error(self) -> Optional[Exception]:
        """Returns the exception from the most recent task execution, or None."""
        with self._lock:
            return self._error

    def cancel(self) -> None:
        """Alias for cancel_task()."""
        self.cancel_task()

    def join(self, timeout: Optional[float] = None) -> bool:
        """Alias for join_worker(timeout)."""
        return self.join_worker(timeout=timeout if timeout is not None else 2.0)

    def start_task(self, target: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """
        Compatibility runner method that starts a background task executing target(*args, **kwargs).
        Passes cooperative cancellation token if requested or if target accepts it.
        Returns True if started, False if a task is already running.
        """
        with self._lock:
            if self.is_running():
                return False
            self._state = TaskState.RUNNING
            self._error = None
            self.cancellation_token.reset()

        def worker_target() -> None:
            try:
                import inspect
                try:
                    sig = inspect.signature(target)
                    param_names = list(sig.parameters.keys())
                except Exception:
                    param_names = []

                if len(args) == 0 and len(param_names) == 1 and "cancel_token" not in kwargs:
                    target(self.cancellation_token, **kwargs)
                elif "cancel_token" in param_names and "cancel_token" not in kwargs:
                    target(*args, cancel_token=self.cancellation_token, **kwargs)
                else:
                    try:
                        target(*args, cancel_token=self.cancellation_token, **kwargs)
                    except TypeError:
                        target(*args, **kwargs)

                with self._lock:
                    self._state = TaskState.COMPLETED
            except TaskCancelledException:
                with self._lock:
                    self._state = TaskState.CANCELLED
                logger.info("Task was successfully cancelled.")
            except Exception as ex:
                with self._lock:
                    self._state = TaskState.FAILED
                    self._error = ex
                logger.error(f"Task in start_task failed with error: {ex}\n{traceback.format_exc()}")
            finally:
                with self._lock:
                    if self._state in (TaskState.RUNNING, TaskState.CANCELLING):
                        self._state = TaskState.COMPLETED

        self._worker_thread = threading.Thread(
            target=worker_target,
            name="AsyncTaskWorker",
            daemon=True
        )
        self._worker_thread.start()
        return True

    def submit_task(
        self,
        task_func: Callable[[CancellationToken, Callable[[ProgressEvent], None]], Any],
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception, str], None]] = None,
        on_cancelled: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Launch a background worker task in a daemon thread.
        All callbacks are automatically marshaled to the UI thread via UIEventQueue.
        """
        with self._lock:
            if self._state in (TaskState.RUNNING, TaskState.CANCELLING):
                raise RuntimeError("Cannot start task: another task is currently running.")
            self._state = TaskState.RUNNING
            self._error = None
            self.cancellation_token.reset()

        def progress_bridge(event: ProgressEvent) -> None:
            if on_progress:
                self.ui_queue.post(on_progress, event)

        def worker_target() -> None:
            result = None
            try:
                result = task_func(self.cancellation_token, progress_bridge)
                with self._lock:
                    self._state = TaskState.COMPLETED
                if on_success:
                    self.ui_queue.post(on_success, result)

            except TaskCancelledException:
                with self._lock:
                    self._state = TaskState.CANCELLED
                logger.info("Task was successfully cancelled.")
                if on_cancelled:
                    self.ui_queue.post(on_cancelled)

            except Exception as ex:
                with self._lock:
                    self._state = TaskState.FAILED
                    self._error = ex
                tb = traceback.format_exc()
                logger.error(f"Task failed with error: {ex}\n{tb}")
                if on_error:
                    self.ui_queue.post(on_error, ex, tb)

        self._worker_thread = threading.Thread(
            target=worker_target,
            name="TranslationWorker",
            daemon=True
        )
        self._worker_thread.start()

    def cancel_task(self) -> None:
        """Request cooperative cancellation of the running task."""
        with self._lock:
            if self._state == TaskState.RUNNING:
                self._state = TaskState.CANCELLING
                self.cancellation_token.cancel()
                logger.info("Cancellation requested for running task.")

    def join_worker(self, timeout: float = 2.0) -> bool:
        """
        Wait for the worker thread to exit with a bounded timeout.
        Returns True if the thread has finished, False if timed out.
        """
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            return not self._worker_thread.is_alive()
        return True
