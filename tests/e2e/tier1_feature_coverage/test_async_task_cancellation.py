"""Tier 1 Feature Tests: AsyncTaskManager & Cancellation (F2).
Verifies worker thread lifecycle, thread-safe cancellation tokens, callbacks, and error handling.
"""
import time
import pytest
from tests.e2e.fixtures import CancellationToken, AsyncTaskManager

try:
    from src.launcher.concurrency import (
        CancellationToken as ProjectCancellationToken,
        AsyncTaskManager as ProjectAsyncTaskManager,
    )
except ImportError:
    ProjectCancellationToken = CancellationToken
    ProjectAsyncTaskManager = AsyncTaskManager


def get_token_cls():
    return ProjectCancellationToken if ProjectCancellationToken is not None else CancellationToken


def get_manager_cls():
    return ProjectAsyncTaskManager if ProjectAsyncTaskManager is not None else AsyncTaskManager


class TestAsyncTaskCancellation:
    def test_cancellation_token_initial_and_cancel_state(self):
        """Test F2.1: Cancellation token state changes."""
        token_cls = get_token_cls()
        token = token_cls()
        
        assert token.is_cancelled() is False
        token.cancel()
        assert token.is_cancelled() is True

    def test_cancellation_token_reset_and_callbacks(self):
        """Test F2.2: Cancellation token callbacks and reset functionality."""
        token_cls = get_token_cls()
        token = token_cls()
        callback_called = []

        token.register_callback(lambda: callback_called.append(True))
        assert len(callback_called) == 0

        token.cancel()
        assert len(callback_called) == 1
        assert token.is_cancelled() is True

        token.reset()
        assert token.is_cancelled() is False

    def test_async_task_manager_start_and_complete(self):
        """Test F2.3: Task manager starts background worker and completes cleanly."""
        mgr_cls = get_manager_cls()
        mgr = mgr_cls()
        results = []

        def worker_func(cancel_token):
            for i in range(5):
                if cancel_token.is_cancelled():
                    break
                results.append(i)
                time.sleep(0.01)

        started = mgr.start_task(worker_func)
        assert started is True
        assert mgr.is_running() is True
        
        mgr.join(timeout=2.0)
        assert mgr.is_running() is False
        assert results == [0, 1, 2, 3, 4]
        assert mgr.get_error() is None

    def test_async_task_manager_cancel_running_task(self):
        """Test F2.4: Immediate cancellation halts worker loop prematurely."""
        mgr_cls = get_manager_cls()
        mgr = mgr_cls()
        iterations = []

        def long_running_worker(cancel_token):
            for i in range(100):
                if cancel_token.is_cancelled():
                    break
                iterations.append(i)
                time.sleep(0.01)

        mgr.start_task(long_running_worker)
        time.sleep(0.03)  # Let it run 2-3 iterations
        mgr.cancel()
        mgr.join(timeout=1.0)

        assert mgr.is_running() is False
        assert len(iterations) < 20  # Stopped well before 100

    def test_async_task_manager_handles_worker_exception(self):
        """Test F2.5: Worker exceptions are captured safely without crashing caller."""
        mgr_cls = get_manager_cls()
        mgr = mgr_cls()

        def failing_worker(cancel_token):
            raise ValueError("Test worker intentional failure")

        mgr.start_task(failing_worker)
        mgr.join(timeout=1.0)

        assert mgr.is_running() is False
        err = mgr.get_error()
        assert err is not None
        assert isinstance(err, ValueError)
        assert "Test worker intentional failure" in str(err)

    def test_async_task_manager_prevent_duplicate_concurrent_tasks(self):
        """Test F2.6: Starting a task while one is running returns False."""
        mgr_cls = get_manager_cls()
        mgr = mgr_cls()

        def worker(cancel_token):
            time.sleep(0.1)

        first_started = mgr.start_task(worker)
        second_started = mgr.start_task(worker)

        assert first_started is True
        assert second_started is False
        mgr.join(timeout=1.0)
