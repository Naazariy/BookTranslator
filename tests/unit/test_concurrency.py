"""
Unit tests for Concurrency utilities: CancellationToken, ProgressEvent, UIEventQueue, AsyncTaskManager.
"""
import time
import threading
import pytest
from src.launcher.concurrency import (
    CancellationToken,
    TaskCancelledException,
    ProgressEvent,
    UIEventQueue,
    AsyncTaskManager,
    TaskState,
)


def test_cancellation_token_lifecycle():
    token = CancellationToken()
    assert not token.is_cancelled
    assert not token.is_cancelled()
    assert not token.is_set()

    # Cancel
    token.cancel()
    assert token.is_cancelled
    assert token.is_cancelled()
    assert token.is_set()
    assert token.is_cancelled == True
    assert repr(token.is_cancelled) == "True"

    with pytest.raises(TaskCancelledException):
        token.raise_if_cancelled()

    # Reset
    token.reset()
    assert not token.is_cancelled
    assert not token.is_cancelled()
    assert not token.is_set()
    assert token.is_cancelled == False
    assert repr(token.is_cancelled) == "False"
    # Should not raise
    token.raise_if_cancelled()


def test_progress_event_defaults_and_synchronization():
    # Positional args (stage, current, total, eta_seconds, status_message)
    event1 = ProgressEvent(1, 25, 100, 150.5, "Translating chunk 25/100")
    assert event1.stage == 1
    assert event1.current == 25
    assert event1.total == 100
    assert event1.percentage == 0.25
    assert event1.eta_seconds == 150.5
    assert event1.status_message == "Translating chunk 25/100"
    assert event1.message == "Translating chunk 25/100"
    assert event1.stage_name == "Етап 1: Машинний переклад (NLLB)"

    # Stage 2 keyword args
    event2 = ProgressEvent(stage="STAGE_2_AYA", current=50, total=50, message="Aya refinement done")
    assert event2.percentage == 1.0
    assert event2.status_message == "Aya refinement done"
    assert event2.stage_name == "Етап 2: Літературне редагування (Aya)"

    # Custom stage
    event3 = ProgressEvent(stage="CUSTOM_STAGE", current=0, total=0, status_message="Custom")
    assert event3.stage_name == "CUSTOM_STAGE"


def test_ui_event_queue_thread_safe_posting():
    ui_queue = UIEventQueue(poll_interval_ms=20, max_events_per_tick=50)
    received = []

    def callback(val, key="default"):
        received.append((val, key))

    # Post from multiple threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=lambda idx: ui_queue.post(callback, idx, key=f"k{idx}"), args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Drain queue synchronously
    count = ui_queue.drain_all_synchronously()
    assert count == 10
    assert len(received) == 10
    assert set(val for val, _ in received) == set(range(10))


def test_async_task_manager_success():
    ui_queue = UIEventQueue()
    mgr = AsyncTaskManager(ui_queue)

    results = []

    def sample_task(cancel_tok, prog_cb):
        prog_cb(ProgressEvent(stage=1, current=1, total=2, status_message="Halfway"))
        return "SUCCESS_RESULT"

    mgr.submit_task(
        task_func=sample_task,
        on_progress=lambda e: results.append(("progress", e)),
        on_success=lambda r: results.append(("success", r)),
    )

    # Wait for completion
    finished = mgr.join_worker(timeout=2.0)
    assert finished
    assert mgr.state == TaskState.COMPLETED
    assert not mgr.is_running()

    ui_queue.drain_all_synchronously()
    assert len(results) == 2
    assert results[0][0] == "progress"
    assert results[0][1].current == 1
    assert results[1] == ("success", "SUCCESS_RESULT")


def test_async_task_manager_cancellation():
    ui_queue = UIEventQueue()
    mgr = AsyncTaskManager(ui_queue)

    events = []

    def cancellable_task(cancel_tok, prog_cb):
        for i in range(100):
            cancel_tok.raise_if_cancelled()
            time.sleep(0.01)
        return "SHOULD_NOT_REACH"

    mgr.submit_task(
        task_func=cancellable_task,
        on_cancelled=lambda: events.append("CANCELLED"),
        on_success=lambda r: events.append(f"SUCCESS_{r}"),
    )

    assert mgr.is_running()
    time.sleep(0.03)
    mgr.cancel_task()

    finished = mgr.join_worker(timeout=2.0)
    assert finished
    assert mgr.state == TaskState.CANCELLED

    ui_queue.drain_all_synchronously()
    assert events == ["CANCELLED"]


def test_async_task_manager_error():
    ui_queue = UIEventQueue()
    mgr = AsyncTaskManager(ui_queue)

    errors = []

    def failing_task(cancel_tok, prog_cb):
        raise ValueError("Simulated translation failure")

    mgr.submit_task(
        task_func=failing_task,
        on_error=lambda ex, tb: errors.append((str(ex), tb)),
    )

    finished = mgr.join_worker(timeout=2.0)
    assert finished
    assert mgr.state == TaskState.FAILED

    ui_queue.drain_all_synchronously()
    assert len(errors) == 1
    assert "Simulated translation failure" in errors[0][0]
    assert "ValueError" in errors[0][1]


def test_async_task_manager_cannot_double_submit():
    ui_queue = UIEventQueue()
    mgr = AsyncTaskManager(ui_queue)

    barrier = threading.Barrier(2)

    def blocking_task(cancel_tok, prog_cb):
        barrier.wait()
        time.sleep(0.05)

    mgr.submit_task(task_func=blocking_task)
    barrier.wait()

    # Attempt second submit while first is running
    with pytest.raises(RuntimeError, match="another task is currently running"):
        mgr.submit_task(task_func=lambda tok, cb: None)

    mgr.join_worker(timeout=2.0)
