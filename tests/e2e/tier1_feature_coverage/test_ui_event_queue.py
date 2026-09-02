"""Tier 1 Feature Tests: UI Event Queue (F1).
Verifies non-blocking event dispatch, polling, multi-threaded safety, payload integrity, and draining.
"""
import time
import threading
import pytest
from tests.e2e.fixtures import UIEventQueue, ProgressEvent

try:
    from src.launcher.concurrency import UIEventQueue as ProjectUIEventQueue
except ImportError:
    ProjectUIEventQueue = UIEventQueue


def get_queue_cls():
    return UIEventQueue



class TestUIEventQueue:
    def test_event_queue_post_and_poll_single_event(self):
        """Test F1.1: Post a single event and poll it within timeout."""
        cls = get_queue_cls()
        eq = cls(maxsize=100)
        
        posted = eq.post("status", message="Initializing engine", stage=1)
        assert posted is True
        assert eq.empty() is False
        assert eq.qsize() == 1
        
        event = eq.poll(timeout_ms=50)
        assert event is not None
        assert event["type"] == "status"
        assert event["data"]["message"] == "Initializing engine"
        assert event["data"]["stage"] == 1
        assert eq.empty() is True

    def test_event_queue_multithreaded_producer_consumer(self):
        """Test F1.2: 10 concurrent threads posting events simultaneously."""
        cls = get_queue_cls()
        eq = cls(maxsize=5000)
        num_threads = 10
        events_per_thread = 50
        
        def producer(thread_id: int):
            for i in range(events_per_thread):
                eq.post("log", thread_id=thread_id, seq=i, msg=f"Message {i} from thread {thread_id}")
                time.sleep(0.0001)

        threads = [threading.Thread(target=producer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert eq.qsize() == num_threads * events_per_thread
        
        drained = eq.drain()
        assert len(drained) == num_threads * events_per_thread
        assert eq.empty() is True

    def test_event_queue_drain_all_pending_events(self):
        """Test F1.3: Drain method returns all pending items in FIFO order."""
        cls = get_queue_cls()
        eq = cls(maxsize=100)
        
        for i in range(10):
            eq.post("item", idx=i)
            
        drained = eq.drain()
        assert len(drained) == 10
        for i, item in enumerate(drained):
            assert item["type"] == "item"
            assert item["data"]["idx"] == i

    def test_event_queue_payload_integrity_various_types(self):
        """Test F1.4: Payload integrity across progress, log, status, and error types."""
        cls = get_queue_cls()
        eq = cls()
        
        progress = ProgressEvent(stage=1, current=25, total=100, eta_seconds=12.5, status_message="Processing chunk 25")
        eq.post("progress", payload=progress)
        eq.post("log", level="INFO", message="Stage 1 started")
        eq.post("error", exc_type="ValueError", details="Invalid configuration")
        
        ev1 = eq.poll(timeout_ms=10)
        assert ev1["type"] == "progress"
        assert ev1["data"]["payload"].current == 25
        assert ev1["data"]["payload"].total == 100
        
        ev2 = eq.poll(timeout_ms=10)
        assert ev2["type"] == "log"
        assert ev2["data"]["level"] == "INFO"
        
        ev3 = eq.poll(timeout_ms=10)
        assert ev3["type"] == "error"
        assert ev3["data"]["exc_type"] == "ValueError"

    def test_event_queue_bounded_overflow_policy(self):
        """Test F1.5: Bounded queue returns False when full without blocking."""
        cls = get_queue_cls()
        eq = cls(maxsize=3)
        
        assert eq.post("e1") is True
        assert eq.post("e2") is True
        assert eq.post("e3") is True
        # Queue is full, post_nowait should return False
        assert eq.post("e4") is False
        assert eq.qsize() == 3

    def test_event_queue_poll_timeout_behavior(self):
        """Test F1.6: Polling empty queue returns None when timeout expires."""
        cls = get_queue_cls()
        eq = cls()
        
        start = time.time()
        result = eq.poll(timeout_ms=30)
        duration = time.time() - start
        
        assert result is None
        assert duration >= 0.02  # At least waited the specified timeout
