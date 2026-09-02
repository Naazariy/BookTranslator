"""Tier 2 Boundary Tests: Log Queue Flooding Under High Contention (F3).
Verifies burst handling of 10,000 log events, multi-threaded producer contention, 1,000-line circular buffer trimming, and 50ms flush rate.
"""
import time
import threading
import pytest
from tests.e2e.fixtures import UIEventQueue, ThrottledLogBuffer


class TestBoundaryLogQueueFlooding:
    def test_burst_10000_logs_no_deadlock(self):
        """Test B3.1: Burst of 10,000 log messages queued and drained without freezing."""
        eq = UIEventQueue(maxsize=20000)
        start_time = time.time()
        
        for i in range(10000):
            posted = eq.post("log", msg=f"Log message {i}")
            assert posted is True
            
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Fast non-blocking enqueue
        
        drained = eq.drain()
        assert len(drained) == 10000
        assert eq.empty() is True

    def test_high_contention_multithreaded_log_producers(self):
        """Test B3.2: 20 worker threads flooding log events simultaneously."""
        eq = UIEventQueue(maxsize=50000)
        num_threads = 20
        logs_per_thread = 500
        
        def logger_worker(t_id: int):
            for i in range(logs_per_thread):
                eq.post("log", thread=t_id, seq=i, payload="High contention test record")

        threads = [threading.Thread(target=logger_worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert eq.qsize() == num_threads * logs_per_thread
        all_events = eq.drain()
        assert len(all_events) == 10000

    def test_circular_buffer_memory_cap_1000_lines(self):
        """Test B3.3: ThrottledLogBuffer strictly caps stored lines to max_lines=1000."""
        buf = ThrottledLogBuffer(max_lines=1000, flush_interval_ms=50)
        
        for i in range(5000):
            buf.append(f"Line {i}")
            
        stored = buf.get_lines()
        assert len(stored) == 1000
        # Check that it retained the most recent 1,000 lines (4000..4999)
        assert stored[0] == "Line 4000"
        assert stored[-1] == "Line 4999"

    def test_throttled_flusher_frequency_and_interval(self):
        """Test B3.4: Buffer flushes on interval condition (50ms)."""
        buf = ThrottledLogBuffer(max_lines=1000, flush_interval_ms=50)
        
        buf.append("First log")
        # Flush immediately after first append
        flushed1 = buf.flush()
        assert flushed1 == ["First log"]
        
        buf.append("Second log")
        # Immediately after flush, should_flush should be False until interval expires
        assert buf.should_flush() is False
        
        time.sleep(0.06)  # Wait > 50ms
        assert buf.should_flush() is True
        flushed2 = buf.flush()
        assert flushed2 == ["Second log"]
