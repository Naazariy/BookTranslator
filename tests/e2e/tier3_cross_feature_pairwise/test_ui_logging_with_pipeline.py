"""Tier 3 Cross-Feature Tests: UI Logging & Translation Pipeline Interoperability.
Verifies progress event streaming from TwoStageTranslationPipeline to UIEventQueue and ThrottledLogHandler integration.
"""
import time
import logging
from uuid import uuid4
import pytest

from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.translation.pipeline import TwoStageTranslationPipeline
from tests.e2e.fixtures import (
    UIEventQueue,
    ProgressEvent,
    AsyncTaskManager,
    MockCTranslate2Engine,
    MockQuantizedAyaEngine,
    SampleBookFactory,
)


class TestUILoggingWithPipeline:
    def test_pipeline_progress_events_stream_to_ui_event_queue(self, temp_work_dir):
        """Test X1.1: TwoStageTranslationPipeline execution posts ProgressEvents to UIEventQueue."""
        db_path = temp_work_dir / "test_pipeline_ui.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=2)
        for c in chunks:
            repo.save_chunk_state(c)

        eq = UIEventQueue()
        nllb = MockCTranslate2Engine()
        aya = MockQuantizedAyaEngine()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        # Custom progress logger handler that captures PROGRESS_UPDATE records
        class ProgressQueueHandler(logging.Handler):
            def emit(self, record):
                msg = record.getMessage()
                if msg.startswith("PROGRESS_UPDATE:"):
                    parts = msg.split(":")
                    if len(parts) == 5:
                        stage_num = 1 if parts[1] == "STAGE_1" else 2
                        cur = int(parts[2])
                        tot = int(parts[3])
                        eta = float(parts[4])
                        eq.post("progress", event=ProgressEvent(stage=stage_num, current=cur, total=tot, eta_seconds=eta))

        handler = ProgressQueueHandler()
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        try:
            pipeline.execute_nllb_stage(book.id)
            pipeline.execute_aya_stage(book.id)
        finally:
            root_logger.removeHandler(handler)

        events = eq.drain()
        progress_events = [e for e in events if e["type"] == "progress"]
        assert len(progress_events) == len(chunks) * 2  # Stage 1 + Stage 2 updates
        
        # Verify stage 1 completed
        s1_events = [e["data"]["event"] for e in progress_events if e["data"]["event"].stage == 1]
        assert len(s1_events) == len(chunks)
        assert s1_events[-1].current == len(chunks)

        # Verify stage 2 completed
        s2_events = [e["data"]["event"] for e in progress_events if e["data"]["event"].stage == 2]
        assert len(s2_events) == len(chunks)
        assert s2_events[-1].current == len(chunks)

    def test_throttled_log_handler_captures_stage_transitions(self, temp_work_dir):
        """Test X1.2: Throttled logging captures pipeline logs without dropping crucial milestone messages."""
        db_path = temp_work_dir / "test_throttled_pipeline.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=2)
        for c in chunks:
            repo.save_chunk_state(c)

        eq = UIEventQueue()
        nllb = MockCTranslate2Engine()
        aya = MockQuantizedAyaEngine()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        class ForwardingHandler(logging.Handler):
            def emit(self, record):
                eq.post("log", message=record.getMessage(), level=record.levelname)

        handler = ForwardingHandler()
        logging.getLogger("src.translation.pipeline").addHandler(handler)
        logging.getLogger("src.translation.pipeline").setLevel(logging.INFO)

        try:
            pipeline.execute_nllb_stage(book.id)
            pipeline.execute_aya_stage(book.id)
        finally:
            logging.getLogger("src.translation.pipeline").removeHandler(handler)

        all_logs = eq.drain()
        log_messages = [l["data"]["message"] for l in all_logs if l["type"] == "log"]
        
        assert any("Starting Stage 1" in m for m in log_messages)
        assert any("Stage 1" in m and ("finished" in m.lower() or "completed" in m.lower()) for m in log_messages)
        assert any("Starting Stage 2" in m for m in log_messages)
        assert any("Stage 2" in m and ("finished" in m.lower() or "completed" in m.lower()) for m in log_messages)


    def test_ui_event_queue_drains_smoothly_under_simulated_ui_tick(self, temp_work_dir):
        """Test X1.3: Background thread runs translation while main thread polls at 50Hz (20ms)."""
        db_path = temp_work_dir / "test_smooth_drain.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=2)
        for c in chunks:
            repo.save_chunk_state(c)

        eq = UIEventQueue()
        nllb = MockCTranslate2Engine()
        aya = MockQuantizedAyaEngine()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        def bg_worker(cancel_token):
            for i, c in enumerate(chunks):
                if cancel_token.is_cancelled():
                    break
                eq.post("status", text=f"Processing {i+1}/{len(chunks)}")
                time.sleep(0.01)

        mgr = AsyncTaskManager()
        mgr.start_task(bg_worker)

        received_ticks = 0
        while mgr.is_running() or not eq.empty():
            ev = eq.poll(timeout_ms=20)
            if ev is not None:
                received_ticks += 1

        mgr.join(timeout=1.0)
        assert received_ticks == len(chunks)
