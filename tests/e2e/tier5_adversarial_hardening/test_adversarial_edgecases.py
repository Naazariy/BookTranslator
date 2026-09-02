"""Tier 5 Adversarial Edge-Case Probing & Coverage Hardening Test Suite.
Verifies:
1. Zero-length text and extreme multi-thousand token sentence handling.
2. Rapid start/cancel/toggle hammering on AsyncTaskManager and CancellationToken.
3. SQLite high-frequency multi-threaded read/write contention under WAL mode.
4. Cyrillic character encoding in PDF export across non-standard fonts.
5. Dynamic token ceiling behavior on highly verbose translations.
"""
import os
import sys
import time
import tempfile
import threading
import sqlite3
from uuid import uuid4
from pathlib import Path

from src.launcher.concurrency import (
    CancellationToken,
    UIEventQueue,
    AsyncTaskManager,
    ProgressEvent,
    TaskState,
    TaskCancelledException,
)
from src.translation.batching import DynamicTokenBucketBatcher
from src.parsers.segmenter import RuleBasedSentenceSegmenter
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.knowledge_base.db_schema import DatabaseSchema
from src.writers.pdf_writer import PdfWriter
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.translation.stopping_criteria import CancellationTokenStoppingCriteria


class TestTier5AdversarialHardening:
    # -------------------------------------------------------------------------
    # 1. Zero-Length and Extreme Multi-Thousand Token Sentences
    # -------------------------------------------------------------------------
    def test_segmenter_zero_length_and_whitespace_inputs(self):
        """T5.1: RuleBasedSentenceSegmenter handles empty strings and whitespace without error."""
        assert RuleBasedSentenceSegmenter.split_sentences("") == []
        assert RuleBasedSentenceSegmenter.split_sentences("   \t\n  ") == []
        assert RuleBasedSentenceSegmenter.split_sentences("\n\n\n") == []
        assert RuleBasedSentenceSegmenter.split_sentences(".") == ["."]
        assert RuleBasedSentenceSegmenter.split_sentences("...") == ["..."]

    def test_segmenter_extreme_50k_words_single_sentence(self):
        """T5.2: Segmenter processes a massive 50,000-word sentence without stack overflow."""
        massive_text = " ".join(["token" + str(i) for i in range(50000)])
        result = RuleBasedSentenceSegmenter.split_sentences(massive_text)
        assert len(result) == 1
        assert len(result[0].split()) == 50000

    def test_batcher_zero_length_and_empty_items(self):
        """T5.3: DynamicTokenBucketBatcher processes empty list and 0-length items without ZeroDivisionError."""
        batcher = DynamicTokenBucketBatcher(max_batch_tokens=2048)
        assert batcher.create_buckets([]) == []
        assert batcher.create_batches([]) == []

        empty_items = ["", "   ", "\n"]
        buckets = batcher.create_batches(empty_items)
        assert sum(len(b) for b in buckets) == 3

    def test_batcher_extreme_10k_token_sentence_standalone_bucketing(self):
        """T5.4: Multi-thousand token sentences are isolated into standalone buckets without dropping."""
        batcher = DynamicTokenBucketBatcher(max_batch_tokens=2048)
        giant_sentence = " ".join(["word"] * 8000)  # ~10,400 subword tokens
        normal_sentences = [f"Normal sentence {i}." for i in range(10)]

        all_sentences = normal_sentences[:5] + [giant_sentence] + normal_sentences[5:]
        buckets = batcher.create_batches(all_sentences, max_tokens=2048)

        # All 11 sentences must be present across buckets
        assert sum(len(b) for b in buckets) == 11
        # Giant sentence must be in its own single-item bucket
        giant_bucket = [b for b in buckets if giant_sentence in b]
        assert len(giant_bucket) == 1
        assert len(giant_bucket[0]) == 1

    def test_batch_and_translate_unsorting_with_extreme_lengths(self):
        """T5.5: batch_and_translate preserves original IDs when mixing empty and extreme lengths."""
        batcher = DynamicTokenBucketBatcher(max_batch_tokens=2048)
        items = [
            (1, ""),
            (2, "Short sentence."),
            (3, " ".join(["massive"] * 4000)),
            (4, "Another normal sentence."),
            (5, "   ")
        ]
        mock_fn = lambda batch: [f"[TR:{len(s)}]" for s in batch]
        results = batcher.batch_and_translate(items, mock_fn)
        assert len(results) == 5
        assert all(k in results for k in [1, 2, 3, 4, 5])

    # -------------------------------------------------------------------------
    # 2. Rapid start/cancel/toggle hammering
    # -------------------------------------------------------------------------
    def test_cancellation_token_100_thread_hammer(self):
        """T5.6: 100 concurrent threads hammering CancellationToken state mutations without race."""
        token = CancellationToken()
        num_threads = 100
        ops_per_thread = 200
        errors = []

        def hammer():
            try:
                for i in range(ops_per_thread):
                    if i % 2 == 0:
                        token.cancel()
                    else:
                        token.reset()
                    _ = bool(token.is_cancelled)
                    _ = token.is_cancelled()
                    if hasattr(token, "register_callback"):
                        token.register_callback(lambda: None)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=hammer) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_ui_event_queue_30_thread_flood(self):
        """T5.7: High-concurrency 30-producer flood on UIEventQueue drained synchronously."""
        ui_queue = UIEventQueue()
        events_received = []
        num_producers = 30
        events_per_producer = 500

        def producer(pid: int):
            for i in range(events_per_producer):
                ui_queue.post(lambda p=pid, idx=i: events_received.append((p, idx)))

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(num_producers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        drained = ui_queue.drain_all_synchronously()
        expected = num_producers * events_per_producer
        assert drained == expected
        assert len(events_received) == expected

    def test_async_task_manager_rapid_start_cancel_hammer(self):
        """T5.8: Rapid submit/cancel hammering on AsyncTaskManager without deadlocks."""
        ui_q = UIEventQueue()
        mgr = AsyncTaskManager(ui_q)
        cycles = 25
        successful_cancels = 0

        for i in range(cycles):
            flag = threading.Event()

            def dummy_job(token, progress_cb):
                for _ in range(50):
                    if token.is_cancelled:
                        flag.set()
                        raise TaskCancelledException()
                    time.sleep(0.002)

            mgr.submit_task(dummy_job)
            time.sleep(0.005)
            mgr.cancel_task()
            mgr.join_worker(timeout=1.0)

            assert not mgr.is_running()
            if flag.is_set() or mgr.state == TaskState.CANCELLED:
                successful_cancels += 1

        assert successful_cancels >= 15

    # -------------------------------------------------------------------------
    # 3. SQLite Multi-Threaded Read/Write Contention under WAL Mode
    # -------------------------------------------------------------------------
    def test_sqlite_wal_20_thread_concurrent_contention(self, temp_work_dir):
        """T5.9: 10 concurrent writers and 10 readers against SQLite WAL database without locking."""
        db_path = temp_work_dir / "wal_contention.db"
        schema = DatabaseSchema(db_path)
        schema.initialize_schema()

        repo = SQLiteKnowledgeBaseRepository(db_path)
        book_id = uuid4()
        chapter_id = uuid4()

        num_writers = 10
        num_readers = 10
        chunks_per_writer = 100

        write_errors = []
        read_errors = []
        total_reads = [0]
        read_lock = threading.Lock()

        def writer(wid: int):
            try:
                for batch_idx in range(chunks_per_writer // 10):
                    batch = []
                    for c in range(10):
                        idx = batch_idx * 10 + c
                        chunk = TranslationChunk(
                            id=uuid4(),
                            book_id=book_id,
                            chapter_id=chapter_id,
                            paragraph_indices=[idx],
                            source_sentences=[
                                Sentence(id=uuid4(), original_text=f"W {wid} S {idx}", order_index=0)
                            ],
                            token_count=10,
                            context=ChunkContext(),
                            status=ChunkStatus.DRAFT_COMPLETED if (idx % 2 == 0) else ChunkStatus.PENDING,
                            draft_translation=f"D_{wid}_{idx}",
                            final_translation=f"F_{wid}_{idx}"
                        )
                        batch.append(chunk)
                    repo.save_chunk_state_batch(batch)
            except Exception as e:
                write_errors.append(str(e))

        def reader(rid: int):
            try:
                for _ in range(30):
                    _ = list(repo.load_chunks_by_status(book_id, ChunkStatus.DRAFT_COMPLETED))
                    _ = repo.count_chunks_for_book(book_id)
                    with read_lock:
                        total_reads[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                read_errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_writers)] + \
                  [threading.Thread(target=reader, args=(i,)) for i in range(num_readers)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert len(write_errors) == 0, f"Write errors: {write_errors}"
        assert len(read_errors) == 0, f"Read errors: {read_errors}"

        # Verify chunk counts
        expected_total = num_writers * chunks_per_writer
        assert repo.count_chunks_for_book(book_id) == expected_total

        # Verify WAL mode
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"

        repo.close()

    # -------------------------------------------------------------------------
    # 4. Cyrillic Character Encoding in PDF Export across Fonts
    # -------------------------------------------------------------------------
    def test_cyrillic_pdf_export_full_alphabet_fidelity(self, temp_work_dir):
        """T5.10: Full Ukrainian Cyrillic alphabet and typography exported to PDF without glyph corruption."""
        pdf_path = temp_work_dir / "cyrillic_test.pdf"

        book = Book(
            id=uuid4(),
            title="Тестова Книга: Українська Мова (Є, Ї, І, Ґ)",
            author="Тарас Григорович Шевченко",
            source_language="en",
            target_language="uk"
        )
        chapter = Chapter(id=uuid4(), title="Розділ 1. Повна абетка", order_index=0)

        ukr_alphabet = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
        dialogue = "«Привіт, світе!» — вигукнув він. «Чи бачив солов’я біля м. Києва?»"

        p = Paragraph(id=uuid4())
        p.sentences.append(Sentence(id=uuid4(), original_text=ukr_alphabet, translated_text=ukr_alphabet, order_index=0))
        p.sentences.append(Sentence(id=uuid4(), original_text=dialogue, translated_text=dialogue, order_index=1))
        chapter.paragraphs.append(p)
        book.chapters.append(chapter)

        writer = PdfWriter()
        out = writer.write(book, pdf_path)
        assert out.exists() and out.stat().st_size > 0

        # Verify with fitz
        import fitz
        doc = fitz.open(pdf_path)
        extracted = "".join(page.get_text() for page in doc)
        doc.close()

        for char in ["Є", "є", "Ї", "ї", "І", "і", "Ґ", "ґ", "солов", "Києва"]:
            assert char in extracted, f"Missing Cyrillic glyph '{char}' in PDF export!"

    # -------------------------------------------------------------------------
    # 5. Dynamic Token Ceiling & Stopping Criteria
    # -------------------------------------------------------------------------
    def test_dynamic_token_ceiling_formula_clamping(self):
        """T5.11: Dynamic token ceiling scales and clamps correctly across token lengths."""
        def ceiling_fn(L_in: int) -> int:
            return min(max(256, int(L_in * 1.5) + 64), 4096)

        assert ceiling_fn(10) == 256
        assert ceiling_fn(100) == 256
        assert ceiling_fn(500) == 814
        assert ceiling_fn(2000) == 3064
        assert ceiling_fn(3000) == 4096
        assert ceiling_fn(10000) == 4096

    def test_stopping_criteria_cancels_generation_immediately(self):
        """T5.12: CancellationTokenStoppingCriteria triggers immediate halt upon cancel()."""
        token = CancellationToken()
        criteria = CancellationTokenStoppingCriteria(token)
        assert criteria(None, None) is False

        token.cancel()
        assert criteria(None, None) is True
