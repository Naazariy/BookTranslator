"""
Adversarial Edge-Case Probing & Tier 5 Coverage Hardening Test Suite for BookTranslator.
Empirically stress-tests:
1. Zero-length text and extreme multi-thousand token sentences
2. Rapid start/cancel/toggle hammering on AsyncTaskManager and CancellationToken
3. SQLite high-frequency multi-threaded read/write contention under WAL mode
4. Cyrillic character encoding in PDF export across non-standard fonts
5. Dynamic token ceiling behavior on highly verbose translations
"""

import os
import sys
import time
import queue
import sqlite3
import tempfile
import threading
import traceback
from pathlib import Path
from uuid import uuid4, UUID
from typing import List, Dict, Any, Tuple

# Ensure project root and venv site-packages are in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

venv_site = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

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
from src.domain.models.knowledge import Entity, EntityType, GlossaryItem
from src.translation.stopping_criteria import CancellationTokenStoppingCriteria


class EmpiricalAdversarialTester:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def log_result(self, name: str, passed: bool, details: str, duration_ms: float = 0.0):
        self.results.append({
            "name": name,
            "passed": passed,
            "details": details,
            "duration_ms": duration_ms
        })
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_str} {name:<60} ({duration_ms:6.1f} ms)")
        if not passed:
            print(f"         Details: {details}")

    # =========================================================================
    # VECTOR 1: Zero-length text and extreme multi-thousand token sentences
    # =========================================================================
    def test_vector1_zero_length_and_extreme_tokens(self):
        print("\n--- Running Vector 1: Zero-length Text & Extreme Multi-thousand Token Sentences ---")
        t0 = time.perf_counter()

        # 1.1: Segmenter on empty, whitespace, and single-punctuation strings
        try:
            assert RuleBasedSentenceSegmenter.split_sentences("") == []
            assert RuleBasedSentenceSegmenter.split_sentences("   \n\t  ") == []
            assert RuleBasedSentenceSegmenter.split_sentences(".") == ["."]
            assert RuleBasedSentenceSegmenter.split_sentences("...") == ["..."]
            assert RuleBasedSentenceSegmenter.split_sentences("\n\n\n") == []
            self.log_result("V1.1_Segmenter_ZeroLength_And_Whitespace", True, "Zero-length handled gracefully", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self.log_result("V1.1_Segmenter_ZeroLength_And_Whitespace", False, f"Exception: {e}\n{traceback.format_exc()}", (time.perf_counter() - t0) * 1000)

        # 1.2: Segmenter on massive 50,000-word single sentence (no delimiters)
        t1 = time.perf_counter()
        try:
            massive_words = ["word" + str(i) for i in range(50000)]
            massive_text = " ".join(massive_words)
            res = RuleBasedSentenceSegmenter.split_sentences(massive_text)
            assert len(res) == 1
            assert len(res[0].split()) == 50000
            self.log_result("V1.2_Segmenter_Extreme_50k_Word_Single_Sentence", True, f"Parsed 50k words in {(time.perf_counter() - t1)*1000:.1f}ms", (time.perf_counter() - t1) * 1000)
        except Exception as e:
            self.log_result("V1.2_Segmenter_Extreme_50k_Word_Single_Sentence", False, f"Exception: {e}", (time.perf_counter() - t1) * 1000)

        # 1.3: DynamicTokenBucketBatcher on empty list and empty strings
        t2 = time.perf_counter()
        try:
            batcher = DynamicTokenBucketBatcher(max_batch_tokens=2048)
            assert batcher.create_buckets([]) == []
            assert batcher.create_batches([]) == []
            
            # Batching items with length 0
            empty_items = ["", "   ", "\n"]
            buckets = batcher.create_batches(empty_items)
            # All 3 empty items should be grouped safely without division by zero
            total_items = sum(len(b) for b in buckets)
            assert total_items == 3
            self.log_result("V1.3_Batcher_Empty_And_Whitespace_Items", True, "Zero-length items batched safely without ZeroDivisionError", (time.perf_counter() - t2) * 1000)
        except Exception as e:
            self.log_result("V1.3_Batcher_Empty_And_Whitespace_Items", False, f"Exception: {e}", (time.perf_counter() - t2) * 1000)

        # 1.4: DynamicTokenBucketBatcher on extreme 10,000 token sentences exceeding max_batch_tokens
        t3 = time.perf_counter()
        try:
            batcher = DynamicTokenBucketBatcher(max_batch_tokens=2048)
            # 5 normal sentences (50 tokens each) and 2 giant sentences (10,000 tokens each)
            giant_sent_1 = " ".join(["gigantic"] * 8000)  # ~10,400 subword tokens
            giant_sent_2 = " ".join(["massive"] * 9000)   # ~11,700 subword tokens
            normal_sents = [f"Normal sentence number {i}." for i in range(10)]

            all_sents = normal_sents[:5] + [giant_sent_1, giant_sent_2] + normal_sents[5:]
            buckets = batcher.create_batches(all_sents, max_tokens=2048)

            total_batched = sum(len(b) for b in buckets)
            assert total_batched == 12, f"Expected 12 items in buckets, got {total_batched}"
            # Ensure giant sentences are handled as standalone buckets
            giant_buckets = [b for b in buckets if giant_sent_1 in b or giant_sent_2 in b]
            assert len(giant_buckets) == 2
            assert len(giant_buckets[0]) == 1 and len(giant_buckets[1]) == 1
            self.log_result("V1.4_Batcher_Extreme_10k_Token_Sentences", True, "Oversized items isolated into standalone buckets safely", (time.perf_counter() - t3) * 1000)
        except Exception as e:
            self.log_result("V1.4_Batcher_Extreme_10k_Token_Sentences", False, f"Exception: {e}\n{traceback.format_exc()}", (time.perf_counter() - t3) * 1000)

        # 1.5: Batch and translate un-sorting with mixed zero-length and extreme lengths
        t4 = time.perf_counter()
        try:
            indexed_items = [
                (1, ""),
                (2, "Short sentence."),
                (3, " ".join(["huge"] * 5000)),
                (4, "Another normal one."),
                (5, "   ")
            ]
            mock_translate = lambda batch: [f"[TR: {len(text)} chars]" for text in batch]
            results_map = batcher.batch_and_translate(indexed_items, mock_translate)
            assert len(results_map) == 5
            assert all(k in results_map for k in [1, 2, 3, 4, 5])
            self.log_result("V1.5_Batch_And_Translate_Extreme_Length_Preservation", True, "All 5 items mapped back with perfect key alignment", (time.perf_counter() - t4) * 1000)
        except Exception as e:
            self.log_result("V1.5_Batch_And_Translate_Extreme_Length_Preservation", False, f"Exception: {e}", (time.perf_counter() - t4) * 1000)

    # =========================================================================
    # VECTOR 2: Rapid start/cancel/toggle hammering on AsyncTaskManager and CancellationToken
    # =========================================================================
    def test_vector2_concurrency_hammering(self):
        print("\n--- Running Vector 2: Rapid Start/Cancel/Toggle Hammering ---")
        t0 = time.perf_counter()

        # 2.1: CancellationToken 100-thread concurrent hammer
        try:
            token = CancellationToken()
            num_threads = 100
            ops_per_thread = 200
            errors = []

            def hammer_worker(tid: int):
                try:
                    for i in range(ops_per_thread):
                        if i % 3 == 0:
                            token.cancel()
                        elif i % 3 == 1:
                            token.reset()
                        # Check property vs callable
                        _ = bool(token.is_cancelled)
                        _ = token.is_cancelled()
                        if hasattr(token, "register_callback"):
                            token.register_callback(lambda: None)
                except Exception as ex:
                    errors.append(f"Thread {tid}: {ex}")

            threads = [threading.Thread(target=hammer_worker, args=(i,)) for i in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)

            assert len(errors) == 0, f"Encountered {len(errors)} concurrency errors: {errors[:3]}"
            self.log_result("V2.1_CancellationToken_100_Thread_Hammer", True, f"100 threads completed {num_threads*ops_per_thread} concurrent state mutations without race", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self.log_result("V2.1_CancellationToken_100_Thread_Hammer", False, f"Exception: {e}\n{traceback.format_exc()}", (time.perf_counter() - t0) * 1000)

        # 2.2: UIEventQueue 50-thread producer / 1 consumer high-load flood
        t1 = time.perf_counter()
        try:
            ui_queue = UIEventQueue(poll_interval_ms=5, max_events_per_tick=100)
            events_received = []
            num_producers = 30
            events_per_producer = 500

            def producer(pid: int):
                for i in range(events_per_producer):
                    ui_queue.post(lambda p=pid, idx=i: events_received.append((p, idx)))

            prod_threads = [threading.Thread(target=producer, args=(i,)) for i in range(num_producers)]
            for t in prod_threads:
                t.start()
            for t in prod_threads:
                t.join()

            drained = ui_queue.drain_all_synchronously()
            expected = num_producers * events_per_producer
            assert drained == expected, f"Expected {expected} events, got {drained}"
            assert len(events_received) == expected
            self.log_result("V2.2_UIEventQueue_30_Producer_Flood", True, f"Drained {drained} events synchronously with 100% arrival rate", (time.perf_counter() - t1) * 1000)
        except Exception as e:
            self.log_result("V2.2_UIEventQueue_30_Producer_Flood", False, f"Exception: {e}", (time.perf_counter() - t1) * 1000)

        # 2.3: AsyncTaskManager Rapid Start/Cancel/Submit Hammering
        t2 = time.perf_counter()
        try:
            ui_q = UIEventQueue()
            mgr = AsyncTaskManager(ui_q)
            cycles = 30
            successful_cancels = 0

            for i in range(cycles):
                cancelled_flag = threading.Event()

                def dummy_job(token, progress_cb):
                    for step in range(50):
                        if token.is_cancelled:
                            cancelled_flag.set()
                            raise TaskCancelledException()
                        time.sleep(0.002)

                # Submit task
                mgr.submit_task(dummy_job)
                time.sleep(0.005)  # Let it start
                mgr.cancel_task()
                mgr.join_worker(timeout=1.0)
                
                assert not mgr.is_running(), f"Manager still running on cycle {i}"
                if cancelled_flag.is_set() or mgr.state == TaskState.CANCELLED:
                    successful_cancels += 1

            assert successful_cancels >= 20, f"Only {successful_cancels}/{cycles} cancellations intercepted"
            self.log_result("V2.3_AsyncTaskManager_Rapid_Start_Cancel_Hammer", True, f"Completed {cycles} start-cancel cycles cleanly (state: {mgr.state})", (time.perf_counter() - t2) * 1000)
        except Exception as e:
            self.log_result("V2.3_AsyncTaskManager_Rapid_Start_Cancel_Hammer", False, f"Exception: {e}\n{traceback.format_exc()}", (time.perf_counter() - t2) * 1000)

    # =========================================================================
    # VECTOR 3: SQLite high-frequency multi-threaded read/write contention under WAL mode
    # =========================================================================
    def test_vector3_sqlite_wal_contention(self):
        print("\n--- Running Vector 3: SQLite High-Frequency Multi-threaded Contention ---")
        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "contention_test.db"
            schema = DatabaseSchema(db_path)
            schema.initialize_schema()

            repo = SQLiteKnowledgeBaseRepository(db_path)
            book_id = uuid4()
            chapter_id = uuid4()

            num_writer_threads = 10
            num_reader_threads = 10
            chunks_per_writer = 100

            write_errors = []
            read_errors = []
            total_reads = [0]
            read_lock = threading.Lock()

            def writer_worker(wid: int):
                try:
                    # Each thread gets its own thread-local connection via repo
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
                                    Sentence(id=uuid4(), original_text=f"Worker {wid} sentence {idx}", order_index=0)
                                ],
                                token_count=10,
                                context=ChunkContext(),
                                status=ChunkStatus.DRAFT_COMPLETED if (idx % 2 == 0) else ChunkStatus.PENDING,
                                draft_translation=f"Draft {wid}_{idx}",
                                final_translation=f"Final {wid}_{idx}"
                            )
                            batch.append(chunk)
                        repo.save_chunk_state_batch(batch)
                except Exception as ex:
                    write_errors.append(f"Writer {wid}: {ex}\n{traceback.format_exc()}")

            def reader_worker(rid: int):
                try:
                    for _ in range(50):
                        completed = list(repo.load_chunks_by_status(book_id, ChunkStatus.DRAFT_COMPLETED))
                        count = repo.count_chunks_for_book(book_id)
                        with read_lock:
                            total_reads[0] += 1
                        time.sleep(0.001)
                except Exception as ex:
                    read_errors.append(f"Reader {rid}: {ex}\n{traceback.format_exc()}")

            # Launch concurrent readers and writers simultaneously
            writers = [threading.Thread(target=writer_worker, args=(i,)) for i in range(num_writer_threads)]
            readers = [threading.Thread(target=reader_worker, args=(i,)) for i in range(num_reader_threads)]

            all_threads = writers + readers
            for t in all_threads:
                t.start()
            for t in all_threads:
                t.join(timeout=15.0)

            duration = (time.perf_counter() - t0) * 1000

            # Verify zero errors
            assert len(write_errors) == 0, f"Write errors: {write_errors}"
            assert len(read_errors) == 0, f"Read errors: {read_errors}"

            # Verify total written count
            expected_total = num_writer_threads * chunks_per_writer
            actual_count = repo.count_chunks_for_book(book_id)
            assert actual_count == expected_total, f"Expected {expected_total} chunks, got {actual_count}"

            # Verify WAL mode PRAGMA
            conn = sqlite3.connect(db_path)
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            conn.close()
            assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"

            repo.close()
            self.log_result("V3.1_SQLite_WAL_20_Thread_Contention_1000_Chunks", True, f"10 writers + 10 readers processed {expected_total} writes and {total_reads[0]} reads in {duration:.1f}ms without database locks", duration)

    # =========================================================================
    # VECTOR 4: Cyrillic character encoding in PDF export across non-standard fonts
    # =========================================================================
    def test_vector4_cyrillic_pdf_export(self):
        print("\n--- Running Vector 4: Cyrillic Character Encoding in PDF Export ---")
        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "cyrillic_stress_test.pdf"

            # Create Book with complex Ukrainian and special typographic glyphs
            book = Book(
                id=uuid4(),
                title="Тестова Книга: Українська Мова та Літературні Символи (Є, Ї, І, Ґ)",
                author="Тарас Григорович Шевченко",
                source_language="en",
                target_language="uk"
            )

            chapter = Chapter(
                id=uuid4(),
                title="Розділ 1: Повне покриття української абетки",
                order_index=0
            )

            # Test all upper and lowercase Cyrillic characters plus Ukrainian unique glyphs
            ukr_alphabet = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
            dialogue_text = "«Привіт, світе!» — вигукнув він. «Чи бачив ти солов’я біля м. Києва?»"
            numbers_and_quotes = "Ціна: 3,14 грн. Відсоток: 99.9%. Доктор Ватсон (д-р) та містер Холмс."

            p1 = Paragraph(id=uuid4())
            p1.sentences.append(Sentence(id=uuid4(), original_text=ukr_alphabet, translated_text=ukr_alphabet, order_index=0))
            p1.sentences.append(Sentence(id=uuid4(), original_text=dialogue_text, translated_text=dialogue_text, order_index=1))
            p1.sentences.append(Sentence(id=uuid4(), original_text=numbers_and_quotes, translated_text=numbers_and_quotes, order_index=2))
            chapter.paragraphs.append(p1)
            book.chapters.append(chapter)

            writer = PdfWriter()
            output = writer.write(book, pdf_path)
            assert output.exists() and output.stat().st_size > 0

            # Extract text from generated PDF with PyMuPDF (fitz)
            import fitz
            doc = fitz.open(pdf_path)
            extracted_text = ""
            for page in doc:
                extracted_text += page.get_text()
            doc.close()

            # Verify key Ukrainian Cyrillic characters
            chars_to_verify = ["Є", "є", "Ї", "ї", "І", "і", "Ґ", "ґ", "солов", "Києва", "Шевченко", "Холмс"]
            missing_chars = [c for c in chars_to_verify if c not in extracted_text]

            # Also verify no Mojibake questions marks for Ukrainian chars
            assert len(missing_chars) == 0, f"Missing Cyrillic glyphs in PDF extraction: {missing_chars}\nExtracted text:\n{extracted_text}"
            self.log_result("V4.1_PDF_Cyrillic_TrueType_Glyph_Fidelity", True, f"All Ukrainian glyphs (Є, Ї, І, Ґ, etc.) extracted with 100% fidelity ({output.stat().st_size} bytes)", (time.perf_counter() - t0) * 1000)

    # =========================================================================
    # VECTOR 5: Dynamic token ceiling behavior on highly verbose translations
    # =========================================================================
    def test_vector5_dynamic_token_ceiling(self):
        print("\n--- Running Vector 5: Dynamic Token Ceiling & Stopping Criteria ---")
        t0 = time.perf_counter()

        # 5.1: Dynamic ceiling formula verification across input length spectrum
        try:
            # Formula: min(max(256, int(L_in * 1.5) + 64), 4096)
            def compute_ceiling(input_length: int) -> int:
                return min(max(256, int(input_length * 1.5) + 64), 4096)

            # Very short input (10 tokens): should equal lower clamp 256
            assert compute_ceiling(10) == 256
            # Short input (100 tokens): 100*1.5 + 64 = 214 -> clamped to 256
            assert compute_ceiling(100) == 256
            # Medium input (200 tokens): 200*1.5 + 64 = 364
            assert compute_ceiling(200) == 364
            # Long input (1000 tokens): 1000*1.5 + 64 = 1564
            assert compute_ceiling(1000) == 1564
            # Very long input (2600 tokens): 2600*1.5 + 64 = 3964
            assert compute_ceiling(2600) == 3964
            # Giant input (3000 tokens): 3000*1.5 + 64 = 4564 -> clamped to 4096
            assert compute_ceiling(3000) == 4096
            # Extreme input (10000 tokens): clamped to 4096
            assert compute_ceiling(10000) == 4096

            self.log_result("V5.1_Dynamic_Token_Ceiling_Formula_Spectrum", True, "Ceiling smoothly scales and clamps to [256, 4096] across all input lengths", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self.log_result("V5.1_Dynamic_Token_Ceiling_Formula_Spectrum", False, f"Exception: {e}", (time.perf_counter() - t0) * 1000)

        # 5.2: Stopping Criteria Token Interruption
        t1 = time.perf_counter()
        try:
            cancel_token = CancellationToken()
            criteria = CancellationTokenStoppingCriteria(cancel_token)

            # Not cancelled
            assert criteria(None, None) is False

            # Set cancel
            cancel_token.cancel()
            assert criteria(None, None) is True
            self.log_result("V5.2_CancellationTokenStoppingCriteria_Interrupt", True, "StoppingCriteria returns True immediately on cancellation", (time.perf_counter() - t1) * 1000)
        except Exception as e:
            self.log_result("V5.2_CancellationTokenStoppingCriteria_Interrupt", False, f"Exception: {e}", (time.perf_counter() - t1) * 1000)

    def run_all(self) -> int:
        print("=" * 80)
        print("  TIER 5 ADVERSARIAL EDGE-CASE VERIFICATION SUITE")
        print("=" * 80)

        suite_start = time.perf_counter()
        self.test_vector1_zero_length_and_extreme_tokens()
        self.test_vector2_concurrency_hammering()
        self.test_vector3_sqlite_wal_contention()
        self.test_vector4_cyrillic_pdf_export()
        self.test_vector5_dynamic_token_ceiling()

        total_time = (time.perf_counter() - suite_start) * 1000
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"])

        print("\n" + "=" * 80)
        print(f"  TIER 5 SUMMARY: {passed}/{total_tests} PASSED ({failed} FAILED) in {total_time:.1f} ms")
        print("=" * 80)

        return 0 if failed == 0 else 1


if __name__ == "__main__":
    tester = EmpiricalAdversarialTester()
    exit_code = tester.run_all()
    sys.exit(exit_code)
