# Test Infrastructure & Architecture: BookTranslator Performance Optimization

## 1. Overview & Objectives

The **BookTranslator** test suite is engineered as an opaque-box, requirement-driven, 4-tier End-to-End (E2E) testing framework. It validates all three architectural pillars of performance optimization (UI & Concurrency, ML Translation Pipelines, Database & Parsers) under high contention, boundary stresses, cross-feature interactions, and realistic literary book translation workloads.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            E2E Test Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Tier 1: Feature Coverage (>=5 test cases per feature across F1 - F12)       │
│  ├─ UI Event Queue (F1)               ├─ Async Task Cancellation (F2)      │
│  ├─ Throttled Log Buffer (F3)          ├─ Decoupled Translation Runner (F4)  │
│  ├─ CTranslate2 NLLB (F5)              ├─ Quantized Aya-23-8B Engine (F6)    │
│  ├─ Dynamic Token Batching (F7)        ├─ Stopping Criteria & Memory (F8)   │
│  ├─ SQLite WAL & Batch Tx (F9)         ├─ Compound Indexing & Query (F10)   │
│  ├─ O(S) DOM Reconciliation (F11)     └─ Rule-Based Segmenter (F12)        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Tier 2: Boundary & Corner Cases                                            │
│  ├─ Empty & Extreme Length Inputs      ├─ Rapid Multi-Stage Cancellation    │
│  ├─ High-Contention Log Flooding       └─ SQLite WAL Concurrent Readers/Writers│
├─────────────────────────────────────────────────────────────────────────────┤
│  Tier 3: Cross-Feature Pairwise Combinations                                │
│  ├─ UI Logging + Pipeline Event Stream ├─ Token Batching + CTranslate2 + DB │
│  ├─ Aya Refinement + Glossary DB + Cancel └─ DOM Reconcile + PDF/TXT Export │
├─────────────────────────────────────────────────────────────────────────────┤
│  Tier 4: Real-World Workloads                                               │
│  ├─ Multi-Chapter Ukrainian Prose E2E  ├─ Crash/Interruption Recovery       │
│  └─ High-Fidelity Literary DOM PDF Output                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 4-Tier Test Architecture

### Tier 1: Feature Coverage (Unit-to-End Component Tests)
Each core feature defined in `PROJECT.md` is guaranteed at least 5 independent, verifiable test cases:
1. **F1 (UI Event Queue)**: Verifies non-blocking post, main-thread polling, multi-producer FIFO ordering, payload dataclass preservation, and capacity management.
2. **F2 (AsyncTaskManager & Cancellation)**: Tests worker thread lifecycle, thread-safe `CancellationToken` signaling, graceful cancellation propagation, and exception propagation.
3. **F3/F4 (Throttled Logging & Translation Runner)**: Validates 50ms timer flushing, 1,000-line circular buffer capping, and headless runner execution.
4. **F5 (CTranslate2 NLLB Engine)**: Validates Flores-200 language code translation, batch translation accuracy, JIT INT8/FP16 execution, and memory load/unload.
5. **F6 (Quantized Aya-23-8B Engine)**: Validates in-memory prompt template caching (no repetitive disk I/O), glossary term injection, markdown/header cleanup, and 4-bit quantization interface.
6. **F7 (Dynamic Token-Bucket Batching)**: Validates token grouping up to `max_tokens=2048`, padding waste reduction from ~48% to ~2.5%, single oversized sequence handling, and order preservation.
7. **F8 (Stopping Criteria & Memory Polish)**: Validates PyTorch generation early halting on `CancellationToken` and $O(N)$ progress tracking.
8. **F9/F10 (SQLite WAL & Indexing)**: Tests `PRAGMA journal_mode=WAL`, `idx_chunk_book_status` index presence and query plans, and 50-chunk batch `executemany` transactions.
9. **F11 ($O(S)$ DOM Reconciliation)**: Validates dictionary lookup `{sentence.id: sentence}` for document reconstruction, multi-sentence chunk writeback, and linear time scaling.
10. **F12 (Rule-Based Sentence Segmenter)**: Verifies 0% failure rate across English abbreviations, Ukrainian Cyrillic abbreviations ("м. Київ", "вул."), dialogue with guillemets/em-dashes, numbers/decimals, and time notation.

### Tier 2: Boundary & Corner Cases
Stress tests edge conditions that cause software crashes, hangs, or race conditions:
- **Empty & Extreme Inputs**: 0-byte files, single-character sentences, 10,000-token sentences, chapters with zero paragraphs.
- **Rapid Cancellation**: Immediate pre-start cancel, cancel during Stage 1 NLLB, cancel during Stage 2 Aya, cancel during DB commit, repeated rapid start/stop toggles.
- **Log Queue Flooding**: 10,000 log events dispatched simultaneously across 10 threads without deadlock or memory exhaustion.
- **Database Concurrency & Recovery**: Multiple concurrent reader threads querying while a writer thread executes batch transactions under WAL mode; SQLite busy retry handling.

### Tier 3: Cross-Feature Pairwise Combinations
Validates subsystem interoperability and data integrity across pipeline boundaries:
- **UI Logging ↔ Translation Pipeline**: Ensures progress events (`ProgressEvent`) and log streams emit in sync without blocking the translation thread.
- **Dynamic Batcher ↔ CTranslate2 ↔ SQLite Batching**: Verifies variable-sized token buckets correctly feed CTranslate2 inference and flush status updates to SQLite in batch transactions.
- **Aya Engine ↔ SQLite Glossary ↔ Cancellation Token**: Verifies live glossary extraction from SQLite, dynamic prompt compilation, and mid-generation cancellation.
- **DOM Reconciliation ↔ PDF/TXT Writers**: Verifies reconstructed `Book` DOM translates into valid Unicode PDF files and formatted text documents.

### Tier 4: Real-World Workloads
Simulates full end-to-end production scenarios:
- **End-to-End Literary Book Translation**: Multi-chapter Ukrainian literary prose translated through Stage 1 and Stage 2 with glossary consistency.
- **Interruption & Checkpoint Recovery**: Simulates pipeline termination at 50% progress, verifies restart resumes solely from pending chunks with 0 redundant re-translations.
- **High-Fidelity PDF Export**: Complete book rendered with Cyrillic characters, dialogue dashes, and chapter hierarchy.

---

## 3. Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py
├── test_harness.py                       # Master test runner & reporter
└── e2e/
    ├── __init__.py
    ├── fixtures.py                       # Shared test fixtures, mock engines, sample data
    ├── tier1_feature_coverage/
    │   ├── __init__.py
    │   ├── test_ui_event_queue.py
    │   ├── test_async_task_cancellation.py
    │   ├── test_ctranslate2_nllb.py
    │   ├── test_quantized_aya.py
    │   ├── test_dynamic_token_batching.py
    │   ├── test_sqlite_wal_indexing.py
    │   ├── test_dom_reconciliation.py
    │   └── test_rule_based_segmentation.py
    ├── tier2_boundary_corner/
    │   ├── __init__.py
    │   ├── test_boundary_empty_and_extremes.py
    │   ├── test_boundary_rapid_cancellation.py
    │   ├── test_boundary_log_queue_flooding.py
    │   └── test_boundary_database_concurrency.py
    ├── tier3_cross_feature_pairwise/
    │   ├── __init__.py
    │   ├── test_ui_logging_with_pipeline.py
    │   ├── test_batching_with_ctranslate2_sqlite.py
    │   ├── test_aya_refinement_with_glossary_db.py
    │   └── test_dom_reconciliation_with_pdf_export.py
    └── tier4_real_world_workloads/
        ├── __init__.py
        ├── test_full_book_translation_pipeline.py
        ├── test_checkpoint_interruption_recovery.py
        └── test_literary_dom_to_pdf_export.py
```

---

## 4. Execution & Verification

### Running All Tests via Test Harness
```bash
python tests/test_harness.py
```

### Running Specific Tiers
```bash
python tests/test_harness.py --tier 1
python tests/test_harness.py --tier 2
python tests/test_harness.py --tier 3
python tests/test_harness.py --tier 4
```

### Running via Pytest
```bash
pytest tests/e2e/ -v
```

---

## 5. Quality & Pass Criteria
- **Pass Rate**: 100% (0 failures, 0 errors).
- **Execution Time**: < 30 seconds for the entire test suite on standard CPU.
- **Isolation**: Every test creates isolated temporary directories and databases, guaranteeing no state leakage.
- **Zero Mock Facades**: All tests verify actual functional logic and interface contracts.
