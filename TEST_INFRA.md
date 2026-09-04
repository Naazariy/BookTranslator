# E2E Test Infra: BookTranslator V2

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation internal design.
- Verification mechanism operates strictly through application entry points (CLI, `TranslationRunner`, domain I/O, SQLite state).
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing.

## Feature Inventory & Test Coverage Goals
| # | Feature | Requirement Source | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Real-World) |
|---|---------|-------------------|:----------------:|:-----------------:|:-----------------:|:-------------------:|
| 1 | Sentence Immutability & Normalization | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ |
| 2 | Content Identity (SHA-256 & Deterministic Job) | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ |
| 3 | 7-State Segment Lifecycle | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ |
| 4 | Aya Failure Visibility (No Silent Fallback) | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ |
| 5 | Deterministic Stage 2 Generation | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ |
| 6 | Document Writers Status Gating (Accepted Only)| ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ |
| 7 | DB Migrations & Schema Versioning | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ |
| 8 | Scoped Knowledge Hierarchy (BOOK>SERIES>DOMAIN>GLOBAL) | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ |
| 9 | EntityProfile (Autolock >= 0.90 & Forbidden Variants) | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ |
| 10 | Pre-translation Whole-Book Analysis | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ |
| 11 | Localized Entity Mention Indexing | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ |
| 12 | Paragraph TranslationSegment Refinement | ORIGINAL_REQUEST § R3 | >=5 | >=5 | ✓ | ✓ |
| 13 | ContextBuilder & TokenBudget Allocation | ORIGINAL_REQUEST § R3 | >=5 | >=5 | ✓ | ✓ |
| 14 | editing_prompt_v2 Structured JSON Output | ORIGINAL_REQUEST § R3 | >=5 | >=5 | ✓ | ✓ |
| 15 | Modular Quality Validators (8 Validators) | ORIGINAL_REQUEST § R4 | >=5 | >=5 | ✓ | ✓ |
| 16 | Bounded Repair Loop (max 2 retries) | ORIGINAL_REQUEST § R4 | >=5 | >=5 | ✓ | ✓ |
| 17 | Auditable Quality Reports Persistence | ORIGINAL_REQUEST § R4 | >=5 | >=5 | ✓ | ✓ |
| 18 | Quality Cases Regression Suite | ORIGINAL_REQUEST § R1, Survey | >=5 | >=5 | ✓ | ✓ |

## Test Architecture
- **Regression Corpus**: `tests/regression/quality_cases.json`
  - Structure: list of test cases with `id`, `category` (character_name, polysemy, item, gender_agreement, rpg_tag), `source_en`, `expected_uk_canonical`, `forbidden_variants`, `metadata`.
  - Runner: `tests/regression/test_quality_cases.py` using `pytest`.
- **E2E Test Runner**: `tests/test_harness.py` & `pytest tests/e2e/` (or `tests/integration/`)
  - Runs full translation lifecycle on mock book content.
  - Verifies that:
    1. `Sentence.original_text` remains untouched across preprocessing, translation, and export.
    2. Modifying file content produces a distinct `Book.id` and `job_id`.
    3. Scoped entities resolve properly with book-level overrides taking precedence over global.
    4. Forbidden variants for locked entities fail QA and trigger repair.
    5. Malformed LLM outputs are routed to repair and never marked as `ACCEPTED` without passing QA.
    6. Output files contain only `ACCEPTED` segments.

## Coverage Thresholds
- Minimum Tier 1 tests: >=5 per feature area
- Minimum Tier 2 tests: >=5 boundary and corner cases per feature area
- Minimum Tier 3 tests: Pairwise feature combination tests
- Minimum Tier 4 tests: Real-world document workloads (multi-chapter books, complex dialogues, nested entities)
