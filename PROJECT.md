# Project: BookTranslator V2 MVP

## Architecture
- **Stage 1 (Machine Translation)**: High-throughput auxiliary draft translation powered by CTranslate2 NLLB-200. Operates at sentence level to generate initial candidate translations.
- **Stage 2 (Contextual Literary Refinement)**: Paragraph-level semantic refinement powered by Quantized Aya-23-8B (`TranslationSegment`). Assembles context via `ContextBuilder` and `TokenBudget` (current source paragraph, fallible NLLB draft, previous 1–2 approved Ukrainian paragraphs, active locked entities/terms, chapter summary). Eliminates 1:1 sentence constraint inside paragraphs while strictly preserving paragraph boundaries.
- **Data & State Management**:
  - `Sentence.original_text` is strictly immutable; `Sentence.normalized_source_text` holds preprocessed text.
  - `Book.id` is derived deterministically from `sha256(file_bytes)`.
  - `job_id` is a composite deterministic fingerprint of `(pipeline_version, model_config, prompt_hash, kb_revision)`.
  - 7-state `SegmentStatus` lifecycle: `PENDING` -> `DRAFT_COMPLETED` -> `EDITED` -> `VALIDATING` -> `ACCEPTED` / `REVIEW_REQUIRED` / `FAILED`.
- **Scoped Knowledge Base & Whole-Book Analysis**:
  - Migration framework (`src/knowledge_base/migrations/`) with `schema_version`.
  - Scoped hierarchy: `BOOK > SERIES > DOMAIN > GLOBAL`.
  - `EntityProfile` model with auto-locking at confidence $\ge 0.90$, aliases, grammatical gender, allowed forms, and forbidden variant rejection.
  - `src/analysis/book_analyzer.py` for whole-book pre-translation candidate extraction.
  - Localized `EntityMention` indexing per segment.
- **Modular Quality Assurance & Bounded Repair**:
  - Modular validator architecture under `src/quality/validators/` (completeness, empty translation, entity consistency, glossary consistency, gender agreement, numbers/units, control tokens, dialogue integrity).
  - Bounded repair engine `src/quality/repair.py` (max 2 retries with targeted diagnostic feedback).
  - Segments that fail retries transition to `REVIEW_REQUIRED` (never silently accepted).
  - Auditable `quality_reports` persisted in SQLite.
  - Document writers (`txt_writer.py`, `pdf_writer.py`) consume only `ACCEPTED` segments by default.

## Feature Inventory
| # | Feature ID | Feature Name & Description | Milestone | Source |
|---|------------|----------------------------|-----------|--------|
| 1 | F01_REGRESSION_CORPUS | Create `tests/regression/quality_cases.json` reproducing V1 quality failures | M_TEST | ORIGINAL_REQUEST § R1, Survey |
| 2 | F02_REGRESSION_RUNNER | Regression runner `tests/regression/test_quality_cases.py` asserting test cases | M_TEST | ORIGINAL_REQUEST § R1, Survey |
| 3 | F03_E2E_TIER1_FEATURE | Opaque-box Tier 1 Feature Coverage test cases (>=5 per feature) | M_TEST | Dual Track Protocol |
| 4 | F04_E2E_TIER2_BOUNDARY | Opaque-box Tier 2 Boundary & Corner test cases (>=5 per feature) | M_TEST | Dual Track Protocol |
| 5 | F05_E2E_TIER3_PAIRWISE | Opaque-box Tier 3 Pairwise cross-feature combination test cases | M_TEST | Dual Track Protocol |
| 6 | F06_E2E_TIER4_WORKLOAD | Opaque-box Tier 4 Real-world document application scenarios | M_TEST | Dual Track Protocol |
| 7 | F07_TEST_READY_SIGNAL | Creation of `TEST_READY.md` signaling E2E test suite readiness | M_TEST | Dual Track Protocol |
| 8 | F08_SENTENCE_IMMUTABILITY | Strict immutability of `Sentence.original_text` across all stages | M1 | ORIGINAL_REQUEST § R1 |
| 9 | F09_NORMALIZED_SOURCE_TEXT | Dedicated `Sentence.normalized_source_text` for preprocessing & unit conversion | M1 | ORIGINAL_REQUEST § R1 |
| 10 | F10_PREPROCESS_ISOLATION | Update `UnitConverter` & `PreprocessingPipeline` to leave `original_text` untouched | M1 | ORIGINAL_REQUEST § R1 |
| 11 | F12_BOOK_SHA256_ID | Base `Book.id` on SHA-256 of document content (`sha256(file_bytes)`) | M1 | ORIGINAL_REQUEST § R1 |
| 12 | F12_JOB_ID_FINGERPRINT | Generate deterministic `job_id` and fingerprint (version, model, prompt, KB) | M1 | ORIGINAL_REQUEST § R1 |
| 13 | F13_SEGMENT_STATUS_LIFECYCLE | 7-state lifecycle: PENDING -> DRAFT_COMPLETED -> EDITED -> VALIDATING -> ACCEPTED/REVIEW_REQUIRED/FAILED | M1 | ORIGINAL_REQUEST § R1 |
| 14 | F14_AYA_FAILURE_VISIBILITY | Visible failure on malformed Aya output; eliminate silent fallback to draft marked success | M1 | ORIGINAL_REQUEST § R1 |
| 15 | F15_DETERMINISTIC_STAGE2_CONFIG | Deterministic Stage 2 config (temp=0.0, do_sample=False, top_p=1.0, rep_penalty=1.02) | M1 | ORIGINAL_REQUEST § R1 |
| 16 | F16_WRITERS_ACCEPTED_ONLY | Ensure document writers consume only `ACCEPTED` segments by default | M1 | ORIGINAL_REQUEST § R1 |
| 17 | F17_DB_MIGRATIONS_SYSTEM | Database migrations system (`src/knowledge_base/migrations/`) with `schema_version` | M2 | ORIGINAL_REQUEST § R2 |
| 18 | F18_SCOPE_HIERARCHY | Scope hierarchy: `BOOK > SERIES > DOMAIN > GLOBAL` in models & repository | M2 | ORIGINAL_REQUEST § R2 |
| 19 | F19_ENTITY_PROFILE_MODEL | `EntityProfile` with source, canonical, aliases, forms, gender, policy, confidence, locked | M2 | ORIGINAL_REQUEST § R2 |
| 20 | F20_ENTITY_AUTOLOCK | Auto-lock entities at confidence >= 0.90 | M2 | ORIGINAL_REQUEST § R2 |
| 21 | F21_FORBIDDEN_VARIANT_REJECTION | Reject forbidden variants for locked entities (e.g. Cherry rejects Вишня) | M2 | ORIGINAL_REQUEST § R2 |
| 22 | F22_BOOK_ANALYZER | `src/analysis/book_analyzer.py` for pre-translation entity candidate extraction & classification | M2 | ORIGINAL_REQUEST § R2 |
| 23 | F23_ENTITY_MENTION_INDEX | Index entity mentions by segment (`EntityMention`) for precise localized retrieval | M2 | ORIGINAL_REQUEST § R2 |
| 24 | F24_TRANSLATION_SEGMENT_MODEL | `TranslationSegment` representing paragraph-level semantic units for Stage 2 | M3 | ORIGINAL_REQUEST § R3 |
| 25 | F25_NLLB_SENTENCE_DRAFT_AUX | Keep NLLB as auxiliary sentence-draft generator feeding `TranslationSegment` | M3 | ORIGINAL_REQUEST § R3 |
| 26 | F26_CONTEXT_BUILDER | `ContextBuilder` (`src/context/builder.py`) assembling paragraph, drafts, history, locked terms | M3 | ORIGINAL_REQUEST § R3 |
| 27 | F27_TOKEN_BUDGET_CALCULATOR | Tokenizer-aware `TokenBudget` dynamically managing prompt budget | M3 | ORIGINAL_REQUEST § R3 |
| 28 | F28_EDITING_PROMPT_V2 | Deploy `data/prompts/editing_prompt_v2.md` with structured JSON keyed by segment IDs | M3 | ORIGINAL_REQUEST § R3 |
| 29 | F29_STAGE2_PARAGRAPH_PIPELINE | Refactor `aya_editing_engine.py` and `pipeline.py` to refine `TranslationSegment`s | M3 | ORIGINAL_REQUEST § R3 |
| 30 | F30_MODULAR_VALIDATOR_ARCHITECTURE | Refactor `src/quality/pipeline.py` into modular validator architecture in `src/quality/validators/` | M4 | ORIGINAL_REQUEST § R4 |
| 31 | F31_SPECIFIC_VALIDATORS | Implement 8 modular validators (completeness, empty, entity, glossary, gender, units, tokens, dialogue) | M4 | ORIGINAL_REQUEST § R4 |
| 32 | F32_BOUNDED_REPAIR_ENGINE | Implement bounded repair loop (`src/quality/repair.py`) with max 2 retries with QA diagnostics | M4 | ORIGINAL_REQUEST § R4 |
| 33 | F33_PERSIST_AUDITABLE_QA_REPORTS | Persist auditable quality reports for every segment in SQLite table `quality_reports` | M4 | ORIGINAL_REQUEST § R4 |
| 34 | F34_QUALITY_RUNNER_INTEGRATION | Integrate `QualityPipeline` and repair loop directly into `TranslationRunner` before reconciliation | M4 | ORIGINAL_REQUEST § R4 |
| 35 | F35_E2E_INTEGRATION_AND_HARDENING | 100% pass of E2E test suite (Tiers 1-4) followed by Tier 5 adversarial coverage hardening | M_FINAL | Project Pattern Final Milestone |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M_TEST | E2E Testing Track | Design & implement opaque-box test suite, regression corpus `quality_cases.json`, and publish `TEST_READY.md` | none | IN_PROGRESS |
| M1 | Pipeline Safety, Content Identity & Immutability | Sentence immutability, `normalized_source_text`, SHA-256 book ID & fingerprint, 7-state lifecycle, deterministic Aya config, writers gating | none | IN_PROGRESS |
| M2 | Scoped Knowledge Base & Whole-Book Analysis | Migrations runner, `BOOK>SERIES>DOMAIN>GLOBAL`, `EntityProfile` (autolock >= 0.90, forbidden variants), `BookAnalyzer`, `EntityMention` | M1 (models & contracts) | PLANNED |
| M3 | Paragraph-Aware Contextual Translation | `TranslationSegment`, `ContextBuilder`, `TokenBudget`, `editing_prompt_v2.md`, Stage 2 paragraph pipeline | M1, M2 | PLANNED |
| M4 | Modular QA Architecture & Bounded Repair Pass | Modular validators (`src/quality/validators/`), `BoundedRepairEngine` (max 2 retries), SQLite QA reports, `TranslationRunner` integration | M1, M2, M3 | PLANNED |
| M_FINAL | Final Milestone (E2E Test Pass & Adversarial Hardening) | Phase 1: 100% pass of E2E test suite (Tiers 1-4); Phase 2: Tier 5 adversarial hardening with Challenger | M_TEST, M4 | PLANNED |

## Interface Contracts
### Preprocessing ↔ Domain Models
- `Sentence.original_text: str` is frozen / immutable after instantiation.
- `Sentence.normalized_source_text: Optional[str]` holds converted units and preprocessed source text.
- `Book.id: UUID = UUID(hex=hashlib.sha256(file_bytes).hexdigest()[:32])`.
- `job_id: str = f"{book_id}_{hashlib.sha256(composite_config_bytes).hexdigest()[:16]}"`.

### Knowledge Base ↔ Translation / Analysis
- Scope precedence: `BOOK > SERIES > DOMAIN > GLOBAL`.
- `EntityProfile`:
  - `id: UUID`, `scope: ScopeLevel`, `scope_id: Optional[str]`
  - `source_name: str`, `canonical_target: str`, `aliases: List[str]`
  - `allowed_target_forms: List[str]`, `forbidden_target_forms: List[str]`
  - `grammatical_gender: Optional[str]`, `translation_policy: str`
  - `confidence: float`, `locked: bool` (locked if confidence >= 0.90)
- `EntityMention`:
  - `id: UUID`, `entity_id: UUID`, `segment_id: UUID`, `char_start: int`, `char_end: int`, `surface_form: str`

### ContextBuilder ↔ Aya Editing Engine
- `TranslationSegment`:
  - `id: UUID`, `paragraph_id: UUID`, `sentence_ids: List[UUID]`, `source_text: str`, `draft_text: str`, `status: SegmentStatus`
- `ContextBuilder.build_context(segment, prev_segments, entities, summary, token_budget) -> PromptContext`:
  - Returns prompt parameters adhering to `TokenBudget`.
- Output format:
  ```json
  {
    "segments": [
      {
        "id": "<segment_uuid>",
        "translation": "<refined ukrainian text>"
      }
    ]
  }
  ```

### QualityPipeline & Repair ↔ TranslationRunner
- `QualityPipeline.validate_segment(segment: TranslationSegment, context: ValidationContext) -> QualityReport`:
  - Returns `QualityReport` with `is_valid: bool`, `issues: List[QualityIssue]`, `severity: IssueSeverity`.
- `BoundedRepairEngine.repair_segment(segment, report, max_retries=2) -> TranslationSegment`:
  - Retries up to 2 times prompting LLM with detected QA diagnostics.
  - If valid -> `status = SegmentStatus.ACCEPTED`.
  - If still invalid after 2 retries -> `status = SegmentStatus.REVIEW_REQUIRED`.

## Code Layout
- `src/domain/models/`:
  - `document.py` (`Sentence`, `Paragraph`, `Chapter`, `Book`)
  - `segment.py` (`TranslationSegment`, `SegmentStatus`)
  - `knowledge.py` (`ScopeLevel`, `EntityProfile`, `EntityMention`, `GlossaryItem`)
  - `quality.py` (`QualityIssue`, `QualityReport`, `IssueSeverity`)
- `src/knowledge_base/`:
  - `migrations/` (`migration_runner.py`, `001_initial_schema.py`, etc.)
  - `sqlite_repository.py`
- `src/analysis/`:
  - `book_analyzer.py`
- `src/context/`:
  - `builder.py`, `token_budget.py`
- `src/translation/`:
  - `aya_editing_engine.py`, `pipeline.py`, `nllb_engine.py`
- `src/quality/`:
  - `pipeline.py`
  - `repair.py`
  - `validators/` (`base.py`, `completeness.py`, `empty_translation.py`, `entity_consistency.py`, `glossary_consistency.py`, `gender_agreement.py`, `numbers_and_units.py`, `control_tokens.py`, `dialogue_integrity.py`)
- `src/launcher/`:
  - `translation_runner.py`, `cli.py`, `app_context.py`
- `src/writers/`:
  - `txt_writer.py`, `pdf_writer.py`
- `tests/`:
  - `regression/` (`quality_cases.json`, `test_quality_cases.py`)
  - `unit/`, `integration/`, `test_harness.py`
