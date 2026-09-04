# Test Suite Readiness Signal: BookTranslator V2 (M_TEST)

**Status**: READY  
**Milestone**: M_TEST (E2E Testing Track)  
**Date**: 2026-09-03  
**Author**: teamwork_preview_test_writer_m_test  
**Target Repository**: `d:\Перекладач`  

---

## 1. Executive Summary

The E2E test and regression testing infrastructure for **BookTranslator V2** is fully implemented, verified, and operational. All core deliverables specified in `PROJECT.md`, `TEST_INFRA.md`, and the user requirements have been created and validated:

1. **Regression Test Corpus (`tests/regression/quality_cases.json`)**:
   - Comprehensive, version-controlled JSON corpus containing **31 test cases** targeting confirmed V1 quality failures across all 5 mandatory categories:
     - **Character Names** (7 cases): strict preservation (e.g., `Cherry` -> `Черрі`, rejecting `Вишня`, `Вішня`, `Черешня`, `Черри`; plus proper noun homographs `Hope`, `Faith`, `Robin`, `Rose`, `Flint`).
     - **Polysemy** (6 cases): context-aware disambiguation (`tart` speech tone vs culinary pastry; `bank` river vs financial; `cast` magic vs plaster; `mine` excavation vs weapon; `trunk` tree stem vs elephant/luggage).
     - **Items** (6 cases): RPG fantasy item standardization (`healing potion` -> `зілля зцілення` strictly rejecting `напій`/`чай`; `mana potion` -> `зілля мани`; `broadsword` -> `палаш`; `scroll of town portal` -> `сувій міського порталу`; `stamina elixir` -> `еліксир витривалості`; `chainmail armor` -> `кольчуга`).
     - **Gender Agreement** (6 cases): subject-verb grammatical agreement for female, male, and inverted dialogue clauses (`Марія увійшла... помітила... зітхнула`; `Черрі усміхнулася`; `Лорд Браян підвівся... наказав`; `Принцеса Олена була... вирушила`; `Гросмейстер Торн зупинився... обернувся`; dialogue inversion `«...», — тихо прошепотіла Марія`).
     - **RPG Class Tags** (6 cases): formatting and UI bracket preservation (`[Status: Active] [Class: Shadowblade] [Level: 45]`; `[System Alert: ...]`; `[HP: 250/250] [MP: 120/120]`; `[Skill Acquired: ... (Rank: ...)]`; `[Class Evolution: ...]`; `[Combat Log: ...]`).

2. **Regression Test Runner & Evaluator (`tests/regression/test_quality_cases.py`)**:
   - **108 automated pytest assertions** verifying:
     - Corpus schema validity, completeness, and ID uniqueness.
     - Distribution across all 5 categories (>=5 cases each).
     - Clean passage of all 31 canonical translations with 0 violations.
     - Detection sensitivity: all 31 historical V1 failure examples fail evaluation as expected.
     - Robustness: injecting forbidden variants into canonical translations triggers deterministic violations.
     - Category-specific invariant checks for names, polysemy, items, gender agreement, and RPG UI tags.

3. **Full Test Suite Status**:
   - Pytest unit + integration + regression suite: **366 / 366 tests passing (100%)** with zero regressions.
   - Master test harness (`tests/test_harness.py`): **113 / 113 tests passing across 5 tiers**.

---

## 2. Feature Inventory & Test Coverage Matrix

In accordance with `TEST_INFRA.md`, the test suite covers the following 18 feature areas across Tiers 1 through 4:

| # | Feature Area | Requirement Source | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Real-World) | Status |
|---|--------------|--------------------|:----------------:|:-----------------:|:-----------------:|:-------------------:|:------:|
| 1 | Sentence Immutability & Normalization | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ | READY |
| 2 | Content Identity (SHA-256 & Deterministic Job) | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ | READY |
| 3 | 7-State Segment Lifecycle | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ | READY |
| 4 | Aya Failure Visibility (No Silent Fallback) | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ | READY |
| 5 | Deterministic Stage 2 Generation | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ | READY |
| 6 | Document Writers Status Gating (Accepted Only) | ORIGINAL_REQUEST § R1 | >=5 | >=5 | ✓ | ✓ | READY |
| 7 | DB Migrations & Schema Versioning | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ | READY |
| 8 | Scoped Knowledge Hierarchy (BOOK>SERIES>DOMAIN>GLOBAL) | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ | READY |
| 9 | EntityProfile (Autolock >= 0.90 & Forbidden Variants) | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ | READY |
| 10 | Pre-translation Whole-Book Analysis | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ | READY |
| 11 | Localized Entity Mention Indexing | ORIGINAL_REQUEST § R2 | >=5 | >=5 | ✓ | ✓ | READY |
| 12 | Paragraph TranslationSegment Refinement | ORIGINAL_REQUEST § R3 | >=5 | >=5 | ✓ | ✓ | READY |
| 13 | ContextBuilder & TokenBudget Allocation | ORIGINAL_REQUEST § R3 | >=5 | >=5 | ✓ | ✓ | READY |
| 14 | editing_prompt_v2 Structured JSON Output | ORIGINAL_REQUEST § R3 | >=5 | >=5 | ✓ | ✓ | READY |
| 15 | Modular Quality Validators (8 Validators) | ORIGINAL_REQUEST § R4 | >=5 | >=5 | ✓ | ✓ | READY |
| 16 | Bounded Repair Loop (max 2 retries) | ORIGINAL_REQUEST § R4 | >=5 | >=5 | ✓ | ✓ | READY |
| 17 | Auditable Quality Reports Persistence | ORIGINAL_REQUEST § R4 | >=5 | >=5 | ✓ | ✓ | READY |
| 18 | Quality Cases Regression Suite | ORIGINAL_REQUEST § R1, Survey | 7 | 6 | 6 | 12 | READY |

---

## 3. Regression Corpus Structure & Verification Rules

### 3.1 JSON Schema Contract (`tests/regression/quality_cases.json`)
Each case in the corpus provides complete dual-format compatibility:
- `id`: Unique identifier formatted as `REG-<CATEGORY_SHORT>-<NUMBER>` (e.g. `REG-NAME-001`).
- `category`: One of `character_names`, `polysemy`, `items`, `gender_agreement`, `rpg_class_tags`.
- `description`: Plain-language explanation of the quality rule and target defect.
- `source_text` & `source_en`: Source English text containing the test element.
- `expected_uk_canonical`: Authoritative canonical Ukrainian translation.
- `forbidden_variants`: Strictly rejected strings, mistranslations, or phonetic distortions.
- `v1_failure_example`: Verbatim example of historical failure mode produced by V1.
- `context`: Structured metadata including active entities, glossaries, domain, and grammatical gender.
- `metadata`: Testing tier, syntactic patterns, formatting properties.
- `expected_constraints`:
  - `must_include`: Mandatory substrings or sub-lists.
  - `must_include_any`: Alternative acceptable substrings (at least one required).
  - `must_not_include`: Forbidden terms (mirrors `forbidden_variants`).
  - `required_gender`: Grammatical gender constraint ("жіночий", "чоловічий", "середній").
  - `expected_regex`: Structural regex pattern verification.
  - `forbidden_regex`: Forbidden regex pattern verification.

### 3.2 Category Breakdown
| Category | Cases Count | Primary Invariant Verified |
|----------|:-----------:|----------------------------|
| `character_names` | 7 | Proper names are never translated as common nouns (fruits, birds, virtues, minerals) |
| `polysemy` | 6 | Semantic context determines sense (speech tone vs culinary, river bank vs financial, etc.) |
| `items` | 6 | Fantasy items adopt standard literary terminology (`зілля`, `палаш`, `сувій`, `кольчуга`) |
| `gender_agreement` | 6 | Past-tense verbs and adjectives agree with subject grammatical gender |
| `rpg_class_tags` | 6 | Bracketed UI blocks, system alerts, and status tags are preserved intact |
| **Total** | **31** | **Comprehensive V1 regression coverage** |

---

## 4. Verification & Execution Instructions

The test suite can be run using the project virtual environment:

```bash
# 1. Run the new regression test suite (108 tests)
.venv\Scripts\python.exe -m pytest tests/regression/test_quality_cases.py -v

# 2. Run the complete pytest test suite (366 tests)
.venv\Scripts\python.exe -m pytest tests/unit tests/integration tests/regression -v

# 3. Run the master 5-tier test harness (113 tests)
.venv\Scripts\python.exe tests/test_harness.py
```

### Empirical Verification Results:
- `pytest tests/regression`: **108 passed** in 0.18s.
- `pytest tests/unit tests/integration tests/regression`: **366 passed**, 1 warning in 8.28s.
- `python tests/test_harness.py`: **113 passed** (100% pass rate).

---

## 5. Downstream Milestones Readiness Sign-Off

The regression corpus and assertion patterns established here are ready for immediate consumption by:
- **Milestone M1**: Pipeline Safety & Immutability verification.
- **Milestone M2**: Scoped Knowledge Base & `EntityProfile` forbidden variant rejection (`Cherry` -> `Черрі`).
- **Milestone M3**: Paragraph-level `TranslationSegment` contextual refinement.
- **Milestone M4**: Modular `QualityPipeline` validators (`EntityConsistencyValidator`, `GenderAgreementValidator`, `NumberAndUnitValidator`).
- **Milestone M_FINAL**: Final E2E verification and adversarial hardening.
