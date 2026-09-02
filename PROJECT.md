# Project: BookTranslator Architectural Review & Quality Remediation

## Architecture
- **Stage 1 (Machine Translation)**: High-throughput draft translation powered by CTranslate2 NLLB-200, dynamic token-bucket batching ($B_{tokens}=2048$), and SQLite 50-chunk WAL transactions.
- **Stage 2 (Literary Refinement)**: High-precision stylistic editing powered by Quantized Aya-23-8B / Aya Expanse (4-bit NF4 / GGUF), in-memory prompt templates, 2-sentence overlap context, multi-pass regex sanitization, and atomic WAL checkpointing.
- **Supporting Infrastructure**: Rule-based sentence segmentation (15 rules), deterministic UUID DOM tree (`Book` -> `Chapter` -> `Paragraph` -> `Sentence`), $O(S)$ dictionary DOM reconciliation, and 50Hz non-blocking CustomTkinter UI.

## Feature Inventory & Issue Mapping
| # | Requirement / Issue | Description | Status | Reference Artifact |
|---|---------------------|-------------|--------|-------------------|
| 1 | Pipeline Architecture Review | Comprehensive mapping of text lifecycle from ingestion to output | COMPLETED | `architectural_review.md` |
| 2 | Issue 1: Sentence/Paragraph Duplication | Stale NLLB drafts retained when Aya merges sentences ($M < N$) | ANALYZED & PLANNED | `root_cause_analysis.md`, `improvement_plan.md § 1` |
| 3 | Issue 2: Missing Text (Dropped Sentences) | Prompt lacks `{source_text}`; segmenter lookahead omissions | ANALYZED & PLANNED | `root_cause_analysis.md`, `improvement_plan.md § 2` |
| 4 | Issue 3: Technical Artifact Leaks | Dynamic `# Header` and `<tag_1>` leaking past static sanitizer | ANALYZED & PLANNED | `root_cause_analysis.md`, `improvement_plan.md § 3` |
| 5 | Issue 4: Inconsistent Names / Terms | Missing `{source_text}` in prompt; unpopulated SQLite glossary | ANALYZED & PLANNED | `root_cause_analysis.md`, `improvement_plan.md § 4` |
| 6 | Issue 5: Hybrid Portmanteaus ("Смачнissimo") | `repetition_penalty = 1.15` suppresses Ukrainian roots -> subword bleed | ANALYZED & PLANNED | `root_cause_analysis.md`, `improvement_plan.md § 5` |
| 7 | Issue 6: Inconsistent Unit Conversions | Zero deterministic converter; stochastic LLM mental math | ANALYZED & PLANNED | `root_cause_analysis.md`, `improvement_plan.md § 6` |
| 8 | Actionable Improvement Plan | Concrete implementable remediation blueprint across all 6 issues | COMPLETED | `improvement_plan.md` |

## Milestones & Status
| # | Name | Scope | Status | Deliverable |
|---|------|-------|--------|-------------|
| M1 | Multi-Agent Survey & Exploration | 3 parallel Explorers (Architecture, Structural Issues, Linguistic Issues) | DONE | `explorer_arch_1/`, `explorer_struct_1/`, `explorer_ling_1/` |
| M2 | Synthesis & Root Cause Analysis | Formal analysis and tracing of all 6 issues with code/test evidence | DONE | `root_cause_analysis.md`, `architectural_review.md` |
| M3 | Actionable Improvement Plan Artifact | Concrete implementation code/prompt blueprint for all 6 issues | DONE | `improvement_plan.md` |
| M4 | Multi-Agent Review & Forensic Audit | Verification by Reviewer and Forensic Auditor | IN_PROGRESS | `reviewer_1/`, `auditor_1/`, `GATE_STATUS.md` |
| M5 | Final Notification & Delivery | Verification completion and formal report back to caller agent | PENDING | `handoff.md` |

## Interface Contracts & Data Models
- `Book` DOM: `Book` -> `List[Chapter]` -> `List[Paragraph]` -> `List[Sentence]` with MD5 deterministic UUIDs.
- `TranslationChunk`: `chunk_id: UUID`, `book_id: UUID`, `target_sentence_ids: List[UUID]`, `context_sentence_ids: List[UUID]`, `status: ChunkStatus`.
- `_sanitize_output`: `text: str -> str` multi-pass cleaning ensuring zero leaking headers, markdown fences, preambles, or unmapped inline tags.
