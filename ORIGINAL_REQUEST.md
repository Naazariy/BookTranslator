# Original User Request

## 2026-08-15T11:05:20Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Full team

Implement the performance optimizations detailed in the `optimization_plan.md` report across the entire BookTranslator application. The implementation should cover all three pillars (UI, ML, DB) concurrently.

Working directory: d:\Перекладач
Integrity mode: development

## Requirements

### R1. UI & Concurrency
Refactor the GUI (`gui.py`) to prevent UI freezes. Implement a non-blocking `post_to_ui()` event queue, an `AsyncTaskManager`, and throttled log buffers as specified in the optimization plan.

### R2. ML Translation Pipelines
Optimize `nllb_engine.py` and `aya_editing_engine.py`. Integrate CTranslate2 and BitsAndBytes for 4-bit/GGUF quantization to reduce VRAM footprint. Implement dynamic token-bucket batching to eliminate padding waste. Installing new Python dependencies is explicitly permitted.

### R3. Database and Parsers
Optimize SQLite transactions by enabling WAL mode, compound indexing, and chunk batch transactions. Refactor the document parser and `pdf_writer.py` to use O(S) dictionary lookups for DOM tree reconstruction instead of nested loops.

## Acceptance Criteria

### Execution & Performance
- [ ] The application launches successfully and the GUI remains fully responsive during a translation task.
- [ ] The benchmark scripts in `scratch/` execute successfully and show empirical performance gains (e.g. VRAM usage dropped to ~4.8GB, database write throughput increased).
- [ ] All three optimization pillars (UI, ML, DB) are fully implemented without breaking the core translation logic or losing translation accuracy.

## Follow-up — 2026-08-15T16:11:50Z

Please resume your work and report the final status. Did you complete the victory synthesis before the server restarted?

## 2026-08-16T14:00:52Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Small focused team (single self-contained bug fix)

This is a single self-contained fix; keep it small and focused.
Виправити баг дублювання речень на межах чанків (overlap bug) у проєкті BookTranslator, щоб кожне речення перекладалося і потрапляло у фінальний документ рівно один раз.

Working directory: d:\Перекладач
Integrity mode: development

## Requirements

### R1. Джерело правди — `Sentence`, а не `Chunk`
Переклад повинен писатися не в `chunk.draft_translation` / `chunk.final_translation` (суцільний рядок на весь чанк), а напряму в поле результату, прив'язане до `sentence.id`. Забезпечити наявність `order_index` у `Sentence` для коректної збірки.

### R2. Розділення ролей Overlap-речень у чанку
У `TranslationChunk` чітко розділити:
- `context_sentence_ids`: тільки для довідки (стиль/термінологія), НЕ перезаписують переклад.
- `target_sentence_ids`: саме ці речення перекладаються в цьому чанку.
Кожне речення документа має бути в рівно **одному** `target_sentence_ids` серед усіх чанків. Overlap реалізується виключно через `context_sentence_ids`.

### R3. Оновлення NLLB Stage
Запис перекладу здійснювати по `sentence_id`. Ігнорувати контекст.
```python
def execute_nllb_stage(chunks: list[TranslationChunk]) -> None:
    # Перекладати лише target_sentences.
    # Записувати в sentence.draft_translation
```

### R4. Оновлення Aya Stage
Запис перекладу здійснювати по `sentence_id`. Додати програмну перевірку після парсингу відповіді LLM: ігнорувати будь-які ID у відповіді, яких не було в `target_sentence_ids`, навіть якщо LLM порушила інструкцію і повернула зайве.

### R5. Збірка (Writer) за `order_index`
Збірка документа має відбуватися виключно шляхом ітерації речень у їх глобальному порядку (`order_index`), без конкатенації тексту чанків. `Writer` ніколи не бере текст із `chunk`.

### R6. Що НЕ робити
Не вирішувати проблему постфактум (детекція й видалення дублікатів після збірки через string similarity). Не залишати запис перекладу в `chunk.draft_translation`/`final_translation` як суцільний рядок.

## Acceptance Criteria

### Verification & Testing
- [ ] Створено новий regression-тест, що явно перевіряє відсутність дублікатів (речення на межі overlap між двома чанками не повинно з'являтися у фінальному документі двічі).
- [ ] Оновлено логіку побудови чанків: кожне речення є `target` рівно в одному чанку.
- [ ] Оновлено `nllb.py` та `aya.py`: запис результату відбувається по `sentence.id`.
- [ ] Оновлено `writer`: збірка документа відбувається виключно за `order_index` речень.
- [ ] Надано коротке пояснення, які файли й функції змінились і чому (без переписування всього проєкту з нуля).

## 2026-09-02T07:53:31Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Full team

Implement the concrete code changes, prompt adjustments, and architectural updates detailed in `improvement_plan.md` across the BookTranslator application to definitively resolve translation quality issues (duplication, dropped text, artifacts, terminology inconsistencies, hallucinations, and unit conversions).

Working directory: d:\Перекладач
Integrity mode: development

## Verification Resources
- `improvement_plan.md`: The detailed roadmap containing exact code snippets and regex patterns to implement.
- Existing pytest suite (agents are authorized to update existing tests to match the new behavior if necessary).

## Requirements

### R1. Implement Core Pipeline & Prompts (M1-M3)
Apply the fixes for duplication ($M < N$ merge clearing in `pipeline.py` / `translation_runner.py`), missing text (rebuild `editing_prompt.md` to include `{source_text}` and expand segmenter lookahead), and technical artifacts (implement dynamic multi-pass regex in `_sanitize_output`).

### R2. Implement Linguistics & Unit Conversion (M4-M6)
Adjust Aya hyperparameters (repetition penalty calibration). Implement dynamic chunk-level glossary injection and Cyrillic-Latin mixed-script post-processing. Create and integrate the deterministic `UnitConverter` module (`src/preprocessing/unit_converter.py`) for precise metric conversions.

## Acceptance Criteria

### Verification & Testing
- [ ] All code and prompt changes specified in `improvement_plan.md` have been fully integrated into the `src/` and `data/` directories.
- [ ] The full pytest suite passes successfully (100% pass rate). Existing tests broken by the new intentional behavior have been updated.
- [ ] New unit tests have been written specifically to cover `UnitConverter` logic and the new `_sanitize_output` regex paths, and they pass.
- [ ] Running an end-to-end test translation confirms the pipeline no longer produces duplicates, dropped sentences, or unhandled markdown headers.

