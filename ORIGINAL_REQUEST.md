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

## 2026-09-03T08:14:20Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Small focused team (single self-contained bug fix)

This is a single self-contained fix; keep it small and focused.
Виправити 4 підтверджених проблеми другого раунду тестування BookTranslator у суворому порядку черговості (1 -> 3 -> 2 -> 4) із зупинкою на перевірці тестів (`pytest tests/unit -x`) після кожного кроку:
1. Проблема 1 (найпростіша й локалізована): обробка обірваних escape-артефактів (`\"...`) при неповному JSON.
2. Проблема 3 (wiring): активація та прокидання `convert_units` через Settings, DI-контейнер та CLI.
3. Проблема 2 (модель даних і промпт): розділення статусу термінів глосарія на підтверджені (`reviewed=True`) та авто-екстраговані (`reviewed=False`).
4. Проблема 4 (гендерне узгодження): підтримка `grammatical_gender` у моделі `GlossaryItem` та явне передавання роду в промпт для перекладача.

Working directory: d:\Перекладач
Integrity mode: development

## Verification Resources
- Покрокова верифікація: `pytest tests/unit -x` після кожного виконаного пункту
- Повний набір тестів на фініші: `pytest tests/unit tests/integration`
- Комплексний системний прогін: `python tests/test_harness.py`

## Requirements

### R1. Крок 1 — Проблема 1: Обробка обірваних escape-артефактів (`\"...`)
- У `src/translation/aya_editing_engine.py` (`_parse_json_response()`, Strategy 3) додати обробку обірваних рядків перед фінальним збереженням:
  - Відрізати кінцевий бекслеш (`\`), якщо рядок закінчується на нього.
  - Виконати unescape для послідовностей `\\"`, `\\n`, `\\\\`.
  - Зняти залишковий початковий бекслеш/лапку (`\\"` або незакриту `"`).
- Додати `logger.debug` при переході від Strategy 1 до Strategy 2/3.
- Додати regression-тест `test_aya_parse_json_response_with_truncated_escaped_value` у `tests/unit/test_translation.py`.

### R2. Крок 2 — Проблема 3: Активація та підключення конвертації одиниць виміру
- У `src/config/settings.py` додати поле `convert_units: bool = True`.
- Прокинути `convert_units=settings.provided.convert_units` у DI-контейнери (`src/launcher/app_context.py` та `src/config/app_context.py`).
- Додати прапорець `--no-unit-conversion` у CLI `src/launcher/cli.py`.
- Додати unit-тест `test_preprocessing_pipeline_convert_units_enabled_by_default_from_settings`.

### R3. Крок 3 — Проблема 2: Розділення статусу термінів глосарія (`reviewed`)
- У `src/domain/models/knowledge.py` додати поле `reviewed: bool = False` до моделі `GlossaryItem`.
- В `src/preprocessing/pipeline.py` під час автоматичної екстракції створювати `GlossaryItem` зі статусом `reviewed=False`.
- В `src/translation/aya_editing_engine.py` розділити формування `glossary_str` у промпті:
  - Для `reviewed=True`: обов'язкові відповідники у форматі `- {source} => {target}`.
  - Для `reviewed=False`: рекомендація однакового перекладу та транслітерації повторюваних імен без нав'язування заглушки `- {source}`.
- Оновити фікстури у тестах для сумісності з новим полем.

### R4. Крок 4 — Проблема 4: Гендерне узгодження персонажів (`grammatical_gender`)
- Додати поле `grammatical_gender: Optional[str] = None` до `GlossaryItem` (значення "чоловічий", "жіночий", "середній").
- У промпті `aya_editing_engine.py` для `reviewed`-термінів додавати граматичний рід персонажа: `- {source} => {target}, граматичний рід: {gender}`.
- В `src/quality/pipeline.py` додати евристичну перевірку (WARNING) на невідповідність родових закінчень дієслів минулого часу імені персонажа.

## Acceptance Criteria

### Verification & Testing
- [ ] Обірване на середині значення з бекслешами коректно парситься без артефактів; regression-тест проходить.
- [ ] `settings.convert_units` за замовчуванням увімкнено (`True`), прапорець `--no-unit-conversion` дозволяє вимкнути його з CLI; тест wiring проходить.
- [ ] Авто-екстраговані імена більше не блокують транслітерацію суворим знаком тотожності `=>`, а підтверджені терміни передаються моделі з точним перекладом та родом.
- [ ] 100% тестів (`pytest tests/unit tests/integration` та `python tests/test_harness.py`) проходять без помилок та регресій.


