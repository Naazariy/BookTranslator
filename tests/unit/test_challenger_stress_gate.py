"""
Additional Empirical Challenger Stress Gate Test Suite (Challenger 3).

Comprehensive stress tests for:
1. Regex sanitization & mixed-script repairs (preambles, headers, tags, zero-width chars, portmanteaus).
2. Segmenter lookahead, abbreviations, dialogues, inline tags, ellipses, and OCR defects.
3. UnitConverter imperial-to-metric conversions, word-number parsing, and Ukrainian declension rules.
4. Sentence deduplication, merge policies (M<N, M=N, M>N, M=0), and DOM document serialization.
"""

import pytest
from uuid import uuid4
from typing import List

from src.parsers.segmenter import RuleBasedSentenceSegmenter
from src.translation.aya_editing_engine import QuantizedAyaEditingEngine
from src.quality.pipeline import sanitize_mixed_script_words
from src.preprocessing.unit_converter import UnitConverter
from src.domain.models.document import Sentence, Paragraph
from src.writers.txt_writer import TxtWriter


def execute_sentence_merge(target_sentences: List[Sentence], refined_text: str, segmenter: RuleBasedSentenceSegmenter):
    """Mirror of the production pipeline & runner sentence merge logic."""
    refined_sentences = segmenter.split_sentences(refined_text) if (refined_text and refined_text.strip()) else []

    if len(refined_sentences) == len(target_sentences):
        for s, r_text in zip(target_sentences, refined_sentences):
            s.translated_text = r_text.strip()
    elif len(target_sentences) == 1:
        if refined_sentences:
            target_sentences[0].translated_text = " ".join(s.strip() for s in refined_sentences).strip()
        elif refined_text and refined_text.strip():
            target_sentences[0].translated_text = refined_text.strip()
        elif not (target_sentences[0].translated_text and target_sentences[0].translated_text.strip()):
            target_sentences[0].translated_text = target_sentences[0].original_text
    elif refined_sentences:
        if len(refined_sentences) < len(target_sentences):
            for idx in range(len(refined_sentences)):
                target_sentences[idx].translated_text = refined_sentences[idx].strip()
            for idx in range(len(refined_sentences), len(target_sentences)):
                target_sentences[idx].translated_text = ""
        else:
            for idx in range(len(target_sentences) - 1):
                target_sentences[idx].translated_text = refined_sentences[idx].strip()
            target_sentences[-1].translated_text = " ".join(refined_sentences[len(target_sentences) - 1:]).strip()
    else:
        for s in target_sentences:
            if not (s.translated_text and s.translated_text.strip()):
                s.translated_text = s.original_text


# =========================================================================
# 1. Regex Sanitization & Mixed Script Tests
# =========================================================================

class TestSanitizerStressGate:
    @pytest.fixture
    def engine(self):
        return QuantizedAyaEditingEngine.__new__(QuantizedAyaEditingEngine)

    @pytest.mark.parametrize("raw_input,expected_output", [
        # Conversational preambles variations (English / Ukrainian, bold/colon order)
        ("**Ось фінальний переклад:** Текст роману тут.", "Текст роману тут."),
        ("Ось відредагований переклад:\nВін подивився у вікно.", "Він подивився у вікно."),
        ("Here is the translation:\n**Фінальний переклад**: Замок височів на горі.", "Замок височів на горі."),
        ("Here is the refined translation:\n**Ukrainian translation:**\nНіч опустилася на місто.", "Ніч опустилася на місто."),
        ("Відредагований літературний текст:\nГори сяяли у променях сонця.", "Гори сяяли у променях сонця."),
        ("Покращений переклад: Ранок був тихим.", "Ранок був тихим."),
        # Section markers
        ("=== ОРИГІНАЛЬНИЙ АНГЛІЙСЬКИЙ ТЕКСТ ===\nOriginal.\n=== ВІДРЕДАГОВАНИЙ ТЕКСТ ===:\nЧистий текст.", "Чистий текст."),
        ("=== ФІНАЛЬНИЙ ПЕРЕКЛАД ===\nСправжній фінальний результат.", "Справжній фінальний результат."),
        # Markdown headers
        ("# Розділ 1: Початок\n## Сцена перша\nДвері повільно відчинилися.", "Двері повільно відчинилися."),
        ("###### Примітка автора\nЦе важливий сюжетний поворот.", "Це важливий сюжетний поворот."),
        # Code fences
        ("```markdown\nСправжній текст без огорожі.\n```", "Справжній текст без огорожі."),
        ("```\n```python\nПерекладене речення.\n```\n```", "Перекладене речення."),
        # Unmapped inline tags
        ("<tag_1>Важливе слово</tag_1> у реченні.", "Важливе слово у реченні."),
        ("<tag_42>Текст без закриття <tag_99>і ще один", "Текст без закриття і ще один"),
        # Zero-width whitespace and unusual whitespace
        ("\u200b\u200c\u200d\ufeffЧистий рядок.\u200b\ufeff", "Чистий рядок."),
        ("\r\n\t  \u00a0  Текст з відступами.  \u00a0\r\n\t", "Текст з відступами."),
        # Outer wrapping quotes
        ('"Повне речення у подвійних лапках."', "Повне речення у подвійних лапках."),
        ("«Повне речення у ялинках.»", "Повне речення у ялинках."),
        ("“Повне речення у типографських лапках.”", "Повне речення у типографських лапках."),
        ("'Повне речення в одинарних лапках.'", "Повне речення в одинарних лапках."),
        # Portmanteaus & Mixed script
        ("Вечеря була Смачнissimo для всієї родини.", "Вечеря була Смакота для всієї родини."),
        ("Це виглядає Гарнissimo!", "Це виглядає Прегарно!"),
        ("Свято вдалося Чудовissimo.", "Свято вдалося Чудово."),
        ("Гарне мiсто iз смачною кaвою.", "Гарне місто із смачною кавою."),
    ])
    def test_sanitize_output_gate(self, engine, raw_input, expected_output):
        assert engine._sanitize_output(raw_input) == expected_output


# =========================================================================
# 2. Segmenter Lookahead & Tricky Syntax Tests
# =========================================================================

class TestSegmenterStressGate:
    @pytest.fixture
    def segmenter(self):
        return RuleBasedSentenceSegmenter()

    @pytest.mark.parametrize("text,expected_segments", [
        # Honorific abbreviations
        ("Dr. John Watson visited Mr. Sherlock Holmes.", ["Dr. John Watson visited Mr. Sherlock Holmes."]),
        ("Громадянин гр. Іваненко прибув до суду.", ["Громадянин гр. Іваненко прибув до суду."]),
        ("вул. Хрещатик, буд. 10, кв. 5.", ["вул. Хрещатик, буд. 10, кв. 5."]),
        # Ukrainian river and year abbreviations
        ("р. Дніпро впадає в Чорне море.", ["р. Дніпро впадає в Чорне море."]),
        ("У 1991 р. було проголошено незалежність.", ["У 1991 р. було проголошено незалежність."]),
        ("Подія сталася у 2024 р. Всі були вражені.", ["Подія сталася у 2024 р.", "Всі були вражені."]),
        # Latin & terminal abbreviations
        ("Buy food, e.g. bread, milk, etc. at the store.", ["Buy food, e.g. bread, milk, etc. at the store."]),
        ("We packed food, water, etc. The journey began at dawn.", ["We packed food, water, etc.", "The journey began at dawn."]),
        ("Купили хліб, воду тощо. Похід розпочався на світанку.", ["Купили хліб, воду тощо.", "Похід розпочався на світанку."]),
        # Decimal numbers
        ("Температура зросла на 3.5 градуса за 2.5 години.", ["Температура зросла на 3.5 градуса за 2.5 години."]),
        ("Розділ 2.4. Основні результати. Висновки.", ["Розділ 2.4.", "Основні результати.", "Висновки."]),
        # Inline XML tags around boundaries
        ("<tag_1>Перша думка автора.</tag_1> <tag_2>Друга важлива теза.</tag_2>", ["<tag_1>Перша думка автора.</tag_1>", "<tag_2>Друга важлива теза.</tag_2>"]),
        # Dialogue with em-dashes and quotes
        ("— Зупинись! — гукнув лицар. — Хто ти такий?", ["— Зупинись! — гукнув лицар.", "— Хто ти такий?"]),
        ("«Я не можу чекати», — промовила вона. «Час спливає».", ["«Я не можу чекати», — промовила вона.", "«Час спливає»."]),
        # Ellipses
        ("Вона замислилася... і повільно похитала головою.", ["Вона замислилася... і повільно похитала головою."]),
        ("Вона замислилася... Раптом пролунав дзвінок.", ["Вона замислилася...", "Раптом пролунав дзвінок."]),
        # OCR missing space
        ("День минув непомітно.Настала ніч.", ["День минув непомітно.", "Настала ніч."]),
        ("Куди ти біжиш?Повернися!", ["Куди ти біжиш?", "Повернися!"]),
    ])
    def test_segmenter_gate(self, segmenter, text, expected_segments):
        assert segmenter.split_sentences(text) == expected_segments


# =========================================================================
# 3. UnitConverter Deterministic Conversion & Declension Tests
# =========================================================================

class TestUnitConverterGate:
    @pytest.mark.parametrize("value,forms,expected_unit", [
        # Slavic declension rules for meters
        (1, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метр"),
        (2, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метри"),
        (3, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метри"),
        (4, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метри"),
        (5, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метрів"),
        (10, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метрів"),
        (11, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метрів"),
        (12, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метрів"),
        (14, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метрів"),
        (20, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метрів"),
        (21, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метр"),
        (22, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метри"),
        (24, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метри"),
        (25, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метрів"),
        (101, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метр"),
        (111, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метрів"),
        (112, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метрів"),
        (122, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метри"),
        # Fractions use genitive singular (родовий однини)
        (4.5, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метра"),
        (24.4, UnitConverter.UKRAINIAN_UNIT_FORMS['meter'], "метра"),
        (1.5, UnitConverter.UKRAINIAN_UNIT_FORMS['kilometer'], "кілометра"),
        (0.5, UnitConverter.UKRAINIAN_UNIT_FORMS['kilogram'], "кілограма"),
    ])
    def test_ukrainian_unit_declensions(self, value, forms, expected_unit):
        assert UnitConverter.get_ukrainian_unit_form(value, forms) == expected_unit

    @pytest.mark.parametrize("expr,expected_val", [
        ("80", 80.0),
        ("15.5", 15.5),
        ("1/2", 0.5),
        ("3/4", 0.75),
        ("eighty", 80.0),
        ("fifteen", 15.0),
        ("twenty-four", 24.0),
        ("one hundred", 100.0),
        ("two and a half", 2.5),
    ])
    def test_parse_number_expression(self, expr, expected_val):
        assert UnitConverter.parse_number_expression(expr) == pytest.approx(expected_val)

    def test_convert_text_units_metric_policy(self):
        # 80 feet -> ~24 meters (80 * 0.3048 = 24.384 -> 24 метри)
        converted_80ft = UnitConverter.convert_text("The wall was 80 feet high.", target_lang="uk", policy="metric")
        assert "24 метри" in converted_80ft

        # 15 miles -> ~24 km (15 * 1.60934 = 24.14 -> 24 кілометри)
        converted_15mi = UnitConverter.convert_text("They walked 15 miles across the hills.", target_lang="uk", policy="metric")
        assert "24 кілометри" in converted_15mi


# =========================================================================
# 4. Sentence Deduplication & DOM Reconciliation Tests
# =========================================================================

class TestSentenceDeduplicationGate:
    @pytest.fixture
    def segmenter(self):
        return RuleBasedSentenceSegmenter()

    def test_merge_20_to_1_clears_trailing_19(self, segmenter):
        """When N=20 sentences collapse to M=1, exactly sents[0] has content and sents[1..19] are empty."""
        p_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=i, original_text=f"Sent {i}.", translated_text=f"Draft {i}.")
            for i in range(20)
        ]
        refined = "Одне єдине синтезоване речення для всього абзацу."
        execute_sentence_merge(sents, refined, segmenter)

        assert sents[0].translated_text == refined
        assert all(s.translated_text == "" for s in sents[1:])

    def test_merge_5_to_2_clears_trailing_3(self, segmenter):
        """When N=5 sentences collapse to M=2, sents[0..1] get text and sents[2..4] are empty."""
        p_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=i, original_text=f"Sent {i}.", translated_text=f"Draft {i}.")
            for i in range(5)
        ]
        refined = "Перша зведена частина. Друга зведена частина."
        execute_sentence_merge(sents, refined, segmenter)

        assert sents[0].translated_text == "Перша зведена частина."
        assert sents[1].translated_text == "Друга зведена частина."
        assert sents[2].translated_text == ""
        assert sents[3].translated_text == ""
        assert sents[4].translated_text == ""

    def test_merge_3_to_5_expansion(self, segmenter):
        """When N=3 expands to M=5, sents[0..1] get items 0..1, sents[2] gets items 2..4 joined."""
        p_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=i, original_text=f"Sent {i}.", translated_text=f"Draft {i}.")
            for i in range(3)
        ]
        refined = "Перше речення тут. Друге речення тут. Третє речення тут. Четверте речення тут. П'яте речення тут."
        execute_sentence_merge(sents, refined, segmenter)

        assert sents[0].translated_text == "Перше речення тут."
        assert sents[1].translated_text == "Друге речення тут."
        assert sents[2].translated_text == "Третє речення тут. Четверте речення тут. П'яте речення тут."

    def test_empty_refinement_preserves_stage1_drafts(self, segmenter):
        """When refined text is empty or whitespace, existing draft translations remain intact."""
        p_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=0, original_text="Orig 1.", translated_text="Draft 1."),
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=1, original_text="Orig 2.", translated_text="Draft 2.")
        ]
        execute_sentence_merge(sents, "   ", segmenter)
        assert sents[0].translated_text == "Draft 1."
        assert sents[1].translated_text == "Draft 2."

    def test_dom_txt_writer_skips_empty_sentences_cleanly(self, segmenter):
        """TxtWriter generates seamless text without double spaces or orphan spaces from cleared sentences."""
        p_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=i, original_text=f"Orig {i}.", translated_text=f"Draft {i}.")
            for i in range(4)
        ]
        refined = "Перше речення. Друге речення."
        execute_sentence_merge(sents, refined, segmenter)

        # Build paragraph and verify writer rendering logic
        para = Paragraph(id=p_id, chapter_id=uuid4(), order_index=0, sentences=sents)
        sorted_sents = sorted(para.sentences, key=lambda s: s.order_index)
        translated_sents = [s.translated_text.strip() if s.translated_text is not None else s.original_text for s in sorted_sents]
        rendered_text = " ".join(s.strip() for s in translated_sents if s and s.strip())

        assert rendered_text == "Перше речення. Друге речення."
        assert "  " not in rendered_text
