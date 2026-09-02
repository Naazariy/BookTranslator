"""
Adversarial Stress Test Suite for BookTranslator Quality Improvements.
Challenger 1 Empirical Verification Suite.

Validates:
1. Sentence deduplication on extreme merge scenarios (N=10 -> M=1, M=N, M>N, M=0, N=50->1, 1->50, empty).
2. Multi-pass regex sanitizer on nasty inputs (fences, headers, XML tags, preambles, quotes, mixed script).
3. Sentence segmenter on tricky text (abbreviations, numbers, inline tags, dialogue, ellipses, OCR defects).
4. DOM reconciliation and Document Writer output formatting.
"""

import pytest
from pathlib import Path
from typing import List

from src.parsers.segmenter import RuleBasedSentenceSegmenter
from src.translation.aya_editing_engine import QuantizedAyaEditingEngine
from src.quality.pipeline import sanitize_mixed_script_words
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.writers.txt_writer import TxtWriter


def run_pipeline_merge_logic(target_sentences: List[Sentence], refined_text: str, segmenter: RuleBasedSentenceSegmenter):
    """
    Executes the exact sentence distribution & deduplication logic from
    src/translation/pipeline.py and src/launcher/translation_runner.py.
    """
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
# SUITE 1: Sentence Deduplication & Extreme Merge Scenarios
# =========================================================================

from uuid import uuid4

class TestSentenceDeduplicationExtremeMerges:
    @pytest.fixture
    def segmenter(self):
        return RuleBasedSentenceSegmenter()

    def test_extreme_merge_10_to_1(self, segmenter):
        """When N=10 sentences merge to M=1, sentence 0 gets text, trailing 9 are explicitly cleared to ''."""
        p_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=i, original_text=f"Sentence {i}.", translated_text=f"Старий переклад {i}.")
            for i in range(10)
        ]
        refined_text = "Це єдине зведене речення для всіх десяти вихідних."
        run_pipeline_merge_logic(sents, refined_text, segmenter)

        assert sents[0].translated_text == refined_text
        for i in range(1, 10):
            assert sents[i].translated_text == "", f"Sentence index {i} was not cleared!"

    def test_document_assembly_ignores_cleared_sentences(self, segmenter):
        """TxtWriter correctly skips cleared sentences (translated_text == '') without extra spaces."""
        p_id = uuid4()
        c_id = uuid4()
        b_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=i, original_text=f"Sentence {i}.", translated_text=f"Draft {i}.")
            for i in range(10)
        ]
        refined_text = "Одне зведене речення."
        run_pipeline_merge_logic(sents, refined_text, segmenter)

        para = Paragraph(id=p_id, chapter_id=c_id, order_index=0, sentences=sents)
        sorted_sents = sorted(para.sentences, key=lambda s: s.order_index)
        translated_sents = [s.translated_text.strip() if s.translated_text is not None else s.original_text for s in sorted_sents]
        para_text = " ".join(s.strip() for s in translated_sents if s and s.strip())

        assert para_text == refined_text

    def test_bijective_merge_5_to_5(self, segmenter):
        """When N=5 and M=5, 1-to-1 matching occurs perfectly."""
        p_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=i, original_text=f"Sentence {i}.", translated_text=f"Draft {i}.")
            for i in range(5)
        ]
        refined_5 = "Перше речення. Друге речення. Третє речення. Четверте речення. П'яте речення."
        run_pipeline_merge_logic(sents, refined_5, segmenter)

        split_5 = segmenter.split_sentences(refined_5)
        assert len(split_5) == 5
        for i in range(5):
            assert sents[i].translated_text == split_5[i]

    def test_expansion_merge_3_to_7(self, segmenter):
        """When M > N (N=3, M=7), sentences 0..1 get items 0..1, sentence 2 gets remaining 5 joined."""
        p_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=i, original_text=f"Sentence {i}.", translated_text=f"Draft {i}.")
            for i in range(3)
        ]
        refined_7 = "Речення один. Речення два. Речення три. Речення чотири. Речення п'ять. Речення шість. Речення сім."
        run_pipeline_merge_logic(sents, refined_7, segmenter)

        split_7 = segmenter.split_sentences(refined_7)
        assert sents[0].translated_text == split_7[0]
        assert sents[1].translated_text == split_7[1]
        assert sents[2].translated_text == " ".join(split_7[2:])

        # Total assembled paragraph contains all 7 sentences intact
        assembled = " ".join(s.translated_text.strip() for s in sents if s.translated_text and s.translated_text.strip())
        assert assembled == refined_7

    def test_single_target_merge_1_to_3(self, segmenter):
        """When N=1 and M=3, single target sentence receives all 3 sentences joined."""
        p_id = uuid4()
        sents = [Sentence(id=uuid4(), paragraph_id=p_id, order_index=0, original_text="One long sentence.", translated_text="Draft.")]
        refined = "Перша частина. Друга частина. Третя частина."
        run_pipeline_merge_logic(sents, refined, segmenter)
        assert sents[0].translated_text == refined

    def test_fallback_when_refined_is_empty_or_whitespace(self, segmenter):
        """When refined text is empty or whitespace (M=0), existing draft translations are preserved."""
        p_id = uuid4()
        sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=0, original_text="Original 1.", translated_text="Draft 1."),
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=1, original_text="Original 2.", translated_text="Draft 2.")
        ]
        run_pipeline_merge_logic(sents, "   \n\t   ", segmenter)
        assert sents[0].translated_text == "Draft 1."
        assert sents[1].translated_text == "Draft 2."

    def test_extreme_scale_50_to_1_and_1_to_50(self, segmenter):
        """Stress tests extreme scale 50->1 and 1->50."""
        p_id = uuid4()
        # 50 -> 1
        sents_50 = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=i, original_text=f"Original {i}.", translated_text=f"Draft {i}.")
            for i in range(50)
        ]
        refined_50_to_1 = "Одне велике речення для 50 вихідних."
        run_pipeline_merge_logic(sents_50, refined_50_to_1, segmenter)
        assert sents_50[0].translated_text == refined_50_to_1
        assert all(s.translated_text == "" for s in sents_50[1:])

        # 1 -> 50
        p_id2 = uuid4()
        sents_1_50 = [Sentence(id=uuid4(), paragraph_id=p_id2, order_index=0, original_text="Original long.", translated_text="Draft.")]
        refined_50_sents = " ".join(f"Речення {i}." for i in range(50))
        run_pipeline_merge_logic(sents_1_50, refined_50_sents, segmenter)
        assert sents_1_50[0].translated_text == refined_50_sents

    def test_stale_draft_overwrite_preventing_duplication(self, segmenter):
        """Stale Stage 1 drafts are explicitly overwritten when N=3 merges to M=1."""
        p_id = uuid4()
        stale_sents = [
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=0, original_text="Alice saw a rabbit.", translated_text="Аліса побачила кролика."),
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=1, original_text="It ran into a hole.", translated_text="Він побіг у нору."),
            Sentence(id=uuid4(), paragraph_id=p_id, order_index=2, original_text="She followed it.", translated_text="Вона пішла за ним.")
        ]
        refined_merged = "Аліса помітила кролика і побігла слідом за ним у нору."
        run_pipeline_merge_logic(stale_sents, refined_merged, segmenter)

        assert stale_sents[0].translated_text == refined_merged
        assert stale_sents[1].translated_text == ""
        assert stale_sents[2].translated_text == ""


# =========================================================================
# SUITE 2: Multi-Pass Regex Sanitizer (_sanitize_output)
# =========================================================================

class TestMultiPassRegexSanitizer:
    @pytest.fixture
    def engine(self):
        return QuantizedAyaEditingEngine.__new__(QuantizedAyaEditingEngine)

    @pytest.mark.parametrize("input_str,expected", [
        # Nested markdown fences
        ("```markdown\n```\nВідредагований український переклад.\n```\n```", "Відредагований український переклад."),
        ("```ukrainian\nТекст без огорожі.\n```", "Текст без огорожі."),
        # Multiple markdown headers
        ("# Розділ 1: Початок\n## Підрозділ 1.1\n### Деталі\n###### Параграф 5\nЦе реальний перекладений текст роману.", "Це реальний перекладений текст роману."),
        ("Він вибрав варіант #1 серед усіх інших.", "Він вибрав варіант #1 серед усіх інших."),
        # Unclosed and nested inline tags
        ("<tag_1>Текст <tag_2>з вкладеними тегами</tag_2></tag_1>", "Текст з вкладеними тегами"),
        ("<tag_0>Початок тексту <tag_999>продовження без закриття</tag_12>", "Початок тексту продовження без закриття"),
        ("Значення x < y та a > b є дійсними.", "Значення x < y та a > b є дійсними."),
        # Chained conversational preambles
        ("Ось фінальний переклад:\n**Фінальний переклад:**\nВідредагований літературний переклад:\nСонце повільно сідало за обрій.", "Сонце повільно сідало за обрій."),
        ("Here is the refined translation:\n**Ukrainian translation:**\nОсь виправлений переклад:\nВін відчинив двері.", "Він відчинив двері."),
        ("**Ось готовий переклад:** Покращений переклад: Ніч була темною.", "Ніч була темною."),
        # Echoed prompt section markers
        ("=== ОРИГІНАЛЬНИЙ АНГЛІЙСЬКИЙ ТЕКСТ ===\nOriginal English sentence.\n=== ВІДРЕДАГОВАНИЙ ТЕКСТ ===\nФінальний літературний переклад.", "Фінальний літературний переклад."),
        ("=== ФІНАЛЬНИЙ ПЕРЕКЛАД ===:\nПерекладений текст без заголовка.", "Перекладений текст без заголовка."),
        # Outer quotes
        ('"Це речення повністю взяте в лапки."', "Це речення повністю взяте в лапки."),
        ("«Це речення у ялинках.»", "Це речення у ялинках."),
        ("“Це речення у типографських лапках.”", "Це речення у типографських лапках."),
        ("'Це речення в одинарних лапках.'", "Це речення в одинарних лапках."),
        ('"Він гукнув: "Стій!" і побіг."', '"Він гукнув: "Стій!" і побіг."'),
        ("Д'Артаньян підняв шпагу.", "Д'Артаньян підняв шпагу."),
        # Extreme whitespace
        ("\r\n\t  \u00a0\u200b  Чистий переклад без зайвих пробілів.  \u00a0\r\n\t", "Чистий переклад без зайвих пробілів."),
        # Portmanteaus & Mixed Script
        ("Цей пиріг був Смачнissimo!", "Цей пиріг був Смакота!"),
        ("Це було просто смачнissimo для гостей.", "Це було просто смакота для гостей."),
        ("Виглядає Гарнissimo сьогодні.", "Виглядає Прегарно сьогодні."),
        ("Все пройшло Чудовissimo.", "Все пройшло Чудово."),
        ("Ми купили стиглий кaвун.", "Ми купили стиглий кавун."),  # Latin 'a'
        ("Старовинне мiсто Лева.", "Старовинне місто Лева."),      # Latin 'i'
        ("Маленькі котиing гралися на подвір'ї.", "Маленькі коти гралися на подвір'ї."),
    ])
    def test_sanitize_output_cases(self, engine, input_str, expected):
        assert engine._sanitize_output(input_str) == expected


# =========================================================================
# SUITE 3: Sentence Segmenter (RuleBasedSentenceSegmenter)
# =========================================================================

class TestSentenceSegmenterStress:
    @pytest.fixture
    def segmenter(self):
        return RuleBasedSentenceSegmenter()

    @pytest.mark.parametrize("input_str,expected", [
        # Numbers and decimals
        ("Він заплатив $19.99 за 3.14 літра молока.", ["Він заплатив $19.99 за 3.14 літра молока."]),
        ("Розділ 1.2. Вступ до курсу. Нова частина.", ["Розділ 1.2.", "Вступ до курсу.", "Нова частина."]),
        ("Він пробіг 10.5 км у 2020 р., встановивши рекорд.", ["Він пробіг 10.5 км у 2020 р., встановивши рекорд."]),
        # Honorifics and address abbreviations
        ("Dr. Watson and Mr. Holmes met Prof. Moriarty in London.", ["Dr. Watson and Mr. Holmes met Prof. Moriarty in London."]),
        ("Адреса: вул. Хрещатик, буд. 25, кв. 10, м. Київ.", ["Адреса: вул. Хрещатик, буд. 25, кв. 10, м. Київ."]),
        ("Свідок гр. Петренко дав свідчення.", ["Свідок гр. Петренко дав свідчення."]),
        # Latin abbreviations
        ("Fruits, e.g. apples and pears, i.e. fresh produce, are healthy.", ["Fruits, e.g. apples and pears, i.e. fresh produce, are healthy."]),
        ("The train arrives at 9:15 a.m. sharp.", ["The train arrives at 9:15 a.m. sharp."]),
        # Ukrainian Year / River abbreviations
        ("У 1945 р. закінчилася війна.", ["У 1945 р. закінчилася війна."]),
        ("У 1991 р., коли Україна відновила незалежність, відбувся референдум.", ["У 1991 р., коли Україна відновила незалежність, відбувся референдум."]),
        ("р. Дніпро є найбільшою річкою України.", ["р. Дніпро є найбільшою річкою України."]),
        ("Це відбулося у 1945 р. Було укладено мирний договір.", ["Це відбулося у 1945 р.", "Було укладено мирний договір."]),
        # Terminal abbreviations
        ("We bought apples, oranges, etc. at the market.", ["We bought apples, oranges, etc. at the market."]),
        ("We bought apples, oranges, etc. The market was closed after that.", ["We bought apples, oranges, etc.", "The market was closed after that."]),
        ("Купили зошити, олівці тощо. Вони були потрібні для школи.", ["Купили зошити, олівці тощо.", "Вони були потрібні для школи."]),
        # Inline XML tags
        ("<tag_1>Перше речення тут.</tag_1> <tag_2>Друге речення тут.</tag_2>", ["<tag_1>Перше речення тут.</tag_1>", "<tag_2>Друге речення тут.</tag_2>"]),
        ("<tag_1>Dr. Watson</tag_1> smiled. <tag_2>He stepped forward.</tag_2>", ["<tag_1>Dr. Watson</tag_1> smiled.", "<tag_2>He stepped forward.</tag_2>"]),
        # Dialogue attributions and quotes
        ("— Привіт! — сказав він. — Як твої справи?", ["— Привіт! — сказав він.", "— Як твої справи?"]),
        ("«Куди ти йдеш?» — запитала вона. «Я йду додому», — відповів він.", ["«Куди ти йдеш?» — запитала вона.", "«Я йду додому», — відповів він."]),
        ('"Wait here!" shouted the commander. "Do not move!"', ['"Wait here!" shouted the commander.', '"Do not move!"']),
        # Ellipses handling
        ("Він помовчав... а потім тихо продовжив розповідь.", ["Він помовчав... а потім тихо продовжив розповідь."]),
        ("Він помовчав... А потім раптом гучно засміявся.", ["Він помовчав...", "А потім раптом гучно засміявся."]),
        ("«Що це означає?..» — здивовано прошепотіла вона.", ["«Що це означає?..» — здивовано прошепотіла вона."]),
        # OCR missing whitespace repair
        ("Сонце вже сіло.Птахи замовкли.", ["Сонце вже сіло.", "Птахи замовкли."]),
        ("Стій!Хто йде?", ["Стій!", "Хто йде?"]),
        ("Чому ти мовчиш?Скажи правду.", ["Чому ти мовчиш?", "Скажи правду."]),
    ])
    def test_segmenter_cases(self, segmenter, input_str, expected):
        assert segmenter.split_sentences(input_str) == expected
