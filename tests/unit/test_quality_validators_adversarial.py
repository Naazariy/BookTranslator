"""
tests/unit/test_quality_validators_adversarial.py

Comprehensive Empirical Adversarial Stress Test Suite for BookTranslator V2
Modular QA Validators (src/quality/validators/) and Bounded Repair Engine.

Author: Challenger 1 (Milestone 4)
"""
from uuid import uuid4
import pytest

from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.domain.models.knowledge import EntityProfile, GlossaryItem, ScopeLevel
from src.context.builder import PromptContext
from src.quality.models import QASeverity, QAReport
from src.quality.pipeline import QualityPipeline
from src.quality.repair import BoundedRepairEngine
from src.quality.validators import (
    EmptyTranslationValidator,
    CompletenessValidator,
    EntityConsistencyValidator,
    GlossaryConsistencyValidator,
    GenderAgreementValidator,
    NumbersAndUnitsValidator,
    ControlTokenValidator,
    DialogueIntegrityValidator,
)


def _make_segment(source_text: str, target_text: str = "", **kwargs) -> TranslationSegment:
    return TranslationSegment(
        id=uuid4(),
        book_id=uuid4(),
        chapter_id=uuid4(),
        paragraph_id=uuid4(),
        source_text=source_text,
        draft_translation=target_text,
        refined_translation=target_text,
        final_translation=target_text,
        status=kwargs.pop("status", SegmentStatus.EDITED),
        **kwargs,
    )


# ============================================================================
# 1. EmptyTranslationValidator Adversarial Tests
# ============================================================================

class TestEmptyTranslationAdversarial:
    """Stress-tests EmptyTranslationValidator against invisible and adversarial inputs."""

    @pytest.fixture
    def validator(self):
        return EmptyTranslationValidator()

    def test_none_and_empty_string(self, validator):
        seg_none = _make_segment("Source paragraph.")
        seg_none.final_translation = None
        seg_none.refined_translation = None
        seg_none.draft_translation = None
        v_none = validator.validate(seg_none)
        assert len(v_none) == 1
        assert v_none[0].severity == QASeverity.CRITICAL
        assert v_none[0].rule_code == "EmptyTranslation"

        seg_empty = _make_segment("Source paragraph.", "")
        v_empty = validator.validate(seg_empty)
        assert len(v_empty) == 1
        assert v_empty[0].severity == QASeverity.CRITICAL

    def test_whitespace_variations(self, validator):
        for ws in ["   ", "\t\t", "\n\n", "\r\n\r\n", " \t \r\n \t "]:
            seg = _make_segment("Source paragraph.", ws)
            v = validator.validate(seg)
            assert len(v) == 1, f"Failed to catch whitespace: {repr(ws)}"
            assert v[0].severity == QASeverity.CRITICAL

    def test_pipeline_catches_zero_width_spaces_and_nulls_via_completeness(self):
        """
        Adversarial attack: zero-width space (\u200b), BOM (\ufeff), null byte (\x00).
        Python's str.strip() does not consider \u200b or \x00 as whitespace.
        We verify whether the full QualityPipeline / CompletenessValidator catches
        these deceptive single-character translations on non-trivial source text.
        """
        qp = QualityPipeline()
        source = "This is a meaningful source paragraph with enough length to test."

        for deceptive_char in ["\u200b", "\ufeff", "\u200c", "\u200d", "\x00"]:
            seg = _make_segment(source, deceptive_char)
            report = qp.validate_segment(seg)
            # Must fail validation (cannot pass QA with deceptive invisible character)
            assert not report.is_valid, f"Pipeline silently accepted deceptive character: {repr(deceptive_char)}"
            assert report.has_critical(), f"Expected CRITICAL violation for {repr(deceptive_char)}"


# ============================================================================
# 2. CompletenessValidator Adversarial Tests
# ============================================================================

class TestCompletenessAdversarial:
    """Stress-tests length ratios, sentence drops, repetition loops, and truncation."""

    @pytest.fixture
    def validator(self):
        return CompletenessValidator()

    def test_exact_ratio_low_boundary_35_percent(self, validator):
        # 100 source characters
        source = "A" * 100

        # Case 1: 34 characters (34.0% < 35.0%) -> MUST trigger
        target_34 = "Б" * 34
        seg_34 = _make_segment(source, target_34)
        v_34 = validator.validate(seg_34)
        assert any(v.rule_code == "LENGTH_RATIO_TOO_LOW" for v in v_34)

        # Case 2: 35 characters (35.0% == 35.0%) -> MUST NOT trigger ratio < 0.35
        target_35 = "Б" * 35 + "."
        seg_35 = _make_segment(source, target_35)
        v_35 = validator.validate(seg_35)
        assert not any(v.rule_code == "LENGTH_RATIO_TOO_LOW" for v in v_35)

    def test_exact_ratio_high_boundary_280_percent(self, validator):
        # 100 source characters
        source = "A" * 100

        # Case 1: 280 characters (280.0% == 280.0%) -> MUST NOT trigger ratio > 2.80
        target_280 = "Б" * 279 + "."
        seg_280 = _make_segment(source, target_280)
        v_280 = validator.validate(seg_280)
        assert not any(v.rule_code == "LENGTH_RATIO_TOO_HIGH" for v in v_280)

        # Case 2: 281 characters (281.0% > 280.0%) -> MUST trigger ratio > 2.80
        target_281 = "Б" * 280 + "."
        seg_281 = _make_segment(source, target_281)
        v_281 = validator.validate(seg_281)
        assert any(v.rule_code == "LENGTH_RATIO_TOO_HIGH" for v in v_281)

    def test_short_string_guards(self, validator):
        # src_len < 25: does not trigger ratio too low (avoids false positives on "Yes." -> "Так.")
        seg_short_ok = _make_segment("No, never.", "Ні.")
        v_short = validator.validate(seg_short_ok)
        assert not any(v.rule_code == "LENGTH_RATIO_TOO_LOW" for v in v_short)

        # src_len < 15: does not trigger ratio too high (avoids false positives on "Ah!" -> "Ого-го-го!")
        seg_high_ok = _make_segment("Ah!", "Ого-го-го-го-го!")
        v_high = validator.validate(seg_high_ok)
        assert not any(v.rule_code == "LENGTH_RATIO_TOO_HIGH" for v in v_high)

    def test_sentence_count_drop_boundary(self, validator):
        # 4 sentences in source: 0.7 * 4 = 2.8.
        # If target has 2 sentences: 2 < 2.8 -> triggers SentenceCountDrop
        source_4 = "First sentence. Second sentence. Third sentence. Fourth sentence."
        seg_2 = _make_segment(source_4, "Перше речення тут. Друге речення тут.")
        v_drop = validator.validate(seg_2)
        assert any(v.rule_code == "SentenceCountDrop" for v in v_drop)

        # If target has 3 sentences: 3 >= 2.8 -> does NOT trigger SentenceCountDrop
        seg_3 = _make_segment(source_4, "Перше речення. Друге речення. Третє речення.")
        v_nodrop = validator.validate(seg_3)
        assert not any(v.rule_code == "SentenceCountDrop" for v in v_nodrop)

    def test_consecutive_duplicate_hallucination_loop(self, validator):
        source = "He kept shouting into the dark abyss."
        # Repetition loop: exact same sentence 3 times
        loop_target = "Він кричав у темну безодню. Він кричав у темну безодню. Він кричав у темну безодню."
        seg_loop = _make_segment(source, loop_target)
        v_loop = validator.validate(seg_loop)
        dups = [v for v in v_loop if v.rule_code == "ConsecutiveDuplicateSentences"]
        assert len(dups) >= 1
        assert dups[0].severity == QASeverity.WARNING

    def test_trailing_connectors_with_and_without_punctuation(self, validator):
        source = "He walked slowly into the forest and."

        # Cut off trailing conjunctions without punctuation -> MUST trigger
        for conj in ["і", "та", "або", "чи", "а", "але", "до", "з", "що", "як"]:
            target_cut = f"Він повільно пішов до лісу {conj}"
            seg_cut = _make_segment(source, target_cut)
            v_cut = validator.validate(seg_cut)
            assert any(v.rule_code == "TRUNCATION_TRAILING_CONNECTOR" for v in v_cut), f"Failed for '{conj}'"

        # Conjunction properly punctuated (e.g. question) -> MUST NOT trigger
        target_punctuated = "Він спитав, що?"
        seg_punc = _make_segment(source, target_punctuated)
        v_punc = validator.validate(seg_punc)
        assert not any(v.rule_code == "TRUNCATION_TRAILING_CONNECTOR" for v in v_punc)

    def test_trailing_escape_character(self, validator):
        source = "The door suddenly slammed shut."
        target_esc = "Двері раптово зачинилися.\\"
        seg_esc = _make_segment(source, target_esc)
        v_esc = validator.validate(seg_esc)
        assert any(v.rule_code == "TRUNCATION_TRAILING_ESCAPE" for v in v_esc)


# ============================================================================
# 3. EntityConsistencyValidator Adversarial Tests
# ============================================================================

class TestEntityConsistencyAdversarial:
    """Stress-tests Ukrainian inflections, homographs, and word boundaries."""

    @pytest.fixture
    def validator(self):
        return EntityConsistencyValidator()

    def test_cherry_forbidden_inflections_comprehensive(self, validator):
        cherry = EntityProfile(
            source_name="Cherry",
            canonical_target="Черрі",
            forbidden_target_forms=["Вишня", "Вішня", "Черешня"],
            locked=True,
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Cherry was waiting outside.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[cherry],
        )

        # Test all standard case inflections of forbidden "Вишня"
        test_cases = [
            ("Вишня чекала надворі.", "Вишня"),
            ("Він підійшов до Вишні.", "Вишні"),
            ("Він подарував квіти Вишні.", "Вишні"),
            ("Він побачив Вишню біля дверей.", "Вишню"),
            ("Він пішов разом із Вишнею.", "Вишнею"),
            ("Там не було жодної з Вишень.", "Вишень"),  # Fleeting vowel test
            # Phonetic variant Вішня
            ("Вішнею захоплювалися всі.", "Вішнею"),
            # Черешня
            ("Він покликав Черешню.", "Черешню"),
        ]

        for target_sentence, expected_token in test_cases:
            seg = _make_segment("Cherry was waiting outside.", target_sentence)
            violations = validator.validate(seg, context=ctx)
            forbidden_viols = [v for v in violations if v.rule_code == "FORBIDDEN_ENTITY_VARIANT"]
            assert len(forbidden_viols) >= 1, f"Failed to catch forbidden inflection in: '{target_sentence}'"
            assert forbidden_viols[0].severity == QASeverity.CRITICAL
            assert forbidden_viols[0].suggested_fix == "Черрі"

    def test_faith_homograph_vs_imperative_verb(self, validator):
        """
        Adversarial test: Faith -> Віра (forbidden variant for character Faith).
        'Вір мені!' is an imperative verb ('Believe me!'), NOT an inflection of 'Віра'.
        The validator must reject 'Віра' / 'Вірою', but MUST NOT flag 'Вір мені!'.
        """
        faith = EntityProfile(
            source_name="Faith",
            canonical_target="Фейт",
            forbidden_target_forms=["Віра"],
            locked=True,
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Faith looked at him and said: Believe me!",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[faith],
        )

        # 1. Forbidden name used -> MUST trigger
        seg_bad1 = _make_segment("Faith looked at him.", "Віра подивилася на нього.")
        v_bad1 = validator.validate(seg_bad1, context=ctx)
        assert any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in v_bad1)

        # 2. Forbidden instrumental case 'Вірою' -> MUST trigger
        seg_bad2 = _make_segment("Faith was praised.", "Всі захоплювалися Вірою.")
        v_bad2 = validator.validate(seg_bad2, context=ctx)
        assert any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in v_bad2)

        # 3. Imperative verb 'Вір мені!' -> MUST NOT trigger false positive!
        seg_imperative = _make_segment(
            "Faith looked at him and said: Believe me!",
            "Фейт подивилася на нього і сказала: Вір мені!"
        )
        v_imperative = validator.validate(seg_imperative, context=ctx)
        assert not any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in v_imperative), (
            "False positive: imperative verb 'Вір' was incorrectly flagged as forbidden form of 'Віра'!"
        )

    def test_substring_word_boundary_defense(self, validator):
        """
        Target contains a word that has 'вишн' or 'віра' as a substring, but is an unrelated word.
        e.g. 'підвищення' (promotion), 'перевірка' (verification).
        Word boundary regex must prevent false positives.
        """
        cherry = EntityProfile(
            source_name="Cherry",
            canonical_target="Черрі",
            forbidden_target_forms=["Вишня"],
            locked=True,
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Cherry asked about the promotion.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[cherry],
        )

        seg_clean = _make_segment(
            "Cherry asked about the promotion.",
            "Черрі запитала про підвищення на роботі."
        )
        v = validator.validate(seg_clean, context=ctx)
        assert not any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in v)

    def test_entity_not_in_source_is_ignored(self, validator):
        """
        If 'Cherry' is in global entities but NOT in the source paragraph,
        mentioning cherries ('вишня') in a culinary context must not trigger violations.
        """
        cherry = EntityProfile(
            source_name="Cherry",
            canonical_target="Черрі",
            forbidden_target_forms=["Вишня"],
            locked=True,
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="He planted a fruit tree in the garden.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[cherry],
        )

        seg = _make_segment(
            "He planted a fruit tree in the garden.",
            "Він посадив вишню у саду."
        )
        v = validator.validate(seg, context=ctx)
        assert len(v) == 0


# ============================================================================
# 4. GenderAgreementValidator Adversarial Tests
# ============================================================================

class TestGenderAgreementAdversarial:
    """Stress-tests Ukrainian gender agreement, syntax inversion, particles, and homonyms."""

    @pytest.fixture
    def validator(self):
        return GenderAgreementValidator()

    def test_direct_and_inverted_mismatches_all_genders(self, validator):
        maria = GlossaryItem(source_term="Maria", target_term="Марія", grammatical_gender="жіночий")
        john = GlossaryItem(source_term="John", target_term="Джон", grammatical_gender="чоловічий")
        monster = GlossaryItem(source_term="Monster", target_term="Чудовисько", grammatical_gender="середній")

        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Maria said. John went. Monster growled.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[maria, john, monster],
        )

        # 1. Feminine character with masculine verb (Direct & Inverted)
        seg_fem_dir = _make_segment("Maria said.", "Марія сказав усе.")
        assert any(v.rule_code == "GenderAgreementMismatch" for v in validator.validate(seg_fem_dir, context=ctx))

        seg_fem_inv = _make_segment("Maria said.", "— Так, — сказав Марія.")
        assert any(v.rule_code == "GenderAgreementMismatch" for v in validator.validate(seg_fem_inv, context=ctx))

        # 2. Masculine character with feminine verb (Direct & Inverted)
        seg_masc_dir = _make_segment("John said.", "Джон сказала правду.")
        assert any(v.rule_code == "GenderAgreementMismatch" for v in validator.validate(seg_masc_dir, context=ctx))

        seg_masc_inv = _make_segment("John said.", "— Так, — сказала Джон.")
        assert any(v.rule_code == "GenderAgreementMismatch" for v in validator.validate(seg_masc_inv, context=ctx))

        # 3. Neuter character with masculine verb
        seg_neu_dir = _make_segment("Monster growled.", "Чудовисько загарчав у темряві.")
        assert any(v.rule_code == "GenderAgreementMismatch" for v in validator.validate(seg_neu_dir, context=ctx))

    def test_intervening_adverbs_and_particles(self, validator):
        maria = GlossaryItem(source_term="Maria", target_term="Марія", grammatical_gender="жіночий")
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Maria did not want.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[maria],
        )

        # "Марія зовсім не хотів" (intervening adverbs 'зовсім', 'не') -> MUST catch mismatch 'хотів'
        seg = _make_segment("Maria did not want.", "Марія зовсім не хотів іти туди.")
        v = validator.validate(seg, context=ctx)
        assert any(v.rule_code == "GenderAgreementMismatch" for v in v)

    def test_prepositional_object_and_oblique_false_positive_resistance(self, validator):
        maria = GlossaryItem(source_term="Maria", target_term="Марія", grammatical_gender="жіночий")
        john = GlossaryItem(source_term="John", target_term="Джон", grammatical_gender="чоловічий")
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="He came to Maria and saw John.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[maria, john],
        )

        # "до Марії підійшов" -> 'Марії' is in prepositional phrase (preposition 'до').
        # Masculine verb 'підійшов' belongs to omitted 'Він', NOT to 'Марія'.
        seg_prep = _make_segment("He came to Maria.", "До Марії підійшов незнайомець.")
        assert not any(v.rule_code == "GenderAgreementMismatch" for v in validator.validate(seg_prep, context=ctx))

        # "покликав Джона" -> 'Джона' is oblique case object.
        # Verb 'покликала' refers to feminine subject, 'Джона' is object.
        seg_oblique = _make_segment("She called John.", "Вона покликала Джона додому.")
        assert not any(v.rule_code == "GenderAgreementMismatch" for v in validator.validate(seg_oblique, context=ctx))

    def test_compound_subjects_avoid_false_positives(self, validator):
        john = GlossaryItem(source_term="John", target_term="Джон", grammatical_gender="чоловічий")
        maria = GlossaryItem(source_term="Maria", target_term="Марія", grammatical_gender="жіночий")
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="John and Maria went.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[john, maria],
        )

        # "Джон і Марія пішли" -> Compound subject followed by plural verb
        seg = _make_segment("John and Maria went.", "Джон і Марія пішли разом до лісу.")
        assert len(validator.validate(seg, context=ctx)) == 0


# ============================================================================
# 5. NumbersAndUnitsValidator Adversarial Tests
# ============================================================================

class TestNumbersAndUnitsAdversarial:
    """Stress-tests Arabic numbers, decimal points/commas, Roman numerals, and currencies."""

    @pytest.fixture
    def validator(self):
        return NumbersAndUnitsValidator()

    def test_decimal_comma_vs_dot(self, validator):
        # Source has 3.14, Ukrainian translation has 3,14
        seg = _make_segment("The value of pi is 3.14 approximately.", "Значення пі становить приблизно 3,14.")
        v = validator.validate(seg)
        assert len([x for x in v if x.rule_code == "NUMBER_MISMATCH"]) == 0

    def test_spelled_out_small_numbers(self, validator):
        # 1 -> "одна", 2 -> "дві", 3 -> "три"
        seg1 = _make_segment("There was 1 apple.", "Була одна яблуко.")
        assert len(validator.validate(seg1)) == 0

        seg2 = _make_segment("There were 2 cats.", "Там було дві кішки.")
        assert len(validator.validate(seg2)) == 0

        seg3 = _make_segment("There were 5 soldiers.", "Там було п'ять солдатів.")
        assert len(validator.validate(seg3)) == 0

    def test_roman_numerals_vs_english_pronoun_i(self, validator):
        # English pronoun "I" in source MUST NOT be flagged as missing Roman numeral "I"
        seg_pronoun = _make_segment("I went to the tavern yesterday.", "Вчора я пішов до корчми.")
        v_pronoun = validator.validate(seg_pronoun)
        assert not any(v.rule_code == "ROMAN_NUMERAL_MISMATCH" for v in v_pronoun)

        # True Roman numeral "Chapter I" or "Part IV" -> MUST be preserved
        seg_chapter = _make_segment("Chapter IV: The Return.", "Розділ четвертий: Повернення.")
        v_chapter = validator.validate(seg_chapter)
        assert any(v.rule_code == "ROMAN_NUMERAL_MISMATCH" for v in v_chapter)

    def test_currencies_and_percentages(self, validator):
        # Source has $ and %
        seg_curr = _make_segment("The price was $50 with a 10% discount.", "Ціна була 50 доларів зі знижкою 10 відсотків.")
        assert len(validator.validate(seg_curr)) == 0

        # Currency missing
        seg_miss = _make_segment("The price was $50.", "Ціна становила 50 монет.")
        v_miss = validator.validate(seg_miss)
        assert any(v.rule_code == "CURRENCY_PERCENTAGE_MISMATCH" for v in v_miss)


# ============================================================================
# 6. ControlTokenValidator Adversarial Tests
# ============================================================================

class TestControlTokenAdversarial:
    """Stress-tests model stop tokens, code fences, prompt templates, and AI chatter."""

    @pytest.fixture
    def validator(self):
        return ControlTokenValidator()

    def test_all_control_token_variants(self, validator):
        tokens = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|startoftext|>",
            "<|fim_prefix|>", "<|system|>", "<|user|>", "<|assistant|>"
        ]
        for tok in tokens:
            seg = _make_segment("Hello.", f"Привіт.{tok}")
            v = validator.validate(seg)
            assert any(v.rule_code == "ResidualControlToken" for v in v), f"Missed token {tok}"
            assert any(v.severity == QASeverity.CRITICAL for v in v)

    def test_prompt_template_headers_leakage(self, validator):
        headers = [
            "=== ПОПЕРЕДНІЙ КОНТЕКСТ ===",
            "=== ГЛОСАРІЙ ТА СУТНОСТІ ===",
            "=== ПАРАГРАФ ДЛЯ РЕДАГУВАННЯ ===",
            "=== ВІДРЕДАГОВАНИЙ JSON ===",
            "ОРИГІНАЛ (EN):",
            "ЧОРНОВИЙ ПЕРЕКЛАД NLLB",
        ]
        for hdr in headers:
            seg = _make_segment("Text.", f"{hdr}\nПерекладений текст.")
            v = validator.validate(seg)
            assert any(v.rule_code == "LEAKED_PROMPT_TEMPLATE" for v in v), f"Missed header {hdr}"

    def test_conversational_chatter_and_refusals(self, validator):
        phrases = [
            "Sure, here is the translation: Привіт.",
            "As an AI language model, I cannot fulfill this request.",
            "Ось переклад: Текст оповідання.",
            "Звісно, ось переклад параграфа:",
            "Вибачте, як штучний інтелект я не можу це зробити.",
        ]
        for phrase in phrases:
            seg = _make_segment("Story paragraph.", phrase)
            v = validator.validate(seg)
            assert any(v.rule_code == "CONVERSATIONAL_CHATTER" for v in v), f"Missed chatter: {phrase}"


# ============================================================================
# 7. DialogueIntegrityValidator Adversarial Tests
# ============================================================================

class TestDialogueIntegrityAdversarial:
    """Stress-tests typography, dialogue dashes, and speaker turn preservation."""

    @pytest.fixture
    def validator(self):
        return DialogueIntegrityValidator()

    def test_dialogue_hyphen_vs_em_dash(self, validator):
        # ASCII hyphen at start of speech -> WARNING
        seg_hyphen = _make_segment("- Hello, said John.", "- Привіт, — сказав Джон.")
        v_hyphen = validator.validate(seg_hyphen)
        assert any(v.rule_code == "DIALOGUE_HYPHEN_INSTEAD_OF_DASH" for v in v_hyphen)

        # Typographic em-dash -> NO WARNING
        seg_dash = _make_segment("— Hello, said John.", "— Привіт, — сказав Джон.")
        v_dash = [v for v in validator.validate(seg_dash) if v.rule_code == "DIALOGUE_HYPHEN_INSTEAD_OF_DASH"]
        assert len(v_dash) == 0

    def test_unpaired_chevrons_and_straight_quotes(self, validator):
        # Unpaired chevron
        seg_unpair = _make_segment("Title.", "Він прочитав «Кобзар на столі.")
        v_unpair = validator.validate(seg_unpair)
        assert any(v.rule_code == "UNPAIRED_QUOTATION_MARKS" for v in v_unpair)

        # Balanced ASCII quotes -> triggers INFO (ASCII_QUOTES_IN_DIALOGUE)
        seg_ascii = _make_segment("Title.", 'Він прочитав "Кобзар" на столі.')
        v_ascii = validator.validate(seg_ascii)
        assert any(v.rule_code == "ASCII_QUOTES_IN_DIALOGUE" for v in v_ascii)
        assert any(v.severity == QASeverity.INFO for v in v_ascii)

    def test_speaker_turns_collapsed(self, validator):
        # 3 dialogue lines in source collapsed into 1 in target
        source_turns = "— Who are you?\n— I am the traveler.\n— Welcome."
        target_collapsed = "— Хто ти? Я мандрівник. Ласкаво просимо."
        seg = _make_segment(source_turns, target_collapsed)
        v = validator.validate(seg)
        assert any(v.rule_code == "SPEAKER_TURN_COLLAPSE" for v in v)


# ============================================================================
# 8. BoundedRepairEngine Adversarial Convergence Tests
# ============================================================================

class TestBoundedRepairAdversarial:
    """Stress-tests repair loops, diagnostic prompts, and lifecycle termination."""

    def test_repair_remedies_multiple_violations_simultaneously(self):
        qp = QualityPipeline()
        cherry = EntityProfile(
            source_name="Cherry",
            canonical_target="Черрі",
            forbidden_target_forms=["Вишня"],
            locked=True,
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Cherry said warmly: 42 apples.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[cherry],
        )

        # Translation has BOTH forbidden entity ("Вишня") and missing number 42
        seg = _make_segment(
            "Cherry said warmly: 42 apples.",
            "Вишня тепло сказала: багато яблук."
        )

        initial_report = qp.validate_segment(seg, context=ctx)
        assert not initial_report.is_valid
        assert len(initial_report.violations) >= 2

        # Mock engine repairs both on attempt 1
        def smart_engine(s, prompt):
            return "Черрі тепло сказала: 42 яблука."

        repair_engine = BoundedRepairEngine(editing_engine=smart_engine, quality_pipeline=qp, max_retries=2)
        repaired_seg, final_report = repair_engine.repair_segment(seg, initial_report, context=ctx)

        assert repaired_seg.status == SegmentStatus.ACCEPTED
        assert repaired_seg.final_translation == "Черрі тепло сказала: 42 яблука."
        assert final_report.is_valid is True

    def test_repair_prompt_contains_all_diagnostic_directives(self):
        engine = BoundedRepairEngine()
        seg = _make_segment("Cherry said.", "Вишня сказав.")
        report = QAReport(segment_id=seg.id)
        report.add_violation(
            EntityConsistencyValidator().make_violation(
                rule_code="FORBIDDEN_ENTITY_VARIANT",
                message="Forbidden variant 'Вишня' detected.",
                severity=QASeverity.CRITICAL,
                forbidden_form="Вишня",
                suggested_fix="Черрі",
            )
        )

        prompt = engine.synthesize_repair_prompt(seg, report, seg.draft_translation)
        assert "ЗАБОРОНЕНО вживати: \"Вишня\"" in prompt
        assert "Обов'язковий відповідник: \"Черрі\"" in prompt
        assert "JSON" in prompt

    def test_repair_engine_graceful_recovery_on_llm_exceptions(self):
        """Verify that BoundedRepairEngine handles LLM crashes without unhandled exceptions."""
        qp = QualityPipeline()
        seg = _make_segment("Hello", "Wrong")
        rep = QAReport(segment_id=seg.id, is_valid=False)

        def crashing_engine(s, p):
            raise RuntimeError("Inference connection failed")

        engine = BoundedRepairEngine(editing_engine=crashing_engine, quality_pipeline=qp, max_retries=2)
        repaired_seg, final_rep = engine.repair_segment(seg, rep)

        assert repaired_seg.status == SegmentStatus.REVIEW_REQUIRED
        assert repaired_seg.repair_attempts == 2
        assert final_rep.status == "REVIEW_REQUIRED"
        assert not final_rep.is_valid


# ============================================================================
# 9. Deep Edge Cases & Documented Architectural Boundaries
# ============================================================================

class TestDeepEdgeCasesAndBoundaries:
    """Documented boundaries and empirical edge cases discovered during challenge."""

    def test_masculine_names_ending_in_a_and_feminine_in_consonant(self):
        """Validates that non-canonical morphological endings are supported in GenderAgreementValidator."""
        val = GenderAgreementValidator()
        mykola = GlossaryItem(source_term="Mykola", target_term="Микола", grammatical_gender="чоловічий")
        eliz = GlossaryItem(source_term="Elizabeth", target_term="Елізабет", grammatical_gender="жіночий")

        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Mykola and Elizabeth spoke.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[mykola, eliz],
        )

        # Correct agreements
        seg_ok = _make_segment("Mykola and Elizabeth spoke.", "Микола пішов додому. Елізабет сказала все.")
        assert len(val.validate(seg_ok, context=ctx)) == 0

        # Inverted mismatches
        seg_bad = _make_segment("Mykola and Elizabeth spoke.", "Микола пішла додому. Елізабет сказав все.")
        v = val.validate(seg_bad, context=ctx)
        assert len([x for x in v if x.rule_code == "GenderAgreementMismatch"]) == 2

    def test_short_source_invisible_character_boundary(self):
        """
        Documents the boundary where single invisible characters bypass
        str.strip() and short length thresholds (< 25 chars).
        """
        qp = QualityPipeline()
        # Short source (< 25 chars): EmptyTranslationValidator uses .strip()
        # \u200b is category Cf, so .strip() does not remove it.
        seg = _make_segment("Good night.", "\u200b")
        rep = qp.validate_segment(seg)
        # EmptyTranslationValidator does not strip \u200b, and CompletenessValidator requires len >= 25.
        # This test documents this boundary behavior.
        assert rep.is_valid is True

    def test_roman_numeral_chapter_one_limitation(self):
        """
        Documents that single Roman numeral 'I' in 'Chapter I' is not matched
        because len('I') < 2 and 'I' not in ('V', 'X', 'L', 'C', 'D', 'M').
        """
        val = NumbersAndUnitsValidator()
        romans_i = val._extract_roman_numerals("Chapter I: The Beginning")
        # In the current implementation, 'I' is not included in the extracted set.
        assert "I" not in romans_i

        # Roman numerals with len >= 2 work properly:
        romans_ii = val._extract_roman_numerals("Chapter II: The Beginning")
        assert "II" in romans_ii

