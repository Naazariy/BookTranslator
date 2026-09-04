"""
tests/unit/test_challenger_stress_m4_validators.py

Challenger 1 Adversarial Stress Test Suite for Milestone 4 (Phase 4).
Empirically stress-tests all 8 modular validators in src/quality/validators/
under boundary conditions, adversarial attacks, Ukrainian inflections, Unicode edge cases,
length ratio limits, dialogue typography, and prompt token leakage.
"""

from uuid import uuid4
import pytest

from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.domain.models.knowledge import EntityProfile, GlossaryItem, ScopeLevel
from src.context.builder import PromptContext
from src.quality.models import QASeverity
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
        **kwargs,
    )


# ============================================================================
# 1. EmptyTranslationValidator Adversarial Stress Tests
# ============================================================================

class TestEmptyTranslationValidatorStress:
    """Stress-test EmptyTranslationValidator against boundary representations of emptiness."""

    def test_whitespace_variations(self):
        val = EmptyTranslationValidator()
        # Tabs, carriage returns, vertical tabs, form feeds
        whitespace_cases = [
            "",
            "   ",
            "\t\t\t",
            "\n\r\n",
            " \t \r \n \v \f ",
        ]
        for ws in whitespace_cases:
            seg = _make_segment("Valid source.", ws)
            violations = val.validate(seg)
            assert len(violations) == 1, f"Failed for whitespace case: {repr(ws)}"
            assert violations[0].severity == QASeverity.CRITICAL
            assert violations[0].rule_code == "EmptyTranslation"

    def test_null_translation_attributes(self):
        val = EmptyTranslationValidator()
        seg = _make_segment("Valid source.")
        seg.final_translation = None
        seg.refined_translation = None
        seg.draft_translation = None

        violations = val.validate(seg)
        assert len(violations) == 1
        assert violations[0].severity == QASeverity.CRITICAL
        # extract_target_text defaults to empty string "", triggering empty text rule
        assert violations[0].rule_code == "EmptyTranslation"

    def test_explicit_none_target_text(self):
        val = EmptyTranslationValidator()
        seg = _make_segment("Valid source.")
        # When extract_target_text cannot extract anything and translated_text is None
        violations = val.validate(seg, translated_text="")
        assert len(violations) == 1
        assert violations[0].rule_code == "EmptyTranslation"

    def test_unicode_null_and_zero_width_limitation(self):
        """
        Empirical finding: str.strip() does NOT strip non-whitespace Unicode control/format chars.
        \\x00 (Cc) and \\u200b (Cf) are not removed by str.strip(), thus EmptyTranslationValidator
        alone considers them non-empty unless caught downstream by length ratio or token checks.
        """
        val = EmptyTranslationValidator()
        # Null byte
        seg_null = _make_segment("Source", "\x00")
        v_null = val.validate(seg_null)
        # EmptyTranslationValidator does not strip \x00, so it returns 0 violations here
        assert len(v_null) == 0

        # Zero-width space
        seg_zw = _make_segment("Source", "\u200b")
        v_zw = val.validate(seg_zw)
        assert len(v_zw) == 0

    def test_non_empty_passes(self):
        val = EmptyTranslationValidator()
        seg = _make_segment("Source", "Переклад")
        assert len(val.validate(seg)) == 0


# ============================================================================
# 2. CompletenessValidator Adversarial Stress Tests
# ============================================================================

class TestCompletenessValidatorStress:
    """Stress-test CompletenessValidator against boundary ratios, truncations, and repetitions."""

    def test_length_ratio_boundary_35_percent(self):
        val = CompletenessValidator()
        # Source len = 100
        source = "A" * 100

        # Case 1: 34% (34 chars) -> triggers LENGTH_RATIO_TOO_LOW
        seg_below = _make_segment(source, "Б" * 34)
        v_below = val.validate(seg_below)
        assert any(v.rule_code == "LENGTH_RATIO_TOO_LOW" for v in v_below)

        # Case 2: 36% (36 chars) -> does NOT trigger LENGTH_RATIO_TOO_LOW
        seg_above = _make_segment(source, "Б" * 36)
        v_above = val.validate(seg_above)
        assert not any(v.rule_code == "LENGTH_RATIO_TOO_LOW" for v in v_above)

    def test_length_ratio_boundary_280_percent(self):
        val = CompletenessValidator()
        # Source len = 20
        source = "A" * 20

        # Case 1: 290% (58 chars) -> triggers LENGTH_RATIO_TOO_HIGH
        seg_above = _make_segment(source, "Б" * 58)
        v_above = val.validate(seg_above)
        assert any(v.rule_code == "LENGTH_RATIO_TOO_HIGH" for v in v_above)

        # Case 2: 270% (54 chars) -> does NOT trigger LENGTH_RATIO_TOO_HIGH
        seg_below = _make_segment(source, "Б" * 54)
        v_below = val.validate(seg_below)
        assert not any(v.rule_code == "LENGTH_RATIO_TOO_HIGH" for v in v_below)

    def test_short_source_gating(self):
        val = CompletenessValidator()
        # Source len < 25 chars should NOT trigger LENGTH_RATIO_TOO_LOW
        # e.g., "Stop please." (12 chars) -> "Стій." (5 chars)
        seg_short = _make_segment("Stop please.", "Стій.")
        v_short = val.validate(seg_short)
        assert not any(v.rule_code == "LENGTH_RATIO_TOO_LOW" for v in v_short)

    def test_truncation_adversarial_patterns(self):
        val = CompletenessValidator()

        # Broken trailing backslash
        seg_esc = _make_segment("Complete sentence.", "Неповне речення\\")
        assert any(v.rule_code == "TRUNCATION_TRAILING_ESCAPE" for v in val.validate(seg_esc))

        # Trailing conjunctions/prepositions in TRAILING_CONNECTOR_PATTERN
        cutoffs = [
            "Він хотів сказати, але",
            "Ми пішли туди і",
            "Це було важко, та",
            "Я не знаю, чи",
            "Він побіг до",
            "Він зайшов у",
            "Хлопець, який",
        ]
        for cut in cutoffs:
            seg = _make_segment("Complete sentence in English.", cut)
            v = val.validate(seg)
            assert any(v.rule_code == "TRUNCATION_TRAILING_CONNECTOR" for v in v), f"Failed to detect cutoff in '{cut}'"

        # Valid sentences ending with terminal punctuation despite containing connectors
        valid = [
            "Він не знав, як.",
            "Вона спитала: чи?",
            "Це було саме те, до чого ми прагнули.",
            "Він пішов додому і ліг спати.",
        ]
        for val_text in valid:
            seg = _make_segment("Complete sentence in English.", val_text)
            v = val.validate(seg)
            assert not any(v.rule_code == "TRUNCATION_TRAILING_CONNECTOR" for v in v), f"False positive on '{val_text}'"

    def test_sentence_count_drop_ratio(self):
        val = CompletenessValidator()

        # 10 sentences in source -> 6 in target (ratio 0.6 < 0.7)
        src_10 = " ".join(f"Sentence {i}." for i in range(10))
        tgt_6 = " ".join(f"Речення {i}." for i in range(6))
        seg_drop = _make_segment(src_10, tgt_6)
        v_drop = val.validate(seg_drop)
        assert any(v.rule_code == "SentenceCountDrop" for v in v_drop)

        # 10 sentences in source -> 8 in target (ratio 0.8 >= 0.7)
        tgt_8 = " ".join(f"Речення {i}." for i in range(8))
        seg_ok = _make_segment(src_10, tgt_8)
        v_ok = val.validate(seg_ok)
        assert not any(v.rule_code == "SentenceCountDrop" for v in v_ok)

    def test_sentence_combination_strictness(self):
        """
        Documented strictness behavior: combining 2 sentences into 1 sentence
        yields ratio 0.5 (< 0.7), triggering SentenceCountDrop (CRITICAL).
        """
        val = CompletenessValidator()
        seg_combine = _make_segment("He stood up. He walked away.", "Він підвівся і пішов геть.")
        v_comb = val.validate(seg_combine)
        assert any(v.rule_code == "SentenceCountDrop" for v in v_comb)

    def test_consecutive_near_duplicates(self):
        val = CompletenessValidator()

        # Repetition hallucination
        s1 = "Лицар підійшов до воріт замку і постукав у двері."
        s2 = "Лицар підійшов до воріт замку та постукав у двері."
        seg_rep = _make_segment("The knight approached the gates.", f"{s1} {s2}")
        v_rep = val.validate(seg_rep)
        assert any(v.rule_code == "ConsecutiveDuplicateSentences" for v in v_rep)

        # Distinct consecutive sentences
        s3 = "Він озирнувся навколо у пошуках схованки."
        seg_diff = _make_segment("The knight approached. He looked around.", f"{s1} {s3}")
        v_diff = val.validate(seg_diff)
        assert not any(v.rule_code == "ConsecutiveDuplicateSentences" for v in v_diff)


# ============================================================================
# 3. EntityConsistencyValidator Adversarial Stress Tests
# ============================================================================

class TestEntityConsistencyValidatorStress:
    """Stress-test EntityConsistencyValidator on inflections, forbidden variants, and false positive traps."""

    def test_cherry_forbidden_inflections(self):
        val = EntityConsistencyValidator()
        cherry = EntityProfile(
            source_name="Cherry",
            canonical_target="Черрі",
            forbidden_target_forms=["Вишня", "Вішня", "Черешня", "Черри"],
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

        # Test full paradigm of inflections for "Вишня"
        forbidden_in_context = [
            ("На порозі стояла Вишня.", "Вишня"),
            ("Він підійшов до Вишні.", "Вишні"),
            ("Він зустрів Вишню біля саду.", "Вишню"),
            ("Він розмовляв із Вишнею цілий вечір.", "Вишнею"),
            ("Там не було жодної з Вишень.", "Вишень"),
            # Dialectal "Вішня"
            ("Це була Вішня.", "Вішня"),
            ("Він захоплювався Вішнею.", "Вішнею"),
            # Fruit translation "Черешня"
            ("Він покликав Черешню.", "Черешню"),
            ("Він пішов із Черешнею.", "Черешнею"),
            # Russianized "Черри"
            ("Це був Черри.", "Черри"),
        ]

        for sentence, expected_match in forbidden_in_context:
            seg = _make_segment("Cherry was waiting outside.", sentence)
            violations = val.validate(seg, context=ctx)
            assert any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in violations), (
                f"Failed to flag forbidden inflection in: '{sentence}'"
            )
            crit = next(v for v in violations if v.rule_code == "FORBIDDEN_ENTITY_VARIANT")
            assert crit.severity == QASeverity.CRITICAL
            assert crit.suggested_fix == "Черрі"

    def test_faith_name_vs_imperative_verb_distinction(self):
        """
        Adversarial edge case from dispatch:
        Character 'Faith' -> canonical 'Фейт', forbidden 'Віра'.
        Forbidden inflection 'Вірою' must be caught.
        Imperative verb 'Вір мені!' must NOT trigger false positive!
        """
        val = EntityConsistencyValidator()
        faith = EntityProfile(
            source_name="Faith",
            canonical_target="Фейт",
            forbidden_target_forms=["Віра"],
            locked=True,
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Faith asked him for help.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[faith],
        )

        # 1. Real forbidden mentions (Nominative, Instrumental)
        seg_bad_nom = _make_segment("Faith asked him for help.", "Віра попросила його про допомогу.")
        v_bad_nom = val.validate(seg_bad_nom, context=ctx)
        assert any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in v_bad_nom)

        seg_bad_inst = _make_segment("Faith asked him for help.", "Він захоплювався Вірою та її сміливістю.")
        v_bad_inst = val.validate(seg_bad_inst, context=ctx)
        assert any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in v_bad_inst)

        # 2. Adversarial Imperative Verb: "Вір мені!" (Believe me!)
        # Must NOT trigger FORBIDDEN_ENTITY_VARIANT
        seg_imperative = _make_segment("Faith said: 'Believe me!'", "Фейт сказала: «Вір мені!»")
        v_imperative = val.validate(seg_imperative, context=ctx)
        assert not any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in v_imperative), (
            "False positive: Imperative verb 'Вір' triggered forbidden entity 'Віра'!"
        )

        # Prefix verbs like "Повір"
        seg_prefix = _make_segment("Faith told him: trust me.", "Фейт сказала йому: повір мені.")
        v_prefix = val.validate(seg_prefix, context=ctx)
        assert not any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in v_prefix)

    def test_canonical_inflections_accepted(self):
        """Verify inflected forms of canonical target are recognized as present."""
        val = EntityConsistencyValidator()
        gandalf = EntityProfile(
            source_name="Gandalf",
            canonical_target="Ґандальф",
            locked=True,
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="They met Gandalf on the road.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[gandalf],
        )

        # Instrumental case: "Ґандальфом"
        seg_inst = _make_segment("They met Gandalf on the road.", "Вони зустрілися з Ґандальфом на дорозі.")
        assert len(val.validate(seg_inst, context=ctx)) == 0

        # Genitive case: "Ґандальфа"
        seg_gen = _make_segment("They met Gandalf on the road.", "Вони не бачили Ґандальфа вже багато днів.")
        assert len(val.validate(seg_gen, context=ctx)) == 0


# ============================================================================
# 4. GlossaryConsistencyValidator Adversarial Stress Tests
# ============================================================================

class TestGlossaryConsistencyValidatorStress:
    """Stress-test GlossaryConsistencyValidator on multi-word terms, stem inflections, and boundary matching."""

    def test_multi_word_approved_term_inflections(self):
        """Approved multi-word terms (target_term) recognize case-inflected word stems."""
        val = GlossaryConsistencyValidator()
        potion = GlossaryItem(
            source_term="healing potion",
            target_term="зілля зцілення",
            reviewed=True,
            locked=True,
            forbidden_variants=["лікувальний напій", "чай"],
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="He found a healing potion in the chest.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[potion],
        )

        # Inflection: "зіллям зцілення" (instrumental)
        seg_inst = _make_segment(
            "He found a healing potion in the chest.",
            "Він скористався зіллям зцілення зі скрині."
        )
        assert len(val.validate(seg_inst, context=ctx)) == 0

    def test_multi_word_forbidden_variant_matching(self):
        """
        Tests forbidden variant detection for exact match.
        Documented finding: multi-word forbidden variants require explicit list
        or nominative match; naive inflection does not inflect internal adjectives.
        """
        val = GlossaryConsistencyValidator()
        potion = GlossaryItem(
            source_term="healing potion",
            target_term="зілля зцілення",
            reviewed=True,
            locked=True,
            forbidden_variants=["лікувальний напій", "чай"],
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="He found a healing potion in the chest.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[potion],
        )

        # Exact forbidden phrase: "лікувальний напій"
        seg_forbid = _make_segment(
            "He found a healing potion in the chest.",
            "Він випив лікувальний напій зі скрині."
        )
        v_forbid = val.validate(seg_forbid, context=ctx)
        assert any(v.rule_code == "FORBIDDEN_GLOSSARY_VARIANT" for v in v_forbid)

        # Single word forbidden variant: "чай"
        seg_tea = _make_segment(
            "He found a healing potion in the chest.",
            "Він випив трав'яний чай замість зілля."
        )
        v_tea = val.validate(seg_tea, context=ctx)
        assert any(v.rule_code == "FORBIDDEN_GLOSSARY_VARIANT" for v in v_tea)

    def test_partial_word_boundary_isolation(self):
        """Ensure term 'art' is not falsely matched in 'smart' or 'cart'."""
        val = GlossaryConsistencyValidator()
        art_term = GlossaryItem(
            source_term="art",
            target_term="мистецтво",
            reviewed=True,
            locked=True,
        )
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="He was smart and fast.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[art_term],
        )
        # Source has 'smart', NOT 'art'
        seg = _make_segment("He was smart and fast.", "Він був розумним і швидким.")
        assert len(val.validate(seg, context=ctx)) == 0


# ============================================================================
# 5. GenderAgreementValidator Adversarial Stress Tests
# ============================================================================

class TestGenderAgreementValidatorStress:
    """Stress-test GenderAgreementValidator on inverted clauses, compound subjects, and oblique cases."""

    def test_direct_speech_inverted_verb(self):
        val = GenderAgreementValidator()
        olena = GlossaryItem(source_term="Olena", target_term="Олена", grammatical_gender="жіночий")
        taras = GlossaryItem(source_term="Taras", target_term="Тарас", grammatical_gender="чоловічий")
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Olena said. Taras said.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[olena, taras],
        )

        # 1. Correct inverted dialogue reporting verbs:
        # "— Я готова, — сказала Олена." (fem + fem)
        # "— Я теж, — відповів Тарас." (masc + masc)
        seg_ok = _make_segment(
            "Olena said. Taras said.",
            "— Я готова, — сказала Олена. — Я теж, — відповів Тарас."
        )
        assert len(val.validate(seg_ok, context=ctx)) == 0

        # 2. Inverted dialogue reporting verbs with MISMATCH:
        # "— Я готова, — сказав Олена." (masc verb + fem name)
        seg_bad_fem = _make_segment(
            "Olena said.",
            "— Я готова, — сказав Олена."
        )
        v_fem = val.validate(seg_bad_fem, context=ctx)
        assert any(v.rule_code == "GenderAgreementMismatch" for v in v_fem)

        # "— Я тут, — відповіла Тарас." (fem verb + masc name)
        seg_bad_masc = _make_segment(
            "Taras said.",
            "— Я тут, — відповіла Тарас."
        )
        v_masc = val.validate(seg_bad_masc, context=ctx)
        assert any(v.rule_code == "GenderAgreementMismatch" for v in v_masc)

    def test_clause_initial_inversion(self):
        val = GenderAgreementValidator()
        anna = GlossaryItem(source_term="Anna", target_term="Анна", grammatical_gender="жіночий")
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Anna came home.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[anna],
        )

        # Initial verb inversion: "Прийшла Анна додому." (fem + fem -> valid)
        seg_ok = _make_segment("Anna came home.", "Прийшла Анна додому ввечері.")
        assert len(val.validate(seg_ok, context=ctx)) == 0

        # Initial verb mismatch: "Прийшов Анна додому." (masc verb + fem subject -> mismatch)
        seg_bad = _make_segment("Anna came home.", "Прийшов Анна додому ввечері.")
        v_bad = val.validate(seg_bad, context=ctx)
        assert any(v.rule_code == "GenderAgreementMismatch" for v in v_bad)

    def test_compound_subjects_with_plural_verbs(self):
        """Compound subjects ('Марія та Петро') taking plural past verbs ('пішли') must NOT fail."""
        val = GenderAgreementValidator()
        maria = GlossaryItem(source_term="Maria", target_term="Марія", grammatical_gender="жіночий")
        peter = GlossaryItem(source_term="Peter", target_term="Петро", grammatical_gender="чоловічий")
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Maria and Peter left.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[maria, peter],
        )

        seg_plural = _make_segment(
            "Maria and Peter left.",
            "Марія та Петро швидко пішли до лісу."
        )
        violations = val.validate(seg_plural, context=ctx)
        assert len(violations) == 0, f"Plural past verb triggered gender mismatch: {violations}"

    def test_oblique_object_cases_resistance(self):
        """Ensure characters appearing in oblique case objects do NOT trigger false mismatches."""
        val = GenderAgreementValidator()
        olena = GlossaryItem(source_term="Olena", target_term="Олена", grammatical_gender="жіночий")
        ivan = GlossaryItem(source_term="Ivan", target_term="Іван", grammatical_gender="чоловічий")
        ctx = PromptContext(
            segment_id=uuid4(),
            prompt_text="",
            target_source="Ivan called Olena. Ivan gave a book to Olena.",
            target_draft="",
            previous_context="",
            entities_text="",
            active_entities=[olena, ivan],
        )

        # 1. Accusative object: "Іван покликав Олену."
        # Subject is Іван (masc) -> покликав (masc). Олену is accusative object.
        seg_acc = _make_segment("Ivan called Olena.", "Іван голосно покликав Олену до столу.")
        assert len(val.validate(seg_acc, context=ctx)) == 0

        # 2. Dative object: "Іван дав Олені книгу."
        seg_dat = _make_segment("Ivan gave a book to Olena.", "Іван передав Олені старовинну книгу.")
        assert len(val.validate(seg_dat, context=ctx)) == 0

        # 3. Prepositional object: "Іван думав про Олену."
        seg_prep = _make_segment("Ivan thought about Olena.", "Іван довго думав про Олену.")
        assert len(val.validate(seg_prep, context=ctx)) == 0


# ============================================================================
# 6. NumbersAndUnitsValidator Adversarial Stress Tests
# ============================================================================

class TestNumbersAndUnitsValidatorStress:
    """Stress-test NumbersAndUnitsValidator on Arabic, Roman, fractions, currencies, and word numbers."""

    def test_arabic_integers_and_decimals(self):
        val = NumbersAndUnitsValidator()

        # Decimal with comma in Ukrainian: 3.14 -> 3,14
        seg_dec = _make_segment("Pi is approximately 3.14.", "Число пі дорівнює приблизно 3,14.")
        assert len(val.validate(seg_dec)) == 0

        # Number altered: 100 -> 200
        seg_alt = _make_segment("He had 100 coins.", "У нього було 200 монет.")
        v_alt = val.validate(seg_alt)
        assert any(v.rule_code == "NUMBER_MISMATCH" for v in v_alt)
        assert any(v.severity == QASeverity.CRITICAL for v in v_alt)

    def test_roman_numerals_and_chapter_context(self):
        val = NumbersAndUnitsValidator()

        # Preserved Roman numeral
        seg_ok = _make_segment("In Chapter XII, he arrived.", "У Розділі XII він нарешті прибув.")
        assert len(val.validate(seg_ok)) == 0

        # Missing Roman numeral
        seg_bad = _make_segment("In Chapter XII, he arrived.", "У дванадцятому розділі він прибув.")
        v_bad = val.validate(seg_bad)
        assert any(v.rule_code == "ROMAN_NUMERAL_MISMATCH" for v in v_bad)

    def test_word_numbers_for_small_cardinals(self):
        val = NumbersAndUnitsValidator()

        # 1 -> "один", "одна", "одне"
        seg_1 = _make_segment("He saw 1 dog.", "Він побачив одного собаку.")
        assert len(val.validate(seg_1)) == 0

        # 3 -> "три", "трьох"
        seg_3 = _make_segment("There were 3 swords.", "Там лежало три мечі.")
        assert len(val.validate(seg_3)) == 0

    def test_currencies_and_percentage_preservation(self):
        val = NumbersAndUnitsValidator()

        # $100 -> 100$ or 100 доларів
        seg_curr_sym = _make_segment("Cost was $100.", "Вартість становила 100$.")
        assert len(val.validate(seg_curr_sym)) == 0

        seg_curr_word = _make_segment("Cost was $100.", "Вартість становила 100 доларів.")
        assert len(val.validate(seg_curr_word)) == 0

        # Currency missing completely
        seg_curr_miss = _make_segment("Cost was $100.", "Вартість становила 100 монет.")
        v_miss = val.validate(seg_curr_miss)
        assert any(v.rule_code == "CURRENCY_PERCENTAGE_MISMATCH" for v in v_miss)

        # Percentage %
        seg_pct_sym = _make_segment("Increased by 50%.", "Зросла на 50%.")
        assert len(val.validate(seg_pct_sym)) == 0

        seg_pct_word = _make_segment("Increased by 50%.", "Зросла на 50 відсотків.")
        assert len(val.validate(seg_pct_word)) == 0


# ============================================================================
# 7. ControlTokenValidator Adversarial Stress Tests
# ============================================================================

class TestControlTokenValidatorStress:
    """Stress-test ControlTokenValidator on model tokens, code fences, JSON tags, and chatter."""

    def test_residual_model_tokens(self):
        val = ControlTokenValidator()

        tokens = [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<|thought|>",
            "<|fim_prefix|>",
            "<|system|>",
        ]
        for tok in tokens:
            seg = _make_segment("Hello.", f"Привіт.{tok}")
            violations = val.validate(seg)
            assert any(v.rule_code == "ResidualControlToken" for v in violations), f"Failed to catch {tok}"
            crit = next(v for v in violations if v.rule_code == "ResidualControlToken")
            assert crit.severity == QASeverity.CRITICAL

    def test_technical_tags(self):
        val = ControlTokenValidator()
        tags = ["<tag_0>", "</tag_0>", "<tag_123>", "</tag_999>"]
        for tag in tags:
            seg = _make_segment("Text.", f"{tag}Текст.{tag}")
            v = val.validate(seg)
            assert any(v.rule_code == "LeakedTechnicalTag" for v in v)

    def test_markdown_code_fences_and_json_payloads(self):
        val = ControlTokenValidator()
        payloads = [
            "```json\n{\"id\": \"1\"}\n```",
            "```\nSome text\n```",
            '{"segments": [{"id": "uuid", "translation": "Текст"}]}',
            '{"id": "1", "translation": "Текст"}',
        ]
        for p in payloads:
            seg = _make_segment("English text.", p)
            v = val.validate(seg)
            assert any(v.rule_code == "LEAKED_JSON_OR_CODE_FENCE" for v in v), f"Failed to catch fence in {p}"
            assert any(v.severity == QASeverity.CRITICAL for v in v)

    def test_prompt_template_indicators(self):
        val = ControlTokenValidator()
        templates = [
            "=== ПОПЕРЕДНІЙ КОНТЕКСТ ===\nТекст",
            "=== ГЛОСАРІЙ ТА СУТНОСТІ ===\nТекст",
            "=== ПАРАГРАФ ДЛЯ РЕДАГУВАННЯ ===\nТекст",
            "ОРИГІНАЛ (EN): Hello",
            "ЧОРНОВИЙ ПЕРЕКЛАД NLLB: Привіт",
        ]
        for t in templates:
            seg = _make_segment("English text.", t)
            v = val.validate(seg)
            assert any(v.rule_code == "LEAKED_PROMPT_TEMPLATE" for v in v), f"Failed to catch template in {t}"
            assert any(v.severity == QASeverity.CRITICAL for v in v)

    def test_conversational_chatter_preambles(self):
        val = ControlTokenValidator()
        chatters = [
            "Here is the translation: Ось переклад тексту.",
            "Sure, here is the refined Ukrainian text: Добрий день.",
            "As an AI language model, I cannot translate this.",
            "Ось переклад: Сонце зійшло над горами.",
            "Звісно, ось переклад даного фрагмента: Привіт.",
            "Вибачте, як штучний інтелект я не маю почуттів.",
        ]
        for chat in chatters:
            seg = _make_segment("English source text.", chat)
            v = val.validate(seg)
            assert any(v.rule_code == "CONVERSATIONAL_CHATTER" for v in v), f"Failed to catch chatter: {chat}"
            assert any(v.severity == QASeverity.CRITICAL for v in v)


# ============================================================================
# 8. DialogueIntegrityValidator Adversarial Stress Tests
# ============================================================================

class TestDialogueIntegrityValidatorStress:
    """Stress-test DialogueIntegrityValidator on dash typography, quotation pairing, and speaker turns."""

    def test_ascii_dialogue_hyphen_vs_em_dash(self):
        val = DialogueIntegrityValidator()

        # Direct speech starting with hyphen -> WARNING
        seg_hyphen = _make_segment("- Where are you going?", "- Куди ти йдеш?")
        v_hyphen = val.validate(seg_hyphen)
        assert any(v.rule_code == "DIALOGUE_HYPHEN_INSTEAD_OF_DASH" for v in v_hyphen)

        # Direct speech starting with proper em-dash -> OK
        seg_dash = _make_segment("— Where are you going?", "— Куди ти йдеш?")
        v_dash = [v for v in val.validate(seg_dash) if v.rule_code == "DIALOGUE_HYPHEN_INSTEAD_OF_DASH"]
        assert len(v_dash) == 0

        # Hyphen inside compound word should NOT trigger dialogue warning
        seg_compound = _make_segment("It was blue-green.", "Він був темно-синій.")
        assert not any(v.rule_code == "DIALOGUE_HYPHEN_INSTEAD_OF_DASH" for v in val.validate(seg_compound))

    def test_unpaired_chevrons_and_quotes(self):
        val = DialogueIntegrityValidator()

        # Unpaired opening chevron
        seg_unpair_open = _make_segment("Book title.", "Він читав «Кобзар.")
        v_open = val.validate(seg_unpair_open)
        assert any(v.rule_code == "UNPAIRED_QUOTATION_MARKS" for v in v_open)

        # Unpaired closing chevron
        seg_unpair_close = _make_segment("Book title.", "Він читав Кобзар».")
        v_close = val.validate(seg_unpair_close)
        assert any(v.rule_code == "UNPAIRED_QUOTATION_MARKS" for v in v_close)

        # Odd count straight ASCII quotes
        seg_odd_ascii = _make_segment("Quote.", 'Він сказав: "Привіт.')
        v_odd = val.validate(seg_odd_ascii)
        assert any(v.rule_code == "UNPAIRED_QUOTATION_MARKS" for v in v_odd)

        # Properly paired chevrons -> OK
        seg_paired = _make_segment("Book title.", "Він читав «Кобзар».")
        assert not any(v.rule_code == "UNPAIRED_QUOTATION_MARKS" for v in val.validate(seg_paired))

    def test_speaker_turn_collapse(self):
        val = DialogueIntegrityValidator()

        # Source has 3 distinct speaker turns
        src_3_turns = "— Hello, said Tom.\n— Hi, said Anna.\n— Goodbye, said Tom."
        
        # Target collapses into single line
        tgt_collapsed = "— Привіт, сказав Том, а Анна відповіла привіт і Том попрощався."
        seg_coll = _make_segment(src_3_turns, tgt_collapsed)
        v_coll = val.validate(seg_coll)
        assert any(v.rule_code == "SPEAKER_TURN_COLLAPSE" for v in v_coll)

        # Target preserves 3 turns
        tgt_3_turns = "— Привіт, — сказав Том.\n— Привіт, — відповіла Анна.\n— До побачення, — сказав Том."
        seg_ok = _make_segment(src_3_turns, tgt_3_turns)
        v_ok = val.validate(seg_ok)
        assert not any(v.rule_code == "SPEAKER_TURN_COLLAPSE" for v in v_ok)
