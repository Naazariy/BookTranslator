"""
tests/unit/test_quality_modular_validators.py

Unit tests for all 8 modular QA validators in src/quality/validators/.
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
# 1. EmptyTranslationValidator Tests
# ============================================================================

def test_empty_translation_flags_blank_and_none():
    val = EmptyTranslationValidator()

    seg_blank = _make_segment("Hello world.", "   ")
    v_blank = val.validate(seg_blank)
    assert len(v_blank) == 1
    assert v_blank[0].severity == QASeverity.CRITICAL
    assert v_blank[0].rule_code == "EmptyTranslation"

    seg_none = _make_segment("Hello world.")
    seg_none.final_translation = None
    seg_none.refined_translation = None
    seg_none.draft_translation = None
    v_none = val.validate(seg_none)
    assert len(v_none) == 1
    assert v_none[0].severity == QASeverity.CRITICAL

    seg_ok = _make_segment("Hello world.", "Привіт, світе.")
    assert len(val.validate(seg_ok)) == 0


# ============================================================================
# 2. CompletenessValidator Tests
# ============================================================================

def test_completeness_flags_extreme_length_ratios():
    val = CompletenessValidator()

    # Suspiciously short: 100 chars -> 10 chars (<35%)
    src_long = "This is a very long paragraph that contains a lot of descriptive text and several sentences together."
    seg_short = _make_segment(src_long, "Коротко.")
    v_short = val.validate(seg_short)
    assert any(v.rule_code == "LENGTH_RATIO_TOO_LOW" for v in v_short)
    assert any(v.severity == QASeverity.CRITICAL for v in v_short)

    # Excessively long: 22 chars -> 120 chars (>280%)
    seg_long = _make_segment("This is a source text.", "Це надзвичайно довгий переклад який повторює одне й те саме знову і знову без жодної на те потреби та причини взагалі.")
    v_long = val.validate(seg_long)
    assert any(v.rule_code == "LENGTH_RATIO_TOO_HIGH" for v in v_long)


def test_completeness_flags_truncation_and_escapes():
    val = CompletenessValidator()

    # Trailing connector (and no terminal punctuation)
    seg_cut = _make_segment("He went to the forest and met a bear.", "Він пішов до лісу і")
    v_cut = val.validate(seg_cut)
    assert any(v.rule_code == "TRUNCATION_TRAILING_CONNECTOR" for v in v_cut)

    # Trailing escape
    seg_esc = _make_segment("He opened the door.", "Він відчинив двері.\\")
    v_esc = val.validate(seg_esc)
    assert any(v.rule_code == "TRUNCATION_TRAILING_ESCAPE" for v in v_esc)


def test_completeness_flags_sentence_count_drop_and_duplicates():
    val = CompletenessValidator()

    # 4 sentences in source, 1 in target
    src_multi = "Sentence one. Sentence two. Sentence three. Sentence four."
    seg_drop = _make_segment(src_multi, "Лише одне речення.")
    v_drop = val.validate(seg_drop)
    assert any(v.rule_code == "SentenceCountDrop" for v in v_drop)

    # Consecutive duplicates
    tgt_dup = "Він пішов додому ввечері. Він пішов додому увечері."
    seg_dup = _make_segment("He went home in the evening. He went home in the evening.", tgt_dup)
    v_dup = val.validate(seg_dup)
    assert any(v.rule_code == "ConsecutiveDuplicateSentences" for v in v_dup)


# ============================================================================
# 3. EntityConsistencyValidator Tests
# ============================================================================

def test_entity_consistency_flags_forbidden_variants():
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
        target_source="Cherry smiled at the boy.",
        target_draft="",
        previous_context="",
        entities_text="",
        active_entities=[cherry],
    )

    # Forbidden variant "Вишня"
    seg_bad = _make_segment("Cherry smiled at the boy.", "Вишня усміхнулася хлопчику.")
    v_bad = val.validate(seg_bad, context=ctx)
    assert any(v.rule_code == "FORBIDDEN_ENTITY_VARIANT" for v in v_bad)
    crit = next(v for v in v_bad if v.rule_code == "FORBIDDEN_ENTITY_VARIANT")
    assert crit.severity == QASeverity.CRITICAL
    assert crit.suggested_fix == "Черрі"

    # Canonical target "Черрі"
    seg_ok = _make_segment("Cherry smiled at the boy.", "Черрі усміхнулася хлопчикові.")
    assert len(val.validate(seg_ok, context=ctx)) == 0


def test_entity_consistency_flags_missing_locked_entity():
    val = EntityConsistencyValidator()

    gandalf = EntityProfile(
        source_name="Gandalf",
        canonical_target="Ґандальф",
        locked=True,
    )
    ctx = PromptContext(
        segment_id=uuid4(),
        prompt_text="",
        target_source="Gandalf arrived at sunset.",
        target_draft="",
        previous_context="",
        entities_text="",
        active_entities=[gandalf],
    )

    seg_missing = _make_segment("Gandalf arrived at sunset.", "Старий маг прибув на заході сонця.")
    v_missing = val.validate(seg_missing, context=ctx)
    assert any(v.rule_code == "MISSING_LOCKED_ENTITY" for v in v_missing)


# ============================================================================
# 4. GlossaryConsistencyValidator Tests
# ============================================================================

def test_glossary_consistency_flags_missing_and_forbidden_terms():
    val = GlossaryConsistencyValidator()

    item = GlossaryItem(
        source_term="healing potion",
        target_term="зілля зцілення",
        reviewed=True,
        locked=True,
        forbidden_variants=["лікувальний напій", "чай"],
    )
    ctx = PromptContext(
        segment_id=uuid4(),
        prompt_text="",
        target_source="He drank the healing potion.",
        target_draft="",
        previous_context="",
        entities_text="",
        active_entities=[item],
    )

    # Forbidden variant
    seg_forbid = _make_segment("He drank the healing potion.", "Він випив лікувальний напій.")
    v_forbid = val.validate(seg_forbid, context=ctx)
    assert any(v.rule_code == "FORBIDDEN_GLOSSARY_VARIANT" for v in v_forbid)

    # Missing approved term
    seg_miss = _make_segment("He drank the healing potion.", "Він випив невідому рідину.")
    v_miss = val.validate(seg_miss, context=ctx)
    assert any(v.rule_code == "MISSING_GLOSSARY_TERM" for v in v_miss)

    # Correct term
    seg_ok = _make_segment("He drank the healing potion.", "Він випив зілля зцілення.")
    assert len(val.validate(seg_ok, context=ctx)) == 0


# ============================================================================
# 5. GenderAgreementValidator Tests
# ============================================================================

def test_gender_agreement_flags_mismatches_and_resists_false_positives():
    val = GenderAgreementValidator()

    maria = GlossaryItem(source_term="Maria", target_term="Марія", grammatical_gender="жіночий")
    john = GlossaryItem(source_term="John", target_term="Джон", grammatical_gender="чоловічий")
    ctx = PromptContext(
        segment_id=uuid4(),
        prompt_text="",
        target_source="Maria said. John went.",
        target_draft="",
        previous_context="",
        entities_text="",
        active_entities=[maria, john],
    )

    # Mismatch: Maria + сказав, John + пішла
    seg_mismatch = _make_segment("Maria said. John went.", "Марія сказав правду. Джон пішла геть.")
    v_mismatch = val.validate(seg_mismatch, context=ctx)
    assert len([v for v in v_mismatch if v.rule_code == "GenderAgreementMismatch"]) == 2

    # Correct agreement
    seg_ok = _make_segment("Maria said. John went.", "Марія сказала правду. Джон пішов геть.")
    assert len(val.validate(seg_ok, context=ctx)) == 0

    # False-positive resistance: oblique object ("покликала Джона")
    seg_fp = _make_segment("Maria called John.", "Марія покликала Джона до себе.")
    assert len(val.validate(seg_fp, context=ctx)) == 0


# ============================================================================
# 6. NumbersAndUnitsValidator Tests
# ============================================================================

def test_numbers_and_units_preservation():
    val = NumbersAndUnitsValidator()

    # Missing number 42
    seg_num = _make_segment("There were 42 apples on the table.", "На столі було багато яблук.")
    v_num = val.validate(seg_num)
    assert any(v.rule_code == "NUMBER_MISMATCH" for v in v_num)

    # Preserved number
    seg_num_ok = _make_segment("There were 42 apples on the table.", "На столі було 42 яблука.")
    assert len(val.validate(seg_num_ok)) == 0

    # Spelled-out numeral recognition (1 -> "одне")
    seg_spelled = _make_segment("Target 1.", "Одне цільове речення.")
    assert len(val.validate(seg_spelled)) == 0

    # Missing Roman numeral
    seg_roman = _make_segment("In Chapter IV, he returned.", "У четвертому розділі він повернувся.")
    v_roman = val.validate(seg_roman)
    assert any(v.rule_code == "ROMAN_NUMERAL_MISMATCH" for v in v_roman)


# ============================================================================
# 7. ControlTokenValidator Tests
# ============================================================================

def test_control_tokens_and_chatter_detection():
    val = ControlTokenValidator()

    # Residual control token
    seg_ctrl = _make_segment("Hello.", "Привіт.<|im_end|>")
    v_ctrl = val.validate(seg_ctrl)
    assert any(v.rule_code == "ResidualControlToken" for v in v_ctrl)

    # Technical tag
    seg_tech = _make_segment("Word.", "<tag_1>Слово.</tag_1>")
    v_tech = val.validate(seg_tech)
    assert any(v.rule_code == "LeakedTechnicalTag" for v in v_tech)

    # Leaked code fence / JSON
    seg_json = _make_segment("Text.", "```json\n{\"segments\": [{\"id\": \"1\", \"translation\": \"Текст.\"}]}\n```")
    v_json = val.validate(seg_json)
    assert any(v.rule_code == "LEAKED_JSON_OR_CODE_FENCE" for v in v_json)

    # Conversational chatter
    seg_chat = _make_segment("Hello.", "Here is the translation: Привіт.")
    v_chat = val.validate(seg_chat)
    assert any(v.rule_code == "CONVERSATIONAL_CHATTER" for v in v_chat)


# ============================================================================
# 8. DialogueIntegrityValidator Tests
# ============================================================================

def test_dialogue_integrity_validation():
    val = DialogueIntegrityValidator()

    # ASCII hyphen instead of em-dash
    seg_hyphen = _make_segment("- Hello, said John.", "- Привіт, — сказав Джон.")
    v_hyphen = val.validate(seg_hyphen)
    assert any(v.rule_code == "DIALOGUE_HYPHEN_INSTEAD_OF_DASH" for v in v_hyphen)

    # Proper em-dash
    seg_dash = _make_segment("— Hello, said John.", "— Привіт, — сказав Джон.")
    v_dash = [v for v in val.validate(seg_dash) if v.rule_code == "DIALOGUE_HYPHEN_INSTEAD_OF_DASH"]
    assert len(v_dash) == 0

    # Unpaired chevrons
    seg_unpair = _make_segment("Book title.", "Книга «Таємничий острів.")
    v_unpair = val.validate(seg_unpair)
    assert any(v.rule_code == "UNPAIRED_QUOTATION_MARKS" for v in v_unpair)
