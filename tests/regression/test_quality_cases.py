"""
Regression Test Suite for BookTranslator V2 Quality Cases.

Loads tests/regression/quality_cases.json and validates each test case against
the quality assertion patterns required for V2, ensuring:
1. Strict character name preservation (e.g. 'Cherry' -> 'Черрі', strictly rejecting 'Вишня').
2. Accurate polysemy disambiguation (e.g. 'tart' as sharp speech vs fruit pie).
3. Consistent fantasy item translation (e.g. 'healing potion' -> 'зілля зцілення').
4. Grammatical gender agreement between subjects and past-tense verbs.
5. Preservation of bracketed LitRPG/UI class tags and status banners.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.quality.pipeline import (
    is_masculine_past_verb,
    is_feminine_past_verb,
    is_neuter_past_verb,
)

CORPUS_PATH = Path(__file__).parent / "quality_cases.json"


def load_corpus() -> Dict[str, Any]:
    """Loads and returns the quality regression corpus from JSON."""
    assert CORPUS_PATH.exists(), f"Corpus file not found at {CORPUS_PATH}"
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "Corpus root must be a JSON object"
    assert "cases" in data, "Corpus must contain 'cases' array"
    return data


CORPUS = load_corpus()
ALL_CASES = CORPUS["cases"]
CASE_MAP = {c["id"]: c for c in ALL_CASES}


class QualityCaseEvaluator:
    """
    Evaluator implementing the quality assertion patterns for regression cases.
    Used by test runners and adaptable by the V2 Quality Pipeline validators.
    """

    @staticmethod
    def evaluate(
        case: Dict[str, Any],
        translated_text: str,
    ) -> List[str]:
        """
        Evaluates translated_text against constraints defined in case.
        Returns a list of violation messages (empty if completely valid).
        """
        violations: List[str] = []
        if not translated_text or not translated_text.strip():
            return ["Translation text is empty or whitespace."]

        constraints = case.get("expected_constraints", {})
        forbidden_variants = case.get("forbidden_variants", []) or constraints.get("must_not_include", [])

        # 1. Check forbidden variants (case-insensitive substring/word matching)
        for forbidden in forbidden_variants:
            if not forbidden:
                continue
            # Use regex word boundary for alphanumeric words, literal search for symbols
            if re.search(r"^\w+$", forbidden, re.UNICODE):
                pattern = rf"\b{re.escape(forbidden)}\b"
                if re.search(pattern, translated_text, re.IGNORECASE):
                    violations.append(f"Forbidden variant detected: '{forbidden}' in '{translated_text}'")
            else:
                if forbidden.lower() in translated_text.lower():
                    violations.append(f"Forbidden phrase detected: '{forbidden}' in '{translated_text}'")

        # 2. Check must_include constraints
        must_include = constraints.get("must_include", [])
        for req in must_include:
            if not req:
                continue
            if isinstance(req, list):
                if not any(sub.lower() in translated_text.lower() for sub in req if sub):
                    violations.append(f"None of required alternative elements {req} found in translation '{translated_text}'")
            else:
                if req.lower() not in translated_text.lower():
                    violations.append(f"Required element '{req}' missing from translation '{translated_text}'")

        # 2b. Check must_include_any constraints
        must_include_any = constraints.get("must_include_any", [])
        if must_include_any:
            if not any(sub.lower() in translated_text.lower() for sub in must_include_any if sub):
                violations.append(
                    f"None of the alternative elements {must_include_any} found in translation '{translated_text}'"
                )

        # 3. Check expected_regex pattern
        expected_regex = constraints.get("expected_regex")
        if expected_regex:
            if not re.search(expected_regex, translated_text):
                violations.append(f"Translation does not match expected regex: r'{expected_regex}'")

        # 4. Check forbidden_regex pattern
        forbidden_regex = constraints.get("forbidden_regex")
        if forbidden_regex:
            if re.search(forbidden_regex, translated_text):
                violations.append(f"Translation matches forbidden regex: r'{forbidden_regex}'")

        # 5. Check grammatical gender agreement if required
        required_gender = constraints.get("required_gender")
        if required_gender:
            gender_violation = QualityCaseEvaluator._check_gender_agreement(
                case, translated_text, required_gender
            )
            if gender_violation:
                violations.append(gender_violation)

        return violations

    @staticmethod
    def _check_gender_agreement(
        case: Dict[str, Any],
        translated_text: str,
        required_gender: str,
    ) -> Optional[str]:
        """
        Validates past-tense verb agreement for the required grammatical gender.
        """
        # Find tokens
        tokens = re.findall(r"[а-яіїєґА-ЯІЇЄҐa-zA-Z'’ʼ‘`]+", translated_text)

        if required_gender == "жіночий":
            # Check for conflicting masculine past verbs near subject
            for tok in tokens:
                if is_masculine_past_verb(tok):
                    # Check if token is explicitly a verb that contradicts female subject
                    # Ensure it's not a noun homograph
                    subject_terms = case.get("expected_constraints", {}).get("must_include", [])
                    return f"Gender mismatch: expected feminine past tense, but found masculine verb '{tok}'"

        elif required_gender == "чоловічий":
            for tok in tokens:
                if is_feminine_past_verb(tok):
                    return f"Gender mismatch: expected masculine past tense, but found feminine verb '{tok}'"

        elif required_gender == "середній":
            for tok in tokens:
                if is_masculine_past_verb(tok) or is_feminine_past_verb(tok):
                    return f"Gender mismatch: expected neuter past tense, but found non-neuter verb '{tok}'"

        return None


# ============================================================================
# Schema & Integrity Tests
# ============================================================================

def test_quality_cases_json_exists_and_parses():
    """Verify that quality_cases.json exists, is valid JSON, and has correct root structure."""
    assert CORPUS_PATH.exists()
    assert CORPUS.get("version") == "2.0.0"
    assert "description" in CORPUS
    assert len(ALL_CASES) >= 25, f"Expected >= 25 cases, found {len(ALL_CASES)}"


def test_quality_cases_covers_all_five_mandatory_categories():
    """Verify all 5 required defect categories are represented with at least 5 cases each."""
    required_categories = {
        "character_names",
        "polysemy",
        "items",
        "gender_agreement",
        "rpg_class_tags",
    }
    counts: Dict[str, int] = {}
    for case in ALL_CASES:
        cat = case["category"]
        counts[cat] = counts.get(cat, 0) + 1

    missing = required_categories - set(counts.keys())
    assert not missing, f"Missing required categories in corpus: {missing}"

    for cat in required_categories:
        assert counts[cat] >= 5, f"Category '{cat}' has only {counts[cat]} cases (expected >= 5)"


def test_quality_cases_unique_identifiers():
    """Verify that all test case IDs are unique and properly formatted."""
    ids = [c["id"] for c in ALL_CASES]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {[x for x in ids if ids.count(x) > 1]}"
    for cid in ids:
        assert re.match(r"^REG-[A-Z]+-\d{3}$", cid), f"ID '{cid}' does not match format REG-XXX-000"


def test_quality_cases_mandatory_fields_present():
    """Verify that every case contains all mandatory fields."""
    mandatory_fields = [
        "id",
        "category",
        "description",
        "source_text",
        "source_en",
        "expected_uk_canonical",
        "forbidden_variants",
        "v1_failure_example",
        "expected_constraints",
    ]
    for case in ALL_CASES:
        cid = case["id"]
        for field in mandatory_fields:
            assert field in case, f"Case '{cid}' missing mandatory field '{field}'"
            assert case[field], f"Case '{cid}' has empty value for '{field}'"

        # Verify source_en matches source_text
        assert case["source_en"] == case["source_text"], f"Case '{cid}' source_en mismatch with source_text"

        # Verify expected_constraints has must_include and must_not_include
        constraints = case["expected_constraints"]
        assert "must_include" in constraints, f"Case '{cid}' missing must_include"
        assert "must_not_include" in constraints, f"Case '{cid}' missing must_not_include"


# ============================================================================
# Parameterized Validation Tests
# ============================================================================

@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c["id"])
def test_canonical_translation_passes_all_quality_constraints(case: Dict[str, Any]):
    """
    Every expected canonical translation in the corpus must pass all defined
    quality constraints with zero violations.
    """
    canonical_text = case["expected_uk_canonical"]
    violations = QualityCaseEvaluator.evaluate(case, canonical_text)
    assert not violations, (
        f"Case '{case['id']}' canonical translation failed evaluation:\n"
        f"Text: '{canonical_text}'\n"
        f"Violations: {violations}"
    )


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c["id"])
def test_v1_failure_example_triggers_violations(case: Dict[str, Any]):
    """
    Every V1 failure example must fail evaluation, proving that the assertion
    pattern successfully catches the historical defect.
    """
    v1_text = case["v1_failure_example"]
    violations = QualityCaseEvaluator.evaluate(case, v1_text)
    assert len(violations) > 0, (
        f"Case '{case['id']}' V1 failure example unexpectedly PASSED evaluation:\n"
        f"V1 text: '{v1_text}'\n"
        f"Expected violation from forbidden variants: {case.get('forbidden_variants')}"
    )


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c["id"])
def test_forbidden_variants_injection_triggers_failure(case: Dict[str, Any]):
    """
    Injecting any forbidden variant into a valid translation must trigger
    a quality failure.
    """
    forbidden_variants = case.get("forbidden_variants", [])
    if not forbidden_variants:
        pytest.skip(f"Case '{case['id']}' has no forbidden variants")

    first_forbidden = forbidden_variants[0]
    corrupted_text = f"{case['expected_uk_canonical']} {first_forbidden}"
    violations = QualityCaseEvaluator.evaluate(case, corrupted_text)
    assert any(first_forbidden.lower() in v.lower() for v in violations), (
        f"Case '{case['id']}': injecting forbidden variant '{first_forbidden}' "
        f"did not trigger an explicit violation! Violations: {violations}"
    )


# ============================================================================
# Category-Specific Deep Invariant Tests
# ============================================================================

class TestCharacterNameInvariants:
    """Specialized tests for character names (R1 / R2)."""

    def test_cherry_strictly_rejects_vyshnya_and_vishnya(self):
        """Cherry must never be translated as fruit (Вишня, Вішня, Черешня)."""
        cherry_cases = [c for c in ALL_CASES if "cherry" in c["source_text"].lower()]
        assert len(cherry_cases) >= 2, "Need at least 2 Cherry regression cases"

        for case in cherry_cases:
            for bad_word in ["Вишня", "Вішня", "Черешня", "Вишенька"]:
                bad_text = f"{bad_word} стояла біля дверей."
                violations = QualityCaseEvaluator.evaluate(case, bad_text)
                assert any(bad_word.lower() in v.lower() for v in violations)

    def test_proper_name_homographs_prefer_transliteration(self):
        """Names like Hope, Faith, Robin, Rose, Flint must not become common Ukrainian nouns."""
        homograph_cases = [
            c for c in ALL_CASES
            if c["id"] in ("REG-NAME-003", "REG-NAME-004", "REG-NAME-005", "REG-NAME-006", "REG-NAME-007")
        ]
        assert len(homograph_cases) == 5
        for case in homograph_cases:
            assert case["expected_constraints"]["required_gender"] in ("жіночий", "чоловічий")
            for forbidden in case["forbidden_variants"]:
                corrupted = f"{forbidden} відповіла тихо."
                violations = QualityCaseEvaluator.evaluate(case, corrupted)
                assert len(violations) > 0


class TestPolysemyInvariants:
    """Specialized tests for context-aware polysemy disambiguation."""

    def test_tart_speech_tone_vs_culinary_pastry(self):
        """Verify that 'tart reply' rejects 'пиріг' and 'blackberry tart' rejects 'їдкий'."""
        case_tone = CASE_MAP["REG-POLY-001"]
        case_food = CASE_MAP["REG-POLY-002"]

        # Tone case must reject pastry
        assert "пиріг" in case_tone["forbidden_variants"]
        assert "тарт" in case_tone["forbidden_variants"]
        assert QualityCaseEvaluator.evaluate(case_tone, "Вона дала йому пиріг.")

        # Food case must reject speech tone adjectives
        assert "їдкий" in case_food["forbidden_variants"]
        assert "дошкульний" in case_food["forbidden_variants"]
        assert QualityCaseEvaluator.evaluate(case_food, "Шинкар подав теплий їдкий пиріжок.")

    def test_bank_river_vs_financial_institution(self):
        """Verify that 'river bank' strictly rejects financial 'банк'."""
        case_bank = CASE_MAP["REG-POLY-003"]
        assert "банк" in case_bank["forbidden_variants"]
        bad_text = "Мандрівники сіли у банку річки."
        assert QualityCaseEvaluator.evaluate(case_bank, bad_text)

    def test_cast_magic_vs_plaster_or_theatrical(self):
        """Verify that 'cast a spell' strictly rejects 'гіпс' or 'акторський склад'."""
        case_cast = CASE_MAP["REG-POLY-004"]
        assert "гіпс" in case_cast["forbidden_variants"]
        bad_text = "Чаклун наклав гіпс навколо союзників."
        assert QualityCaseEvaluator.evaluate(case_cast, bad_text)


class TestItemInvariants:
    """Specialized tests for fantasy item terminology."""

    def test_potion_strictly_rejects_generic_beverage(self):
        """Potion must be translated as 'зілля', strictly rejecting 'напій' or 'чай'."""
        potion_cases = [c for c in ALL_CASES if c["category"] == "items" and "potion" in c["source_text"].lower()]
        assert len(potion_cases) >= 2
        for case in potion_cases:
            assert "зілля" in case["expected_uk_canonical"]
            assert any("напій" in f for f in case["forbidden_variants"])
            bad_text = "Воїн випив лікувальний напій і пішов."
            assert QualityCaseEvaluator.evaluate(case, bad_text)

    def test_broadsword_and_scroll_terminology(self):
        """Weapons and scrolls must adhere to fantasy literary standards."""
        case_sword = CASE_MAP["REG-ITEM-003"]
        case_scroll = CASE_MAP["REG-ITEM-004"]

        assert "широкий ніж" in case_sword["forbidden_variants"]
        assert "рулон" in case_scroll["forbidden_variants"]


class TestGenderAgreementInvariants:
    """Specialized tests for grammatical gender agreement."""

    def test_female_protagonists_require_feminine_past_verbs(self):
        """Female characters must agree with feminine verbs (Марія увійшла, Черрі усміхнулася)."""
        female_cases = [
            c for c in ALL_CASES
            if c["category"] == "gender_agreement"
            and c["expected_constraints"].get("required_gender") == "жіночий"
        ]
        assert len(female_cases) >= 4

        for case in female_cases:
            # Injecting masculine past verb must trigger violation
            masculine_corrupted = f"{case['expected_uk_canonical']} Він пішов геть."
            violations = QualityCaseEvaluator.evaluate(case, masculine_corrupted)
            assert any("gender" in v.lower() or "mismatch" in v.lower() for v in violations), (
                f"Expected gender mismatch violation for {case['id']}: {violations}"
            )

    def test_male_protagonists_require_masculine_past_verbs(self):
        """Male characters must agree with masculine verbs (Лорд Браян підвівся)."""
        male_cases = [
            c for c in ALL_CASES
            if c["category"] == "gender_agreement"
            and c["expected_constraints"].get("required_gender") == "чоловічий"
        ]
        assert len(male_cases) >= 2

        for case in male_cases:
            # Injecting feminine past verb must trigger violation
            feminine_corrupted = f"{case['expected_uk_canonical']} Вона пішла геть."
            violations = QualityCaseEvaluator.evaluate(case, feminine_corrupted)
            assert any("gender" in v.lower() or "feminine" in v.lower() or "mismatch" in v.lower() for v in violations), (
                f"Expected gender mismatch violation for {case['id']}: {violations}"
            )


class TestRpgTagInvariants:
    """Specialized tests for RPG class tags and system notification formatting."""

    def test_bracketed_status_tags_strictly_preserved(self):
        """Ensure '[' and ']' are never stripped or converted to raw text."""
        rpg_cases = [c for c in ALL_CASES if c["category"] == "rpg_class_tags"]
        assert len(rpg_cases) >= 5

        for case in rpg_cases:
            assert "[" in case["expected_uk_canonical"]
            assert "]" in case["expected_uk_canonical"]

            # Stripped bracket version must fail
            stripped = case["expected_uk_canonical"].replace("[", "").replace("]", "")
            violations = QualityCaseEvaluator.evaluate(case, stripped)
            assert len(violations) > 0, f"Stripped brackets unexpectedly passed for {case['id']}"

    def test_no_technical_tags_leaked_in_rpg_output(self):
        """Ensure no raw '<tag_...>' placeholders leak into translation."""
        rpg_cases = [c for c in ALL_CASES if c["category"] == "rpg_class_tags"]
        for case in rpg_cases:
            assert "<tag_" in case["forbidden_variants"]
            leaked_text = f"{case['expected_uk_canonical']} <tag_1>"
            violations = QualityCaseEvaluator.evaluate(case, leaked_text)
            assert any("<tag_" in v for v in violations)
