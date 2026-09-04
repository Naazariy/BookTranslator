"""
src/quality/validators/gender_agreement.py

Validator checking character grammatical gender agreement in Ukrainian translations,
specifically past-tense verbs and predicate constructions.
"""
from __future__ import annotations

import re
from typing import Optional, List, Dict, Set, Tuple, Any
from src.domain.models.segment import TranslationSegment
from src.context.builder import PromptContext
from src.parsers.segmenter import RuleBasedSentenceSegmenter
from src.quality.base import BaseValidator
from src.quality.models import QAViolation, QASeverity

# ============================================================================
# Ukrainian Past-Tense Verb Gender Agreement Heuristics & Lexical Sets
# ============================================================================

MASCULINE_PAST_COMMON_VERBS: Set[str] = {
    "сказав", "відповів", "запитав", "спитав", "промовив", "вигукнув", "шепнув", "подумав",
    "подивився", "пішов", "прийшов", "вийшов", "зайшов", "підійшов", "відійшов",
    "знайшов", "сів", "встав", "підвівся", "побіг", "поїхав", "побачив", "глянув",
    "зиркнув", "поглянув", "знав", "хотів", "зробив", "взяв", "дав", "стояв", "лежав", "сидів",
    "повернувся", "усміхнувся", "посміхнувся", "кивнув", "зітхнув", "заперечив",
    "погодився", "здивувався", "нагадав", "помітив", "відчув", "згадав", "мовив",
    "скрикнув", "скривився", "додав", "зауважив", "продовжив", "похитав", "помовчав",
    "зрозумів", "зустрів", "залишив", "залишився", "попросив", "відкрив", "закрив",
    "почав", "закінчив", "встиг", "перестав", "намагався", "спробував", "вирішив",
    "розповів", "почув", "шепотів", "пробурмотів", "вибіг",
    "був", "жив", "пив", "мив", "бив", "шив", "вів", "плив", "грів", "дув",
    "допоміг", "приніс", "заніс", "виніс", "втік", "утік", "привів", "провів", "вивів",
    "переміг", "виріс", "зберіг", "заснув", "ліг", "пробіг", "досяг", "помер",
}

FEMININE_PAST_COMMON_VERBS: Set[str] = {
    "сказала", "відповіла", "запитала", "спитала", "промовила", "вигукнула", "шепнула", "подумала",
    "подивилася", "пішла", "прийшла", "вийшла", "зайшла", "підійшла", "відійшла",
    "знайшла", "сіла", "встала", "підвелася", "побігла", "поїхала", "побачила", "глянула",
    "зиркнула", "поглянула", "знала", "хотіла", "зробила", "взяла", "дала", "стояла", "лежала", "сиділа",
    "повернулася", "усміхнулася", "посміхнулася", "кивнула", "зітхнула", "заперечила",
    "погодилася", "здивувалася", "нагадала", "помітила", "відчула", "згадала", "мовила",
    "скрикнула", "скривилася", "додала", "зауважила", "продовжила", "похитала", "помовчала",
    "зрозуміла", "зустріла", "залишила", "залишилася", "попросила", "відкрила", "закрила",
    "почала", "закінчила", "встигла", "перестала", "намагалася", "спробувала", "вирішила",
    "розповіла", "почула", "шепотіла", "пробурмотіла", "вибігла",
    "була", "допомогла", "принесла", "занесла", "винесла", "втекла", "утекла",
    "привела", "провела", "вивела", "перемогла", "виросла", "зберегла", "заснула",
    "лягла", "пробігла", "досягла", "померла",
}

NON_VERBS_MASCULINE_ENDINGS: Set[str] = {
    "київ", "львів", "харків", "чернігів", "острів", "рукав", "вплив", "порив",
    "спів", "гнів", "лев", "рев", "посів", "перелив", "зістав", "став", "норов", "покров",
    "мотив", "масив", "прорив", "приплив", "відплив"
}

NON_VERBS_FEMININE_ENDINGS: Set[str] = {
    "школа", "сила", "хвала", "смола", "скеля", "акула", "бджола", "зола", "куля",
    "могила", "жила", "стріла", "світла", "темна", "тепла", "біла", "чорна", "мила",
    "правила", "джерела", "крила", "тіла", "діла", "кола", "числа", "дзеркала",
    "весла", "горла", "стебла", "вітрила", "гасла", "масла", "сідла", "зала", "брила"
}

NEUTER_PAST_COMMON_VERBS: Set[str] = {
    "пішло", "прийшло", "вийшло", "зайшло", "знайшло", "підійшло", "відійшло",
    "сталося", "було", "сказало", "відповіло", "запитало", "почало", "закінчилося",
    "зробило", "побачило", "почуло", "зникло", "впало", "замовкло", "спало",
    "лежало", "стояло", "сиділо", "загарчало", "заплакало", "дихало",
    "допомогло", "принесло", "перемогло", "виросло", "зберегло", "заснуло",
}

NON_VERBS_NEUTER_ENDINGS: Set[str] = {
    "коло", "число", "джерело", "дзеркало", "весло", "горло", "стебло", "вітрило",
    "гасла", "масло", "сідло", "жито", "золото", "місто", "тісто", "сало", "крило",
    "світло", "тепло", "добро", "зло", "село", "помело", "свердло"
}

PREPOSITIONS: Set[str] = {
    "до", "для", "з", "із", "зі", "про", "від", "біля", "на", "у", "в", "по",
    "через", "за", "під", "над", "перед", "коло", "проти", "без", "крім", "щодо",
    "поруч", "між", "серед", "крізь", "повз", "заради", "навколо", "після", "замість",
    "завдяки", "поза", "посеред"
}

CLAUSE_DELIMITERS: Set[str] = {
    "але", "що", "щоб", "бо", "тому", "коли", "як", "якщо", "хоч", "хоча",
    "проте", "однак", "і", "та", "а"
}

SUBJECT_PRONOUNS: Set[str] = {
    "я", "ти", "він", "вона", "воно", "ми", "ви", "вони", "хто", "хтось", "ніхто", "кожен"
}

ADVERBS_AND_PARTICLES: Set[str] = {
    "не", "тихо", "швидко", "знову", "раптом", "вже", "теж", "нарешті", "тільки",
    "лише", "спокійно", "впевнено", "сердито", "радісно", "сумно", "ледве", "потім",
    "так", "ні", "одразу", "щойно", "завжди", "ніколи", "разом", "навіть",
    "ще", "тепер", "мабуть", "все", "зовсім", "точно", "навряд", "мало", "багато"
}


def _clean_apostrophes(w: str) -> str:
    return w.replace("’", "'").replace("ʼ", "'").replace("‘", "'").replace("`", "'")


def is_masculine_past_verb(token: str) -> bool:
    w = _clean_apostrophes(token.lower())
    if w in NON_VERBS_MASCULINE_ENDINGS:
        return False
    if w in MASCULINE_PAST_COMMON_VERBS:
        return True
    if len(w) >= 3 and re.search(r'[аеєиуюя]в$', w):
        return True
    if len(w) >= 5 and re.search(r'[аеєиіїоуюя]в(?:ся|сь)$', w):
        return True
    return False


def is_feminine_past_verb(token: str) -> bool:
    w = _clean_apostrophes(token.lower())
    if w in NON_VERBS_FEMININE_ENDINGS:
        return False
    if w in FEMININE_PAST_COMMON_VERBS:
        return True
    if len(w) >= 4 and re.search(r'(?:[аеєиіїоуюя]|[йш])ла$', w):
        return True
    if len(w) >= 6 and re.search(r'(?:[аеєиіїоуюя]|[йш])ла(?:ся|сь)$', w):
        return True
    return False


def is_neuter_past_verb(token: str) -> bool:
    w = _clean_apostrophes(token.lower())
    if w in NON_VERBS_NEUTER_ENDINGS:
        return False
    if w in NEUTER_PAST_COMMON_VERBS:
        return True
    if len(w) >= 4 and re.search(r'(?:[аеєиіїоуюя]|[йш])ло$', w):
        return True
    if len(w) >= 6 and re.search(r'(?:[аеєиіїоуюя]|[йш])ло(?:ся|сь)$', w):
        return True
    return False


class GenderAgreementValidator(BaseValidator):
    """
    Validates grammatical gender agreement between named characters and past-tense verbs.
    Emits WARNING when masculine/feminine/neuter mismatches occur in subject-predicate clauses.
    """
    name: str = "GenderAgreementValidator"
    description: str = "Checks character grammatical gender agreement with past-tense verbs"
    default_severity: QASeverity = QASeverity.WARNING

    def validate(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        translated_text: Optional[str] = None,
        glossary: Optional[List[Any]] = None,
        chunk: Any = None,
        **kwargs: Any,
    ) -> List[QAViolation]:
        violations: List[QAViolation] = []
        target_text = self.extract_target_text(segment, translated_text=translated_text).strip()
        if not target_text:
            return violations

        active_glossary = self._gather_glossary(segment, context, glossary, chunk=chunk)
        characters_with_gender = [
            (item, self._normalize_gender(getattr(item, "grammatical_gender", None)))
            for item in (active_glossary or [])
            if self._normalize_gender(getattr(item, "grammatical_gender", None)) is not None
        ]

        if not characters_with_gender:
            return violations

        translated_sents = RuleBasedSentenceSegmenter.split_sentences(target_text)
        reported_mismatches: Set[Tuple[str, str]] = set()

        for sent_str in translated_sents:
            token_matches = list(re.finditer(r"[а-яіїєґА-ЯІЇЄҐa-zA-Z'’ʼ‘`]+", sent_str))
            tokens = [m.group(0) for m in token_matches]

            for char_item, gender in characters_with_gender:
                target_name = getattr(char_item, "target_term", None) or getattr(char_item, "canonical_target", None)
                source_name = getattr(char_item, "source_term", None) or getattr(char_item, "source_name", None)

                names_to_check = []
                if target_name:
                    names_to_check.append(target_name)
                if source_name and source_name != target_name:
                    names_to_check.append(source_name)

                for full_name in names_to_check:
                    name_parts = [p.lower() for p in full_name.split() if len(p) >= 3]
                    if not name_parts and len(full_name.strip()) >= 2:
                        name_parts = [full_name.strip().lower()]
                    if not name_parts:
                        continue

                    for idx, tok in enumerate(tokens):
                        tok_l = tok.lower()

                        # Check if preceded by preposition (prepositional object, not subject)
                        if idx > 0 and tokens[idx - 1].lower() in PREPOSITIONS:
                            continue

                        matches_name = False
                        is_oblique = False

                        tok_clean = _clean_apostrophes(tok_l)
                        for part in name_parts:
                            part_clean = _clean_apostrophes(part)
                            nom_ends_in_consonant = part_clean[-1] not in "аяеєієюїо"
                            nom_ends_in_a_ya = part_clean[-1] in "ая"
                            nom_ends_in_o = part_clean[-1] == "о"

                            if tok_clean == part_clean:
                                matches_name = True
                                break
                            elif len(part_clean) >= 4:
                                stem = part_clean[:-1] if (nom_ends_in_a_ya or nom_ends_in_o) else part_clean
                                if tok_clean.startswith(stem):
                                    if nom_ends_in_consonant and (tok_clean[-1] in "аяуюіеє" or tok_clean.endswith(("ом", "ем", "єм", "ові", "єві"))):
                                        is_oblique = True
                                    elif nom_ends_in_a_ya and (tok_clean[-1] in "иуюіїоеє" or tok_clean.endswith(("ою", "єю"))):
                                        is_oblique = True
                                    elif nom_ends_in_o and (tok_clean[-1] in "ауіе" or tok_clean.endswith(("ом", "ові", "еві"))):
                                        is_oblique = True
                                    matches_name = True
                                    break

                        if not matches_name or is_oblique:
                            continue

                        # Subject candidate at `idx`. Inspect candidate verbs:
                        # 1. Verb following subject: [Subject] [Adverb/Particle]* [Verb]
                        candidate_verb_after = None
                        v_idx = idx + 1
                        while v_idx < len(tokens) and tokens[v_idx].lower() in ADVERBS_AND_PARTICLES:
                            v_idx += 1

                        if v_idx < len(tokens):
                            tok_cand = tokens[v_idx]
                            if tok_cand.lower() not in CLAUSE_DELIMITERS:
                                span_between = sent_str[token_matches[idx].end():token_matches[v_idx].start()]
                                if not any(ch in span_between for ch in "—–-»\"”'!?…:;."):
                                    if is_masculine_past_verb(tok_cand) or is_feminine_past_verb(tok_cand) or is_neuter_past_verb(tok_cand):
                                        candidate_verb_after = tok_cand

                        # 2. Verb preceding subject (dialogue reporting or clause-initial): [Verb] [Adverb/Particle]* [Subject]
                        candidate_verb_before = None
                        v_idx = idx - 1
                        while v_idx >= 0 and tokens[v_idx].lower() in ADVERBS_AND_PARTICLES:
                            v_idx -= 1

                        if v_idx >= 0:
                            tok_cand = tokens[v_idx]
                            if (is_masculine_past_verb(tok_cand) or is_feminine_past_verb(tok_cand) or is_neuter_past_verb(tok_cand)) and \
                               tok_cand.lower() not in CLAUSE_DELIMITERS and tok_cand.lower() not in PREPOSITIONS:
                                span_between = sent_str[token_matches[v_idx].end():token_matches[idx].start()]
                                if not any(ch in span_between for ch in "»\"”!?…:;."):
                                    is_valid_inversion = False
                                    if v_idx == 0:
                                        is_valid_inversion = True
                                    elif tokens[v_idx - 1].lower() in CLAUSE_DELIMITERS:
                                        is_valid_inversion = True
                                    else:
                                        preceding_snip = sent_str[:token_matches[v_idx].start()].rstrip()
                                        if any(ch in preceding_snip[-6:] for ch in "—–-»\"”'!?"):
                                            is_valid_inversion = True
                                        elif all(tokens[i].lower() in ADVERBS_AND_PARTICLES for i in range(v_idx)):
                                            is_valid_inversion = True
                                        elif tokens[0].lower() in PREPOSITIONS:
                                            is_valid_inversion = True

                                    if is_valid_inversion:
                                        candidate_verb_before = tok_cand

                        verbs_to_evaluate = []
                        if candidate_verb_after:
                            verbs_to_evaluate.append(candidate_verb_after)
                        elif candidate_verb_before:
                            verbs_to_evaluate.append(candidate_verb_before)

                        for v_tok in verbs_to_evaluate:
                            mismatch = False
                            if gender == "жіночий":
                                if is_masculine_past_verb(v_tok) or is_neuter_past_verb(v_tok):
                                    mismatch = True
                            elif gender == "чоловічий":
                                if is_feminine_past_verb(v_tok) or is_neuter_past_verb(v_tok):
                                    mismatch = True
                            elif gender == "середній":
                                if is_masculine_past_verb(v_tok) or is_feminine_past_verb(v_tok):
                                    mismatch = True

                            if mismatch:
                                display_name = target_name or full_name
                                mismatch_key = (display_name, v_tok.lower())
                                if mismatch_key not in reported_mismatches:
                                    reported_mismatches.add(mismatch_key)
                                    violations.append(
                                        self.make_violation(
                                            rule_code="GenderAgreementMismatch",
                                            message=(
                                                f"Potential grammatical gender mismatch: character "
                                                f"'{display_name}' ({gender} рід) appears with "
                                                f"conflicting past-tense verb '{v_tok}'."
                                            ),
                                            severity=QASeverity.WARNING,
                                            target_snippet=sent_str,
                                            forbidden_form=v_tok,
                                            suggested_fix=f"Change verb '{v_tok}' to {gender} form.",
                                            metadata={"character": display_name, "gender": gender, "verb": v_tok},
                                        )
                                    )

        return violations

    def _gather_glossary(
        self,
        segment: Any,
        context: Optional[PromptContext],
        glossary: Optional[List[Any]],
        chunk: Any = None,
    ) -> List[Any]:
        if glossary is not None:
            return glossary
        if context:
            if hasattr(context, "active_entities") and context.active_entities:
                return context.active_entities
            if hasattr(context, "glossary") and context.glossary:
                return context.glossary
        obj = chunk or segment
        if hasattr(obj, "context") and hasattr(obj.context, "active_glossary"):
            return obj.context.active_glossary
        if hasattr(obj, "glossary") and obj.glossary:
            return obj.glossary
        return []

    def _normalize_gender(self, g: Optional[str]) -> Optional[str]:
        if not g:
            return None
        g_str = str(g).strip().lower()
        if g_str in ("чоловічий", "masculine", "male", "ч", "m"):
            return "чоловічий"
        if g_str in ("жіночий", "feminine", "female", "ж", "f"):
            return "жіночий"
        if g_str in ("середній", "neuter", "с", "n"):
            return "середній"
        return None
