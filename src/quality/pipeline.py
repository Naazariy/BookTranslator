"""
Quality Assurance and Validation Pipeline for BookTranslator.

Provides Cyrillic-Latin mixed-script sanitization, Latin homoglyph repair,
portmanteau normalization, and translation quality validation.
"""

from __future__ import annotations
import re
import logging
from typing import Optional, Dict, List, Set, Tuple

from src.domain.interfaces.quality import IQualityChecker
from src.domain.models.chunk import TranslationChunk
from src.domain.models.quality import QualityReport, ValidationIssue, IssueSeverity
from src.domain.models.knowledge import GlossaryItem

logger = logging.getLogger(__name__)

# Latin to Ukrainian Cyrillic homoglyphs
LATIN_TO_CYRILLIC_HOMOGLYPHS: Dict[str, str] = {
    'a': 'а', 'A': 'А',
    'c': 'с', 'C': 'С',
    'e': 'е', 'E': 'Е',
    'i': 'і', 'I': 'І',
    'o': 'о', 'O': 'О',
    'p': 'р', 'P': 'Р',
    's': 'с', 'S': 'С',
    'x': 'х', 'X': 'Х',
    'y': 'у', 'Y': 'У',
    'j': 'ј', 'J': 'Ј',
    'B': 'В', 'H': 'Н', 'K': 'К', 'M': 'М', 'T': 'Т',
    'ï': 'ї', 'Ï': 'Ї'
}

# Cyrillic to Latin homoglyphs (for repairing Latin brand names or words)
CYRILLIC_TO_LATIN_HOMOGLYPHS: Dict[str, str] = {
    'а': 'a', 'А': 'A',
    'с': 'c', 'С': 'C',
    'е': 'e', 'Е': 'E',
    'і': 'i', 'І': 'I',
    'о': 'o', 'О': 'O',
    'р': 'p', 'Р': 'P',
    'с': 's', 'С': 'S',
    'х': 'x', 'Х': 'X',
    'у': 'y', 'У': 'Y',
    'В': 'B', 'Н': 'H', 'К': 'K', 'М': 'M', 'Т': 'T'
}

# Common hybrid portmanteau replacements
KNOWN_PORTMANTEAU_FIXES: Dict[str, str] = {
    "смачнissimo": "смакота",
    "Смачнissimo": "Смакота",
    "смачниссимо": "смакота",
    "Смачниссимо": "Смакота",
    "смачненькissimo": "дуже смачненько",
    "Смачненькissimo": "Дуже смачненько",
    "гарнissimo": "прегарно",
    "Гарнissimo": "Прегарно",
    "прекраснissimo": "прекрасно",
    "Прекраснissimo": "Прекрасно",
    "чудовissimo": "чудово",
    "Чудовissimo": "Чудово",
}


def sanitize_mixed_script_words(text: str) -> str:
    """
    Cleans mixed Cyrillic-Latin words, repairs homoglyph leakage,
    strips unwanted foreign suffixes from Ukrainian stems, and normalizes
    hybrid portmanteaus (e.g. 'Смачнissimo' -> 'Смакота').
    """
    if not text:
        return ""

    # 1. Direct known portmanteau substitutions
    for port, replacement in KNOWN_PORTMANTEAU_FIXES.items():
        if port in text:
            text = re.sub(r'\b' + re.escape(port) + r'\b', replacement, text)

    # 2. Match any word containing both Cyrillic and Latin characters
    mixed_word_pattern = re.compile(
        r'\b(?=[а-яіїєґА-ЯІЇЄҐa-zA-Z\']*[а-яіїєґА-ЯІЇЄҐ])(?=[а-яіїєґА-ЯІЇЄҐa-zA-Z\']*[a-zA-Z])[а-яіїєґА-ЯІЇЄҐa-zA-Z\']+\b'
    )

    def repair_word(match: re.Match) -> str:
        word = match.group(0)

        # Check for direct portmanteau match case-insensitively
        word_lower = word.lower()
        if "смачнissimo" in word_lower or "смачниссимо" in word_lower:
            return "Смакота" if word[0].isupper() else "смакота"
        if "гарнissimo" in word_lower:
            return "Прегарно" if word[0].isupper() else "прегарно"
        if "чудовissimo" in word_lower:
            return "Чудово" if word[0].isupper() else "чудово"

        # Check if word ends with common Latin superlative or foreign suffixes
        suffix_match = re.search(r'([а-яіїєґА-ЯІЇЄҐ]+)(?:issimo|able|ed|ing|like|folk|ness|tion|ment)$', word, re.IGNORECASE)
        if suffix_match:
            stem = suffix_match.group(1)
            if stem.lower() in ("смачн", "смач"):
                return "Смакота" if word[0].isupper() else "смакота"
            elif stem.lower() in ("гарн", "чудов", "прекрасн"):
                return f"Пре{stem.lower()}о" if word[0].isupper() else f"пре{stem.lower()}о"
            # Strip the Latin suffix
            return re.sub(r'[A-Za-z]+$', '', word)

        cyr_chars = re.findall(r'[а-яіїєґА-ЯІЇЄҐ]', word)
        lat_chars = re.findall(r'[a-zA-Z]', word)

        # Predominantly Cyrillic word with stray Latin homoglyphs (e.g. 'мiсто', 'свiт', 'рiка')
        if len(cyr_chars) >= len(lat_chars):
            repaired = [LATIN_TO_CYRILLIC_HOMOGLYPHS.get(ch, ch) for ch in word]
            # Strip trailing non-homoglyph Latin debris if any
            repaired_str = "".join(repaired)
            return re.sub(r'(?<=[а-яіїєґА-ЯІЇЄҐ])[a-zA-Z]+$', '', repaired_str)
        else:
            # Predominantly Latin word with stray Cyrillic homoglyphs
            repaired = [CYRILLIC_TO_LATIN_HOMOGLYPHS.get(ch, ch) for ch in word]
            return "".join(repaired)

    return mixed_word_pattern.sub(repair_word, text)


# ============================================================================
# Ukrainian Past-Tense Verb Gender Agreement Heuristics
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
    # Exclude -ів from open regex to avoid matching genitive plural nouns (кроків, слів, років)
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


class QualityPipeline(IQualityChecker):
    """
    Validates translated chunks and post-processes translation output
    to guarantee formatting and linguistic integrity.
    """

    def __init__(self, sanitize_mixed_script: bool = True):
        self.sanitize_mixed_script = sanitize_mixed_script

    def post_process(self, translated_text: str) -> str:
        """
        Cleans and normalizes translation text.
        """
        if not translated_text:
            return ""

        cleaned = translated_text
        if self.sanitize_mixed_script:
            cleaned = sanitize_mixed_script_words(cleaned)

        return cleaned

    def clean_text(self, text: str) -> str:
        """Alias for post_process."""
        return self.post_process(text)

    def validate(
        self,
        original_chunk: TranslationChunk,
        translated_text: str,
        glossary: Optional[List[GlossaryItem]] = None
    ) -> QualityReport:
        """
        Validates the translation quality of a chunk.
        """
        report = QualityReport(is_passed=True, translation_score=1.0)

        # 1. Check for empty translation
        if not translated_text or not translated_text.strip():
            report.is_passed = False
            report.translation_score = 0.0
            report.issues.append(ValidationIssue(
                rule_name="EmptyTranslation",
                severity=IssueSeverity.CRITICAL,
                message="The translated text is empty."
            ))
            return report

        # 2. Check for leftover mixed-script words
        mixed_word_pattern = re.compile(
            r'\b(?=[а-яіїєґА-ЯІЇЄҐa-zA-Z\']*[а-яіїєґА-ЯІЇЄҐ])(?=[а-яіїєґА-ЯІЇЄҐa-zA-Z\']*[a-zA-Z])[а-яіїєґА-ЯІЇЄҐa-zA-Z\']+\b'
        )
        mixed_matches = mixed_word_pattern.findall(translated_text)
        if mixed_matches:
            report.issues.append(ValidationIssue(
                rule_name="MixedScriptWord",
                severity=IssueSeverity.WARNING,
                message=f"Detected mixed Cyrillic-Latin word(s): {', '.join(mixed_matches[:5])}"
            ))
            report.translation_score = max(0.0, report.translation_score - (0.1 * len(mixed_matches)))

        # 3. Check for leaked or unmapped technical tags (e.g. <tag_1>)
        leaked_tags = re.findall(r'</?tag_\d+>', translated_text)
        if leaked_tags:
            report.issues.append(ValidationIssue(
                rule_name="LeakedTechnicalTag",
                severity=IssueSeverity.WARNING,
                message=f"Detected unresolved technical tags in translation: {', '.join(set(leaked_tags))}"
            ))
            report.translation_score = max(0.0, report.translation_score - 0.2)

        # 4. Check for leaked model control tokens (<|...|>)
        leaked_control_tokens = re.findall(r'<\|[^|>\n]{1,40}\|>', translated_text)
        if leaked_control_tokens:
            report.issues.append(ValidationIssue(
                rule_name="ResidualControlToken",
                severity=IssueSeverity.CRITICAL,
                message=f"Detected residual model control tokens in translation: {', '.join(set(leaked_control_tokens))}"
            ))
            report.translation_score = max(0.0, report.translation_score - 0.3)

        from src.parsers.segmenter import RuleBasedSentenceSegmenter
        from difflib import SequenceMatcher

        translated_sents = RuleBasedSentenceSegmenter.split_sentences(translated_text) if translated_text else []

        # 5. Check for consecutive duplicate sentences (similarity > 0.8)
        for i in range(len(translated_sents) - 1):
            s1 = translated_sents[i].strip()
            s2 = translated_sents[i + 1].strip()
            if len(s1) > 5 and len(s2) > 5:
                ratio = SequenceMatcher(None, s1, s2).ratio()
                if ratio > 0.8:
                    report.issues.append(ValidationIssue(
                        rule_name="ConsecutiveDuplicateSentences",
                        severity=IssueSeverity.WARNING,
                        message=f"Detected consecutive near-duplicate sentences (similarity {ratio:.2f}): '{s1}' and '{s2}'"
                    ))
                    report.translation_score = max(0.0, report.translation_score - 0.2)

        # 6. Check for substantial sentence count drop (< 0.7 * expected)
        target_ids = None
        if hasattr(original_chunk, "target_sentence_ids"):
            try:
                raw_target_ids = getattr(original_chunk, "target_sentence_ids", None)
                target_ids = set(raw_target_ids) if raw_target_ids else None
            except AttributeError:
                target_ids = None

        source_sents = []
        if hasattr(original_chunk, "source_sentences"):
            try:
                source_sents = getattr(original_chunk, "source_sentences", None) or []
            except AttributeError:
                source_sents = []

        context_ids = set()
        if hasattr(original_chunk, "context_sentence_ids"):
            try:
                raw_context_ids = getattr(original_chunk, "context_sentence_ids", None)
                if raw_context_ids:
                    context_ids = set(raw_context_ids)
            except AttributeError:
                context_ids = set()

        if target_ids is not None and source_sents:
            expected_count = len([s for s in source_sents if getattr(s, "id", None) in target_ids])
        elif source_sents:
            expected_count = len([s for s in source_sents if getattr(s, "id", None) not in context_ids])
        else:
            expected_count = 0

        if expected_count > 1 and len(translated_sents) < 0.7 * expected_count:
            report.issues.append(ValidationIssue(
                rule_name="SentenceCountDrop",
                severity=IssueSeverity.CRITICAL,
                message=f"Translated sentence count ({len(translated_sents)}) dropped significantly below expected ({expected_count}, ratio: {len(translated_sents)/expected_count:.2f} < 0.7)."
            ))
            report.translation_score = max(0.0, report.translation_score - 0.4)

        # 7. Check for grammatical gender agreement with past-tense verbs (heuristic WARNING)
        active_glossary = glossary
        if active_glossary is None:
            if hasattr(original_chunk, "context") and hasattr(original_chunk.context, "active_glossary"):
                active_glossary = original_chunk.context.active_glossary
            elif hasattr(original_chunk, "glossary"):
                active_glossary = getattr(original_chunk, "glossary")
            else:
                active_glossary = []

        def _normalize_gender(g: Optional[str]) -> Optional[str]:
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

        characters_with_gender = [
            (item, _normalize_gender(getattr(item, "grammatical_gender", None)))
            for item in (active_glossary or [])
            if _normalize_gender(getattr(item, "grammatical_gender", None)) is not None
        ]

        if characters_with_gender and translated_sents:
            reported_mismatches: Set[Tuple[str, str]] = set()
            for sent_str in translated_sents:
                token_matches = list(re.finditer(r"[а-яіїєґА-ЯІЇЄҐa-zA-Z'’ʼ‘`]+", sent_str))
                tokens = [m.group(0) for m in token_matches]

                for char_item, gender in characters_with_gender:
                    names_to_check = []
                    if char_item.target_term:
                        names_to_check.append(char_item.target_term)
                    if char_item.source_term and char_item.source_term != char_item.target_term:
                        names_to_check.append(char_item.source_term)

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
                                    # Ensure no dialogue punctuation or clause break separates subject from verb
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
                                    mismatch_key = (char_item.target_term, v_tok.lower())
                                    if mismatch_key not in reported_mismatches:
                                        reported_mismatches.add(mismatch_key)
                                        report.issues.append(ValidationIssue(
                                            rule_name="GenderAgreementMismatch",
                                            severity=IssueSeverity.WARNING,
                                            message=(
                                                f"Potential grammatical gender mismatch: character "
                                                f"'{char_item.target_term}' ({gender} рід) appears with "
                                                f"conflicting past-tense verb '{v_tok}'."
                                            )
                                        ))
                                        report.translation_score = max(0.0, report.translation_score - 0.1)

        if any(issue.severity == IssueSeverity.CRITICAL for issue in report.issues):
            report.is_passed = False

        return report


# Alias for backward compatibility
QualityChecker = QualityPipeline

