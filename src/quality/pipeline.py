"""
Quality Assurance and Validation Pipeline for BookTranslator.

Provides Cyrillic-Latin mixed-script sanitization, Latin homoglyph repair,
portmanteau normalization, and translation quality validation.
"""

from __future__ import annotations
import re
import logging
from typing import Optional, Dict, List

from src.domain.interfaces.quality import IQualityChecker
from src.domain.models.chunk import TranslationChunk
from src.domain.models.quality import QualityReport, ValidationIssue, IssueSeverity

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

    def validate(self, original_chunk: TranslationChunk, translated_text: str) -> QualityReport:
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

        if any(issue.severity == IssueSeverity.CRITICAL for issue in report.issues):
            report.is_passed = False

        return report


# Alias for backward compatibility
QualityChecker = QualityPipeline

