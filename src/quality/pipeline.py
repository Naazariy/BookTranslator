"""
src/quality/pipeline.py

Quality Assurance and Validation Pipeline for BookTranslator V2.
Orchestrates the 8 modular validators, computes aggregate quality scores,
provides Cyrillic-Latin mixed-script sanitization, and ensures full backward
compatibility for existing test suites.
"""

from __future__ import annotations
import re
import logging
from typing import Optional, Dict, List, Set, Tuple, Any, Union
from uuid import UUID, uuid4

from src.domain.interfaces.quality import IQualityChecker
from src.domain.models.chunk import TranslationChunk
from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.domain.models.quality import ValidationIssue, IssueSeverity
from src.domain.models.knowledge import GlossaryItem
from src.context.builder import PromptContext
from src.quality.models import QASeverity, QAViolation, QAReport
from src.quality.base import BaseValidator
from src.quality.validators import (
    DEFAULT_VALIDATORS,
    EmptyTranslationValidator,
    CompletenessValidator,
    EntityConsistencyValidator,
    GlossaryConsistencyValidator,
    GenderAgreementValidator,
    NumbersAndUnitsValidator,
    ControlTokenValidator,
    DialogueIntegrityValidator,
)
from src.quality.validators.gender_agreement import (
    MASCULINE_PAST_COMMON_VERBS,
    FEMININE_PAST_COMMON_VERBS,
    NEUTER_PAST_COMMON_VERBS,
    NON_VERBS_MASCULINE_ENDINGS,
    NON_VERBS_FEMININE_ENDINGS,
    NON_VERBS_NEUTER_ENDINGS,
    PREPOSITIONS,
    CLAUSE_DELIMITERS,
    SUBJECT_PRONOUNS,
    ADVERBS_AND_PARTICLES,
    _clean_apostrophes,
    is_masculine_past_verb,
    is_feminine_past_verb,
    is_neuter_past_verb,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Cyrillic-Latin Homoglyphs & Portmanteau Normalization
# ============================================================================

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

        word_lower = word.lower()
        if "смачнissimo" in word_lower or "смачниссимо" in word_lower:
            return "Смакота" if word[0].isupper() else "смакота"
        if "гарнissimo" in word_lower:
            return "Прегарно" if word[0].isupper() else "прегарно"
        if "чудовissimo" in word_lower:
            return "Чудово" if word[0].isupper() else "чудово"

        suffix_match = re.search(r'([а-яіїєґА-ЯІЇЄҐ]+)(?:issimo|able|ed|ing|like|folk|ness|tion|ment)$', word, re.IGNORECASE)
        if suffix_match:
            stem = suffix_match.group(1)
            if stem.lower() in ("смачн", "смач"):
                return "Смакота" if word[0].isupper() else "смакота"
            elif stem.lower() in ("гарн", "чудов", "прекрасн"):
                return f"Пре{stem.lower()}о" if word[0].isupper() else f"пре{stem.lower()}о"
            return re.sub(r'[A-Za-z]+$', '', word)

        cyr_chars = re.findall(r'[а-яіїєґА-ЯІЇЄҐ]', word)
        lat_chars = re.findall(r'[a-zA-Z]', word)

        if len(cyr_chars) >= len(lat_chars):
            repaired = [LATIN_TO_CYRILLIC_HOMOGLYPHS.get(ch, ch) for ch in word]
            repaired_str = "".join(repaired)
            return re.sub(r'(?<=[а-яіїєґА-ЯІЇЄҐ])[a-zA-Z]+$', '', repaired_str)
        else:
            repaired = [CYRILLIC_TO_LATIN_HOMOGLYPHS.get(ch, ch) for ch in word]
            return "".join(repaired)

    return mixed_word_pattern.sub(repair_word, text)


# ============================================================================
# Quality Pipeline Orchestrator
# ============================================================================

class QualityPipeline(IQualityChecker):
    """
    Modular Quality Assurance Pipeline for BookTranslator V2.
    Orchestrates the sequential execution of modular validators,
    calculates aggregate diagnostic quality scores, and sanitizes output text.
    """

    def __init__(
        self,
        sanitize_mixed_script: bool = True,
        validators: Optional[List[BaseValidator]] = None,
    ):
        self.sanitize_mixed_script = sanitize_mixed_script
        self.validators: List[BaseValidator] = (
            validators if validators is not None else [cls() for cls in DEFAULT_VALIDATORS]
        )

    def post_process(self, translated_text: str) -> str:
        """Cleans and normalizes translation text."""
        if not translated_text:
            return ""

        cleaned = translated_text
        if self.sanitize_mixed_script:
            cleaned = sanitize_mixed_script_words(cleaned)

        return cleaned

    def clean_text(self, text: str) -> str:
        """Alias for post_process."""
        return self.post_process(text)

    def validate_segment(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        translated_text: Optional[str] = None,
        chunk: Any = None,
        **kwargs: Any,
    ) -> QAReport:
        """
        Executes modular QA validation for a TranslationSegment.
        Sequentially invokes the 8 specialized validators and checks for mixed scripts.
        """
        seg_id = getattr(segment, "id", uuid4())
        book_id = getattr(segment, "book_id", None)
        report = QAReport(
            segment_id=seg_id,
            book_id=book_id,
            is_valid=True,
            score=1.0,
            status=SegmentStatus.VALIDATING.value,
        )

        target_text = translated_text
        if target_text is None:
            target_text = (
                segment.final_translation or
                segment.refined_translation or
                segment.translated_text or
                segment.draft_translation or
                ""
            )

        # 1. Run Empty Translation Validator first (short-circuit on empty)
        empty_val = next((v for v in self.validators if isinstance(v, EmptyTranslationValidator)), None)
        if empty_val:
            empty_violations = empty_val.validate(
                segment,
                context=context,
                translated_text=target_text,
                chunk=chunk,
                **kwargs
            )
            if empty_violations:
                for v in empty_violations:
                    report.add_violation(v)
                report.is_valid = False
                report.score = 0.0
                report.status = SegmentStatus.REVIEW_REQUIRED.value
                return report

        # 2. Check for leftover mixed-script words (legacy compatibility)
        mixed_word_pattern = re.compile(
            r'\b(?=[а-яіїєґА-ЯІЇЄҐa-zA-Z\']*[а-яіїєґА-ЯІЇЄҐ])(?=[а-яіїєґА-ЯІЇЄҐa-zA-Z\']*[a-zA-Z])[а-яіїєґА-ЯІЇЄҐa-zA-Z\']+\b'
        )
        mixed_matches = mixed_word_pattern.findall(target_text)
        if mixed_matches:
            mixed_viol = QAViolation(
                validator_name="MixedScriptWord",
                rule_code="MixedScriptWord",
                severity=QASeverity.WARNING,
                message=f"Detected mixed Cyrillic-Latin word(s): {', '.join(mixed_matches[:5])}",
                target_snippet=", ".join(mixed_matches[:5]),
            )
            report.add_violation(mixed_viol)

        # 3. Run all remaining modular validators
        for validator in self.validators:
            if isinstance(validator, EmptyTranslationValidator):
                continue

            try:
                violations = validator.validate(
                    segment,
                    context=context,
                    translated_text=target_text,
                    chunk=chunk,
                    **kwargs
                )
                for violation in violations:
                    report.add_violation(violation)
            except Exception as e:
                logger.error(f"Error executing validator {validator.name} on segment {seg_id}: {e}")

        # Compute validity: valid if and only if NO critical violations
        report.is_valid = not report.has_critical()
        report.score = max(0.0, min(1.0, report.score))
        report.status = SegmentStatus.ACCEPTED.value if report.is_valid else SegmentStatus.REVIEW_REQUIRED.value

        return report

    def validate(
        self,
        original_chunk: Union[TranslationChunk, TranslationSegment, None],
        translated_text: Optional[str] = None,
        glossary: Optional[List[GlossaryItem]] = None,
        context: Optional[PromptContext] = None,
        **kwargs: Any,
    ) -> QAReport:
        """
        Polymorphic validation entry point supporting both TranslationSegment (V2)
        and TranslationChunk (V1 legacy test suites).
        """
        if isinstance(original_chunk, TranslationSegment):
            target_seg = original_chunk
            if translated_text is not None:
                target_seg = original_chunk.model_copy()
                target_seg.refined_translation = translated_text
            return self.validate_segment(target_seg, context=context, glossary=glossary, **kwargs)

        if original_chunk is None:
            ephemeral_segment = TranslationSegment(
                id=uuid4(),
                book_id=uuid4(),
                chapter_id=uuid4(),
                paragraph_id=uuid4(),
                source_text="",
                draft_translation=translated_text,
                refined_translation=translated_text,
                final_translation=translated_text,
            )
            return self.validate_segment(
                ephemeral_segment,
                context=context,
                translated_text=translated_text,
                glossary=glossary,
                chunk=None,
                **kwargs
            )

        # Legacy TranslationChunk handling: extract only target sentences (excluding context sentences)
        source_sents = getattr(original_chunk, "source_sentences", []) or []
        context_ids = set(getattr(original_chunk, "context_sentence_ids", []) or [])
        target_ids = set(getattr(original_chunk, "target_sentence_ids", []) or [])

        if target_ids:
            actual_sents = [s for s in source_sents if getattr(s, "id", None) in target_ids]
        elif context_ids:
            actual_sents = [s for s in source_sents if getattr(s, "id", None) not in context_ids]
        else:
            actual_sents = source_sents

        source_text = " ".join(
            s.original_text for s in actual_sents if getattr(s, "original_text", None)
        ).strip()
        if not source_text and hasattr(original_chunk, "source_text"):
            source_text = getattr(original_chunk, "source_text") or ""

        chunk_id = getattr(original_chunk, "id", None)
        if not isinstance(chunk_id, UUID):
            chunk_id = uuid4()
        book_id = getattr(original_chunk, "book_id", None)
        if not isinstance(book_id, UUID):
            book_id = uuid4()
        chapter_id = getattr(original_chunk, "chapter_id", None)
        if not isinstance(chapter_id, UUID):
            chapter_id = uuid4()

        sentence_ids = [s.id for s in actual_sents if hasattr(s, "id")]

        ephemeral_segment = TranslationSegment(
            id=chunk_id,
            book_id=book_id,
            chapter_id=chapter_id,
            paragraph_id=chunk_id,
            source_text=source_text,
            draft_translation=translated_text,
            refined_translation=translated_text,
            final_translation=translated_text,
            sentence_ids=sentence_ids,
        )

        return self.validate_segment(
            ephemeral_segment,
            context=context,
            translated_text=translated_text,
            glossary=glossary,
            chunk=original_chunk,
            **kwargs
        )


# Alias for backward compatibility
QualityChecker = QualityPipeline
