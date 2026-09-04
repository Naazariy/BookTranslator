"""
src/quality/validators/dialogue_integrity.py

Validator checking dialogue formatting, punctuation integrity, quotation marks,
and preservation of speaker turns in literary Ukrainian translations.
"""
from __future__ import annotations

import re
from typing import Optional, List, Any
from src.domain.models.segment import TranslationSegment
from src.context.builder import PromptContext
from src.quality.base import BaseValidator
from src.quality.models import QAViolation, QASeverity


class DialogueIntegrityValidator(BaseValidator):
    """
    Validates literary dialogue formatting:
    - Checks proper dialogue dash (— instead of ASCII hyphen -)
    - Checks quotation mark typography (Ukrainian chevrons «...» instead of straight quotes ")
    - Detects unpaired quotation marks
    - Checks preservation of distinct speaker turns across multiple lines
    """
    name: str = "DialogueIntegrityValidator"
    description: str = "Verifies dialogue dash standards, chevron typography, and speaker turn preservation"
    default_severity: QASeverity = QASeverity.WARNING

    # Regex for lines starting with ASCII hyphen direct speech
    ASCII_DIALOGUE_HYPHEN_PATTERN = re.compile(r'(?:^|\n)-\s+[а-яіїєґА-ЯІЇЄҐa-zA-Z]')

    def validate(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        translated_text: Optional[str] = None,
        **kwargs: Any,
    ) -> List[QAViolation]:
        violations: List[QAViolation] = []
        source_text = self.extract_source_text(segment).strip()
        target_text = self.extract_target_text(segment, translated_text=translated_text).strip()

        if not target_text or not source_text:
            return violations

        # 1. Dialogue Dash Standards
        if self.ASCII_DIALOGUE_HYPHEN_PATTERN.search(target_text):
            violations.append(
                self.make_violation(
                    rule_code="DIALOGUE_HYPHEN_INSTEAD_OF_DASH",
                    message="Direct speech begins with ASCII hyphen '-' instead of typographic dialogue em-dash '—'.",
                    severity=QASeverity.WARNING,
                    target_snippet=target_text[:80],
                    suggested_fix="Replace leading hyphen with proper dialogue dash '— '.",
                )
            )

        # 2. Unpaired Quotation Marks
        open_chevrons = target_text.count("«")
        close_chevrons = target_text.count("»")
        if open_chevrons != close_chevrons:
            violations.append(
                self.make_violation(
                    rule_code="UNPAIRED_QUOTATION_MARKS",
                    message=f"Unpaired Ukrainian chevrons detected (opened: {open_chevrons}, closed: {close_chevrons}).",
                    severity=QASeverity.WARNING,
                    target_snippet=target_text[:100],
                    suggested_fix="Ensure every opening chevron '«' has a matching closing chevron '»'.",
                )
            )

        ascii_quote_count = target_text.count('"')
        if ascii_quote_count % 2 != 0:
            violations.append(
                self.make_violation(
                    rule_code="UNPAIRED_QUOTATION_MARKS",
                    message=f"Unpaired straight quotation marks '\"' detected (count: {ascii_quote_count}).",
                    severity=QASeverity.WARNING,
                    target_snippet=target_text[:100],
                    suggested_fix="Ensure every quotation mark is properly paired.",
                )
            )

        # 3. Straight ASCII Quotes vs Chevrons (INFO)
        if ascii_quote_count > 0:
            violations.append(
                self.make_violation(
                    rule_code="ASCII_QUOTES_IN_DIALOGUE",
                    message="Translation uses straight ASCII quotes '\"' instead of Ukrainian chevrons «...».",
                    severity=QASeverity.INFO,
                    target_snippet=target_text[:100],
                    suggested_fix="Replace ASCII quotes with typographic chevrons «...».",
                )
            )

        # 4. Preservation of Speaker Turns
        src_dialogue_lines = [
            line.strip() for line in source_text.splitlines()
            if line.strip().startswith(('—', '–', '-', '"', '“', '«'))
        ]
        tgt_dialogue_lines = [
            line.strip() for line in target_text.splitlines()
            if line.strip().startswith(('—', '–', '-', '"', '“', '«'))
        ]

        if len(src_dialogue_lines) > 1 and len(tgt_dialogue_lines) <= 1:
            violations.append(
                self.make_violation(
                    rule_code="SPEAKER_TURN_COLLAPSE",
                    message=(
                        f"Multiple speaker turns in source ({len(src_dialogue_lines)} lines) "
                        f"were collapsed into a single block ({len(tgt_dialogue_lines)} line)."
                    ),
                    severity=QASeverity.WARNING,
                    source_snippet=source_text[:100],
                    target_snippet=target_text[:100],
                    suggested_fix="Format each speaker turn on a new line with a dialogue dash.",
                )
            )

        return violations
