"""
src/quality/validators/empty_translation.py

Validator detecting empty, missing, or whitespace-only translations.
"""
from __future__ import annotations

from typing import Optional, List, Any
from src.domain.models.segment import TranslationSegment
from src.context.builder import PromptContext
from src.quality.base import BaseValidator
from src.quality.models import QAViolation, QASeverity


class EmptyTranslationValidator(BaseValidator):
    """
    Validates that a segment has a non-empty, non-whitespace translation.
    """
    name: str = "EmptyTranslation"
    description: str = "Detects empty, null, or whitespace-only translations"
    default_severity: QASeverity = QASeverity.CRITICAL

    def validate(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        translated_text: Optional[str] = None,
        **kwargs: Any,
    ) -> List[QAViolation]:
        target_text = self.extract_target_text(segment, translated_text=translated_text)

        if target_text is None:
            return [
                self.make_violation(
                    rule_code="EmptyTranslation",
                    message="Translated text is None.",
                    severity=QASeverity.CRITICAL,
                    suggested_fix="Generate translation for the segment.",
                )
            ]

        if not target_text.strip():
            return [
                self.make_violation(
                    rule_code="EmptyTranslation",
                    message="The translated text is empty.",
                    severity=QASeverity.CRITICAL,
                    suggested_fix="Translate the non-empty source text.",
                )
            ]

        return []
