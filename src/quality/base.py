"""
src/quality/base.py

Abstract base validator contract for modular QA checks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Any
from uuid import UUID

from src.domain.models.segment import TranslationSegment
from src.context.builder import PromptContext
from src.quality.models import QAViolation, QASeverity


class BaseValidator(ABC):
    """
    Abstract base class for all modular translation quality validators.
    """
    name: str = "BaseValidator"
    description: str = "Base validator interface"
    default_severity: QASeverity = QASeverity.WARNING

    @abstractmethod
    def validate(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        **kwargs: Any,
    ) -> List[QAViolation]:
        """
        Validates the translation within segment against this validator's rules.
        Returns a list of QAViolations (empty list if valid).
        """
        pass

    def extract_source_text(self, segment: Any, chunk: Any = None) -> str:
        """Safely extracts source text from TranslationSegment or TranslationChunk."""
        if hasattr(segment, "source_text") and segment.source_text is not None:
            return segment.source_text
        obj = chunk or segment
        if hasattr(obj, "source_text") and obj.source_text is not None:
            return obj.source_text
        if hasattr(obj, "source_sentences") and obj.source_sentences:
            return " ".join(
                s.original_text for s in obj.source_sentences if getattr(s, "original_text", None)
            )
        return str(obj or "")

    def extract_target_text(self, segment: Any, translated_text: Optional[str] = None) -> str:
        """Safely extracts translated text prioritizing final_translation > refined_translation > draft."""
        if translated_text is not None:
            return translated_text
        if hasattr(segment, "final_translation") and segment.final_translation:
            return segment.final_translation
        if hasattr(segment, "refined_translation") and segment.refined_translation:
            return segment.refined_translation
        if hasattr(segment, "translated_text") and segment.translated_text:
            return segment.translated_text
        if hasattr(segment, "draft_translation") and segment.draft_translation:
            return segment.draft_translation
        return ""

    def make_violation(
        self,
        rule_code: str,
        message: str,
        severity: Optional[QASeverity] = None,
        source_snippet: Optional[str] = None,
        target_snippet: Optional[str] = None,
        forbidden_form: Optional[str] = None,
        suggested_fix: Optional[str] = None,
        affected_sentence_id: Optional[UUID] = None,
        metadata: Optional[dict] = None,
    ) -> QAViolation:
        """Convenience factory for constructing typed QAViolation instances."""
        return QAViolation(
            validator_name=self.name,
            rule_code=rule_code,
            severity=severity or self.default_severity,
            message=message,
            source_snippet=source_snippet,
            target_snippet=target_snippet,
            forbidden_form=forbidden_form,
            suggested_fix=suggested_fix,
            affected_sentence_id=affected_sentence_id,
            metadata=metadata or {},
        )
