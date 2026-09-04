"""
src/quality/models.py

Typed domain models for quality assurance diagnostics, violations, and reports.
Supports structured feedback for BoundedRepairEngine and backward compatibility
with legacy ValidationIssue and QualityReport.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from uuid import UUID, uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class QASeverity(str, Enum):
    """
    Severity levels for QA violations.
    - CRITICAL: Segment cannot be accepted; triggers repair loop or transitions to REVIEW_REQUIRED/FAILED.
    - WARNING: Sub-optimal translation or stylistic inconsistency; penalizes score.
    - INFO: Stylistic suggestion or minor formatting observation; minimal score impact.
    """
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class QAViolation(BaseModel):
    """
    Structured record of a specific quality rule breach.
    Provides fine-grained diagnostic snippets and repair hints for the BoundedRepairEngine.
    """
    id: UUID = Field(default_factory=uuid4)
    validator_name: str
    rule_code: str
    severity: QASeverity
    message: str
    source_snippet: Optional[str] = None
    target_snippet: Optional[str] = None
    forbidden_form: Optional[str] = None
    suggested_fix: Optional[str] = None
    affected_sentence_id: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_validation_issue(self) -> Any:
        """
        Backward-compatibility bridge to legacy domain.models.quality.ValidationIssue.
        """
        from src.domain.models.quality import ValidationIssue, IssueSeverity
        sev_map = {
            QASeverity.CRITICAL: IssueSeverity.CRITICAL,
            QASeverity.WARNING: IssueSeverity.WARNING,
            QASeverity.INFO: IssueSeverity.INFO,
        }
        return ValidationIssue(
            rule_name=self.rule_code or self.validator_name,
            severity=sev_map.get(self.severity, IssueSeverity.WARNING),
            message=self.message,
            affected_sentence_id=self.affected_sentence_id,
        )


class QAReport(BaseModel):
    """
    Comprehensive quality audit report for a TranslationSegment.
    Aggregates all violations, computes validation status, and records diagnostic scores.
    """
    id: UUID = Field(default_factory=uuid4)
    book_id: Optional[UUID] = None
    segment_id: Union[UUID, str]
    violations: List[QAViolation] = Field(default_factory=list)
    is_valid: bool = True
    score: float = 1.0
    validator_scores: Dict[str, float] = Field(default_factory=dict)
    repair_attempts: int = 0
    status: str = "VALIDATING"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    repair_prompt_hint: Optional[str] = None

    @property
    def is_passed(self) -> bool:
        """Backward-compatibility alias for legacy QualityReport.is_passed."""
        return self.is_valid

    @is_passed.setter
    def is_passed(self, value: bool) -> None:
        self.is_valid = value

    @property
    def translation_score(self) -> float:
        """Backward-compatibility alias for legacy QualityReport.translation_score."""
        return self.score

    @translation_score.setter
    def translation_score(self, value: float) -> None:
        self.score = value

    @property
    def issues(self) -> List[Any]:
        """Backward-compatibility alias returning legacy ValidationIssue list."""
        return [v.to_validation_issue() for v in self.violations]

    @property
    def critical_violations(self) -> List[QAViolation]:
        """Returns all violations with CRITICAL severity."""
        return [v for v in self.violations if v.severity == QASeverity.CRITICAL]

    @property
    def warnings(self) -> List[QAViolation]:
        """Returns all violations with WARNING severity."""
        return [v for v in self.violations if v.severity == QASeverity.WARNING]

    def add_violation(self, violation: QAViolation) -> None:
        """Appends violation and dynamically recalculates score and validity."""
        self.violations.append(violation)
        if violation.severity == QASeverity.CRITICAL:
            self.is_valid = False
            self.score = max(0.0, self.score - 0.4)
        elif violation.severity == QASeverity.WARNING:
            self.score = max(0.0, self.score - 0.1)
        elif violation.severity == QASeverity.INFO:
            self.score = max(0.0, self.score - 0.05)

    def has_critical(self) -> bool:
        """Returns True if any CRITICAL violation is present."""
        return any(v.severity == QASeverity.CRITICAL for v in self.violations)

    def format_repair_instructions(self) -> str:
        """
        Synthesizes a concise, targeted Ukrainian instruction prompt for LLM repair.
        """
        critical_fixes = [
            v.suggested_fix or v.message
            for v in self.violations
            if v.severity == QASeverity.CRITICAL
        ]
        warning_fixes = [
            v.suggested_fix or v.message
            for v in self.violations
            if v.severity == QASeverity.WARNING
        ]
        
        lines = ["ВИПРАВТЕ НАСТУПНІ ПОМИЛКИ В ПЕРЕКЛАДІ:"]
        for idx, fix in enumerate(critical_fixes + warning_fixes, 1):
            lines.append(f"{idx}. {fix}")
        lines.append("Збережіть природний український стиль та правильну структуру параграфа.")
        return "\n".join(lines)
