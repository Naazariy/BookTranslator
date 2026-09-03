from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from uuid import UUID


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ValidationIssue(BaseModel):
    rule_name: str
    severity: IssueSeverity
    message: str
    affected_sentence_id: Optional[UUID] = None


class QualityReport(BaseModel):
    is_passed: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    translation_score: float = 1.0
