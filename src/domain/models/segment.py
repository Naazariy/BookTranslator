from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from uuid import UUID, uuid4


class SegmentStatus(str, Enum):
    """
    Explicit 7-state lifecycle for translation segments:
    PENDING -> DRAFT_COMPLETED -> EDITED -> VALIDATING -> ACCEPTED / REVIEW_REQUIRED / FAILED
    """
    PENDING = "PENDING"
    DRAFT_COMPLETED = "DRAFT_COMPLETED"
    EDITED = "EDITED"
    VALIDATING = "VALIDATING"
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


class IllegalStateTransitionError(Exception):
    """Raised when an invalid segment lifecycle state transition is attempted."""
    pass


VALID_SEGMENT_TRANSITIONS: Dict[SegmentStatus, Set[SegmentStatus]] = {
    SegmentStatus.PENDING: {SegmentStatus.DRAFT_COMPLETED, SegmentStatus.FAILED},
    SegmentStatus.DRAFT_COMPLETED: {SegmentStatus.EDITED, SegmentStatus.FAILED},
    SegmentStatus.EDITED: {SegmentStatus.VALIDATING, SegmentStatus.ACCEPTED, SegmentStatus.FAILED},
    SegmentStatus.VALIDATING: {
        SegmentStatus.ACCEPTED,
        SegmentStatus.REVIEW_REQUIRED,
        SegmentStatus.FAILED,
        SegmentStatus.VALIDATING,
    },
    SegmentStatus.REVIEW_REQUIRED: {
        SegmentStatus.VALIDATING,
        SegmentStatus.ACCEPTED,
        SegmentStatus.FAILED,
    },
    SegmentStatus.ACCEPTED: set(),
    SegmentStatus.FAILED: {SegmentStatus.PENDING},
}


def validate_segment_transition(current: SegmentStatus, target: SegmentStatus) -> bool:
    """Checks whether transitioning from current to target status is valid."""
    allowed = VALID_SEGMENT_TRANSITIONS.get(current, set())
    return target in allowed


class TranslationSegment(BaseModel):
    """
    Paragraph-level semantic translation unit for Stage 2 refinement, QA, and export.
    Preserves mappings to constituent sentence IDs, normalized source text, and NLLB drafts.
    """
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    chapter_id: UUID
    paragraph_id: UUID
    order_index: int = 0

    source_text: str
    normalized_text: Optional[str] = None
    sentence_ids: List[UUID] = Field(default_factory=list)

    draft_translation: Optional[str] = None
    refined_translation: Optional[str] = None
    final_translation: Optional[str] = None

    status: SegmentStatus = SegmentStatus.PENDING
    repair_attempts: int = 0
    error_message: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_draft_fields(cls, data: Any) -> Any:
        """Ensures draft_text and draft_translation are synchronized on instantiation."""
        if isinstance(data, dict):
            if "draft_text" in data and "draft_translation" not in data:
                data["draft_translation"] = data["draft_text"]
            elif "draft_translation" in data and "draft_text" not in data:
                data["draft_text"] = data["draft_translation"]
        return data

    @property
    def draft_text(self) -> Optional[str]:
        """Convenience alias for draft_translation matching specification naming."""
        return self.draft_translation

    @draft_text.setter
    def draft_text(self, value: Optional[str]) -> None:
        self.draft_translation = value

    @property
    def translated_text(self) -> Optional[str]:
        """Returns the highest-fidelity available translation for export and inspection."""
        return self.final_translation or self.refined_translation or self.draft_translation

    def transition_to(self, new_status: SegmentStatus) -> None:
        """Atomically transitions segment status enforcing lifecycle rules."""
        if not validate_segment_transition(self.status, new_status):
            raise IllegalStateTransitionError(
                f"Cannot transition TranslationSegment from {self.status} to {new_status}"
            )
        self.status = new_status

    @classmethod
    def from_paragraph(
        cls,
        paragraph: Any,
        book_id: UUID,
        chapter_id: UUID,
        order_index: int = 0,
    ) -> "TranslationSegment":
        """
        Factory creating a TranslationSegment from a document Paragraph.
        Concatenates sentence original/normalized text and populates sentence_ids mapping.
        """
        orig_parts = [s.original_text for s in paragraph.sentences if getattr(s, "original_text", None)]
        source = " ".join(orig_parts).strip() if orig_parts else " ".join(getattr(s, "source_for_translation", "") for s in paragraph.sentences).strip()
        has_norm = any(getattr(s, "normalized_source_text", None) is not None for s in paragraph.sentences)
        normalized = " ".join(getattr(s, "source_for_translation", s.original_text) for s in paragraph.sentences).strip() if has_norm else None
        draft = " ".join(
            s.translated_text for s in paragraph.sentences if getattr(s, "translated_text", None)
        ).strip() or None

        return cls(
            id=paragraph.id,
            book_id=book_id,
            chapter_id=chapter_id,
            paragraph_id=paragraph.id,
            order_index=order_index,
            source_text=source,
            normalized_text=normalized,
            sentence_ids=[s.id for s in paragraph.sentences],
            draft_translation=draft,
            status=SegmentStatus.PENDING if not draft else SegmentStatus.DRAFT_COMPLETED,
        )

