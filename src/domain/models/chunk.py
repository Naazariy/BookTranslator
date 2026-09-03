from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from uuid import UUID, uuid4
from src.domain.models.document import Sentence
from src.domain.models.knowledge import GlossaryItem


class ChunkStatus(str, Enum):
    PENDING = "PENDING"
    DRAFT_COMPLETED = "DRAFT_COMPLETED"
    REFINED = "REFINED"
    FAILED = "FAILED"


class ChunkContext(BaseModel):
    previous_summary: Optional[str] = None
    previous_sentences: List[str] = Field(default_factory=list)
    active_glossary: List[GlossaryItem] = Field(default_factory=list)


class TranslationChunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    book_id: UUID
    chapter_id: UUID
    paragraph_indices: List[int] = Field(default_factory=list)
    source_sentences: List[Sentence] = Field(default_factory=list)
    context_sentence_ids: List[UUID] = Field(default_factory=list)
    target_sentence_ids: List[UUID] = Field(default_factory=list)
    token_count: int = 0
    context: ChunkContext = Field(default_factory=ChunkContext)
    status: ChunkStatus = ChunkStatus.PENDING
    draft_translation: Optional[str] = None
    final_translation: Optional[str] = None
    error_message: Optional[str] = None

    def model_post_init(self, __context) -> None:
        if not self.target_sentence_ids and self.source_sentences:
            ctx_ids = set(self.context_sentence_ids)
            self.target_sentence_ids = [s.id for s in self.source_sentences if s.id not in ctx_ids]

    @property
    def target_sentences(self) -> List[Sentence]:
        """Returns the sentences to be translated in this chunk (ignoring reference context)."""
        target_ids = set(self.target_sentence_ids)
        return [s for s in self.source_sentences if s.id in target_ids]

    @property
    def context_sentences(self) -> List[Sentence]:
        """Returns the reference context sentences for this chunk."""
        ctx_ids = set(self.context_sentence_ids)
        return [s for s in self.source_sentences if s.id in ctx_ids]
