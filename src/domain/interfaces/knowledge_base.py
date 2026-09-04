from typing import Protocol, List, Optional, Iterator
from uuid import UUID
from src.domain.models.knowledge import Entity, GlossaryItem, EntityProfile, EntityMention
from src.domain.models.chunk import TranslationChunk, ChunkStatus


class IKnowledgeBaseRepository(Protocol):
    def add_entity(self, entity: Entity) -> None:
        ...

    def get_glossary(self) -> List[GlossaryItem]:
        ...

    def search_translation_memory(self, source_text: str) -> Optional[str]:
        ...

    def save_chunk_state(self, chunk: TranslationChunk) -> None:
        ...

    def load_chunks_by_status(self, book_id: UUID, status: ChunkStatus) -> Iterator[TranslationChunk]:
        ...

    def count_chunks_for_book(self, book_id: UUID) -> int:
        ...

    def delete_chunks_for_book(self, book_id: UUID) -> None:
        ...

    # Scoped Knowledge Base (Milestone 2)
    def save_entity_profile(self, profile: EntityProfile) -> None:
        ...

    def save_entity_profiles_batch(self, profiles: List[EntityProfile]) -> None:
        ...

    def get_entity_profile(self, profile_id: UUID) -> Optional[EntityProfile]:
        ...

    def get_entity_profiles_scoped(
        self,
        book_id: Optional[str] = None,
        series_id: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[EntityProfile]:
        ...

    def save_entity_mentions_batch(self, mentions: List[EntityMention]) -> None:
        ...

    def get_entity_mentions_for_segment(self, segment_id: UUID) -> List[EntityMention]:
        ...

    def get_entities_for_segment(
        self,
        segment_id: UUID,
        book_id: Optional[str] = None,
        series_id: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[EntityProfile]:
        ...

    def get_scoped_glossary(
        self,
        book_id: Optional[str] = None,
        series_id: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[GlossaryItem]:
        ...
