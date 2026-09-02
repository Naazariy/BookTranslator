from typing import Protocol, List, Optional, Iterator
from uuid import UUID
from src.domain.models.knowledge import Entity, GlossaryItem
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
