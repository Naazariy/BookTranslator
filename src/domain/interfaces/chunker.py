from typing import Protocol, Iterator
from src.domain.models.document import Book
from src.domain.models.chunk import TranslationChunk


class IChunkManager(Protocol):
    def create_chunks_stream(
        self, book: Book, max_tokens: int, overlap_sentences: int
    ) -> Iterator[TranslationChunk]:
        ...
