from typing import Protocol, List, Iterable, Iterator, Optional, Any, Dict
from src.domain.models.chunk import TranslationChunk, ChunkContext
from src.domain.models.knowledge import GlossaryItem


class ITranslationEngine(Protocol):
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        ...


class IEditingEngine(Protocol):
    def refine_chunk(
        self,
        draft_translation: str,
        source_text: str = "",
        context: Optional[ChunkContext] = None,
        glossary: Optional[List[GlossaryItem]] = None,
        cancel_token: Optional[Any] = None
    ) -> str:
        ...

    def refine_chunk_structured(
        self,
        target_sentences: List[Any],
        context_sentences: Optional[List[Any]] = None,
        glossary: Optional[List[GlossaryItem]] = None,
        cancel_token: Optional[Any] = None
    ) -> Dict[int, str]:
        ...


class ITranslationPipeline(Protocol):
    def execute_two_stage_translation(
        self, chunks_stream: Iterable[TranslationChunk]
    ) -> Iterator[TranslationChunk]:
        ...
