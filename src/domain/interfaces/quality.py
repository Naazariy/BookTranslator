from typing import Protocol
from src.domain.models.chunk import TranslationChunk
from src.domain.models.quality import QualityReport


class IQualityChecker(Protocol):
    def validate(self, original_chunk: TranslationChunk, translated_text: str) -> QualityReport:
        ...
