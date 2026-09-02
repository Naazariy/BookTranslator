from typing import Protocol, Optional
from pathlib import Path
from src.domain.models.document import Book


class IDocumentWriter(Protocol):
    def write(self, book: Book, output_path: Path, template_path: Optional[Path] = None) -> Path:
        ...
