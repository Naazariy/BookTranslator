from pathlib import Path
from typing import Optional
from src.domain.interfaces.writer import IDocumentWriter
from src.domain.models.document import Book


class BaseWriter(IDocumentWriter):
    def write(self, book: Book, output_path: Path, template_path: Optional[Path] = None) -> Path:
        raise NotImplementedError("Subclasses must implement the write method")
