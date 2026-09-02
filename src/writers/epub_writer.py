from pathlib import Path
from typing import Optional
from src.writers.base import BaseWriter
from src.domain.models.document import Book

class EpubWriter(BaseWriter):
    def write(self, book: Book, output_path: Path, template_path: Optional[Path] = None) -> Path:
        raise NotImplementedError("EPUB Writing is not yet fully implemented.")
