from pathlib import Path
from src.parsers.base import BaseParser
from src.domain.models.document import Book

class EpubParser(BaseParser):
    def parse(self, file_path: Path) -> Book:
        raise NotImplementedError("EPUB Parsing is not yet fully implemented.")
