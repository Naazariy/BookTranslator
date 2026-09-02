from pathlib import Path
from src.parsers.base import BaseParser
from src.domain.models.document import Book

class DocxParser(BaseParser):
    def parse(self, file_path: Path) -> Book:
        raise NotImplementedError("DOCX Parsing is not yet fully implemented.")
