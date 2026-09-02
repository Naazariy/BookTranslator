from pathlib import Path
from typing import List
from uuid import uuid4

from src.domain.interfaces.parser import IDocumentParser
from src.domain.models.document import Book, Chapter, Paragraph, Sentence


class BaseParser(IDocumentParser):
    def parse(self, file_path: Path) -> Book:
        raise NotImplementedError("Subclasses must implement the parse method")
