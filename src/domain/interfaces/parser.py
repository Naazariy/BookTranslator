from typing import Protocol
from pathlib import Path
from src.domain.models.document import Book


class IDocumentParser(Protocol):
    def parse(self, file_path: Path) -> Book:
        ...
