from pathlib import Path
from src.parsers.base import BaseParser
from src.domain.models.document import Book

class Fb2Parser(BaseParser):
    def parse(self, file_path: Path) -> Book:
        raise NotImplementedError("FB2 Parsing is not yet fully implemented.")
