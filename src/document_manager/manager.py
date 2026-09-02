from pathlib import Path
from typing import Dict, Type
import logging

from src.domain.models.document import Book
from src.domain.interfaces.parser import IDocumentParser
from src.domain.interfaces.writer import IDocumentWriter
from src.parsers.txt_parser import TxtParser
from src.parsers.epub_parser import EpubParser
from src.parsers.docx_parser import DocxParser
from src.parsers.fb2_parser import Fb2Parser
from src.parsers.pdf_parser import PdfParser

from src.writers.txt_writer import TxtWriter
from src.writers.epub_writer import EpubWriter
from src.writers.docx_writer import DocxWriter
from src.writers.fb2_writer import Fb2Writer
from src.writers.pdf_writer import PdfWriter

logger = logging.getLogger(__name__)


class DocumentManager:
    def __init__(self):
        # Register parsers and writers by extension
        self.parsers: Dict[str, Type[IDocumentParser]] = {
            ".txt": TxtParser,
            ".epub": EpubParser,
            ".docx": DocxParser,
            ".fb2": Fb2Parser,
            ".pdf": PdfParser
        }
        self.writers: Dict[str, Type[IDocumentWriter]] = {
            ".txt": TxtWriter,
            ".epub": EpubWriter,
            ".docx": DocxWriter,
            ".fb2": Fb2Writer,
            ".pdf": PdfWriter
        }

    def load_document(self, path: Path) -> Book:
        ext = path.suffix.lower()
        parser_cls = self.parsers.get(ext)
        if not parser_cls:
            raise ValueError(f"No parser found for extension: {ext}")
        
        parser = parser_cls()
        logger.info(f"Parsing document {path} using {parser_cls.__name__}")
        return parser.parse(path)

    def save_document(self, book: Book, output_path: Path) -> Path:
        ext = output_path.suffix.lower()
        writer_cls = self.writers.get(ext)
        if not writer_cls:
            raise ValueError(f"No writer found for extension: {ext}")
        
        writer = writer_cls()
        logger.info(f"Writing document to {output_path} using {writer_cls.__name__}")
        return writer.write(book, output_path)
