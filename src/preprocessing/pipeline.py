"""
Document Preprocessing Pipeline for BookTranslator.

Coordinates entity extraction, glossary preparation, and optional
deterministic unit conversion across chapters and paragraphs before translation.
"""

from __future__ import annotations
import logging
from typing import Optional

from src.domain.models.document import Book
from src.domain.interfaces.knowledge_base import IKnowledgeBaseRepository
from src.preprocessing.unit_converter import UnitConverter

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Orchestrates pre-translation document processing:
    - Unit conversion on book sentences (if enabled)
    - Named Entity Recognition (NER) & Glossary harvesting
    - Alias & Coreference resolution
    """

    def __init__(
        self,
        kb_repo: IKnowledgeBaseRepository,
        unit_converter: Optional[UnitConverter] = None,
        convert_units: bool = False,
        unit_conversion_policy: str = "metric",
        target_language: str = "uk"
    ):
        self.kb_repo = kb_repo
        self.unit_converter = unit_converter or UnitConverter(default_policy=unit_conversion_policy)
        self.convert_units = convert_units
        self.unit_conversion_policy = unit_conversion_policy
        self.target_language = target_language

    def convert_book_units(self, book: Book, policy: Optional[str] = None) -> Book:
        """
        Applies deterministic unit conversion to all sentences within the book.
        """
        active_policy = policy or self.unit_conversion_policy
        target_lang = book.target_language or self.target_language
        logger.info(f"Converting units in book '{book.title}' with policy '{active_policy}' (target_lang={target_lang})")

        converted_count = 0
        for chapter in book.chapters:
            for paragraph in chapter.paragraphs:
                for sentence in paragraph.sentences:
                    if sentence.original_text:
                        original = sentence.original_text
                        self.unit_converter.convert_sentence(
                            sentence,
                            target_lang=target_lang,
                            policy=active_policy
                        )
                        if sentence.original_text != original:
                            converted_count += 1

        logger.info(f"Unit conversion completed: {converted_count} sentence(s) updated.")
        return book

    def process(self, book: Book) -> None:
        """
        Executes full preprocessing workflow on the given book.
        """
        logger.info(f"Starting preprocessing for book: {book.title}")

        # 1. Unit Conversion (if enabled)
        if self.convert_units and self.unit_converter:
            self.convert_book_units(book)

        # 2. Extract Entities (NER) & Populate KB (Future extensible hooks)
        # entities = self.entity_extractor.extract_entities(book)

        # 3. Extract Terms (TF-IDF/Keywords)
        # terms = self.term_extractor.extract_terms(book)

        # 4. Resolve Aliases (Coreference)
        # resolved_entities = self.alias_resolver.resolve_aliases(entities)

        # 5. Save to Knowledge Base
        # for entity in resolved_entities:
        #     self.kb_repo.add_entity(entity)

        logger.info("Preprocessing completed successfully.")
