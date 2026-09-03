"""
Unit tests for HeuristicEntityExtractor and PreprocessingPipeline NER harvesting.
"""
from uuid import uuid4
import pytest

from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.knowledge import GlossaryItem, EntityType
from src.preprocessing.entity_extractor import HeuristicEntityExtractor
from src.preprocessing.pipeline import PreprocessingPipeline
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository


class TestHeuristicEntityExtractor:
    """Tests proper name extraction and stop-word filtering."""

    def test_extract_repeated_entities(self):
        extractor = HeuristicEntityExtractor(min_occurrences=3)
        text = (
            "Sherlock Holmes walked down the street. "
            "Later, Sherlock Holmes met Dr. Watson. "
            "Finally, Sherlock Holmes solved the mystery of Eldoria. "
            "Eldoria was an ancient city. People in Eldoria rejoiced."
        )
        entities = extractor.extract_from_text(text)
        assert "Sherlock Holmes" in entities
        assert "Eldoria" in entities
        # "Sherlock" and "Holmes" individually might also be captured if count >= 3
        # Stop words like "The", "Later", "Finally" must NOT be in entities
        assert "The" not in entities
        assert "Later" not in entities
        assert "Finally" not in entities

    def test_ignore_single_occurrence_entities(self):
        extractor = HeuristicEntityExtractor(min_occurrences=3)
        text = "Alice went to Paris. Bob stayed in London. Charlie visited Rome."
        entities = extractor.extract_from_text(text)
        # All appear only once, so none should pass min_occurrences=3
        assert len(entities) == 0

    def test_stop_words_case_insensitive_filtering(self):
        extractor = HeuristicEntityExtractor(min_occurrences=2)
        text = "The the THE And and AND But but BUT Because because"
        entities = extractor.extract_from_text(text)
        assert len(entities) == 0

    def test_honorifics_and_titles_filtered(self):
        extractor = HeuristicEntityExtractor(min_occurrences=3)
        text = (
            "Mr. Darcy walked into the room. "
            "Mr. Darcy greeted Elizabeth. "
            "Later, Mr. Darcy wrote a letter to Darcy."
        )
        entities = extractor.extract_from_text(text)
        assert "Darcy" in entities
        assert "Mr" not in entities



class TestPreprocessingPipelineEntityExtraction:
    """Tests PreprocessingPipeline integration with HeuristicEntityExtractor and KB repository."""

    @pytest.fixture
    def repo(self, tmp_path):
        db_path = tmp_path / "preproc_kb.db"
        init_db(db_path)
        repository = SQLiteKnowledgeBaseRepository(db_path)
        yield repository
        repository.close()

    def test_pipeline_populates_kb_with_extracted_entities(self, repo):
        # Pre-seed glossary with one custom term
        repo.add_glossary_item(GlossaryItem(
            source_term="Eldoria",
            target_term="Чарівна Ельдорія",
            entity_type=EntityType.LOCATION
        ))

        book = Book(id=uuid4(), title="NER Test Book", source_language="en", target_language="uk")
        ch1 = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        
        sentences = [
            Sentence(id=uuid4(), original_text="Gondor called for aid. Gondor stood strong.", order_index=0),
            Sentence(id=uuid4(), original_text="The beacon of Gondor was lit.", order_index=1),
            Sentence(id=uuid4(), original_text="Eldoria was peaceful. Eldoria was quiet. Eldoria remained.", order_index=2),
        ]
        ch1.paragraphs.append(Paragraph(id=uuid4(), sentences=sentences))
        book.chapters.append(ch1)

        extractor = HeuristicEntityExtractor(min_occurrences=3)
        pipeline = PreprocessingPipeline(
            kb_repo=repo,
            entity_extractor=extractor,
            convert_units=False
        )

        pipeline.process(book)

        glossary = repo.get_glossary()
        glossary_map = {item.source_term: item.target_term for item in glossary}

        # 1. "Gondor" was found 3 times and added as new glossary item
        assert "Gondor" in glossary_map
        assert glossary_map["Gondor"] == "Gondor"

        # 2. "Eldoria" was already present with custom target_term and must NOT be overwritten!
        assert "Eldoria" in glossary_map
        assert glossary_map["Eldoria"] == "Чарівна Ельдорія"
