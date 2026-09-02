"""Tier 2 Boundary Tests: Empty Inputs, Extreme Lengths, and Structural Anomalies.
Verifies pipeline resilience against 0-byte docs, single-character tokens, 10,000-token sentences, and whitespace-only buffers.
"""
from uuid import uuid4
import pytest
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from tests.e2e.fixtures import (
    RuleBasedSentenceSegmenter,
    DynamicTokenBucketBatcher,
    reconcile_document_dom,
    MockCTranslate2Engine,
)


class TestBoundaryEmptyAndExtremes:
    def test_empty_document_pipeline_handling(self):
        """Test B1.1: Document with 0 chapters or 0 paragraphs handled gracefully."""
        book = Book(id=uuid4(), title="Empty Document")
        assert len(book.chapters) == 0
        
        # DOM reconciliation on empty book
        reconciled = reconcile_document_dom(book, [])
        assert len(reconciled.chapters) == 0

    def test_document_with_single_character_sentences(self):
        """Test B1.2: Single-character sentences ('A.', 'Я.', '!', '?') processed safely."""
        text = "A. B. C. Я. ?"
        sents = RuleBasedSentenceSegmenter.split_sentences(text)
        assert len(sents) >= 4

        engine = MockCTranslate2Engine()
        engine.load_model()
        try:
            translations = engine.translate_batch(sents, "en", "uk")
            assert len(translations) == len(sents)
        finally:
            engine.unload_model()

    def test_extreme_sentence_length_10000_tokens(self):
        """Test B1.3: Extreme length sentence (10,000 tokens) does not cause recursion or memory blowup."""
        huge_text = "word " * 10000
        estimated_tokens = DynamicTokenBucketBatcher.estimate_token_length(huge_text)
        assert estimated_tokens >= 9000

        buckets = DynamicTokenBucketBatcher.create_buckets([huge_text], max_tokens=2048)
        # Giant item gets isolated bucket
        assert len(buckets) == 1
        assert len(buckets[0]) == 1

    def test_chapter_with_zero_paragraphs_or_sentences(self):
        """Test B1.4: Empty chapter among normal chapters preserves hierarchy."""
        book = Book(id=uuid4(), title="Hierarchy Test")
        ch1 = Chapter(id=uuid4(), title="Empty Chapter", order_index=0)
        ch2 = Chapter(id=uuid4(), title="Normal Chapter", order_index=1)
        p = Paragraph(id=uuid4())
        s = Sentence(id=uuid4(), original_text="Valid content.", order_index=0)
        p.sentences.append(s)
        ch2.paragraphs.append(p)
        book.chapters.extend([ch1, ch2])

        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book.id,
            chapter_id=ch2.id,
            paragraph_indices=[0],
            source_sentences=[s],
            token_count=10,
            context=ChunkContext(),
            status=ChunkStatus.REFINED,
            final_translation="Дійсний вміст."
        )

        reconciled = reconcile_document_dom(book, [chunk])
        assert len(reconciled.chapters) == 2
        assert len(reconciled.chapters[0].paragraphs) == 0
        assert reconciled.chapters[1].paragraphs[0].sentences[0].translated_text == "Дійсний вміст."

    def test_all_whitespace_and_newline_document(self):
        """Test B1.5: Whitespace, tabs, and newlines do not generate phantom sentences or chunks."""
        text = "\n\n   \t\t \r\n   \n"
        sents = RuleBasedSentenceSegmenter.split_sentences(text)
        assert len(sents) == 0
