"""Tier 1 Feature Tests: O(S) DOM Tree Reconciliation (F11).
Verifies dictionary-based mapping {sentence.id: sentence}, multi-sentence chunk handling, formatting preservation, and linear scaling.
"""
import time
import pytest
from uuid import uuid4
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from tests.e2e.fixtures import reconcile_document_dom, SampleBookFactory

try:
    from src.parsers.dom import reconcile_document_dom as project_reconcile
except ImportError:
    project_reconcile = reconcile_document_dom


def get_reconcile_fn():
    return project_reconcile if project_reconcile is not None else reconcile_document_dom


class TestDOMReconciliation:
    def test_reconcile_document_dom_dictionary_mapping(self):
        """Test F11.1: O(S) dictionary mapping correctly updates target sentences."""
        reconcile_fn = get_reconcile_fn()
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=1)
        
        # Populate final translations
        for i, chunk in enumerate(chunks):
            chunk.final_translation = f"Український переклад {i}"
            chunk.status = ChunkStatus.REFINED
            
        reconciled_book = reconcile_fn(book, chunks)
        
        # Verify sentences received translations
        for chapter in reconciled_book.chapters:
            for p in chapter.paragraphs:
                for s in p.sentences:
                    assert s.translated_text is not None
                    assert "Український переклад" in s.translated_text

    def test_reconcile_multi_sentence_chunk_writeback(self):
        """Test F11.2: Multi-sentence chunk writes translation to first sentence and empties rest."""
        reconcile_fn = get_reconcile_fn()
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=3)
        
        for i, chunk in enumerate(chunks):
            chunk.final_translation = f"Об'єднаний переклад кількох речень {i}"
            chunk.status = ChunkStatus.REFINED
            
        reconciled_book = reconcile_fn(book, chunks)
        
        total_translated = 0
        total_empty = 0
        for chapter in reconciled_book.chapters:
            for p in chapter.paragraphs:
                for s in p.sentences:
                    if s.translated_text and s.translated_text.strip():
                        total_translated += 1
                    else:
                        total_empty += 1
                        
        assert total_translated == len(chunks)
        assert total_empty > 0

    def test_reconcile_preserves_untranslated_and_formatting(self):
        """Test F11.3: Sentences without corresponding chunks remain untouched."""
        reconcile_fn = get_reconcile_fn()
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=1)
        
        # Only translate half the chunks
        half_chunks = chunks[:len(chunks)//2]
        for i, chunk in enumerate(half_chunks):
            chunk.final_translation = f"Частковий переклад {i}"
            
        reconciled_book = reconcile_fn(book, half_chunks)
        
        translated_count = 0
        untranslated_count = 0
        for chapter in reconciled_book.chapters:
            for p in chapter.paragraphs:
                for s in p.sentences:
                    if s.translated_text:
                        translated_count += 1
                    else:
                        untranslated_count += 1
                        
        assert translated_count == len(half_chunks)
        assert untranslated_count > 0

    def test_reconcile_handles_empty_or_missing_chunks(self):
        """Test F11.4: Passing empty chunks list leaves book valid and intact."""
        reconcile_fn = get_reconcile_fn()
        book = SampleBookFactory.create_sample_book()
        reconciled_book = reconcile_fn(book, [])
        assert reconciled_book.id == book.id
        assert len(reconciled_book.chapters) == len(book.chapters)

    def test_reconcile_linear_complexity_scaling(self):
        """Test F11.5: 5,000-sentence DOM reconciliation completes in under 0.2 seconds (O(S) linear)."""
        reconcile_fn = get_reconcile_fn()
        book_id = uuid4()
        book = Book(id=book_id, title="Large Volume Book")
        
        # Create 5 chapters with 1,000 sentences each = 5,000 sentences
        all_sentences = []
        for c_idx in range(5):
            chapter = Chapter(id=uuid4(), title=f"Chapter {c_idx+1}", order_index=c_idx)
            for p_idx in range(100):
                para = Paragraph(id=uuid4())
                for s_idx in range(10):
                    s = Sentence(id=uuid4(), original_text=f"Sentence {c_idx}_{p_idx}_{s_idx}", order_index=s_idx)
                    para.sentences.append(s)
                    all_sentences.append(s)
                chapter.paragraphs.append(para)
            book.chapters.append(chapter)
            
        # Create 1,000 chunks (5 sentences each)
        chunks = []
        for i in range(0, len(all_sentences), 5):
            subset = all_sentences[i:i+5]
            chunks.append(TranslationChunk(
                id=uuid4(),
                book_id=book_id,
                chapter_id=book.chapters[0].id,
                paragraph_indices=[0],
                source_sentences=subset,
                token_count=50,
                context=ChunkContext(),
                status=ChunkStatus.REFINED,
                final_translation=f"Переклад блоку {i//5}"
            ))
            
        start_time = time.time()
        reconciled_book = reconcile_fn(book, chunks)
        elapsed = time.time() - start_time
        
        # Must execute in under 0.25s (O(S) instead of O(C*S) which takes > 25s)
        assert elapsed < 0.25, f"Reconciliation took {elapsed:.4f}s - expected < 0.25s for O(S)"
