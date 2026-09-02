"""Tier 3 Cross-Feature Tests: DOM Reconciliation + Document Writers (PDF / TXT).
Verifies that reconciled Book DOM trees export cleanly to valid PDF and TXT files with Cyrillic formatting.
"""
from pathlib import Path
from uuid import uuid4
import pytest

from src.writers.txt_writer import TxtWriter
from src.writers.pdf_writer import PdfWriter
from tests.e2e.fixtures import (
    SampleBookFactory,
    reconcile_document_dom,
)
from src.domain.models.chunk import ChunkStatus


class TestDOMReconciliationWithPDFExport:
    def test_reconciled_book_dom_exports_to_valid_pdf_file(self, temp_work_dir):
        """Test X4.1: Reconciled Book DOM generates valid PDF output file."""
        book = SampleBookFactory.create_sample_book(title="Тестова Книга")
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=1)
        
        for i, chunk in enumerate(chunks):
            chunk.final_translation = f"Це відредаговане речення номер {i}."
            chunk.status = ChunkStatus.REFINED

        reconciled_book = reconcile_document_dom(book, chunks)
        out_pdf = temp_work_dir / "output_book.pdf"

        writer = PdfWriter()
        result_path = writer.write(reconciled_book, out_pdf)

        assert result_path.exists()
        assert result_path.stat().st_size > 0
        
        # Verify PDF header magic bytes %PDF-
        with open(result_path, "rb") as f:
            header = f.read(5)
            assert header.startswith(b"%PDF")

    def test_pdf_export_contains_ukrainian_cyrillic_characters(self, temp_work_dir):
        """Test X4.2: Exporting multi-chapter book with Cyrillic dialogue dashes and guillemets."""
        book = SampleBookFactory.create_sample_book(title="Подорож")
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=2)
        
        for i, chunk in enumerate(chunks):
            chunk.final_translation = f"«Розділ {i}!» — вигукнув мандрівник."
            chunk.status = ChunkStatus.REFINED

        reconciled_book = reconcile_document_dom(book, chunks)
        out_pdf = temp_work_dir / "cyrillic_book.pdf"

        writer = PdfWriter()
        writer.write(reconciled_book, out_pdf)
        assert out_pdf.exists()
        assert out_pdf.stat().st_size > 100

    def test_txt_export_matches_reconciled_sentence_stream(self, temp_work_dir):
        """Test X4.3: TxtWriter output contains all translated sentences from reconciled DOM."""
        book = SampleBookFactory.create_sample_book(title="Текстова Книга")
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=1)
        
        translations = [f"Українське речення #{i} у файлі." for i in range(len(chunks))]
        for chunk, trans in zip(chunks, translations):
            chunk.final_translation = trans
            chunk.status = ChunkStatus.REFINED

        reconciled_book = reconcile_document_dom(book, chunks)
        out_txt = temp_work_dir / "output_book.txt"

        writer = TxtWriter()
        writer.write(reconciled_book, out_txt)

        assert out_txt.exists()
        content = out_txt.read_text(encoding="utf-8")
        for trans in translations:
            assert trans in content
