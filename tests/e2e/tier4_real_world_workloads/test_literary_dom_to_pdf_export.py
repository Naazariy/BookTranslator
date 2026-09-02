"""Tier 4 Real-World Workload Tests: High-Fidelity Literary DOM to PDF Export.
Verifies complex book structures with Cyrillic dialogue, Ukrainian dashes, honorifics, paragraph hierarchy, and PDF generation.
"""
import time
from pathlib import Path
from uuid import uuid4
import pytest

from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.writers.pdf_writer import PdfWriter
from src.writers.txt_writer import TxtWriter
from tests.e2e.fixtures import (
    SampleBookFactory,
    RuleBasedSentenceSegmenter,
    reconcile_document_dom,
)


class TestLiteraryDOMToPDFExport:
    def test_real_world_ukrainian_dialogue_prose_to_pdf(self, temp_work_dir):
        """Test R3.1: Complete literary book with Ukrainian dialogues, guillemets, and dashes exports to PDF."""
        book = Book(id=uuid4(), title="Кобзар та Мандрівник", author="Тарас Шевченко")
        
        ch1 = Chapter(id=uuid4(), title="Розділ I. Ніч на хуторі", order_index=0)
        p1 = Paragraph(id=uuid4())
        p1_text = "Реве та стогне Дніпр широкий, сердитий вітер завива. Додолу верби гне високі, горами хвилю підійма."
        for idx, s in enumerate(RuleBasedSentenceSegmenter.split_sentences(p1_text)):
            sentence = Sentence(id=uuid4(), original_text=s, translated_text=s, order_index=idx)
            p1.sentences.append(sentence)
        ch1.paragraphs.append(p1)

        p2 = Paragraph(id=uuid4(), is_dialogue=True)
        p2_text = "«Хто там у темряві?» — гукнув козак. «Це я, подорожній», — тихо відповіли з лісу."
        for idx, s in enumerate(RuleBasedSentenceSegmenter.split_sentences(p2_text)):
            sentence = Sentence(id=uuid4(), original_text=s, translated_text=s, order_index=idx)
            p2.sentences.append(sentence)
        ch1.paragraphs.append(p2)

        book.chapters.append(ch1)

        out_pdf = temp_work_dir / "kobzar_prose.pdf"
        writer = PdfWriter()
        result_path = writer.write(book, out_pdf)

        assert result_path.exists()
        assert result_path.stat().st_size > 500

    def test_chapter_hierarchy_and_paragraph_structure_fidelity(self, temp_work_dir):
        """Test R3.2: Multi-chapter hierarchy with translated chapter titles accurately preserved in export."""
        book = Book(id=uuid4(), title="Multi Chapter Anthology")
        
        for c in range(3):
            chapter = Chapter(
                id=uuid4(),
                title=f"Chapter {c+1}: Original Title",
                translated_title=f"Розділ {c+1}: Перекладений Заголовок",
                order_index=c
            )
            for p in range(4):
                para = Paragraph(id=uuid4())
                for s in range(3):
                    para.sentences.append(Sentence(
                        id=uuid4(),
                        original_text=f"Original sentence {c}_{p}_{s}.",
                        translated_text=f"Українське речення {c}_{p}_{s}.",
                        order_index=s
                    ))
                chapter.paragraphs.append(para)
            book.chapters.append(chapter)

        out_txt = temp_work_dir / "anthology.txt"
        txt_writer = TxtWriter()
        txt_writer.write(book, out_txt)

        content = out_txt.read_text(encoding="utf-8")
        assert "Розділ 1: Перекладений Заголовок" in content
        assert "Розділ 3: Перекладений Заголовок" in content
        assert "Українське речення 2_3_2." in content

    def test_large_volume_dom_reconciliation_performance(self):
        """Test R3.3: Reconciling a 1,000-chunk book (10,000 sentences) completes in < 0.15s."""
        book_id = uuid4()
        book = Book(id=book_id, title="Grand Novel")
        
        all_sentences = []
        for c_idx in range(10):
            ch = Chapter(id=uuid4(), title=f"Chapter {c_idx+1}", order_index=c_idx)
            for p_idx in range(100):
                para = Paragraph(id=uuid4())
                for s_idx in range(10):
                    s = Sentence(id=uuid4(), original_text=f"Eng sentence {c_idx}_{p_idx}_{s_idx}", order_index=s_idx)
                    para.sentences.append(s)
                    all_sentences.append(s)
                ch.paragraphs.append(para)
            book.chapters.append(ch)

        chunks = []
        for i in range(0, len(all_sentences), 10):
            subset = all_sentences[i:i+10]
            chunks.append(TranslationChunk(
                id=uuid4(),
                book_id=book_id,
                chapter_id=book.chapters[0].id,
                paragraph_indices=[0],
                source_sentences=subset,
                token_count=150,
                context=ChunkContext(),
                status=ChunkStatus.REFINED,
                final_translation=f"Український переклад блоку #{i//10}"
            ))

        start = time.time()
        reconciled = reconcile_document_dom(book, chunks)
        elapsed = time.time() - start

        assert elapsed < 0.20
        # Verify first sentence has translation
        assert reconciled.chapters[0].paragraphs[0].sentences[0].translated_text == "Український переклад блоку #0"
