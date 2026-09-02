import re
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF

from src.parsers.base import BaseParser
from src.parsers.segmenter import RuleBasedSentenceSegmenter
from src.domain.models.document import Book, Chapter, Paragraph, Sentence


class PdfParser(BaseParser):
    """
    Parser for PDF documents using PyMuPDF (fitz) and RuleBasedSentenceSegmenter.
    Includes de-hyphenation across line breaks and reading-order block sorting.
    """

    def __init__(self, segmenter: Optional[RuleBasedSentenceSegmenter] = None):
        self.segmenter = segmenter or RuleBasedSentenceSegmenter()

    def parse(self, file_path: Path) -> Book:
        book = Book(
            title=file_path.stem,
            source_language="en",
            target_language="uk"
        )

        doc = fitz.open(file_path)

        current_chapter = Chapter(title="Chapter 1", order_index=0)
        chapter_idx = 1

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")

            # blocks: (x0, y0, x1, y1, "text", block_no, block_type)
            # block_type == 0 indicates text
            text_blocks = [b for b in blocks if b[6] == 0]
            # Sort blocks by vertical position y0, then horizontal x0
            text_blocks.sort(key=lambda b: (b[1], b[0]))

            for block in text_blocks:
                raw_text = block[4].strip()
                if not raw_text:
                    continue

                # 1. De-hyphenate words broken across line breaks (e.g. "infor- \n mation" -> "information")
                text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', raw_text)
                text = text.replace('\n', ' ').strip()

                # Chapter detection heuristic: short uppercase text or chapter keyword
                if len(text) < 50 and (text.isupper() or re.match(r'^(?:Chapter|Розділ|ЧАСТИНА|\#)\s+\w+', text, re.IGNORECASE)):
                    if current_chapter.paragraphs:
                        book.chapters.append(current_chapter)

                    current_chapter = Chapter(title=text, order_index=chapter_idx)
                    chapter_idx += 1
                    continue

                # Process paragraph sentences with RuleBasedSentenceSegmenter
                paragraph = Paragraph()
                raw_sentences = self.segmenter.split_sentences(text)

                for s_idx, raw_sentence in enumerate(raw_sentences):
                    if not raw_sentence.strip():
                        continue
                    sentence = Sentence(
                        original_text=raw_sentence.strip(),
                        order_index=s_idx
                    )
                    paragraph.sentences.append(sentence)

                if paragraph.sentences:
                    current_chapter.paragraphs.append(paragraph)

        # Append final chapter if populated
        if current_chapter.paragraphs:
            book.chapters.append(current_chapter)

        doc.close()
        return book
