from pathlib import Path
from typing import Optional

from src.parsers.base import BaseParser
from src.parsers.segmenter import RuleBasedSentenceSegmenter
from src.domain.models.document import Book, Chapter, Paragraph, Sentence


class TxtParser(BaseParser):
    """
    Parser for plain text documents (.txt).
    Utilizes RuleBasedSentenceSegmenter for abbreviation-aware sentence extraction
    and supports multi-encoding fallback for robust file ingestion.
    """

    def __init__(self, segmenter: Optional[RuleBasedSentenceSegmenter] = None):
        self.segmenter = segmenter or RuleBasedSentenceSegmenter()

    def parse(self, file_path: Path) -> Book:
        content = self._read_file_safe(file_path)

        book = Book(
            title=file_path.stem,
            source_language="en",
            target_language="uk"
        )

        # Splitting logic: triple newline as chapter delimiter, double newline as paragraph
        raw_chapters = content.split('\n\n\n')

        for ch_idx, raw_chapter in enumerate(raw_chapters):
            if not raw_chapter.strip():
                continue

            chapter = Chapter(
                title=f"Chapter {ch_idx + 1}",
                order_index=ch_idx
            )

            raw_paragraphs = raw_chapter.split('\n\n')
            for p_idx, raw_paragraph in enumerate(raw_paragraphs):
                if not raw_paragraph.strip():
                    continue

                paragraph = Paragraph()
                # Use high-accuracy rule-based sentence segmenter
                raw_sentences = self.segmenter.split_sentences(raw_paragraph)

                for s_idx, raw_sentence in enumerate(raw_sentences):
                    if not raw_sentence.strip():
                        continue
                    sentence = Sentence(
                        original_text=raw_sentence.strip(),
                        order_index=s_idx
                    )
                    paragraph.sentences.append(sentence)

                if paragraph.sentences:
                    chapter.paragraphs.append(paragraph)

            if chapter.paragraphs:
                book.chapters.append(chapter)

        return book

    @staticmethod
    def _read_file_safe(file_path: Path) -> str:
        """Reads file content with automatic encoding detection fallback."""
        for encoding in ['utf-8', 'utf-8-sig', 'windows-1251', 'cp1252', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
