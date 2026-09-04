from pathlib import Path
from typing import Optional

from src.writers.base import BaseWriter
from src.domain.models.document import Book
from src.domain.models.segment import SegmentStatus


class TxtWriter(BaseWriter):
    def __init__(self, allow_unreviewed: bool = False):
        self.allow_unreviewed = allow_unreviewed

    def write(
        self,
        book: Book,
        output_path: Path,
        template_path: Optional[Path] = None,
        allow_unreviewed: Optional[bool] = None,
    ) -> Path:
        effective_allow_unreviewed = (
            self.allow_unreviewed if allow_unreviewed is None else allow_unreviewed
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for chapter in book.chapters:
                f.write(f"{chapter.translated_title or chapter.title}\n\n")

                for paragraph in chapter.paragraphs:
                    # Sort sentences strictly by order_index for exact literary assembly
                    sorted_sentences = sorted(paragraph.sentences, key=lambda s: s.order_index)
                    translated_sentences = []
                    for s in sorted_sentences:
                        # Gating to ACCEPTED segments by default
                        status = getattr(s, "status", None)
                        if status is not None and not effective_allow_unreviewed:
                            status_str = getattr(status, "value", str(status)).upper()
                            if status_str != SegmentStatus.ACCEPTED.value:
                                continue

                        txt = (
                            s.translated_text.strip()
                            if s.translated_text is not None
                            else s.original_text
                        )
                        translated_sentences.append(txt)

                    para_text = " ".join(s.strip() for s in translated_sentences if s and s.strip())
                    if para_text:
                        f.write(para_text + "\n\n")

                f.write("\n")

        return output_path

