import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
from fpdf import FPDF

from src.writers.base import BaseWriter
from src.domain.models.document import Book

logger = logging.getLogger(__name__)


class PdfWriter(BaseWriter):
    """
    Exports a translated Book DOM tree to PDF format using fpdf2.
    Features robust cross-platform Unicode TrueType font resolution (Windows, Linux, macOS)
    to guarantee flawless rendering of Cyrillic (Ukrainian) characters without glyph corruption.
    """

    FONT_CANDIDATES = [
        # Windows system fonts
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
        # Linux / Unix system fonts
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
        Path(os.path.expanduser("~/.fonts/DejaVuSans.ttf")),
        # macOS system fonts
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/SFNSText.ttf"),
        Path("/Library/Fonts/DejaVuSans.ttf"),
        # Project assets bundled font (if present)
        Path(__file__).resolve().parent.parent / "assets" / "fonts" / "DejaVuSans.ttf",
    ]

    def _resolve_unicode_font(self) -> Tuple[str, Optional[Path]]:
        """
        Scans platform font repositories for a suitable TrueType font supporting Unicode/Cyrillic.
        """
        for candidate in self.FONT_CANDIDATES:
            if candidate.exists() and candidate.is_file():
                font_name = candidate.stem.replace(" ", "")
                return font_name, candidate

        return "helvetica", None

    def write(self, book: Book, output_path: Path, template_path: Optional[Path] = None) -> Path:
        """
        Renders the translated Book DOM into a PDF file at output_path.
        """
        pdf = FPDF()
        pdf.add_page()

        font_name, font_path = self._resolve_unicode_font()

        if font_path is not None:
            try:
                pdf.add_font(font_name, "", str(font_path))
                pdf.set_font(font_name, size=12)
            except Exception as e:
                logger.warning(f"Failed to load TrueType font {font_path}: {e}. Falling back to helvetica.")
                font_name = "helvetica"
                pdf.set_font("helvetica", size=12)
        else:
            pdf.set_font("helvetica", size=12)
            logger.warning("No system Unicode TTF font found. Cyrillic characters may not render properly in helvetica.")

        # Document Title
        pdf.set_font(font_name, size=18)
        title_text = book.title or "Translated Document"
        pdf.cell(0, 12, text=title_text, new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(8)

        # Chapters and Paragraphs
        for chapter in book.chapters:
            # Chapter Header
            pdf.set_font(font_name, size=14)
            chapter_title = chapter.translated_title if chapter.translated_title else chapter.title
            if chapter_title:
                pdf.cell(0, 10, text=chapter_title, new_x="LMARGIN", new_y="NEXT", align='L')
                pdf.ln(3)

            # Paragraph text
            pdf.set_font(font_name, size=11)
            for paragraph in chapter.paragraphs:
                sorted_sentences = sorted(paragraph.sentences, key=lambda s: s.order_index)
                translated_sentences = [
                    s.translated_text.strip() if s.translated_text is not None else s.original_text
                    for s in sorted_sentences
                ]
                # Filter out empty entries and join cleanly
                para_sentences = [s.strip() for s in translated_sentences if s and s.strip()]
                if para_sentences:
                    para_text = " ".join(para_sentences)
                    pdf.multi_cell(0, 7, text=para_text)
                    pdf.ln(3)

            pdf.ln(6)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(output_path))
        return output_path
