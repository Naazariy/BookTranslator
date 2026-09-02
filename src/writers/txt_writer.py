from pathlib import Path
from typing import Optional

from src.writers.base import BaseWriter
from src.domain.models.document import Book


class TxtWriter(BaseWriter):
    def write(self, book: Book, output_path: Path, template_path: Optional[Path] = None) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for chapter in book.chapters:
                f.write(f"{chapter.translated_title or chapter.title}\n\n")
                
                for paragraph in chapter.paragraphs:
                    # Sort sentences strictly by order_index for exact literary assembly
                    sorted_sentences = sorted(paragraph.sentences, key=lambda s: s.order_index)
                    translated_sentences = [
                        s.translated_text.strip() if s.translated_text is not None else s.original_text 
                        for s in sorted_sentences
                    ]
                    para_text = " ".join(s.strip() for s in translated_sentences if s and s.strip())
                    if para_text:
                        f.write(para_text + "\n\n")
                    
                f.write("\n")
                
        return output_path
