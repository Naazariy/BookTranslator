import logging
import re
from typing import Iterator, List, Set, Tuple, Optional
from uuid import uuid4

from src.domain.models.document import Book, Sentence
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.domain.models.knowledge import GlossaryItem
from src.domain.interfaces.chunker import IChunkManager

logger = logging.getLogger(__name__)


def filter_glossary_for_chunk(
    glossary: List[GlossaryItem],
    chunk_text: str,
    draft_text: str = ""
) -> List[GlossaryItem]:
    """
    Filters a list of glossary items to include only terms appearing
    as complete words in either chunk_text (source text) or draft_text.
    Respects the case_sensitive flag of each GlossaryItem.
    """
    if not glossary:
        return []

    combined = f"{chunk_text or ''} {draft_text or ''}"
    if not combined.strip():
        return []

    matching: List[GlossaryItem] = []
    for item in glossary:
        if not item.source_term or not item.source_term.strip():
            continue
        flags = 0 if item.case_sensitive else re.IGNORECASE
        pattern = r'\b' + re.escape(item.source_term.strip()) + r'\b'
        if re.search(pattern, combined, flags):
            matching.append(item)

    return matching


class ChunkManager(IChunkManager):
    filter_glossary_for_chunk = staticmethod(filter_glossary_for_chunk)
    def create_chunks_stream(
        self, book: Book, max_tokens: int = 512, overlap_sentences: int = 2
    ) -> Iterator[TranslationChunk]:
        logger.info(f"Starting chunking for book {book.id} (max_tokens={max_tokens}, overlap_sentences={overlap_sentences})")

        def estimate_tokens(s: Sentence) -> int:
            words = len(s.original_text.split())
            return max(words, int(len(s.original_text) / 3.5), 1)

        for chapter in book.chapters:
            # Flatten all sentences in this chapter with their paragraph indices
            chapter_sentences: List[Tuple[int, Sentence]] = []
            for p_idx, paragraph in enumerate(chapter.paragraphs):
                for sentence in paragraph.sentences:
                    chapter_sentences.append((p_idx, sentence))

            if not chapter_sentences:
                continue

            overlap_buffer: List[Sentence] = []
            curr_target_sentences: List[Sentence] = []
            curr_para_indices: Set[int] = set()
            curr_tokens = 0

            for p_idx, sentence in chapter_sentences:
                s_tokens = estimate_tokens(sentence)

                # Check if adding this sentence exceeds limit (either max_tokens or 10 sentences)
                if curr_target_sentences and (curr_tokens + s_tokens > max_tokens or len(curr_target_sentences) >= 10):
                    context_ids = [s.id for s in overlap_buffer]
                    target_ids = [s.id for s in curr_target_sentences]
                    chunk_source_sentences = overlap_buffer + curr_target_sentences
                    total_tokens = sum(estimate_tokens(s) for s in chunk_source_sentences)

                    yield TranslationChunk(
                        id=uuid4(),
                        book_id=book.id,
                        chapter_id=chapter.id,
                        paragraph_indices=sorted(list(curr_para_indices)),
                        source_sentences=chunk_source_sentences,
                        context_sentence_ids=context_ids,
                        target_sentence_ids=target_ids,
                        token_count=total_tokens,
                        context=ChunkContext(previous_sentences=[s.original_text for s in overlap_buffer]),
                        status=ChunkStatus.PENDING
                    )

                    if overlap_sentences > 0:
                        overlap_buffer = curr_target_sentences[-overlap_sentences:]
                    else:
                        overlap_buffer = []

                    curr_target_sentences = [sentence]
                    curr_para_indices = {p_idx}
                    curr_tokens = s_tokens
                else:
                    curr_target_sentences.append(sentence)
                    curr_para_indices.add(p_idx)
                    curr_tokens += s_tokens

            if curr_target_sentences:
                context_ids = [s.id for s in overlap_buffer]
                target_ids = [s.id for s in curr_target_sentences]
                chunk_source_sentences = overlap_buffer + curr_target_sentences
                total_tokens = sum(estimate_tokens(s) for s in chunk_source_sentences)

                yield TranslationChunk(
                    id=uuid4(),
                    book_id=book.id,
                    chapter_id=chapter.id,
                    paragraph_indices=sorted(list(curr_para_indices)),
                    source_sentences=chunk_source_sentences,
                    context_sentence_ids=context_ids,
                    target_sentence_ids=target_ids,
                    token_count=total_tokens,
                    context=ChunkContext(previous_sentences=[s.original_text for s in overlap_buffer]),
                    status=ChunkStatus.PENDING
                )
