import argparse
import sys
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Set
from uuid import UUID

from src.config.settings import settings
from src.config.app_context import ApplicationContainer
from src.knowledge_base.db_schema import init_db
from src.domain.models.document import Book
from src.domain.models.chunk import TranslationChunk, ChunkStatus
from src.domain.models.segment import SegmentStatus
from src.parsers.segmenter import RuleBasedSentenceSegmenter
from src.launcher.translation_runner import compute_job_fingerprint, _verify_or_save_job_fingerprint



def reconcile_document_dom(
    book: Book,
    refined_chunks: List[TranslationChunk],
    segmenter: Optional[RuleBasedSentenceSegmenter] = None
) -> Book:
    """
    Reconstructs the translated Book DOM tree in O(S) time with sentence-level alignment.
    Sentence is the source of truth. Each sentence is mapped from target_sentence_ids,
    eliminating overlap duplication while preserving paragraph and chapter structures.
    """
    if segmenter is None:
        segmenter = RuleBasedSentenceSegmenter()

    # 1. Build O(S) fast sentence lookup index
    sentence_map = {
        sentence.id: sentence
        for chapter in book.chapters
        for paragraph in chapter.paragraphs
        for sentence in paragraph.sentences
    }

    for chapter in book.chapters:
        if not chapter.translated_title:
            chapter.translated_title = f"[UK] {chapter.title}"

    for chunk in refined_chunks:
        target_ids = set(chunk.target_sentence_ids) if chunk.target_sentence_ids else {s.id for s in chunk.source_sentences if s.id not in chunk.context_sentence_ids}
        target_sents = [s for s in chunk.source_sentences if s.id in target_ids and s.id in sentence_map]
        if not target_sents:
            continue

        has_per_sentence = any(s.translated_text for s in target_sents)
        if has_per_sentence:
            for s in target_sents:
                if s.translated_text is not None:
                    sentence_map[s.id].translated_text = s.translated_text
                    sentence_map[s.id].status = SegmentStatus.ACCEPTED
            continue

        raw_translation = chunk.final_translation or chunk.draft_translation or ""
        if not raw_translation.strip():
            continue

        # Split translation into individual sentences
        translated_sents = segmenter.split_sentences(raw_translation)

        if len(translated_sents) == len(target_sents):
            for src_s, tr_text in zip(target_sents, translated_sents):
                sentence_map[src_s.id].translated_text = tr_text.strip()
                sentence_map[src_s.id].status = SegmentStatus.ACCEPTED
        elif len(target_sents) == 1:
            sentence_map[target_sents[0].id].translated_text = raw_translation.strip()
            sentence_map[target_sents[0].id].status = SegmentStatus.ACCEPTED
        elif translated_sents:
            if len(translated_sents) < len(target_sents):
                for idx in range(len(translated_sents)):
                    sentence_map[target_sents[idx].id].translated_text = translated_sents[idx].strip()
                    sentence_map[target_sents[idx].id].status = SegmentStatus.ACCEPTED
                for idx in range(len(translated_sents), len(target_sents)):
                    sentence_map[target_sents[idx].id].translated_text = ""
            else:
                for idx in range(len(target_sents) - 1):
                    sentence_map[target_sents[idx].id].translated_text = translated_sents[idx].strip()
                    sentence_map[target_sents[idx].id].status = SegmentStatus.ACCEPTED
                sentence_map[target_sents[-1].id].translated_text = " ".join(translated_sents[len(target_sents) - 1:]).strip()
                sentence_map[target_sents[-1].id].status = SegmentStatus.ACCEPTED

    return book



def main():
    parser = argparse.ArgumentParser(description="BookTranslator: Offline Large Document Translation System")
    parser.add_argument("--file", type=str, required=True, help="Path to the input document (e.g. book.txt)")
    parser.add_argument("--out", type=str, required=True, help="Path to the output document (e.g. book_uk.txt)")
    parser.add_argument(
        "--no-unit-conversion",
        action="store_true",
        help="Disable automatic unit conversion during preprocessing"
    )
    args = parser.parse_args()

    if args.no_unit_conversion:
        settings.convert_units = False

    # Configure Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("cli")

    input_path = Path(args.file)
    output_path = Path(args.out)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if settings.hf_token:
        logger.info("Logging into Hugging Face Hub using token from config...")
        from huggingface_hub import login
        login(token=settings.hf_token)

    # Initialize DB Schema
    logger.info("Initializing Knowledge Base Checkpoints database...")
    init_db(settings.db_path)

    # Initialize IoC Container
    logger.info("Initializing Application Container...")
    container = ApplicationContainer()
    container.config.from_pydantic(settings)
    container.wire(modules=[__name__])

    # Resolve dependencies
    doc_manager = container.document_manager()
    preprocessor = container.preprocessing_pipeline()
    chunk_manager = container.chunk_manager()
    translator = container.translation_pipeline()
    kb_repo = container.kb_repository()

    # Configure deterministic Stage 2 generation mode
    if hasattr(translator, "aya_engine") and getattr(settings, "stage2_deterministic", True):
        translator.aya_engine.temperature = getattr(settings, "temperature", 0.0)
        translator.aya_engine.top_p = getattr(settings, "top_p", 1.0)
        translator.aya_engine.repetition_penalty = getattr(settings, "repetition_penalty", 1.02)


    # 1. Parse Document
    logger.info(f"Parsing document: {input_path}")
    book = doc_manager.load_document(input_path)

    with open(input_path, "rb") as f:
        file_bytes = f.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    book.id = UUID(hex=content_hash[:32])

    # Make all IDs deterministic so we can match them after re-parsing
    for c_idx, chapter in enumerate(book.chapters):
        chapter.id = UUID(hex=hashlib.sha256(f"{book.id}_c{c_idx}".encode('utf-8')).hexdigest()[:32])
        for p_idx, paragraph in enumerate(chapter.paragraphs):
            paragraph.id = UUID(hex=hashlib.sha256(f"{chapter.id}_p{p_idx}".encode('utf-8')).hexdigest()[:32])
            for s_idx, sentence in enumerate(paragraph.sentences):
                sentence.id = UUID(hex=hashlib.sha256(f"{paragraph.id}_s{s_idx}".encode('utf-8')).hexdigest()[:32])

    # Compute deterministic job_id and config fingerprint
    job_id, fingerprint, payload = compute_job_fingerprint(
        file_bytes=file_bytes,
        settings=settings,
        kb_repo=kb_repo,
    )
    book.job_id = job_id
    book.fingerprint = fingerprint

    # Verify matching configuration or initialize job fingerprint
    _verify_or_save_job_fingerprint(
        db_path=settings.db_path,
        book_id=book.id,
        job_id=job_id,
        fingerprint=fingerprint,
        payload=payload,
        clear_cache=False,
    )


    # 2. Preprocess (NER, Glossary Setup)
    preprocessor.process(book)

    # 3. Create Chunks Stream (Batch save to DB)
    existing_chunks = kb_repo.count_chunks_for_book(book.id)
    if existing_chunks == 0:
        logger.info("Creating chunk stream and pending checkpoints...")
        chunks_stream = list(chunk_manager.create_chunks_stream(book, max_tokens=settings.max_tokens_per_chunk))
        if hasattr(kb_repo, "save_chunk_state_batch"):
            kb_repo.save_chunk_state_batch(chunks_stream)
        else:
            for chunk in chunks_stream:
                kb_repo.save_chunk_state(chunk)
    else:
        logger.info(f"Found {existing_chunks} existing chunks in DB. Resuming from checkpoints...")

    # 4. Two-Stage Translation
    translator.execute_nllb_stage(book.id)
    translator.execute_aya_stage(book.id)

    # 5. Load Refined Chunks and Rebuild Book using O(S) DOM Reconciliation
    logger.info("Rebuilding translated document with O(S) DOM reconciliation...")
    refined_chunks = list(kb_repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
    reconcile_document_dom(book, refined_chunks)

    # 6. Save Output
    doc_manager.save_document(book, output_path)
    logger.info(f"Translation complete! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
