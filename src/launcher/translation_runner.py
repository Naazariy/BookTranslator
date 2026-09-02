"""
Decoupled programmatic translation job runner bridging GUI and CLI with TwoStageTranslationPipeline.
Eliminates sys.argv spoofing, establishes structured progress observation, and supports cooperative cancellation.
Part of Milestone 1 (UI & Concurrency Optimization) for BookTranslator.
"""
from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union
from uuid import UUID

from src.config.settings import settings
from src.domain.models.chunk import ChunkStatus
from src.knowledge_base.db_schema import init_db
from src.config.app_context import ApplicationContainer
from src.launcher.concurrency import CancellationToken, ProgressEvent

logger = logging.getLogger(__name__)


@dataclass
class TranslationJobConfig:
    """Configuration options for a programmatic translation execution."""
    input_file: Path
    output_file: Path
    cancellation_token: Optional[CancellationToken] = None
    progress_callback: Optional[Callable[[ProgressEvent], None]] = None


@dataclass
class TranslationResult:
    """Result data returned upon successful completion of a translation job."""
    success: bool
    book_id: UUID
    input_file: Path
    output_file: Path
    total_chunks: int = 0
    message: str = ""


def _call_stage_safe(
    stage_func: Callable[..., Any],
    book_id: UUID,
    cancellation_token: CancellationToken,
    progress_callback: Optional[Callable[[ProgressEvent], None]],
) -> Any:
    """
    Safely invoke a pipeline stage function inspecting its parameters to support
    both legacy single-arg signatures and new cooperative-cancellation signatures.
    """
    sig = inspect.signature(stage_func)
    kwargs: dict[str, Any] = {}

    if "cancellation_token" in sig.parameters:
        kwargs["cancellation_token"] = cancellation_token
    elif "cancel_token" in sig.parameters:
        kwargs["cancel_token"] = cancellation_token

    if "progress_callback" in sig.parameters:
        kwargs["progress_callback"] = progress_callback
    elif "progress_cb" in sig.parameters:
        kwargs["progress_cb"] = progress_callback

    return stage_func(book_id, **kwargs)


def count_existing_chunks(input_file: Union[Path, str]) -> int:
    """Helper to check if a translation cache exists for the given file."""
    resolved_input = Path(input_file)
    if not resolved_input.exists():
        return 0
    file_hash = hashlib.md5(str(resolved_input.resolve()).encode("utf-8")).hexdigest()
    book_id = UUID(hex=file_hash)
    init_db(settings.db_path)
    container = ApplicationContainer()
    container.config.from_pydantic(settings)
    kb_repo = container.kb_repository()
    return kb_repo.count_chunks_for_book(book_id)


def execute_translation_job(
    config_or_input: Union[TranslationJobConfig, Path, str, None] = None,
    output_file: Optional[Union[Path, str]] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    input_file: Optional[Union[Path, str]] = None,
    progress_cb: Optional[Callable[[ProgressEvent], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
    clear_cache: bool = False,
    **kwargs: Any,
) -> TranslationResult:
    """
    Programmatic translation execution entrypoint decoupled from sys.argv and CLI parsers.
    Orchestrates:
      1. Schema & IoC Container initialization
      2. Document parsing and deterministic UUID generation
      3. Preprocessing (NER & Glossary)
      4. Chunk stream creation and checkpointing
      5. Stage 1 (NLLB MT Pass) with cooperative cancellation
      6. Stage 2 (Aya Literary Refinement Pass) with cooperative cancellation
      7. O(S) DOM Tree Reconciliation
      8. Document writing
    """
    # Normalize arguments
    if isinstance(config_or_input, TranslationJobConfig):
        resolved_input = Path(config_or_input.input_file)
        resolved_output = Path(config_or_input.output_file)
        tok = config_or_input.cancellation_token or cancel_token or cancellation_token or CancellationToken()
        cb = config_or_input.progress_callback or progress_cb or progress_callback
    else:
        in_path = config_or_input or input_file
        if in_path is None:
            raise ValueError("Input file path must be provided.")
        if output_file is None:
            raise ValueError("Output file path must be provided.")
        resolved_input = Path(in_path)
        resolved_output = Path(output_file)
        tok = cancel_token or cancellation_token or CancellationToken()
        cb = progress_cb or progress_callback

    tok.raise_if_cancelled()

    if not resolved_input.exists():
        raise FileNotFoundError(f"Input file not found: {resolved_input}")

    # Optional Hugging Face Hub login if configured
    if settings.hf_token and not settings.offline_mode:
        try:
            from huggingface_hub import login
            login(token=settings.hf_token)
        except Exception as e:
            logger.debug(f"HF Hub login skipped or failed: {e}")

    # Initialize Database Schema
    logger.info("Initializing Knowledge Base Checkpoints database...")
    init_db(settings.db_path)

    # Initialize Dependency Injection Container
    logger.info("Initializing Application Container...")
    container = ApplicationContainer()
    container.config.from_pydantic(settings)

    # Resolve Services
    doc_manager = container.document_manager()
    preprocessor = container.preprocessing_pipeline()
    chunk_manager = container.chunk_manager()
    translator = container.translation_pipeline()
    kb_repo = container.kb_repository()

    # 1. Parse Document
    tok.raise_if_cancelled()
    if cb:
        cb(
            ProgressEvent(
                stage="PARSING",
                stage_name="Парсинг документа",
                current=0,
                total=100,
                percentage=0.0,
                eta_seconds=0.0,
                status_message=f"Завантаження та парсинг файлу: {resolved_input.name}",
            )
        )
    logger.info(f"Parsing document: {resolved_input}")
    book = doc_manager.load_document(resolved_input)

    # Compute deterministic UUIDs
    file_hash = hashlib.md5(str(resolved_input.resolve()).encode("utf-8")).hexdigest()
    book.id = UUID(hex=file_hash)

    for c_idx, chapter in enumerate(book.chapters):
        chapter.id = UUID(hex=hashlib.md5(f"{book.id}_c{c_idx}".encode("utf-8")).hexdigest())
        for p_idx, paragraph in enumerate(chapter.paragraphs):
            paragraph.id = UUID(hex=hashlib.md5(f"{chapter.id}_p{p_idx}".encode("utf-8")).hexdigest())
            for s_idx, sentence in enumerate(paragraph.sentences):
                sentence.id = UUID(hex=hashlib.md5(f"{paragraph.id}_s{s_idx}".encode("utf-8")).hexdigest())

    # 2. Preprocess (NER & Glossary)
    tok.raise_if_cancelled()
    if cb:
        cb(
            ProgressEvent(
                stage="PREPROCESSING",
                stage_name="Препроцесинг документа",
                current=5,
                total=100,
                percentage=0.05,
                eta_seconds=0.0,
                status_message="Пошук іменованих сутностей та аналіз термінів...",
            )
        )
    logger.info("Running NER and glossary preprocessing...")
    preprocessor.process(book)

    # 3. Create Chunks Stream
    tok.raise_if_cancelled()
    
    if clear_cache and hasattr(kb_repo, "delete_chunks_for_book"):
        logger.info(f"Clearing existing translation cache for book {book.id}")
        kb_repo.delete_chunks_for_book(book.id)

    existing_chunks = kb_repo.count_chunks_for_book(book.id)
    if existing_chunks == 0:
        logger.info("Generating translation chunks stream...")
        chunks_stream = list(chunk_manager.create_chunks_stream(book, max_tokens=settings.max_tokens_per_chunk))
        if hasattr(kb_repo, "save_chunks_batch"):
            kb_repo.save_chunks_batch(chunks_stream)
        elif hasattr(kb_repo, "save_chunk_state_batch"):
            kb_repo.save_chunk_state_batch(chunks_stream)
        else:
            for chunk in chunks_stream:
                kb_repo.save_chunk_state(chunk)
    else:
        logger.info(f"Resuming translation: {existing_chunks} checkpoints found in DB.")

    # 4. Stage 1: NLLB MT Pass
    tok.raise_if_cancelled()
    logger.info("Executing Stage 1 (NLLB Machine Translation)...")
    _call_stage_safe(translator.execute_nllb_stage, book.id, tok, cb)

    # 5. Stage 2: Aya Literary Refinement Pass
    tok.raise_if_cancelled()
    logger.info("Executing Stage 2 (Aya Literary Refinement)...")
    _call_stage_safe(translator.execute_aya_stage, book.id, tok, cb)

    # 6. Rebuild Document DOM Structure (O(S) Lookup)
    tok.raise_if_cancelled()
    if cb:
        cb(
            ProgressEvent(
                stage="REBUILDING",
                stage_name="Збирання документа",
                current=95,
                total=100,
                percentage=0.95,
                eta_seconds=0.0,
                status_message="Реконсиляція DOM-дерева документа...",
            )
        )
    logger.info("Rebuilding document DOM structure...")
    refined_chunks = list(kb_repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))

    # Build O(S) sentence dictionary map
    sentence_map = {s.id: s for chapter in book.chapters for p in chapter.paragraphs for s in p.sentences}

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
            continue

        translation = chunk.final_translation or chunk.draft_translation
        if not translation:
            continue

        from src.parsers.segmenter import RuleBasedSentenceSegmenter
        segmenter = RuleBasedSentenceSegmenter()
        split_sents = segmenter.split_sentences(translation)

        if len(split_sents) == len(target_sents):
            for src_s, tr_t in zip(target_sents, split_sents):
                sentence_map[src_s.id].translated_text = tr_t.strip()
        elif len(target_sents) == 1:
            sentence_map[target_sents[0].id].translated_text = translation.strip()
        elif split_sents:
            if len(split_sents) < len(target_sents):
                for idx in range(len(split_sents)):
                    sentence_map[target_sents[idx].id].translated_text = split_sents[idx].strip()
                for idx in range(len(split_sents), len(target_sents)):
                    sentence_map[target_sents[idx].id].translated_text = ""
            else:
                for idx in range(len(target_sents) - 1):
                    sentence_map[target_sents[idx].id].translated_text = split_sents[idx].strip()
                sentence_map[target_sents[-1].id].translated_text = " ".join(split_sents[len(target_sents) - 1:]).strip()

    # 7. Write Output Document
    tok.raise_if_cancelled()
    logger.info(f"Saving translated document to: {resolved_output}")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    doc_manager.save_document(book, resolved_output)

    if cb:
        cb(
            ProgressEvent(
                stage="COMPLETED",
                stage_name="Завершено",
                current=100,
                total=100,
                percentage=1.0,
                eta_seconds=0.0,
                status_message="Переклад успішно завершено та збережено!",
            )
        )
    logger.info("Translation job completed successfully.")

    # Auto-clear cache on success
    if hasattr(kb_repo, "delete_chunks_for_book"):
        logger.info(f"Clearing translation cache for completed book {book.id}")
        kb_repo.delete_chunks_for_book(book.id)

    return TranslationResult(
        success=True,
        book_id=book.id,
        input_file=resolved_input,
        output_file=resolved_output,
        total_chunks=len(refined_chunks),
        message="Translation job completed successfully.",
    )
