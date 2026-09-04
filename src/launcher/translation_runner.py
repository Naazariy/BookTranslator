"""
Decoupled programmatic translation job runner bridging GUI and CLI with TwoStageTranslationPipeline.
Eliminates sys.argv spoofing, establishes structured progress observation, and supports cooperative cancellation.
Part of Milestone 1 (UI & Concurrency Optimization) for BookTranslator.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID

from src.config.settings import settings
from src.domain.models.chunk import ChunkStatus
from src.domain.models.segment import SegmentStatus, TranslationSegment
from src.knowledge_base.db_schema import init_db
from src.config.app_context import ApplicationContainer
from src.launcher.concurrency import CancellationToken, ProgressEvent

logger = logging.getLogger(__name__)


class IncompatibleJobConfigurationError(Exception):
    """Raised when resuming a translation job with incompatible model, prompt, or pipeline configuration."""
    pass


def compute_job_fingerprint(
    file_bytes: bytes,
    settings: Any,
    pipeline_version: str = "2.0.0",
    kb_repo: Optional[Any] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Computes a deterministic job_id and configuration fingerprint for a translation job.
    Incorporates:
      - sha256(file_bytes) (input document content identity)
      - pipeline version
      - model configurations (NLLB, Aya, temperature, top_p, repetition penalty)
      - prompt template hash
      - knowledge base schema revision
    Returns:
      (job_id, fingerprint, payload_dict)
    """
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    book_id_hex = content_hash[:32]

    prompt_path = getattr(settings, "prompt_file_path", Path("data/prompts/editing_prompt.md"))
    prompt_content = ""
    if isinstance(prompt_path, (str, Path)) and Path(prompt_path).exists():
        try:
            prompt_content = Path(prompt_path).read_text(encoding="utf-8")
        except Exception:
            prompt_content = ""
    prompt_hash = hashlib.sha256(prompt_content.encode("utf-8")).hexdigest()

    kb_revision = 1
    if kb_repo and hasattr(kb_repo, "get_schema_version") and callable(kb_repo.get_schema_version):
        try:
            kb_revision = kb_repo.get_schema_version()
        except Exception:
            kb_revision = 1

    payload = {
        "content_hash": content_hash,
        "pipeline_version": pipeline_version,
        "nllb_model": getattr(settings, "nllb_model_name", ""),
        "temperature": getattr(settings, "temperature", 0.0),
        "aya_temperature": getattr(settings, "aya_temperature", 0.0),
        "do_sample": getattr(settings, "do_sample", False),
        "aya_do_sample": getattr(settings, "aya_do_sample", False),
        "top_p": getattr(settings, "top_p", 1.0),
        "aya_top_p": getattr(settings, "aya_top_p", 1.0),
        "repetition_penalty": getattr(settings, "repetition_penalty", 1.02),
        "aya_repetition_penalty": getattr(settings, "aya_repetition_penalty", 1.02),
        "prompt_hash": prompt_hash,
        "kb_revision": kb_revision,

    }
    payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    fingerprint = hashlib.sha256(payload_bytes).hexdigest()
    job_id = f"{book_id_hex}_{fingerprint[:16]}"
    return job_id, fingerprint, payload


def _verify_or_save_job_fingerprint(
    db_path: Path,
    book_id: UUID,
    job_id: str,
    fingerprint: str,
    payload: Dict[str, Any],
    clear_cache: bool = False,
) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, timeout=10.0) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_fingerprints (
                book_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor = conn.cursor()
        if clear_cache:
            cursor.execute("DELETE FROM job_fingerprints WHERE book_id = ?", (str(book_id),))
            row = None
        else:
            cursor.execute(
                "SELECT job_id, fingerprint, payload_json FROM job_fingerprints WHERE book_id = ?",
                (str(book_id),),
            )
            row = cursor.fetchone()

        if row is not None:
            prev_job_id, prev_fingerprint, prev_payload_json = row
            if prev_fingerprint != fingerprint:
                raise IncompatibleJobConfigurationError(
                    f"Incompatible resume: Configuration fingerprint mismatch for book {book_id}. "
                    f"Recorded: {prev_fingerprint[:16]}, Current: {fingerprint[:16]}. "
                    f"Cannot safely resume translation with modified model, prompt, or pipeline settings. "
                    f"Use clear_cache=True to restart with current settings."
                )

        cursor.execute("""
            INSERT INTO job_fingerprints (book_id, job_id, fingerprint, payload_json, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(book_id) DO UPDATE SET
                job_id = excluded.job_id,
                fingerprint = excluded.fingerprint,
                payload_json = excluded.payload_json,
                updated_at = CURRENT_TIMESTAMP
        """, (str(book_id), job_id, fingerprint, json.dumps(payload, sort_keys=True)))


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
    job_id: Optional[str] = None
    fingerprint: Optional[str] = None
    message: str = ""



def _call_stage_safe(
    stage_func: Callable[..., Any],
    book_id: UUID,
    cancellation_token: CancellationToken,
    progress_callback: Optional[Callable[[ProgressEvent], None]],
    **extra_kwargs: Any,
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

    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    for k, v in extra_kwargs.items():
        if k in sig.parameters or has_var_keyword:
            kwargs[k] = v

    return stage_func(book_id, **kwargs)



def count_existing_chunks(input_file: Union[Path, str]) -> int:
    """Helper to check if a translation cache exists for the given file."""
    resolved_input = Path(input_file)
    if not resolved_input.exists():
        return 0
    with open(resolved_input, "rb") as f:
        file_bytes = f.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    book_id = UUID(hex=content_hash[:32])
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


    # Configure deterministic Stage 2 generation mode
    if hasattr(translator, "aya_engine") and getattr(settings, "stage2_deterministic", True):
        translator.aya_engine.temperature = getattr(settings, "temperature", 0.0)
        translator.aya_engine.top_p = getattr(settings, "top_p", 1.0)
        translator.aya_engine.repetition_penalty = getattr(settings, "repetition_penalty", 1.02)


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

    # Compute deterministic UUIDs from file content bytes
    with open(resolved_input, "rb") as f:
        file_bytes = f.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    book.id = UUID(hex=content_hash[:32])

    for c_idx, chapter in enumerate(book.chapters):
        chapter.id = UUID(hex=hashlib.sha256(f"{book.id}_c{c_idx}".encode("utf-8")).hexdigest()[:32])
        for p_idx, paragraph in enumerate(chapter.paragraphs):
            paragraph.id = UUID(hex=hashlib.sha256(f"{chapter.id}_p{p_idx}".encode("utf-8")).hexdigest()[:32])
            for s_idx, sentence in enumerate(paragraph.sentences):
                sentence.id = UUID(hex=hashlib.sha256(f"{paragraph.id}_s{s_idx}".encode("utf-8")).hexdigest()[:32])

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
        clear_cache=clear_cache,
    )


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

    # 3. Create Chunks Stream & TranslationSegments
    tok.raise_if_cancelled()
    
    if clear_cache and hasattr(kb_repo, "delete_chunks_for_book"):
        logger.info(f"Clearing existing translation cache for book {book.id}")
        kb_repo.delete_chunks_for_book(book.id)

    # Initialize paragraph-level TranslationSegments
    segments: List[TranslationSegment] = []
    seg_order = 0
    for chapter in book.chapters:
        for paragraph in chapter.paragraphs:
            p_text = " ".join(
                getattr(s, "source_for_translation", s.original_text) for s in paragraph.sentences if getattr(s, "source_for_translation", s.original_text)
            ).strip()
            if not p_text:
                continue
            seg = TranslationSegment.from_paragraph(
                paragraph=paragraph,
                book_id=book.id,
                chapter_id=chapter.id,
                order_index=seg_order
            )
            segments.append(seg)
            seg_order += 1

    if hasattr(kb_repo, "save_segments_batch") and segments:
        kb_repo.save_segments_batch(segments)

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
    _call_stage_safe(translator.execute_nllb_stage, book.id, tok, cb, segments=segments)

    # 5. Stage 2: Aya Literary Refinement Pass
    tok.raise_if_cancelled()
    logger.info("Executing Stage 2 (Aya Literary Refinement)...")
    _call_stage_safe(translator.execute_aya_stage, book.id, tok, cb, segments=segments)

    # 6. Stage 3: Quality Assurance & Bounded Repair Pass
    tok.raise_if_cancelled()
    logger.info("Executing Stage 3 (Quality Assurance & Bounded Repair)...")
    if cb:
        cb(
            ProgressEvent(
                stage="QA_REPAIR",
                stage_name="Контроль якості та виправлення",
                current=75,
                total=100,
                percentage=0.75,
                eta_seconds=0.0,
                status_message="Перевірка якості перекладу та автоматичне виправлення помилок...",
            )
        )

    if hasattr(translator, "execute_qa_and_repair_stage"):
        _call_stage_safe(
            translator.execute_qa_and_repair_stage,
            book.id,
            tok,
            cb,
            segments=segments,
        )

    # 7. Rebuild Document DOM Structure (O(S) Lookup with Strict QA Status Gating)
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

    refined_chunks: List[Any] = []
    has_refined_segments = any(
        (seg.refined_translation or seg.status in (SegmentStatus.EDITED, SegmentStatus.ACCEPTED, SegmentStatus.REVIEW_REQUIRED))
        for seg in segments
    )
    if has_refined_segments:
        segment_map = {seg.paragraph_id: seg for seg in segments}
        for chapter in book.chapters:
            for paragraph in chapter.paragraphs:
                seg = segment_map.get(paragraph.id)
                if not seg:
                    continue

                if seg.status == SegmentStatus.ACCEPTED or (seg.status == SegmentStatus.EDITED and (seg.final_translation or seg.refined_translation)):
                    final_text = seg.final_translation or seg.refined_translation
                    try:
                        paragraph.translated_text = final_text
                    except (ValueError, AttributeError):
                        pass
                    try:
                        paragraph.status = SegmentStatus.ACCEPTED
                    except (ValueError, AttributeError):
                        pass
                    if paragraph.sentences:
                        paragraph.sentences[0].translated_text = final_text
                        paragraph.sentences[0].status = SegmentStatus.ACCEPTED
                        for extra_s in paragraph.sentences[1:]:
                            extra_s.translated_text = ""
                            extra_s.status = SegmentStatus.ACCEPTED

                elif seg.status == SegmentStatus.REVIEW_REQUIRED:
                    candidate_text = seg.final_translation or seg.refined_translation or seg.draft_translation
                    try:
                        paragraph.translated_text = candidate_text
                    except (ValueError, AttributeError):
                        pass
                    try:
                        paragraph.status = SegmentStatus.REVIEW_REQUIRED
                    except (ValueError, AttributeError):
                        pass
                    if paragraph.sentences:
                        paragraph.sentences[0].translated_text = candidate_text
                        paragraph.sentences[0].status = SegmentStatus.REVIEW_REQUIRED
                        for extra_s in paragraph.sentences[1:]:
                            extra_s.translated_text = ""
                            extra_s.status = SegmentStatus.REVIEW_REQUIRED

                elif seg.status == SegmentStatus.FAILED:
                    try:
                        paragraph.translated_text = seg.translated_text
                    except (ValueError, AttributeError):
                        pass
                    try:
                        paragraph.status = SegmentStatus.FAILED
                    except (ValueError, AttributeError):
                        pass
                    if paragraph.sentences:
                        paragraph.sentences[0].translated_text = seg.translated_text
                        paragraph.sentences[0].status = SegmentStatus.FAILED
                        for extra_s in paragraph.sentences[1:]:
                            extra_s.translated_text = ""
                            extra_s.status = SegmentStatus.FAILED
                else:
                    try:
                        paragraph.translated_text = seg.translated_text
                    except (ValueError, AttributeError):
                        pass
                    try:
                        paragraph.status = seg.status
                    except (ValueError, AttributeError):
                        pass
                    if paragraph.sentences:
                        paragraph.sentences[0].translated_text = seg.translated_text
                        paragraph.sentences[0].status = seg.status
                        for extra_s in paragraph.sentences[1:]:
                            extra_s.translated_text = ""
                            extra_s.status = seg.status
    else:
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
                        sentence_map[s.id].status = SegmentStatus.ACCEPTED
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
                    sentence_map[src_s.id].status = SegmentStatus.ACCEPTED
            elif len(target_sents) == 1:
                sentence_map[target_sents[0].id].translated_text = translation.strip()
                sentence_map[target_sents[0].id].status = SegmentStatus.ACCEPTED
            elif split_sents:
                if len(split_sents) < len(target_sents):
                    for idx in range(len(split_sents)):
                        sentence_map[target_sents[idx].id].translated_text = split_sents[idx].strip()
                        sentence_map[target_sents[idx].id].status = SegmentStatus.ACCEPTED
                    for idx in range(len(split_sents), len(target_sents)):
                        sentence_map[target_sents[idx].id].translated_text = ""
                else:
                    for idx in range(len(target_sents) - 1):
                        sentence_map[target_sents[idx].id].translated_text = split_sents[idx].strip()
                        sentence_map[target_sents[idx].id].status = SegmentStatus.ACCEPTED
                    sentence_map[target_sents[-1].id].translated_text = " ".join(split_sents[len(target_sents) - 1:]).strip()
                    sentence_map[target_sents[-1].id].status = SegmentStatus.ACCEPTED



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
        total_chunks=len(segments) if has_refined_segments else len(refined_chunks),
        job_id=book.job_id,
        fingerprint=book.fingerprint,
        message="Translation job completed successfully.",
    )

