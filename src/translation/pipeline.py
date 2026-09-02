"""
Two-Stage Translation Pipeline Orchestrator.
Orchestrates Stage 1 (CTranslate2 NLLB Machine Translation) and Stage 2 (Quantized Aya Literary Refinement)
with dynamic token batching, O(N) linear progress indexing, VRAM memory defragmentation,
and stage-differentiated persistence transactions.
"""
from typing import Optional, List, Tuple, Dict, Any, Callable, Iterable, Iterator
from uuid import UUID
from dataclasses import dataclass
import time
import logging
import gc

try:
    import torch
except ImportError:
    torch = None

from src.domain.interfaces.translation import (
    ITranslationEngine,
    IEditingEngine,
    ITranslationPipeline
)
from src.domain.interfaces.knowledge_base import IKnowledgeBaseRepository
from src.domain.models.chunk import TranslationChunk, ChunkStatus, ChunkContext
from src.translation.batching import DynamicTokenBucketBatcher
from src.chunking.manager import filter_glossary_for_chunk

logger = logging.getLogger(__name__)


from typing import Any

@dataclass
class ProgressEvent:
    """Structured progress event emitted during translation pipeline execution."""
    stage: Any = 1
    current: int = 0
    total: int = 0
    eta_seconds: float = 0.0
    status_message: str = ""
    stage_name: str = ""
    percentage: float = 0.0
    speed_chunks_per_sec: float = 0.0
    message: str = ""

    def __post_init__(self) -> None:
        # Calculate percentage if not provided
        if self.percentage == 0.0 and self.total > 0:
            object.__setattr__(self, "percentage", max(0.0, min(1.0, self.current / self.total)))

        # Synchronize status_message and message
        if not self.status_message and self.message:
            object.__setattr__(self, "status_message", self.message)
        elif not self.message and self.status_message:
            object.__setattr__(self, "message", self.status_message)

        # Populate human-readable stage_name if empty
        if not self.stage_name:
            st = str(self.stage).upper()
            if st in ("1", "STAGE_1", "STAGE_1_NLLB"):
                object.__setattr__(self, "stage_name", "Етап 1: Машинний переклад (NLLB)")
            elif st in ("2", "STAGE_2", "STAGE_2_AYA"):
                object.__setattr__(self, "stage_name", "Етап 2: Літературне редагування (Aya)")
            elif st in ("PARSING", "0"):
                object.__setattr__(self, "stage_name", "Парсинг та сегментація документа")
            elif st in ("REBUILDING", "3"):
                object.__setattr__(self, "stage_name", "Збирання вихідного документа")
            else:
                object.__setattr__(self, "stage_name", str(self.stage))


def _is_cancellation_requested(cancel_token: Optional[Any]) -> bool:
    if cancel_token is None:
        return False
    if hasattr(cancel_token, "is_cancelled"):
        checker = getattr(cancel_token, "is_cancelled")
        return checker() if callable(checker) else bool(checker)
    if hasattr(cancel_token, "is_set"):
        return cancel_token.is_set()
    if hasattr(cancel_token, "cancelled"):
        return bool(cancel_token.cancelled)
    return False


class TwoStageTranslationPipeline(ITranslationPipeline):
    """
    Two-Stage Translation Pipeline coordinating NLLB draft generation and Aya post-editing.
    """
    def __init__(
        self,
        nllb_engine: ITranslationEngine,
        aya_engine: IEditingEngine,
        kb_repo: IKnowledgeBaseRepository,
        max_batch_tokens: int = 2048
    ):
        self.nllb_engine = nllb_engine
        self.aya_engine = aya_engine
        self.kb_repo = kb_repo
        self.max_batch_tokens = max_batch_tokens

    def execute_nllb_stage(
        self,
        book_id: UUID,
        cancel_token: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
        progress_cb: Optional[Callable[[ProgressEvent], None]] = None,
        source_lang: str = "en",
        target_lang: str = "uk"
    ) -> None:
        """
        Executes Stage 1: Machine Translation Pass across all pending chunks.
        Uses DynamicTokenBucketBatcher ($B_tokens=2048$) for maximum token throughput
        and persists results in 50-chunk WAL transactions.
        """
        token = cancel_token if cancel_token is not None else cancel_event
        logger.info(f"Starting Stage 1 (NLLB MT Pass) for book {book_id}")

        if hasattr(self.nllb_engine, "load_model"):
            self.nllb_engine.load_model()

        try:
            pending_chunks = list(self.kb_repo.load_chunks_by_status(book_id, ChunkStatus.PENDING))
            total_chunks = len(pending_chunks)
            if total_chunks == 0:
                logger.info("No pending chunks found for Stage 1.")
                return

            start_time = time.perf_counter()

            # 1. Flatten only TARGET source sentences into indexed tuples: ((chunk.id, sentence.id), sentence.original_text)
            indexed_items: List[Tuple[Tuple[UUID, UUID], str]] = []
            for chunk in pending_chunks:
                target_ids = set(chunk.target_sentence_ids) if chunk.target_sentence_ids else {s.id for s in chunk.source_sentences if s.id not in chunk.context_sentence_ids}
                for sentence in chunk.source_sentences:
                    if sentence.id in target_ids:
                        indexed_items.append(((chunk.id, sentence.id), sentence.original_text))

            # 2. Dynamic Token-Bucket Batching
            tokenizer = getattr(self.nllb_engine, "tokenizer", None)
            batcher = DynamicTokenBucketBatcher(tokenizer=tokenizer, max_batch_tokens=self.max_batch_tokens)

            def translation_wrapper(texts: List[str]) -> List[str]:
                return self.nllb_engine.translate_batch(texts, source_lang=source_lang, target_lang=target_lang)

            translated_map = batcher.batch_and_translate(
                indexed_items=indexed_items,
                translation_fn=translation_wrapper,
                cancel_token=token,
                progress_cb=None
            )

            # 3. Write per sentence_id for target sentences (ignoring context)
            batch_to_save: List[TranslationChunk] = []
            BATCH_SAVE_SIZE = 50

            for current_idx, chunk in enumerate(pending_chunks, 1):
                if _is_cancellation_requested(token):
                    logger.info("Stage 1 execution cancelled by user.")
                    break

                target_ids = set(chunk.target_sentence_ids) if chunk.target_sentence_ids else {s.id for s in chunk.source_sentences if s.id not in chunk.context_sentence_ids}
                translated_sents: List[str] = []

                for sentence in chunk.source_sentences:
                    if sentence.id in target_ids:
                        key = (chunk.id, sentence.id)
                        t_text = translated_map.get(key, sentence.original_text)
                        sentence.translated_text = t_text
                        translated_sents.append(t_text)

                chunk.draft_translation = " ".join(translated_sents)
                chunk.status = ChunkStatus.DRAFT_COMPLETED
                batch_to_save.append(chunk)

                # Persist in batches of 50 or on the final chunk
                if len(batch_to_save) >= BATCH_SAVE_SIZE or current_idx == total_chunks:
                    if hasattr(self.kb_repo, "save_chunk_state_batch"):
                        self.kb_repo.save_chunk_state_batch(batch_to_save)
                    elif hasattr(self.kb_repo, "save_chunks_batch"):
                        self.kb_repo.save_chunks_batch(batch_to_save)
                    else:
                        for c in batch_to_save:
                            self.kb_repo.save_chunk_state(c)
                    batch_to_save.clear()

                # Progress & ETA calculation
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / current_idx
                eta_seconds = avg_time * (total_chunks - current_idx)

                logger.info(f"PROGRESS_UPDATE:STAGE_1:{current_idx}:{total_chunks}:{eta_seconds:.1f}")

                if progress_cb is not None:
                    try:
                        event = ProgressEvent(
                            stage=1,
                            current=current_idx,
                            total=total_chunks,
                            eta_seconds=eta_seconds,
                            status_message=f"Stage 1: {current_idx}/{total_chunks} chunks translated"
                        )
                        progress_cb(event)
                    except Exception as e:
                        logger.warning(f"Error in progress callback: {e}")

        finally:
            # Complete VRAM cleanup before moving to next stage
            if hasattr(self.nllb_engine, "unload_model"):
                self.nllb_engine.unload_model()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Stage 1 (NLLB MT Pass) finished.")

    def execute_aya_stage(
        self,
        book_id: UUID,
        cancel_token: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
        progress_cb: Optional[Callable[[ProgressEvent], None]] = None
    ) -> None:
        """
        Executes Stage 2: Aya LLM Literary Refinement Pass across all draft chunks.
        Features single-chunk immediate WAL persistence (0.2ms) to safeguard LLM compute.
        Writes translations per sentence_id and programmatically ignores IDs not in target_sentence_ids.
        """
        token = cancel_token if cancel_token is not None else cancel_event
        logger.info(f"Starting Stage 2 (Aya Editing Pass) for book {book_id}")

        if hasattr(self.aya_engine, "load_model"):
            self.aya_engine.load_model()

        try:
            from src.parsers.segmenter import RuleBasedSentenceSegmenter
            segmenter = RuleBasedSentenceSegmenter()
            draft_chunks = list(self.kb_repo.load_chunks_by_status(book_id, ChunkStatus.DRAFT_COMPLETED))
            total_chunks = len(draft_chunks)
            if total_chunks == 0:
                logger.info("No draft chunks found for Stage 2.")
                return

            all_glossary = self.kb_repo.get_glossary() if hasattr(self.kb_repo, "get_glossary") else []
            start_time = time.perf_counter()

            for current_idx, chunk in enumerate(draft_chunks, 1):
                if _is_cancellation_requested(token):
                    logger.info("Stage 2 execution cancelled by user.")
                    break

                target_ids = set(chunk.target_sentence_ids) if chunk.target_sentence_ids else {s.id for s in chunk.source_sentences if s.id not in chunk.context_sentence_ids}
                target_sentences = [s for s in chunk.source_sentences if s.id in target_ids]

                source_text = " ".join(s.original_text for s in target_sentences)
                draft_translation = chunk.draft_translation or " ".join(s.translated_text for s in target_sentences if s.translated_text)

                context = chunk.context
                if not context.previous_sentences:
                    context_sentences = [s for s in chunk.source_sentences if s.id in chunk.context_sentence_ids]
                    if context_sentences:
                        context = ChunkContext(
                            previous_sentences=[s.original_text for s in context_sentences],
                            active_glossary=context.active_glossary,
                            previous_summary=context.previous_summary
                        )

                # Filter glossary dynamically per chunk
                chunk_glossary = filter_glossary_for_chunk(all_glossary, source_text, draft_translation)

                # Literary refinement via Aya LLM
                refined_text = self.aya_engine.refine_chunk(
                    draft_translation=draft_translation,
                    source_text=source_text,
                    context=context,
                    glossary=chunk_glossary,
                    cancel_token=token
                )

                chunk.final_translation = refined_text.strip() if (refined_text and refined_text.strip()) else chunk.draft_translation
                chunk.status = ChunkStatus.REFINED

                # Write per sentence_id for target sentences and programmatically ignore context sentences
                refined_sentences = segmenter.split_sentences(refined_text) if (refined_text and refined_text.strip()) else []
                if len(refined_sentences) == len(target_sentences):
                    for s, r_text in zip(target_sentences, refined_sentences):
                        s.translated_text = r_text.strip()
                elif len(target_sentences) == 1:
                    if refined_sentences:
                        target_sentences[0].translated_text = " ".join(s.strip() for s in refined_sentences).strip()
                    elif refined_text and refined_text.strip():
                        target_sentences[0].translated_text = refined_text.strip()
                    elif not (target_sentences[0].translated_text and target_sentences[0].translated_text.strip()):
                        target_sentences[0].translated_text = target_sentences[0].original_text
                elif refined_sentences:
                    if len(refined_sentences) < len(target_sentences):
                        for idx in range(len(refined_sentences)):
                            target_sentences[idx].translated_text = refined_sentences[idx].strip()
                        for idx in range(len(refined_sentences), len(target_sentences)):
                            target_sentences[idx].translated_text = ""
                    else:
                        for idx in range(len(target_sentences) - 1):
                            target_sentences[idx].translated_text = refined_sentences[idx].strip()
                        target_sentences[-1].translated_text = " ".join(refined_sentences[len(target_sentences) - 1:]).strip()
                else:
                    for s in target_sentences:
                        if not (s.translated_text and s.translated_text.strip()):
                            s.translated_text = s.original_text

                # Immediate single-chunk commit
                self.kb_repo.save_chunk_state(chunk)

                # Progress & ETA calculation
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / current_idx
                eta_seconds = avg_time * (total_chunks - current_idx)

                logger.info(f"PROGRESS_UPDATE:STAGE_2:{current_idx}:{total_chunks}:{eta_seconds:.1f}")

                if progress_cb is not None:
                    try:
                        event = ProgressEvent(
                            stage=2,
                            current=current_idx,
                            total=total_chunks,
                            eta_seconds=eta_seconds,
                            status_message=f"Stage 2: {current_idx}/{total_chunks} chunks refined"
                        )
                        progress_cb(event)
                    except Exception as e:
                        logger.warning(f"Error in progress callback: {e}")

        finally:
            if hasattr(self.aya_engine, "unload_model"):
                self.aya_engine.unload_model()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Stage 2 (Aya Editing Pass) finished.")

    def execute_two_stage_translation(
        self,
        chunks_stream: Iterable[TranslationChunk]
    ) -> Iterator[TranslationChunk]:
        """
        Streaming two-stage translation for direct chunk processing.
        """
        chunks_list = list(chunks_stream)
        if not chunks_list:
            return

        if hasattr(self.nllb_engine, "load_model"):
            self.nllb_engine.load_model()

        try:
            # Stage 1: Batch MT (target sentences only)
            indexed_items = [
                ((c.id, s.id), s.original_text)
                for c in chunks_list
                for s in ([sent for sent in c.source_sentences if (sent.id in c.target_sentence_ids or (not c.target_sentence_ids and sent.id not in c.context_sentence_ids))])
            ]
            batcher = DynamicTokenBucketBatcher(
                tokenizer=getattr(self.nllb_engine, "tokenizer", None),
                max_batch_tokens=self.max_batch_tokens
            )
            translated_map = batcher.batch_and_translate(
                indexed_items,
                lambda texts: self.nllb_engine.translate_batch(texts, "en", "uk")
            )

            for chunk in chunks_list:
                target_ids = set(chunk.target_sentence_ids) if chunk.target_sentence_ids else {s.id for s in chunk.source_sentences if s.id not in chunk.context_sentence_ids}
                sents = []
                for s in chunk.source_sentences:
                    if s.id in target_ids:
                        t = translated_map.get((chunk.id, s.id), s.original_text)
                        s.translated_text = t
                        sents.append(t)
                chunk.draft_translation = " ".join(sents)
                chunk.status = ChunkStatus.DRAFT_COMPLETED

        finally:
            if hasattr(self.nllb_engine, "unload_model"):
                self.nllb_engine.unload_model()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Stage 2: LLM Refinement
        if hasattr(self.aya_engine, "load_model"):
            self.aya_engine.load_model()

        try:
            from src.parsers.segmenter import RuleBasedSentenceSegmenter
            segmenter = RuleBasedSentenceSegmenter()
            all_glossary = self.kb_repo.get_glossary() if hasattr(self.kb_repo, "get_glossary") else []
            for chunk in chunks_list:
                target_ids = set(chunk.target_sentence_ids) if chunk.target_sentence_ids else {s.id for s in chunk.source_sentences if s.id not in chunk.context_sentence_ids}
                target_sentences = [s for s in chunk.source_sentences if s.id in target_ids]
                source_text = " ".join(s.original_text for s in target_sentences)

                context = chunk.context
                if not context.previous_sentences:
                    context_sentences = [s for s in chunk.source_sentences if s.id in chunk.context_sentence_ids]
                    if context_sentences:
                        context = ChunkContext(
                            previous_sentences=[s.original_text for s in context_sentences],
                            active_glossary=context.active_glossary,
                            previous_summary=context.previous_summary
                        )

                # Filter glossary dynamically per chunk
                chunk_glossary = filter_glossary_for_chunk(all_glossary, source_text, chunk.draft_translation or "")

                refined = self.aya_engine.refine_chunk(
                    draft_translation=chunk.draft_translation or "",
                    source_text=source_text,
                    context=context,
                    glossary=chunk_glossary
                )
                chunk.final_translation = refined.strip() if (refined and refined.strip()) else chunk.draft_translation
                chunk.status = ChunkStatus.REFINED

                refined_sentences = segmenter.split_sentences(refined) if (refined and refined.strip()) else []
                if len(refined_sentences) == len(target_sentences):
                    for s, r_text in zip(target_sentences, refined_sentences):
                        s.translated_text = r_text.strip()
                elif len(target_sentences) == 1:
                    if refined_sentences:
                        target_sentences[0].translated_text = " ".join(s.strip() for s in refined_sentences).strip()
                    elif refined and refined.strip():
                        target_sentences[0].translated_text = refined.strip()
                    elif not (target_sentences[0].translated_text and target_sentences[0].translated_text.strip()):
                        target_sentences[0].translated_text = target_sentences[0].original_text
                elif refined_sentences:
                    if len(refined_sentences) < len(target_sentences):
                        for idx in range(len(refined_sentences)):
                            target_sentences[idx].translated_text = refined_sentences[idx].strip()
                        for idx in range(len(refined_sentences), len(target_sentences)):
                            target_sentences[idx].translated_text = ""
                    else:
                        for idx in range(len(target_sentences) - 1):
                            target_sentences[idx].translated_text = refined_sentences[idx].strip()
                        target_sentences[-1].translated_text = " ".join(refined_sentences[len(target_sentences) - 1:]).strip()
                else:
                    for s in target_sentences:
                        if not (s.translated_text and s.translated_text.strip()):
                            s.translated_text = s.original_text

                yield chunk

        finally:
            if hasattr(self.aya_engine, "unload_model"):
                self.aya_engine.unload_model()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
