"""Shared fixtures, mock ML engines, sample data factories, and contract references for E2E tests."""
import re
import time
import queue
import threading
from pathlib import Path
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Iterator

from src.domain.interfaces.translation import ITranslationEngine, IEditingEngine
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.domain.models.knowledge import Entity, EntityType, GlossaryItem


# ==============================================================================
# 1. Thread-safe Cancellation Token and UI Concurrency Primitives
# ==============================================================================

@dataclass
class ProgressEvent:
    stage: int
    current: int
    total: int
    eta_seconds: float = 0.0
    status_message: str = ""


class CancellationToken:
    """Thread-safe cancellation token matching PROJECT.md interface contract."""
    def __init__(self):
        self._cancelled = threading.Event()
        self._callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            for cb in self._callbacks:
                try:
                    cb()
                except Exception:
                    pass

    def reset(self) -> None:
        self._cancelled.clear()

    def register_callback(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self.is_cancelled():
                cb()
            else:
                self._callbacks.append(cb)


class UIEventQueue:
    """Thread-safe event queue matching PROJECT.md interface contract."""
    def __init__(self, maxsize: int = 10000):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def post(self, event_type: str, **kwargs) -> bool:
        """Non-blocking event dispatch."""
        try:
            self._queue.put_nowait({"type": event_type, "data": kwargs})
            return True
        except queue.Full:
            return False

    def poll(self, timeout_ms: int = 20) -> Optional[Dict[str, Any]]:
        """Polls next event with timeout."""
        try:
            return self._queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return None

    def drain(self, max_events: Optional[int] = None) -> List[Dict[str, Any]]:
        """Drains pending events without blocking."""
        events = []
        count = 0
        while not self._queue.empty():
            if max_events is not None and count >= max_events:
                break
            try:
                events.append(self._queue.get_nowait())
                count += 1
            except queue.Empty:
                break
        return events

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()


class ThrottledLogBuffer:
    """Rate-limited log buffer (50ms interval) with 1,000-line circular buffer."""
    def __init__(self, max_lines: int = 1000, flush_interval_ms: int = 50):
        self.max_lines = max_lines
        self.flush_interval_sec = flush_interval_ms / 1000.0
        self._buffer: List[str] = []
        self._circular_lines: List[str] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

    def append(self, message: str) -> None:
        with self._lock:
            self._buffer.append(message)
            self._circular_lines.append(message)
            if len(self._circular_lines) > self.max_lines:
                # Trim oldest lines
                self._circular_lines = self._circular_lines[-self.max_lines:]

    def should_flush(self) -> bool:
        with self._lock:
            return bool(self._buffer) and (time.time() - self._last_flush >= self.flush_interval_sec)

    def flush(self) -> List[str]:
        with self._lock:
            flushed = self._buffer[:]
            self._buffer.clear()
            self._last_flush = time.time()
            return flushed

    def get_lines(self) -> List[str]:
        with self._lock:
            return list(self._circular_lines)


class AsyncTaskManager:
    """Manages worker thread lifecycle and cancellation."""
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._cancel_token = CancellationToken()
        self._is_running = False
        self._error: Optional[Exception] = None
        self._lock = threading.Lock()

    @property
    def cancel_token(self) -> CancellationToken:
        return self._cancel_token

    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def start_task(self, target: Callable[..., Any], *args, **kwargs) -> bool:
        with self._lock:
            if self._is_running:
                return False
            self._is_running = True
            self._error = None
            self._cancel_token.reset()

        def worker():
            try:
                target(*args, cancel_token=self._cancel_token, **kwargs)
            except Exception as e:
                with self._lock:
                    self._error = e
            finally:
                with self._lock:
                    self._is_running = False

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel_token.cancel()

    def join(self, timeout: Optional[float] = None) -> bool:
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    def get_error(self) -> Optional[Exception]:
        with self._lock:
            return self._error


# ==============================================================================
# 2. Dynamic Token-Bucket Batching Reference
# ==============================================================================

class DynamicTokenBucketBatcher:
    """Groups items by token length to minimize padding waste."""
    FLORES_MAP = {
        "en": "eng_Latn",
        "uk": "ukr_Cyrl",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "pl": "pol_Latn",
    }

    @classmethod
    def resolve_flores_code(cls, lang: str) -> str:
        return cls.FLORES_MAP.get(lang.lower(), lang)

    @staticmethod
    def estimate_token_length(text: str) -> int:
        """Approximates subword token count based on word and character counts."""
        if not text:
            return 0
        words = text.split()
        return max(len(words), int(len(text) / 3.5))

    @classmethod
    def create_buckets(
        cls,
        items: List[Any],
        token_len_fn: Optional[Callable[[Any], int]] = None,
        max_tokens: int = 2048
    ) -> List[List[Any]]:
        if not items:
            return []
        
        if token_len_fn is None:
            token_len_fn = lambda x: cls.estimate_token_length(str(x))

        # Sort with index for order preservation
        indexed_items = list(enumerate(items))
        indexed_items.sort(key=lambda pair: token_len_fn(pair[1]))

        buckets: List[List[Any]] = []
        current_bucket: List[Any] = []
        current_max_len = 0

        for idx, item in indexed_items:
            t_len = max(1, token_len_fn(item))
            if t_len > max_tokens:
                # Oversized item gets its own standalone bucket
                if current_bucket:
                    buckets.append(current_bucket)
                    current_bucket = []
                    current_max_len = 0
                buckets.append([item])
                continue

            new_max_len = max(current_max_len, t_len)
            new_batch_tokens = new_max_len * (len(current_bucket) + 1)

            if current_bucket and new_batch_tokens > max_tokens:
                buckets.append(current_bucket)
                current_bucket = [item]
                current_max_len = t_len
            else:
                current_bucket.append(item)
                current_max_len = new_max_len

        if current_bucket:
            buckets.append(current_bucket)

        return buckets


try:
    from src.parsers.segmenter import RuleBasedSentenceSegmenter
except ImportError:
    class RuleBasedSentenceSegmenter:
        """Fallback stub for RuleBasedSentenceSegmenter."""
        ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "vs.", "e.g.", "i.e.", "etc."}
        @classmethod
        def split_sentences(cls, text: str) -> List[str]:
            return [s.strip() for s in text.split('.') if s.strip()]



# ==============================================================================
# 4. O(S) DOM Tree Reconciliation
# ==============================================================================

def reconcile_document_dom(book: Book, chunks: List[TranslationChunk]) -> Book:
    """O(S) DOM reconciliation mapping chunk translations back into Book DOM."""
    sentence_map: Dict[UUID, Sentence] = {}
    for chapter in book.chapters:
        for paragraph in chapter.paragraphs:
            for s in paragraph.sentences:
                sentence_map[s.id] = s

    for chunk in chunks:
        target_ids = set(chunk.target_sentence_ids) if chunk.target_sentence_ids else {s.id for s in chunk.source_sentences if s.id not in chunk.context_sentence_ids}
        target_sentences = [s for s in chunk.source_sentences if s.id in target_ids and s.id in sentence_map]
        
        has_per_sentence = any(s.translated_text for s in target_sentences)
        if has_per_sentence:
            for s in target_sentences:
                if s.translated_text is not None:
                    sentence_map[s.id].translated_text = s.translated_text
            continue

        translation = chunk.final_translation if chunk.final_translation else chunk.draft_translation
        if not translation:
            continue
        
        sents_to_update = target_sentences if target_sentences else [s for s in chunk.source_sentences if s.id in sentence_map]
        if not sents_to_update:
            continue

        # If single sentence in chunk, direct 1:1 mapping
        if len(sents_to_update) == 1:
            s_id = sents_to_update[0].id
            if s_id in sentence_map:
                sentence_map[s_id].translated_text = translation
        else:
            # Multi-sentence chunk: assign translation to first sentence and clear remaining
            first_id = sents_to_update[0].id
            if first_id in sentence_map:
                sentence_map[first_id].translated_text = translation
            for s_meta in sents_to_update[1:]:
                if s_meta.id in sentence_map:
                    sentence_map[s_meta.id].translated_text = ""

    return book


# ==============================================================================
# 5. Mock CTranslate2 NLLB Engine
# ==============================================================================

class MockCTranslate2Engine(ITranslationEngine):
    """Mock CTranslate2 NLLB engine simulating INT8/FP16 fast translation."""
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.is_loaded = False
        self.translation_dict: Dict[str, str] = {
            "Hello, world!": "Привіт, світе!",
            "It was a dark and stormy night.": "Це була темна і бурхлива ніч.",
            "The wind howled through the trees.": "Вітер вив крізь дерева.",
            "Dr. Watson looked at Mr. Holmes.": "Доктор Ватсон подивився на містера Холмса.",
            "He lived in Kyiv at Khreshchatyk st.": "Він жив у м. Києві на вул. Хрещатик.",
            "\"I will return tomorrow,\" said the captain.": "«Я повернуся завтра», — сказав капітан.",
            "The price was $3.14 per unit.": "Ціна становила $3.14 за одиницю.",
        }

    def load_model(self) -> None:
        self.is_loaded = True

    def unload_model(self) -> None:
        self.is_loaded = False

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded into VRAM. Call load_model() first.")
        
        src_code = DynamicTokenBucketBatcher.resolve_flores_code(source_lang)
        tgt_code = DynamicTokenBucketBatcher.resolve_flores_code(target_lang)

        results = []
        for text in texts:
            if not text or not text.strip():
                results.append("")
                continue
            if text in self.translation_dict:
                results.append(self.translation_dict[text])
            else:
                # Synthetic translation
                results.append(f"[Переклад: {text}]")
        return results


# ==============================================================================
# 6. Mock Quantized Aya-23-8B Engine
# ==============================================================================

class MockQuantizedAyaEngine(IEditingEngine):
    """Mock Quantized Aya-23-8B engine simulating 4-bit literary refinement."""
    def __init__(self, model_name: str = "CohereForAI/aya-23-8B", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.is_loaded = False
        self._prompt_template_cache: Optional[str] = None
        self._disk_read_count = 0

    def load_model(self) -> None:
        self.is_loaded = True
        self._prompt_template_cache = "Context: {context_previous}\nGlossary: {glossary_terms}\nDraft: {draft_text}\nRefined:"

    def unload_model(self) -> None:
        self.is_loaded = False
        self._prompt_template_cache = None

    def refine_chunk(
        self,
        draft_translation: str,
        source_text: Any = "",
        context: Optional[ChunkContext] = None,
        glossary: Optional[List[GlossaryItem]] = None,
        cancel_token: Optional[CancellationToken] = None,
        *args,
        **kwargs
    ) -> str:
        if isinstance(source_text, ChunkContext):
            glossary = context if isinstance(context, list) else glossary
            context = source_text
            source_text = ""
        elif isinstance(context, list) and glossary is None:
            glossary = context
            context = None

        if not self.is_loaded:
            raise RuntimeError("Aya model is not loaded into VRAM. Call load_model() first.")

        token = cancel_token or kwargs.get("cancel_token") or kwargs.get("cancel_event")
        if token and hasattr(token, "is_cancelled") and token.is_cancelled():
            raise RuntimeError("Generation cancelled by user token.")

        if not draft_translation:
            return ""

        refined = draft_translation

        # Apply glossary replacements
        if glossary:
            for item in glossary:
                if item.source_term in refined:
                    refined = refined.replace(item.source_term, item.target_term)

        # Polish literary style and clean synthetic markers
        refined = re.sub(r'\[Переклад:\s*(.*?)\]', r'\1', refined)
        if not refined.startswith("«") and draft_translation.startswith('"'):
            refined = "«" + refined.strip('"') + "»"

        return refined

    def refine_batch(
        self,
        drafts: List[str],
        contexts: List[ChunkContext],
        glossary: List[GlossaryItem],
        cancel_token: Optional[CancellationToken] = None
    ) -> List[str]:
        results = []
        for i, draft in enumerate(drafts):
            ctx = contexts[i] if i < len(contexts) else ChunkContext()
            results.append(self.refine_chunk(draft, "", ctx, glossary, cancel_token=cancel_token))
        return results


# ==============================================================================
# 7. Sample Book and Ukrainian Literary Corpus
# ==============================================================================

class UkrainianLiteraryCorpus:
    PROSE_CHAPTERS = [
        {
            "title": "Розділ 1. Нічна подорож",
            "paragraphs": [
                "It was a dark and stormy night. The wind howled through the trees.",
                "Dr. Watson looked at Mr. Holmes. \"I will return tomorrow,\" said the captain.",
                "He lived in Kyiv at Khreshchatyk st. The price was $3.14 per unit.",
            ]
        },
        {
            "title": "Розділ 2. Зустріч біля Дніпра",
            "paragraphs": [
                "Т. Г. Шевченко народився у 1814 р. Він видатний поет.",
                "«Привіт!» — сказав він. «Як справи?» — запитала вона.",
                "The temperature was -15.5 °C at 6 a.m. It was freezing.",
            ]
        }
    ]


class SampleBookFactory:
    @staticmethod
    def create_sample_book(title: str = "Test Book") -> Book:
        book_id = uuid4()
        book = Book(
            id=book_id,
            title=title,
            author="Author Name",
            source_language="en",
            target_language="uk"
        )
        
        for c_idx, ch_data in enumerate(UkrainianLiteraryCorpus.PROSE_CHAPTERS):
            chapter = Chapter(
                id=uuid4(),
                title=ch_data["title"],
                order_index=c_idx
            )
            for p_idx, p_text in enumerate(ch_data["paragraphs"]):
                paragraph = Paragraph(id=uuid4())
                sentences = RuleBasedSentenceSegmenter.split_sentences(p_text)
                for s_idx, s_text in enumerate(sentences):
                    sentence = Sentence(
                        id=uuid4(),
                        original_text=s_text,
                        order_index=s_idx
                    )
                    paragraph.sentences.append(sentence)
                chapter.paragraphs.append(paragraph)
            book.chapters.append(chapter)
            
        return book

    @staticmethod
    def create_chunks_from_book(book: Book, sentences_per_chunk: int = 2) -> List[TranslationChunk]:
        chunks: List[TranslationChunk] = []
        for chapter in book.chapters:
            curr_sentences = []
            for paragraph in chapter.paragraphs:
                for sentence in paragraph.sentences:
                    curr_sentences.append(sentence)
                    if len(curr_sentences) >= sentences_per_chunk:
                        chunks.append(TranslationChunk(
                            id=uuid4(),
                            book_id=book.id,
                            chapter_id=chapter.id,
                            paragraph_indices=[0],
                            source_sentences=curr_sentences[:],
                            context_sentence_ids=[],
                            target_sentence_ids=[s.id for s in curr_sentences],
                            token_count=sum(len(s.original_text.split()) for s in curr_sentences) * 2,
                            context=ChunkContext(),
                            status=ChunkStatus.PENDING
                        ))
                        curr_sentences.clear()
            if curr_sentences:
                chunks.append(TranslationChunk(
                    id=uuid4(),
                    book_id=book.id,
                    chapter_id=chapter.id,
                    paragraph_indices=[0],
                    source_sentences=curr_sentences[:],
                    context_sentence_ids=[],
                    target_sentence_ids=[s.id for s in curr_sentences],
                    token_count=sum(len(s.original_text.split()) for s in curr_sentences) * 2,
                    context=ChunkContext(),
                    status=ChunkStatus.PENDING
                ))
        return chunks
