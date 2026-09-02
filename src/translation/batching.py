"""
Dynamic Token-Bucket Batching Module for Machine Translation.
Groups sentences by subword token length to eliminate padding waste and minimize attention compute.
"""
from typing import List, Tuple, Dict, Any, Optional, Callable, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


class _DualMethod:
    """Descriptor that allows a method to be called as either an instance method or class method."""
    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __get__(self, instance: Any, owner: Any) -> Callable[..., Any]:
        if instance is not None:
            return lambda *args, **kwargs: self.func(instance, *args, **kwargs)
        return lambda *args, **kwargs: self.func(owner, *args, **kwargs)


class DynamicTokenBucketBatcher:
    """
    Length-sorted dynamic token-bucket batcher.
    Groups items (sentences or objects) by token length into dense batches whose
    total padded token budget does not exceed max_tokens (default 2048).
    Eliminates padding waste across heterogeneous sentence length distributions.
    """
    FLORES_MAP = {
        "en": "eng_Latn",
        "uk": "ukr_Cyrl",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "pl": "pol_Latn",
        "it": "ita_Latn",
        "pt": "por_Latn",
        "zh": "zho_Hans",
        "ja": "jpn_Jpan",
        "cs": "ces_Latn",
        "ro": "ron_Latn",
        "bg": "bul_Cyrl",
        "ru": "rus_Cyrl",
        "nl": "nld_Latn",
        "sv": "swe_Latn",
        "fi": "fin_Latn",
        "da": "dan_Latn",
        "no": "nob_Latn",
        "el": "ell_Grek",
        "tr": "tur_Latn",
        "ar": "arb_Arab",
        "he": "heb_Hebr",
        "hi": "hin_Deva",
        "ko": "kor_Hang",
        "hu": "hun_Latn",
        "sk": "slk_Latn",
        "sl": "slv_Latn",
        "hr": "hrv_Latn",
        "sr": "srp_Cyrl",
    }

    def __init__(self, tokenizer: Optional[Any] = None, max_batch_tokens: int = 2048):
        self.tokenizer = tokenizer
        self.max_batch_tokens = max_batch_tokens

    @classmethod
    def resolve_flores_code(cls, lang: str) -> str:
        """Resolves ISO language code to Flores-200 code standard."""
        if not lang:
            return "ukr_Cyrl"
        return cls.FLORES_MAP.get(lang.strip().lower(), lang.strip())

    @classmethod
    def estimate_token_length_static(cls, text: str) -> int:
        """Fallback static estimation: average 1.3 subword tokens per word + 2 special tokens."""
        if not text:
            return 1
        words = text.split()
        if not words:
            return 1
        return max(1, int(len(words) * 1.3) + 2)

    def estimate_token_length(self, text: str) -> int:
        """Estimates or computes the token length of a given text."""
        if self is not None and not isinstance(self, type) and getattr(self, "tokenizer", None) is not None:
            try:
                if hasattr(self.tokenizer, "encode"):
                    return len(self.tokenizer.encode(text, add_special_tokens=True))
                elif hasattr(self.tokenizer, "tokenize"):
                    return len(self.tokenizer.tokenize(text)) + 2
            except Exception as e:
                logger.debug(f"Tokenizer encoding failed for length estimation: {e}")
        
        return self.estimate_token_length_static(text)

    @_DualMethod
    def create_buckets(
        cls_or_self,
        items: List[T],
        token_len_fn: Optional[Callable[[T], int]] = None,
        max_tokens: Optional[int] = None
    ) -> List[List[T]]:
        """
        Generic token bucketing interface matching PROJECT.md interface contract:
        create_buckets(items: List[T], token_len_fn: Callable[[T], int], max_tokens: int = 2048) -> List[List[T]]
        Supports both class-level (DynamicTokenBucketBatcher.create_buckets) and instance-level (batcher.create_buckets) calls.
        """
        if not items:
            return []

        if isinstance(cls_or_self, type):
            limit = max_tokens if max_tokens is not None else 2048
            len_fn = token_len_fn if token_len_fn is not None else cls_or_self.estimate_token_length_static
        else:
            limit = max_tokens if max_tokens is not None else cls_or_self.max_batch_tokens
            len_fn = token_len_fn if token_len_fn is not None else cls_or_self.estimate_token_length

        # 1. Compute lengths and sort indices by length
        lengths = [max(1, len_fn(item)) for item in items]
        sorted_indices = sorted(range(len(items)), key=lambda i: lengths[i])

        # 2. Form dense buckets
        buckets: List[List[T]] = []
        current_batch_indices: List[int] = []
        current_max_len = 0

        for idx in sorted_indices:
            l = lengths[idx]
            if l > limit:
                # Oversized item gets its own standalone bucket
                if current_batch_indices:
                    buckets.append([items[i] for i in current_batch_indices])
                    current_batch_indices = []
                    current_max_len = 0
                buckets.append([items[idx]])
                continue

            candidate_max_len = max(current_max_len, l)
            candidate_batch_size = len(current_batch_indices) + 1

            if candidate_batch_size * candidate_max_len <= limit:
                current_batch_indices.append(idx)
                current_max_len = candidate_max_len
            else:
                if current_batch_indices:
                    buckets.append([items[i] for i in current_batch_indices])
                current_batch_indices = [idx]
                current_max_len = l

        if current_batch_indices:
            buckets.append([items[i] for i in current_batch_indices])

        return buckets

    @_DualMethod
    def create_batches(
        cls_or_self,
        sentences: List[str],
        max_tokens: Optional[int] = None
    ) -> List[List[str]]:
        """Groups a list of sentence strings into length-sorted dense token buckets."""
        if isinstance(cls_or_self, type):
            len_fn = cls_or_self.estimate_token_length_static
        else:
            len_fn = cls_or_self.estimate_token_length

        return cls_or_self.create_buckets(
            items=sentences,
            token_len_fn=len_fn,
            max_tokens=max_tokens
        )

    def batch_and_translate(
        self,
        indexed_items: List[Tuple[Any, str]],
        translation_fn: Callable[[List[str]], List[str]],
        cancel_token: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None
    ) -> Dict[Any, str]:
        """
        Length-sorts indexed items ((item_id, text)), executes translation in dense buckets,
        and un-sorts the results back into a mapped dictionary {item_id: translated_text}.
        Supports cooperative cancellation via cancel_token / cancel_event.
        """
        if not indexed_items:
            return {}

        token = cancel_token if cancel_token is not None else cancel_event

        # 1. Calculate token lengths
        lengths = [max(1, self.estimate_token_length(text)) for _, text in indexed_items]

        # 2. Sort by length preserving original item indices
        sorted_indices = sorted(range(len(indexed_items)), key=lambda i: lengths[i])

        # 3. Accumulate dense token buckets
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len = 0

        for idx in sorted_indices:
            l = lengths[idx]
            candidate_max_len = max(current_max_len, l)
            candidate_batch_size = len(current_batch) + 1

            if candidate_batch_size * candidate_max_len <= self.max_batch_tokens:
                current_batch.append(idx)
                current_max_len = candidate_max_len
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_max_len = l

        if current_batch:
            batches.append(current_batch)

        logger.info(
            f"DynamicTokenBucketBatcher: Grouped {len(indexed_items)} sentences into {len(batches)} "
            f"dense buckets (max_tokens={self.max_batch_tokens})."
        )

        # 4. Execute translation bucket-by-bucket and populate results map
        results_map: Dict[Any, str] = {}
        total_batches = len(batches)

        for batch_idx, batch_indices in enumerate(batches, 1):
            if _is_cancellation_requested(token):
                logger.info("DynamicTokenBucketBatcher: Cancellation detected. Stopping batch execution.")
                break

            batch_texts = [indexed_items[i][1] for i in batch_indices]
            translated_texts = translation_fn(batch_texts)

            for i, trans_text in zip(batch_indices, translated_texts):
                item_id = indexed_items[i][0]
                results_map[item_id] = trans_text
                
            if progress_cb:
                progress_cb(batch_idx, total_batches)

        return results_map
