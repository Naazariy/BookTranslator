"""
Tokenizer-Aware Token Budget Allocation Engine for Literary Translation.
Dynamically manages context window allocations across prompt components:
Target Paragraph > Active Locked Entities > Sliding Window History > Chapter Summary.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Callable
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class BudgetAllocation:
    """Detailed accounting of token allocation across prompt components."""
    max_prompt_tokens: int
    total_used_tokens: int
    scaffold_tokens: int
    target_tokens: int
    entities_tokens: int
    history_tokens: int
    summary_tokens: int
    remaining_tokens: int
    is_target_truncated: bool = False
    is_entities_truncated: bool = False
    is_history_truncated: bool = False
    is_summary_truncated: bool = False

    @property
    def is_within_budget(self) -> bool:
        return self.total_used_tokens <= self.max_prompt_tokens


class TokenBudget:
    """
    Tokenizer-aware budget allocation engine for Stage 2 Literary Translation.
    Prioritizes: Target Paragraph > Active Entities > Sliding Window History > Chapter Summary.
    Reserves explicit generation headroom (default 1024 tokens).
    """
    def __init__(
        self,
        max_total_tokens: int = 4096,
        max_generation_tokens: int = 1024,
        reserved_scaffold_tokens: int = 450,
        tokenizer: Optional[Any] = None,
        fallback_chars_per_token: float = 3.2,
    ):
        self.max_total_tokens = max_total_tokens
        self.max_generation_tokens = max_generation_tokens
        self.max_prompt_tokens = max(0, max_total_tokens - max_generation_tokens)
        self.reserved_scaffold_tokens = reserved_scaffold_tokens
        self.tokenizer = tokenizer
        self.fallback_chars_per_token = fallback_chars_per_token

    def count_tokens(self, text: Optional[str]) -> int:
        """Calculates token count using tokenizer if available, else empirical estimation."""
        if not text:
            return 0

        # 1. HuggingFace / Llama-cpp Tokenizer Path
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "encode"):
                    try:
                        return len(self.tokenizer.encode(text, add_special_tokens=False))
                    except TypeError:
                        return len(self.tokenizer.encode(text))
                elif hasattr(self.tokenizer, "tokenize"):
                    return len(self.tokenizer.tokenize(text))
            except Exception as e:
                logger.debug(f"Tokenizer token counting failed ({e}); falling back to heuristic.")

        # 2. Empirical Estimation for Ukrainian / English Prose
        words = text.split()
        if not words:
            return 0
        est_words = int(len(words) * 1.35) + 1
        est_chars = int(len(text) / self.fallback_chars_per_token) + 1
        return max(1, max(est_words, est_chars))

    def estimate_tokens(self, text: Optional[str]) -> int:
        """Alias for count_tokens."""
        return self.count_tokens(text)

    def allocate(
        self,
        target_text: str,
        entities: List[Any],
        format_entity_fn: Callable[[Any], str],
        history_paragraphs: List[str],
        summary: Optional[str] = None,
        scaffold_text: Optional[str] = None,
    ) -> Tuple[str, List[Any], List[str], Optional[str], BudgetAllocation]:
        """
        Allocates token budget dynamically according to the strict priority rules:
        1. Target paragraph (source + draft) [Priority 1]
        2. Active locked entities [Priority 2]
        3. Sliding window history (up to 2 paragraphs) [Priority 3]
        4. Chapter summary [Priority 4]

        Returns:
            (fitted_target_text, fitted_entities, fitted_history, fitted_summary, allocation)
        """
        scaffold_tokens = self.count_tokens(scaffold_text) if scaffold_text else self.reserved_scaffold_tokens
        available = max(0, self.max_prompt_tokens - scaffold_tokens)
        remaining = available

        # -------------------------------------------------------------
        # Step 1: Target Paragraph (Priority 1)
        # -------------------------------------------------------------
        target_tokens = self.count_tokens(target_text)
        is_target_truncated = False
        fitted_target = target_text

        if target_tokens <= remaining:
            remaining -= target_tokens
        else:
            is_target_truncated = True
            remaining = 0
            logger.warning(
                f"Target paragraph tokens ({target_tokens}) exceed available prompt budget ({available})."
            )

        # -------------------------------------------------------------
        # Step 2: Active Entities (Priority 2)
        # -------------------------------------------------------------
        fitted_entities: List[Any] = []
        entities_tokens = 0
        is_entities_truncated = False

        if remaining > 0 and entities:
            # Prioritize locked, then high confidence, then those with forbidden forms
            sorted_entities = sorted(
                entities,
                key=lambda e: (
                    getattr(e, "locked", False),
                    getattr(e, "confidence", 1.0),
                    len(getattr(e, "forbidden_target_forms", []) or []) > 0
                ),
                reverse=True
            )

            for ent in sorted_entities:
                ent_line = format_entity_fn(ent)
                line_tokens = self.count_tokens(ent_line + "\n")
                if line_tokens <= remaining:
                    fitted_entities.append(ent)
                    remaining -= line_tokens
                    entities_tokens += line_tokens
                else:
                    is_entities_truncated = True

        # -------------------------------------------------------------
        # Step 3: Sliding Window History (Priority 3: 1–2 Approved Paragraphs)
        # -------------------------------------------------------------
        fitted_history: List[str] = []
        history_tokens = 0
        is_history_truncated = False

        if remaining > 0 and history_paragraphs:
            candidates = [p for p in history_paragraphs if p and p.strip()]
            p_newest = candidates[-1] if candidates else None
            p_older = candidates[-2] if len(candidates) >= 2 else None

            if p_newest:
                t_new = self.count_tokens(p_newest + "\n\n")
                if t_new <= remaining:
                    fitted_history.append(p_newest)
                    remaining -= t_new
                    history_tokens += t_new

                    if p_older:
                        t_old = self.count_tokens(p_older + "\n\n")
                        if t_old <= remaining:
                            fitted_history.insert(0, p_older)
                            remaining -= t_old
                            history_tokens += t_old
                        else:
                            is_history_truncated = True
                else:
                    is_history_truncated = True
                    if remaining >= 60:
                        truncated_p = self._fit_tail_text(p_newest, remaining)
                        if truncated_p:
                            fitted_history.append(truncated_p)
                            t_trunc = self.count_tokens(truncated_p + "\n\n")
                            remaining = max(0, remaining - t_trunc)
                            history_tokens += t_trunc

        # -------------------------------------------------------------
        # Step 4: Chapter Summary (Priority 4)
        # -------------------------------------------------------------
        fitted_summary: Optional[str] = None
        summary_tokens = 0
        is_summary_truncated = False

        if remaining > 0 and summary and summary.strip():
            s_tokens = self.count_tokens(summary.strip())
            if s_tokens <= remaining:
                fitted_summary = summary.strip()
                remaining -= s_tokens
                summary_tokens += s_tokens
            elif remaining >= 30:
                truncated = self._fit_head_text(summary.strip(), max(1, remaining - 10))
                if truncated:
                    fitted_summary = truncated + "..."
                    summary_tokens = self.count_tokens(fitted_summary)
                    remaining = max(0, remaining - summary_tokens)
                is_summary_truncated = True
            else:
                is_summary_truncated = True

        total_used = scaffold_tokens + target_tokens + entities_tokens + history_tokens + summary_tokens
        allocation = BudgetAllocation(
            max_prompt_tokens=self.max_prompt_tokens,
            total_used_tokens=total_used,
            scaffold_tokens=scaffold_tokens,
            target_tokens=target_tokens,
            entities_tokens=entities_tokens,
            history_tokens=history_tokens,
            summary_tokens=summary_tokens,
            remaining_tokens=remaining,
            is_target_truncated=is_target_truncated,
            is_entities_truncated=is_entities_truncated,
            is_history_truncated=is_history_truncated,
            is_summary_truncated=is_summary_truncated,
        )

        return fitted_target, fitted_entities, fitted_history, fitted_summary, allocation

    def _fit_tail_text(self, text: str, max_tokens: int) -> str:
        """Keeps the trailing sentences of text that fit within max_tokens."""
        sentences = re.split(r'(?<=[.!?…])\s+', text)
        result: List[str] = []
        accum_tokens = 0
        for s in reversed(sentences):
            st = self.count_tokens(s)
            if accum_tokens + st <= max_tokens:
                result.insert(0, s)
                accum_tokens += st
            else:
                break
        return " ".join(result) if result else ""

    def _fit_head_text(self, text: str, max_tokens: int) -> str:
        """Keeps the leading words of text that fit within max_tokens."""
        words = text.split()
        res: List[str] = []
        for w in words:
            res.append(w)
            if self.count_tokens(" ".join(res)) > max_tokens:
                res.pop()
                break
        return " ".join(res)
