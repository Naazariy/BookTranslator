"""
src/quality/validators/completeness.py

Validator checking translation completeness: length ratios, dropped sentences,
premature truncation, and consecutive duplicate repetition loops.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional, List, Any
from src.domain.models.segment import TranslationSegment
from src.context.builder import PromptContext
from src.parsers.segmenter import RuleBasedSentenceSegmenter
from src.quality.base import BaseValidator
from src.quality.models import QAViolation, QASeverity


class CompletenessValidator(BaseValidator):
    """
    Validates completeness of translation by detecting:
    - Extreme length ratio anomalies (<35% or >280%)
    - Sentence count drop (<70% of source sentences)
    - Premature truncation (unclosed clauses, trailing conjunctions/escapes)
    - Consecutive near-duplicate sentence repetition loops
    """
    name: str = "CompletenessValidator"
    description: str = "Detects dropped sentences, extreme length discrepancies, truncation, and repetitions"
    default_severity: QASeverity = QASeverity.CRITICAL

    TRAILING_CONNECTOR_PATTERN = re.compile(
        r'(?:,\s*|—\s*|\b(?:і|та|або|чи|а|але|в|у|на|до|з|із|зі|що|як|який|яка|яке|яких)\s*)$',
        re.IGNORECASE
    )

    def validate(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        translated_text: Optional[str] = None,
        chunk: Any = None,
        **kwargs: Any,
    ) -> List[QAViolation]:
        violations: List[QAViolation] = []
        source_text = self.extract_source_text(segment, chunk=chunk).strip()
        target_text = self.extract_target_text(segment, translated_text=translated_text).strip()

        if not target_text or not source_text:
            return violations

        src_len = len(source_text)
        tgt_len = len(target_text)
        ratio = tgt_len / max(src_len, 1)

        # 1. Extreme Length Ratio Check
        if src_len >= 25 and ratio < 0.35:
            violations.append(
                self.make_violation(
                    rule_code="LENGTH_RATIO_TOO_LOW",
                    message=(
                        f"Translation length ({tgt_len} chars) is suspiciously short compared to "
                        f"source ({src_len} chars, ratio: {ratio:.1%} < 35%). Probable sentence omission."
                    ),
                    severity=QASeverity.CRITICAL,
                    source_snippet=source_text[:100],
                    target_snippet=target_text[:100],
                    suggested_fix="Translate the complete source paragraph without dropping clauses or sentences.",
                )
            )
        elif src_len >= 15 and ratio > 2.80:
            violations.append(
                self.make_violation(
                    rule_code="LENGTH_RATIO_TOO_HIGH",
                    message=(
                        f"Translation length ({tgt_len} chars) is excessively long compared to "
                        f"source ({src_len} chars, ratio: {ratio:.1%} > 280%). Probable hallucination or repetition loop."
                    ),
                    severity=QASeverity.CRITICAL,
                    source_snippet=source_text[:100],
                    target_snippet=target_text[:100],
                    suggested_fix="Remove redundant expansions, repetition, or AI commentary.",
                )
            )

        # 2. Premature Truncation & Unfinished Sentence Check
        if target_text.endswith("\\"):
            violations.append(
                self.make_violation(
                    rule_code="TRUNCATION_TRAILING_ESCAPE",
                    message="Translation ends with broken escape character '\\'.",
                    severity=QASeverity.CRITICAL,
                    target_snippet=target_text[-30:],
                    suggested_fix="Remove broken trailing escape character and complete sentence.",
                )
            )
        elif self.TRAILING_CONNECTOR_PATTERN.search(target_text) and not re.search(r'[.!?…»"]$', target_text):
            violations.append(
                self.make_violation(
                    rule_code="TRUNCATION_TRAILING_CONNECTOR",
                    message="Translation appears cut off mid-sentence (ends with a trailing conjunction or preposition).",
                    severity=QASeverity.CRITICAL,
                    target_snippet=target_text[-40:],
                    suggested_fix="Complete the trailing clause and end with proper terminal punctuation.",
                )
            )

        # 3. Sentence Count Drop Check
        translated_sents = RuleBasedSentenceSegmenter.split_sentences(target_text) if target_text else []
        expected_count = self._calculate_expected_sentences(segment, source_text, chunk=chunk)

        if expected_count > 1 and len(translated_sents) < 0.7 * expected_count:
            violations.append(
                self.make_violation(
                    rule_code="SentenceCountDrop",
                    message=(
                        f"Translated sentence count ({len(translated_sents)}) dropped significantly "
                        f"below expected ({expected_count}, ratio: {len(translated_sents)/expected_count:.2f} < 0.7)."
                    ),
                    severity=QASeverity.CRITICAL,
                    suggested_fix=f"Translate all {expected_count} sentences present in the source text.",
                    metadata={"expected_count": expected_count, "actual_count": len(translated_sents)},
                )
            )

        # 4. Consecutive Duplicate Sentences Check
        for i in range(len(translated_sents) - 1):
            s1 = translated_sents[i].strip()
            s2 = translated_sents[i + 1].strip()
            if len(s1) > 5 and len(s2) > 5:
                sim = SequenceMatcher(None, s1, s2).ratio()
                if sim > 0.8:
                    violations.append(
                        self.make_violation(
                            rule_code="ConsecutiveDuplicateSentences",
                            message=f"Detected consecutive near-duplicate sentences (similarity {sim:.2f}): '{s1}' and '{s2}'",
                            severity=QASeverity.WARNING,
                            target_snippet=f"{s1} | {s2}",
                            suggested_fix="Remove the repeated sentence.",
                        )
                    )

        return violations

    def _calculate_expected_sentences(self, segment: Any, source_text: str, chunk: Any = None) -> int:
        """Determines expected sentence count from chunk or segment sentence mappings."""
        obj = chunk or segment
        target_ids = getattr(obj, "target_sentence_ids", None)
        source_sents = getattr(obj, "source_sentences", None)
        context_ids = set(getattr(obj, "context_sentence_ids", []) or [])

        if target_ids and source_sents:
            return len([s for s in source_sents if getattr(s, "id", None) in target_ids])
        if source_sents:
            return len([s for s in source_sents if getattr(s, "id", None) not in context_ids])
        
        sent_ids = getattr(segment, "sentence_ids", None)
        if sent_ids:
            return len(sent_ids)

        return len(RuleBasedSentenceSegmenter.split_sentences(source_text))
