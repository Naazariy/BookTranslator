"""
src/quality/validators/numbers_and_units.py

Validator ensuring accurate preservation of Arabic numbers, Roman numerals,
percentages, and currencies in Ukrainian translations.
"""
from __future__ import annotations

import re
from typing import Optional, List, Any, Set
from src.domain.models.segment import TranslationSegment
from src.context.builder import PromptContext
from src.quality.base import BaseValidator
from src.quality.models import QAViolation, QASeverity


class NumbersAndUnitsValidator(BaseValidator):
    """
    Validates preservation of numerical data, Roman numerals, currencies, and percentages:
    - Considers normalized_text (metric conversions) when available
    - Recognizes literary spelled-out Ukrainian numeral words (1 -> "одне", "один", etc.)
    - Flags missing Arabic numbers as CRITICAL
    - Flags missing Roman numerals, currency symbols, and percentages as WARNING
    """
    name: str = "NumbersAndUnitsValidator"
    description: str = "Verifies accurate preservation of numbers, Roman numerals, percentages, and currencies"
    default_severity: QASeverity = QASeverity.CRITICAL

    ARABIC_NUM_PATTERN = re.compile(r'\b\d+(?:[.,]\d+)?\b')

    ROMAN_NUM_PATTERN = re.compile(
        r'\b(?:(?:Chapter|Part|Section|Act|Scene|Глава|Розділ|Частина)\s+)?((?=[MDCLXVI])M*(?:C[MD]|D?C*)(?:X[CL]|L?X*)(?:I[XV]|V?I*))\b',
        re.IGNORECASE
    )

    CURRENCY_MAP = {
        '$': ('$', 'долар', 'дол.'),
        '€': ('€', 'євро'),
        '£': ('£', 'фунт'),
        '₴': ('₴', 'гривн', 'грн'),
    }

    # Ukrainian word forms for small cardinal/ordinal numbers commonly spelled out in prose
    WORD_NUMBERS = {
        0: ("нуль", "нуля", "нульов"),
        1: ("один", "одна", "одне", "одну", "одні", "одного", "одній", "одним", "перш"),
        2: ("два", "дві", "двох", "двом", "двома", "друг"),
        3: ("три", "трьох", "трьом", "трьома", "трет"),
        4: ("чотири", "чотирьох", "чотирьом", "чотирма", "четверт"),
        5: ("п'ять", "п’ять", "п'яти", "п’яти", "п'ятий", "п’ятий"),
        6: ("шість", "шести", "шост"),
        7: ("сім", "семи", "сьом"),
        8: ("вісім", "восьми", "восьм"),
        9: ("дев'ять", "дев’ять", "дев'яти", "дев’яти", "дев'ят"),
        10: ("десять", "десяти", "десят"),
    }

    def validate(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        translated_text: Optional[str] = None,
        chunk: Any = None,
        **kwargs: Any,
    ) -> List[QAViolation]:
        violations: List[QAViolation] = []

        ref_source = getattr(segment, "normalized_text", None) or self.extract_source_text(segment, chunk=chunk)
        ref_source = (ref_source or "").strip()
        target_text = self.extract_target_text(segment, translated_text=translated_text).strip()

        if not target_text or not ref_source:
            return violations

        # 1. Arabic Numbers Preservation
        source_nums = self._extract_normalized_numbers(ref_source)
        target_nums = self._extract_normalized_numbers(target_text)
        target_vals = [t[1] for t in target_nums]
        target_lower = target_text.lower()

        for raw_num, norm_val in source_nums:
            if norm_val in target_vals:
                continue

            # Check if small integer is rendered as Ukrainian word
            if norm_val.is_integer() and int(norm_val) in self.WORD_NUMBERS:
                words = self.WORD_NUMBERS[int(norm_val)]
                if any(re.search(rf"\b{re.escape(w)}", target_lower) for w in words):
                    continue

            violations.append(
                self.make_violation(
                    rule_code="NUMBER_MISMATCH",
                    message=f"Source number '{raw_num}' not preserved in translated text.",
                    severity=QASeverity.CRITICAL,
                    source_snippet=ref_source[:80],
                    target_snippet=target_text[:80],
                    suggested_fix=f"Preserve number '{raw_num}' accurately in the translation.",
                    metadata={"source_number": raw_num, "normalized": norm_val},
                )
            )

        # 2. Roman Numerals Preservation
        source_romans = self._extract_roman_numerals(ref_source)
        for roman in source_romans:
            roman_pattern = rf"\b{re.escape(roman)}\b"
            if not re.search(roman_pattern, target_text):
                violations.append(
                    self.make_violation(
                        rule_code="ROMAN_NUMERAL_MISMATCH",
                        message=f"Roman numeral '{roman}' present in source was not found in translation.",
                        severity=QASeverity.WARNING,
                        source_snippet=ref_source[:80],
                        target_snippet=target_text[:80],
                        suggested_fix=f"Preserve Roman numeral '{roman}' in translation.",
                        metadata={"roman_numeral": roman},
                    )
                )

        # 3. Percentages & Currencies
        if "%" in ref_source:
            if "%" not in target_text and "відсотк" not in target_text.lower() and "процент" not in target_text.lower():
                violations.append(
                    self.make_violation(
                        rule_code="CURRENCY_PERCENTAGE_MISMATCH",
                        message="Percentage indicator '%' present in source was not preserved in translation.",
                        severity=QASeverity.WARNING,
                        suggested_fix="Ensure percentage '%' or 'відсотків' is preserved.",
                    )
                )

        for sym, terms in self.CURRENCY_MAP.items():
            if sym in ref_source:
                has_curr = any(t in target_text.lower() for t in terms)
                if not has_curr:
                    violations.append(
                        self.make_violation(
                            rule_code="CURRENCY_PERCENTAGE_MISMATCH",
                            message=f"Currency symbol '{sym}' present in source was not found in translation.",
                            severity=QASeverity.WARNING,
                            suggested_fix=f"Preserve currency '{sym}' or write corresponding Ukrainian term.",
                        )
                    )

        return violations

    def _extract_normalized_numbers(self, text: str) -> List[tuple[str, float]]:
        results: List[tuple[str, float]] = []
        for match in self.ARABIC_NUM_PATTERN.finditer(text):
            raw = match.group(0)
            cleaned = raw.replace(" ", "").replace(",", ".")
            try:
                val = float(cleaned)
                results.append((raw, val))
            except ValueError:
                pass
        return results

    def _extract_roman_numerals(self, text: str) -> Set[str]:
        romans: Set[str] = set()
        for match in self.ROMAN_NUM_PATTERN.finditer(text):
            r = match.group(1)
            if not r:
                continue
            r_upper = r.upper()
            if r_upper == "I" and not match.group(0).lower().startswith(("chapter", "part", "section", "глава", "розділ")):
                continue
            if len(r_upper) >= 2 or r_upper in ("V", "X", "L", "C", "D", "M"):
                romans.add(r_upper)
        return romans
