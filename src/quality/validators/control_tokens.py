"""
src/quality/validators/control_tokens.py

Validator detecting residual model tokens (<|...|>), markdown code fences,
leaked JSON payloads, prompt templates, technical tags, and conversational chatter.
"""
from __future__ import annotations

import re
from typing import Optional, List, Any
from src.domain.models.segment import TranslationSegment
from src.context.builder import PromptContext
from src.quality.base import BaseValidator
from src.quality.models import QAViolation, QASeverity


class ControlTokenValidator(BaseValidator):
    """
    Validates output purity by flagging:
    - Residual model control tokens (<|...|>)
    - Leaked technical tags (</?tag_\\d+>)
    - Markdown code fences and JSON wrappers (```json, {"segments":)
    - Prompt template section headers (=== ПОПЕРЕДНІЙ КОНТЕКСТ ===, etc.)
    - Conversational assistant preambles ("Here is the translation", "Ось переклад")
    """
    name: str = "ControlTokenValidator"
    description: str = "Detects leaked model tokens, code fences, JSON artifacts, and conversational chatter"
    default_severity: QASeverity = QASeverity.CRITICAL

    CONTROL_TOKEN_REGEX = re.compile(r'<\|[^|>\n]{1,40}\|>')
    TECHNICAL_TAG_REGEX = re.compile(r'</?tag_\d+>')

    CODE_FENCE_INDICATORS = [
        "```json",
        "```",
        '{"segments":',
        '{"id":',
        '"translation":',
    ]

    PROMPT_TEMPLATE_INDICATORS = [
        "=== ПОПЕРЕДНІЙ КОНТЕКСТ ===",
        "=== ГЛОСАРІЙ ТА СУТНОСТІ ===",
        "=== ПАРАГРАФ ДЛЯ РЕДАГУВАННЯ ===",
        "=== ВІДРЕДАГОВАНИЙ JSON ===",
        "ОРИГІНАЛ (EN):",
        "ЧОРНОВИЙ ПЕРЕКЛАД NLLB",
    ]

    CONVERSATIONAL_PATTERNS = [
        re.compile(r'(?i)\b(?:here is the translation|sure, here is|as an ai language model|here is a translation)\b'),
        re.compile(r'(?i)\b(?:ось переклад|звісно, ось переклад|вибачте, як штучний інтелект|ось відредагований переклад)\b'),
    ]

    def validate(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        translated_text: Optional[str] = None,
        **kwargs: Any,
    ) -> List[QAViolation]:
        violations: List[QAViolation] = []
        target_text = self.extract_target_text(segment, translated_text=translated_text).strip()
        if not target_text:
            return violations

        # 1. Residual Model Control Tokens (CRITICAL)
        leaked_ctrl = self.CONTROL_TOKEN_REGEX.findall(target_text)
        if leaked_ctrl:
            unique_ctrl = list(dict.fromkeys(leaked_ctrl))
            violations.append(
                self.make_violation(
                    rule_code="ResidualControlToken",
                    message=f"Detected residual model control tokens in translation: {', '.join(unique_ctrl)}",
                    severity=QASeverity.CRITICAL,
                    target_snippet=target_text[:100],
                    forbidden_form=", ".join(unique_ctrl),
                    suggested_fix="Remove model control tokens from output.",
                )
            )

        # 2. Leaked Technical Tags (WARNING)
        leaked_tech = self.TECHNICAL_TAG_REGEX.findall(target_text)
        if leaked_tech:
            unique_tech = list(dict.fromkeys(leaked_tech))
            violations.append(
                self.make_violation(
                    rule_code="LeakedTechnicalTag",
                    message=f"Detected unresolved technical tags in translation: {', '.join(unique_tech)}",
                    severity=QASeverity.WARNING,
                    target_snippet=target_text[:100],
                    forbidden_form=", ".join(unique_tech),
                    suggested_fix="Resolve or strip technical tags.",
                )
            )

        # 3. Code Fences & Leaked JSON Data (CRITICAL)
        for fence in self.CODE_FENCE_INDICATORS:
            if fence in target_text:
                violations.append(
                    self.make_violation(
                        rule_code="LEAKED_JSON_OR_CODE_FENCE",
                        message="Detected leaked JSON structure or markdown code fences in translation.",
                        severity=QASeverity.CRITICAL,
                        target_snippet=target_text[:120],
                        suggested_fix="Strip JSON wrappers and markdown code fences; output clean prose.",
                    )
                )
                break

        # 4. Prompt Template Leaks (CRITICAL)
        for tmpl in self.PROMPT_TEMPLATE_INDICATORS:
            if tmpl in target_text:
                violations.append(
                    self.make_violation(
                        rule_code="LEAKED_PROMPT_TEMPLATE",
                        message="Detected leaked prompt template instructions or metadata tags in translation.",
                        severity=QASeverity.CRITICAL,
                        target_snippet=target_text[:120],
                        suggested_fix="Remove prompt template headers and metadata from translation.",
                    )
                )
                break

        # 5. Conversational Chatter (CRITICAL)
        for pattern in self.CONVERSATIONAL_PATTERNS:
            match = pattern.search(target_text)
            if match:
                violations.append(
                    self.make_violation(
                        rule_code="CONVERSATIONAL_CHATTER",
                        message=f"Detected conversational preamble or AI commentary '{match.group(0)}' in translation.",
                        severity=QASeverity.CRITICAL,
                        target_snippet=target_text[:120],
                        forbidden_form=match.group(0),
                        suggested_fix="Remove assistant conversational preamble.",
                    )
                )
                break

        return violations
