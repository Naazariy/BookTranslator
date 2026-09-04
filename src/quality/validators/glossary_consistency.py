"""
src/quality/validators/glossary_consistency.py

Validator ensuring domain terminology, items, spells, and glossary consistency,
flagging missing reviewed terms and rejecting forbidden glossary variants.
"""
from __future__ import annotations

import re
from typing import Optional, List, Any, Set
from src.domain.models.segment import TranslationSegment
from src.context.builder import PromptContext
from src.domain.models.knowledge import GlossaryItem, EntityProfile, _generate_ukrainian_inflections
from src.quality.base import BaseValidator
from src.quality.models import QAViolation, QASeverity


class GlossaryConsistencyValidator(BaseValidator):
    """
    Validates domain terminology and glossary terms:
    - Verifies presence of approved/reviewed terms in translation when source term appears
    - Strictly rejects forbidden glossary variants (e.g. healing potion -> лікувальний напій)
    """
    name: str = "GlossaryConsistencyValidator"
    description: str = "Enforces approved glossary translations and rejects forbidden term variants"
    default_severity: QASeverity = QASeverity.WARNING

    def validate(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        translated_text: Optional[str] = None,
        glossary: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> List[QAViolation]:
        violations: List[QAViolation] = []
        source_text = self.extract_source_text(segment).strip()
        target_text = self.extract_target_text(segment, translated_text=translated_text).strip()

        if not target_text or not source_text:
            return violations

        glossary_items = self._gather_glossary(context, glossary, kwargs.get("entities"))
        if not glossary_items:
            return violations

        for item in glossary_items:
            source_term = getattr(item, "source_term", None) or getattr(item, "source_name", None)
            target_term = getattr(item, "target_term", None) or getattr(item, "canonical_target", None)
            if not source_term or not target_term:
                continue

            # Check if source term is present in source text
            pattern_src = rf"(?<!\w){re.escape(source_term.strip())}(?!\w)"
            if not re.search(pattern_src, source_text, re.IGNORECASE):
                continue

            # 1. Check for Forbidden Variants (CRITICAL)
            forbidden_variants = getattr(item, "forbidden_variants", None) or getattr(item, "forbidden_target_forms", None) or []
            detected_forbidden = []
            for f_var in forbidden_variants:
                if not f_var or not f_var.strip():
                    continue
                # Include Ukrainian case inflections of forbidden variant
                forms_to_check = {f_var.strip()}
                forms_to_check.update(_generate_ukrainian_inflections(f_var.strip()))
                for form in forms_to_check:
                    f_pattern = rf"(?<!\w){re.escape(form)}(?!\w)"
                    match = re.search(f_pattern, target_text, re.IGNORECASE)
                    if match:
                        detected_forbidden.append(match.group(0))
                        break

            if detected_forbidden:
                det_str = ", ".join(dict.fromkeys(detected_forbidden))
                violations.append(
                    self.make_violation(
                        rule_code="FORBIDDEN_GLOSSARY_VARIANT",
                        message=(
                            f"Forbidden translation variant '{det_str}' detected for term '{source_term}'. "
                            f"Approved target: '{target_term}'."
                        ),
                        severity=QASeverity.CRITICAL,
                        source_snippet=self._extract_snippet(source_text, source_term),
                        target_snippet=self._extract_snippet(target_text, det_str),
                        forbidden_form=det_str,
                        suggested_fix=f"Replace '{det_str}' with approved term '{target_term}'.",
                        metadata={"source_term": source_term, "target_term": target_term, "forbidden": det_str},
                    )
                )
                continue

            # 2. Check for Missing Approved Term
            is_reviewed = getattr(item, "reviewed", False)
            is_locked = getattr(item, "locked", False) or getattr(item, "confidence", 0.0) >= 0.90

            if is_reviewed or is_locked:
                if not self._is_term_present(target_term, target_text, getattr(item, "allowed_forms", None)):
                    sev = QASeverity.CRITICAL if is_locked else QASeverity.WARNING
                    violations.append(
                        self.make_violation(
                            rule_code="MISSING_GLOSSARY_TERM",
                            message=f"Approved glossary term '{source_term}' => '{target_term}' missing from translation.",
                            severity=sev,
                            source_snippet=self._extract_snippet(source_text, source_term),
                            target_snippet=target_text[:100],
                            suggested_fix=f"Ensure '{source_term}' is translated as '{target_term}'.",
                            metadata={"source_term": source_term, "target_term": target_term, "is_locked": is_locked},
                        )
                    )

        return violations

    def _gather_glossary(
        self,
        context: Optional[PromptContext],
        glossary: Optional[List[Any]],
        entities: Optional[List[Any]],
    ) -> List[Any]:
        """Gathers glossary and non-character entity objects."""
        items: List[Any] = []
        if glossary:
            items.extend(glossary)
        if entities:
            items.extend(entities)
        if context:
            if hasattr(context, "glossary") and context.glossary:
                items.extend(context.glossary)
            if hasattr(context, "active_entities") and context.active_entities:
                for e in context.active_entities:
                    # Include items that are glossary terms, items, or have forbidden variants
                    if getattr(e, "entity_type", None) in ("item", "location", "term", "concept", "spell") or \
                       getattr(e, "forbidden_variants", None) or getattr(e, "forbidden_target_forms", None):
                        items.append(e)
        return items

    def _is_term_present(self, target_term: str, target_text: str, allowed_forms: Optional[List[str]]) -> bool:
        """Checks if target term or its inflections appear in target text."""
        candidates: Set[str] = {target_term.strip().lower()}
        if allowed_forms:
            for af in allowed_forms:
                if af and af.strip():
                    candidates.add(af.strip().lower())
                    candidates.update(inf.lower() for inf in _generate_ukrainian_inflections(af.strip()))
        candidates.update(inf.lower() for inf in _generate_ukrainian_inflections(target_term.strip()))

        # For multi-word terms like "зілля зцілення", generate inflected stem parts
        words = target_term.strip().split()
        if len(words) > 1:
            stems = [w[:max(3, len(w) - 2)].lower() for w in words]
            # Check if all word stems appear in target text in order
            stem_pattern = r".*?".join(rf"\b{re.escape(stem)}" for stem in stems)
            if re.search(stem_pattern, target_text, re.IGNORECASE | re.DOTALL):
                return True

        for cand in candidates:
            if not cand:
                continue
            pattern = rf"(?<!\w){re.escape(cand)}(?!\w)"
            if re.search(pattern, target_text, re.IGNORECASE):
                return True

        return False

    def _extract_snippet(self, text: str, token: str, window: int = 40) -> str:
        if not token or token not in text:
            return text[:window * 2]
        idx = text.find(token)
        start = max(0, idx - window)
        end = min(len(text), idx + len(token) + window)
        return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
