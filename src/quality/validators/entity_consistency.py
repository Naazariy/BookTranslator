"""
src/quality/validators/entity_consistency.py

Validator enforcing character and named entity consistency, strictly rejecting
forbidden translation variants and requiring locked canonical forms.
"""
from __future__ import annotations

import re
from typing import Optional, List, Any, Set
from src.domain.models.segment import TranslationSegment
from src.context.builder import PromptContext
from src.domain.models.knowledge import EntityProfile, GlossaryItem, _generate_ukrainian_inflections
from src.quality.base import BaseValidator
from src.quality.models import QAViolation, QASeverity


class EntityConsistencyValidator(BaseValidator):
    """
    Validates named entity translations against scoped entity profiles and glossaries:
    - Strictly rejects forbidden variants (e.g., Cherry -> Вишня/Вішня/Черри)
    - Enforces approved canonical targets for locked or high-confidence entities
    """
    name: str = "EntityConsistencyValidator"
    description: str = "Enforces canonical entity targets and strictly rejects forbidden variants"
    default_severity: QASeverity = QASeverity.CRITICAL

    def validate(
        self,
        segment: TranslationSegment,
        context: Optional[PromptContext] = None,
        translated_text: Optional[str] = None,
        entities: Optional[List[Any]] = None,
        glossary: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> List[QAViolation]:
        violations: List[QAViolation] = []
        source_text = self.extract_source_text(segment).strip()
        target_text = self.extract_target_text(segment, translated_text=translated_text).strip()

        if not target_text or not source_text:
            return violations

        active_entities = self._collect_entities(context, entities, glossary)
        if not active_entities:
            return violations

        for item in active_entities:
            profile = self._ensure_entity_profile(item)
            if profile is None:
                continue

            # 1. Verify entity presence in source text
            if not self._is_entity_in_source(profile, source_text):
                continue

            # 2. Check for Forbidden Variants (CRITICAL)
            if profile.is_variant_forbidden(target_text, check_inflections=True):
                detected = profile.find_forbidden_mentions(target_text, check_inflections=True)
                detected_str = ", ".join(detected) if detected else "заборонений варіант"
                target_snip = self._extract_snippet(target_text, detected[0] if detected else "")
                violations.append(
                    self.make_violation(
                        rule_code="FORBIDDEN_ENTITY_VARIANT",
                        message=(
                            f"Forbidden variant '{detected_str}' detected for entity "
                            f"'{profile.source_name}' (canonical target: '{profile.canonical_target}')."
                        ),
                        severity=QASeverity.CRITICAL,
                        source_snippet=self._extract_snippet(source_text, profile.source_name),
                        target_snippet=target_snip,
                        forbidden_form=detected_str,
                        suggested_fix=profile.canonical_target,
                        metadata={
                            "entity_id": str(profile.id),
                            "source_name": profile.source_name,
                            "canonical_target": profile.canonical_target,
                            "detected_forbidden": detected,
                        },
                    )
                )
                continue

            # 3. Check for Locked / High-Confidence Canonical Form Enforcement
            is_locked = profile.locked or (profile.confidence >= 0.90)
            if is_locked and profile.canonical_target:
                if not self._is_canonical_present(profile, target_text):
                    violations.append(
                        self.make_violation(
                            rule_code="MISSING_LOCKED_ENTITY",
                            message=(
                                f"Locked entity '{profile.source_name}' is present in source, "
                                f"but canonical translation '{profile.canonical_target}' is missing."
                            ),
                            severity=QASeverity.CRITICAL,
                            source_snippet=self._extract_snippet(source_text, profile.source_name),
                            target_snippet=target_text[:100],
                            suggested_fix=profile.canonical_target,
                            metadata={
                                "entity_id": str(profile.id),
                                "source_name": profile.source_name,
                                "canonical_target": profile.canonical_target,
                            },
                        )
                    )

        return violations

    def _collect_entities(
        self,
        context: Optional[PromptContext],
        entities: Optional[List[Any]],
        glossary: Optional[List[Any]],
    ) -> List[Any]:
        """Gathers entity objects from context or kwargs."""
        gathered: List[Any] = []
        if entities:
            gathered.extend(entities)
        if glossary:
            gathered.extend(glossary)
        if context:
            if hasattr(context, "active_entities") and context.active_entities:
                gathered.extend(context.active_entities)
            if hasattr(context, "glossary") and context.glossary:
                gathered.extend(context.glossary)
        return gathered

    def _ensure_entity_profile(self, item: Any) -> Optional[EntityProfile]:
        """Normalizes EntityProfile or GlossaryItem into an EntityProfile."""
        if isinstance(item, EntityProfile):
            return item
        if isinstance(item, GlossaryItem):
            return EntityProfile.from_glossary_item(item)
        if hasattr(item, "source_name") and hasattr(item, "canonical_target"):
            try:
                return EntityProfile(**dict(item))
            except Exception:
                return None
        if hasattr(item, "source_term") and hasattr(item, "target_term"):
            try:
                g = GlossaryItem(
                    source_term=item.source_term,
                    target_term=item.target_term,
                    reviewed=getattr(item, "reviewed", True),
                    locked=getattr(item, "locked", False),
                    forbidden_variants=getattr(item, "forbidden_variants", []),
                )
                return EntityProfile.from_glossary_item(g)
            except Exception:
                return None
        return None

    def _is_entity_in_source(self, profile: EntityProfile, source_text: str) -> bool:
        """Checks if entity name or any alias is mentioned in the source paragraph."""
        names_to_check = [profile.source_name] + list(profile.aliases or [])
        for name in names_to_check:
            clean = name.strip()
            if not clean:
                continue
            pattern = rf"(?<!\w){re.escape(clean)}(?!\w)"
            if re.search(pattern, source_text, re.IGNORECASE):
                return True
        return False

    def _is_canonical_present(self, profile: EntityProfile, target_text: str) -> bool:
        """Checks if canonical target or any allowed inflection is present in translation."""
        allowed: Set[str] = {profile.canonical_target.strip().lower()}
        for f in profile.allowed_target_forms or []:
            if f and f.strip():
                allowed.add(f.strip().lower())
                allowed.update(inf.lower() for inf in _generate_ukrainian_inflections(f.strip()))
        allowed.update(inf.lower() for inf in _generate_ukrainian_inflections(profile.canonical_target.strip()))

        for form in allowed:
            if not form:
                continue
            pattern = rf"(?<!\w){re.escape(form)}(?!\w)"
            if re.search(pattern, target_text, re.IGNORECASE):
                return True
        return False

    def _extract_snippet(self, text: str, token: str, window: int = 40) -> str:
        """Extracts contextual window around the matching token."""
        if not token or token not in text:
            return text[:window * 2]
        idx = text.find(token)
        start = max(0, idx - window)
        end = min(len(text), idx + len(token) + window)
        return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
