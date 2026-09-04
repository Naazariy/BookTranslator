"""
Domain models for knowledge representation, entity profiling, scoped glossaries,
and segment-level entity mentions in BookTranslator V2.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional, List, Dict, Any, Set, Tuple, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator, model_validator


class ScopeLevel(str, Enum):
    """
    Hierarchical scope levels with strict precedence:
    BOOK > SERIES > DOMAIN > GLOBAL
    """
    BOOK = "BOOK"
    SERIES = "SERIES"
    DOMAIN = "DOMAIN"
    GLOBAL = "GLOBAL"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_upper = value.strip().upper()
            for member in cls:
                if member.value == val_upper:
                    return member
        return None

    @property
    def priority(self) -> int:
        """Numeric rank for precedence comparison (higher number = higher precedence)."""
        _PRIORITY = {
            "GLOBAL": 1,
            "DOMAIN": 2,
            "SERIES": 3,
            "BOOK": 4,
        }
        return _PRIORITY.get(self.value.upper(), 0)

    def _other_priority(self, other: Any) -> int:
        if isinstance(other, ScopeLevel):
            return other.priority
        if isinstance(other, str):
            try:
                resolved = ScopeLevel(other)
                return resolved.priority
            except (ValueError, KeyError):
                pass
        raise TypeError(f"Cannot compare ScopeLevel with {type(other)}: {other}")

    def __lt__(self, other: Any) -> bool:
        try:
            return self.priority < self._other_priority(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __le__(self, other: Any) -> bool:
        try:
            return self.priority <= self._other_priority(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __gt__(self, other: Any) -> bool:
        try:
            return self.priority > self._other_priority(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __ge__(self, other: Any) -> bool:
        try:
            return self.priority >= self._other_priority(other)
        except (TypeError, ValueError):
            return NotImplemented


SCOPE_PRECEDENCE: List[ScopeLevel] = [
    ScopeLevel.BOOK,
    ScopeLevel.SERIES,
    ScopeLevel.DOMAIN,
    ScopeLevel.GLOBAL,
]

SCOPE_PRIORITY: Dict[ScopeLevel, int] = {
    ScopeLevel.BOOK: 4,
    ScopeLevel.SERIES: 3,
    ScopeLevel.DOMAIN: 2,
    ScopeLevel.GLOBAL: 1,
}


class TranslationPolicy(str, Enum):
    """Policy governing how an entity or term should be handled during translation."""
    PRESERVE_CANONICAL = "preserve_canonical"
    RESTRICTED_VARIANTS = "restricted_variants"
    TRANSLITERATE = "transliterate"
    TRANSLATE = "translate"
    TRANSLATE_MEANING = "translate_meaning"
    KEEP_ORIGINAL = "keep_original"
    ADAPT = "adapt"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_norm = value.strip().lower()
            for member in cls:
                if member.value.lower() == val_norm:
                    return member
        return None


class EntityType(str, Enum):
    CHARACTER = "character"
    LOCATION = "location"
    ORGANIZATION = "organization"
    ITEM = "item"
    MAGIC = "magic"
    TITLE = "title"
    TERM = "term"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_norm = value.strip().lower()
            for member in cls:
                if member.value.lower() == val_norm:
                    return member
        return None


UK_CONSONANTS: Set[str] = set("бвгґджзклмнпрстфхцчшщ")


def _generate_ukrainian_inflections(base_word: str) -> Set[str]:
    """
    Expands a Ukrainian base word into common nominal case inflections
    (-я, -а, -ь, etc.) to ensure comprehensive forbidden variant matching.
    Correctly handles fleeting vowels (випадні голосні) in genitive plural.
    """
    clean = base_word.strip()
    if not clean or len(clean) < 3:
        return {clean}
    forms: Set[str] = {clean}
    lower = clean.lower()

    if lower.endswith("ія"):
        stem = clean[:-1]
        for end in ["ї", "ю", "єю", "є", "ям", "ями", "ях", "й"]:
            forms.add(stem + (end.upper() if clean.isupper() else end))
    elif lower.endswith("я"):
        stem = clean[:-1]
        for end in ["і", "ю", "ею", "єю", "е", "ям", "ями", "ях", "ів"]:
            forms.add(stem + (end.upper() if clean.isupper() else end))
        s_low = stem.lower()
        if s_low.endswith("н") and len(stem) >= 2 and s_low[-2] in UK_CONSONANTS:
            forms.add(stem[:-1] + ("ЕНЬ" if clean.isupper() else "ень"))
        elif s_low.endswith("л") and len(stem) >= 2 and s_low[-2] in UK_CONSONANTS:
            forms.add(stem[:-1] + ("ЕЛЬ" if clean.isupper() else "ель"))
        else:
            forms.add(stem + ("Ь" if clean.isupper() else "ь"))
    elif lower.endswith("а"):
        stem = clean[:-1]
        for end in ["и", "і", "у", "ою", "ею", "о", "ам", "ами", "ах"]:
            forms.add(stem + (end.upper() if clean.isupper() else end))
        s_low = stem.lower()
        if s_low.endswith("к") and len(stem) >= 2 and (s_low[-2] in UK_CONSONANTS or s_low[-2] == "ь"):
            forms.add(stem[:-1] + ("ОК" if clean.isupper() else "ок"))
        elif s_low not in ("вір", "довір", "повір", "вер"):
            forms.add(stem)
    elif lower.endswith("ь"):
        stem = clean[:-1]
        for end in ["я", "ю", "ем", "і", "ів", "ям", "ями", "ях", "е"]:
            forms.add(stem + (end.upper() if clean.isupper() else end))
        if lower == "кремінь":
            for end in ["ем", "і", "ів", "ю", "я", "ям", "ями", "ях"]:
                forms.add("кремен" + end if clean.islower() else "Кремен" + end)
    else:
        # Masculine ending in consonant
        for end in ["а", "у", "ом", "ем", "і", "ів", "ам", "ами", "ах", "е"]:
            forms.add(clean + (end.upper() if clean.isupper() else end))

    return forms


class EntityProfile(BaseModel):
    """
    Rich entity profile supporting scoped resolution, auto-locking,
    variant rejection, and grammatical gender agreement.
    """
    id: UUID = Field(default_factory=uuid4)
    scope: ScopeLevel = ScopeLevel.BOOK
    scope_id: Optional[str] = None
    source_name: str
    canonical_target: str
    aliases: List[str] = Field(default_factory=list)
    allowed_target_forms: List[str] = Field(default_factory=list)
    forbidden_target_forms: List[str] = Field(default_factory=list)
    grammatical_gender: Optional[str] = None
    translation_policy: Union[TranslationPolicy, str] = Field(
        default=TranslationPolicy.PRESERVE_CANONICAL.value
    )
    confidence: float = 1.0
    locked: bool = False
    entity_type: EntityType = EntityType.CHARACTER
    description: Optional[str] = None
    case_sensitive: bool = True
    frequency: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {v}")
        return round(v, 4)

    @model_validator(mode="after")
    def _enforce_autolock(self) -> "EntityProfile":
        # Auto-lock entities when confidence threshold >= 0.90
        if self.confidence >= 0.90:
            self.locked = True
        # Ensure canonical_target is present in allowed_target_forms
        if self.canonical_target and self.canonical_target not in self.allowed_target_forms:
            self.allowed_target_forms.append(self.canonical_target)
        return self

    @property
    def name(self) -> str:
        """Legacy compatibility alias for Entity.name."""
        return self.source_name

    @property
    def canonical_translation(self) -> str:
        """Legacy compatibility alias for Entity.canonical_translation."""
        return self.canonical_target

    def get_effective_forbidden_forms(self, check_inflections: bool = True) -> Set[str]:
        """Returns all forbidden surface forms including Ukrainian case declensions."""
        effective: Set[str] = set()
        for f in self.forbidden_target_forms:
            clean = f.strip()
            if not clean:
                continue
            effective.add(clean)
            if check_inflections:
                effective.update(_generate_ukrainian_inflections(clean))
        return effective

    def is_variant_forbidden(self, text: str, check_inflections: bool = True) -> bool:
        """
        Checks whether the candidate word or translated text contains or equals
        any forbidden variant form using Unicode word boundaries and declension expansion.
        """
        if not text or not self.forbidden_target_forms:
            return False

        effective = self.get_effective_forbidden_forms(check_inflections=check_inflections)

        # 1. Direct equality check (fast token match)
        clean_text = text.strip().lower()
        if any(clean_text == f.lower() for f in effective):
            return True

        # 2. Text scanning (word-boundary regex or substring)
        for form in effective:
            if not form:
                continue
            form_clean = form.strip()
            # If form is a single alphanumeric word, enforce Unicode word boundaries
            if re.search(r"^\w+$", form_clean, re.UNICODE):
                pattern = rf"(?<!\w){re.escape(form_clean)}(?!\w)"
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            else:
                # Multi-word phrase or compound
                pattern = rf"(?<!\w){re.escape(form_clean)}(?!\w)"
                if re.search(pattern, text, re.IGNORECASE):
                    return True

        return False

    def find_forbidden_mentions(self, text: str, check_inflections: bool = True) -> List[str]:
        """Returns all forbidden surface form strings found in the text."""
        if not text or not self.forbidden_target_forms:
            return []

        effective = self.get_effective_forbidden_forms(check_inflections=check_inflections)
        detected: List[str] = []

        for form in effective:
            if not form:
                continue
            form_clean = form.strip()
            pattern = rf"(?<!\w){re.escape(form_clean)}(?!\w)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                detected.append(match.group(0))

        return list(dict.fromkeys(detected))

    def is_variant_allowed(self, text: str) -> bool:
        """Checks if a translation candidate is explicitly allowed and not forbidden."""
        if self.is_variant_forbidden(text, check_inflections=True):
            return False
        allowed = {self.canonical_target.lower()}
        allowed.update(f.lower() for f in self.allowed_target_forms)
        return text.strip().lower() in allowed

    def to_glossary_item(self) -> "GlossaryItem":
        """Converts EntityProfile to backward-compatible GlossaryItem."""
        return GlossaryItem(
            source_term=self.source_name,
            target_term=self.canonical_target,
            entity_type=self.entity_type,
            case_sensitive=self.case_sensitive,
            reviewed=self.locked or (self.confidence >= 0.90),
            grammatical_gender=self.grammatical_gender,
            scope=self.scope,
            scope_id=self.scope_id,
            locked=self.locked,
            forbidden_variants=list(self.forbidden_target_forms),
            allowed_forms=list(self.allowed_target_forms),
            entity_id=self.id,
        )

    @classmethod
    def from_glossary_item(
        cls,
        item: "GlossaryItem",
        scope: Optional[ScopeLevel] = None,
        scope_id: Optional[str] = None,
        translation_policy: Union[TranslationPolicy, str] = TranslationPolicy.PRESERVE_CANONICAL.value,
        confidence: Optional[float] = None,
    ) -> "EntityProfile":
        """Constructs an EntityProfile from a GlossaryItem."""
        effective_scope = scope or getattr(item, "scope", None) or ScopeLevel.BOOK
        effective_scope_id = scope_id or getattr(item, "scope_id", None)
        conf = confidence if confidence is not None else (1.0 if item.reviewed else 0.5)
        is_locked = getattr(item, "locked", None)
        if is_locked is None:
            is_locked = item.reviewed or (conf >= 0.90)

        forbidden = getattr(item, "forbidden_variants", []) or []
        allowed = getattr(item, "allowed_forms", []) or [item.target_term]

        return cls(
            id=getattr(item, "entity_id", None) or uuid4(),
            scope=effective_scope,
            scope_id=effective_scope_id,
            source_name=item.source_term,
            canonical_target=item.target_term,
            allowed_target_forms=allowed,
            forbidden_target_forms=forbidden,
            grammatical_gender=item.grammatical_gender,
            translation_policy=translation_policy,
            confidence=conf,
            locked=is_locked,
            entity_type=item.entity_type,
            case_sensitive=item.case_sensitive,
        )


class EntityMention(BaseModel):
    """
    Represents an occurrence of an entity within a TranslationSegment.
    Provides fine-grained character span indexing for localized context assembly.
    """
    id: UUID = Field(default_factory=uuid4)
    entity_id: UUID
    segment_id: UUID
    sentence_id: Optional[UUID] = None
    char_start: int
    char_end: int
    surface_form: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_offsets(self) -> "EntityMention":
        if self.char_start < 0:
            raise ValueError(f"char_start must be >= 0, got {self.char_start}")
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be greater than char_start ({self.char_start})"
            )
        expected_len = self.char_end - self.char_start
        if len(self.surface_form) != expected_len:
            raise ValueError(
                f"surface_form length ({len(self.surface_form)}) does not match span ({expected_len})"
            )
        return self

    @property
    def span(self) -> Tuple[int, int]:
        """Returns the (char_start, char_end) tuple."""
        return (self.char_start, self.char_end)

    @property
    def length(self) -> int:
        """Returns span character length."""
        return self.char_end - self.char_start

    def verify_surface_form(self, text: str) -> bool:
        """Verifies that the surface_form matches the slice in the provided source text."""
        return text[self.char_start:self.char_end] == self.surface_form


class Entity(BaseModel):
    """Legacy Entity model preserved for full backward compatibility."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    entity_type: EntityType
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    canonical_translation: Optional[str] = None
    frequency: int = 1


class GlossaryItem(BaseModel):
    """
    Glossary item model supporting legacy callers while extending
    with optional V2 scoped fields.
    """
    source_term: str
    target_term: str
    entity_type: EntityType = EntityType.TERM
    case_sensitive: bool = True
    reviewed: bool = False
    grammatical_gender: Optional[str] = None
    # Optional V2 extensions
    scope: Optional[ScopeLevel] = None
    scope_id: Optional[str] = None
    locked: Optional[bool] = None
    forbidden_variants: List[str] = Field(default_factory=list)
    allowed_forms: List[str] = Field(default_factory=list)
    entity_id: Optional[UUID] = None

    def to_entity_profile(
        self,
        scope: Optional[ScopeLevel] = None,
        scope_id: Optional[str] = None,
        translation_policy: Union[TranslationPolicy, str] = TranslationPolicy.PRESERVE_CANONICAL.value,
        confidence: Optional[float] = None,
    ) -> EntityProfile:
        return EntityProfile.from_glossary_item(
            self,
            scope=scope,
            scope_id=scope_id,
            translation_policy=translation_policy,
            confidence=confidence,
        )


class RawEntity(BaseModel):
    name: str
    entity_type: EntityType
    context: Optional[str] = None


class RawTerm(BaseModel):
    term: str
    frequency: int


class CharacterEntity(Entity):
    pass
