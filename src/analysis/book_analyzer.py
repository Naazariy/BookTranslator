"""
Whole-Book Pre-Translation Analyzer and Entity Indexing Engine for BookTranslator V2.

Performs document-wide entity candidate extraction, classification, multi-factor
confidence scoring, auto-locking, policy assignment, and segment-level mention indexing.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from src.domain.models.document import Book
from src.domain.models.knowledge import (
    EntityMention,
    EntityProfile,
    EntityType,
    GlossaryItem,
    ScopeLevel,
    TranslationPolicy,
)
from src.domain.models.segment import SegmentStatus, TranslationSegment

logger = logging.getLogger(__name__)


EXTENDED_STOP_WORDS: Set[str] = {
    # Pronouns & Articles
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "he", "his", "him",
    "she", "her", "hers", "they", "them", "their", "we", "us", "our", "you", "your",
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
    # Conjunctions & Prepositions
    "and", "or", "but", "nor", "so", "for", "yet", "if", "then", "because", "as",
    "in", "on", "at", "to", "from", "by", "with", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "behind", "beyond", "without",
    # Discourse Markers & Sentence Starters
    "however", "although", "though", "meanwhile", "suddenly", "later", "finally",
    "already", "still", "again", "never", "always", "sometimes", "often", "soon",
    "thus", "therefore", "furthermore", "moreover", "otherwise", "nevertheless",
    "perhaps", "maybe", "indeed", "almost", "nearly", "besides", "instead",
    # Auxiliary & Common Verbs
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # Time & General Nouns
    "yesterday", "today", "tomorrow", "tonight", "morning", "evening", "night", "day",
    "chapter", "book", "page", "section", "part", "volume", "title", "author",
    "someone", "everyone", "anyone", "no one", "something", "everything", "anything", "nothing",
}


@dataclass
class BookAnalyzerConfig:
    min_occurrences: int = 2
    auto_lock_threshold: float = 0.90
    sentence_start_penalty: float = -0.40
    title_boost: float = 0.35
    dialogue_verb_boost: float = 0.20
    rpg_tag_boost: float = 0.35
    item_pattern_boost: float = 0.30
    multi_word_boost: float = 0.15


class BookAnalysisResult(BaseModel):
    """Complete pre-translation analysis output for a Book."""
    book_id: UUID
    segments: List[TranslationSegment]
    entity_profiles: List[EntityProfile]
    entity_mentions: List[EntityMention]
    glossary_items: List[GlossaryItem]
    statistics: Dict[str, Any] = Field(default_factory=dict)


class BookAnalyzer:
    """
    Whole-book analyzer extracting entities, computing multi-factor confidence scores,
    auto-locking high-confidence profiles (>= 0.90), and building segment-level mention indexes.
    """

    def __init__(self, config: Optional[BookAnalyzerConfig] = None):
        self.config = config or BookAnalyzerConfig()
        self._init_lexicons()

    def _init_lexicons(self) -> None:
        self.title_gender_map: Dict[str, Optional[str]] = {
            # Masculine Titles -> "чоловічий"
            "lord": "чоловічий", "sir": "чоловічий", "king": "чоловічий", "prince": "чоловічий",
            "grandmaster": "чоловічий", "duke": "чоловічий", "baron": "чоловічий", "count": "чоловічий",
            "mr": "чоловічий", "father": "чоловічий", "brother": "чоловічий", "emperor": "чоловічий",
            # Feminine Titles -> "жіночий"
            "lady": "жіночий", "madam": "жіночий", "queen": "жіночий", "princess": "жіночий",
            "duchess": "жіночий", "baroness": "жіночий", "countess": "жіночий", "mrs": "жіночий",
            "ms": "жіночий", "miss": "жіночий", "mother": "жіночий", "sister": "жіночий", "empress": "жіночий",
            # Neutral / Titles with indeterminate gender
            "dr": None, "doctor": None, "prof": None, "professor": None, "captain": None,
            "commander": None, "archmage": None, "elder": None, "general": None, "master": None
        }

        self.character_verbs: Set[str] = {
            "said", "replied", "whispered", "smiled", "sighed", "nodded", "asked",
            "looked", "drew", "shouted", "yelled", "laughed", "murmured", "gasped",
            "stood", "walked", "stepped", "turned", "glanced", "peered", "paused",
            "knocked", "reloaded", "entered", "ordered", "spoke",
            "adjusted", "read", "held", "watched", "sat", "felt"
        }

        self.homograph_registry: Dict[str, Dict[str, Any]] = {
            "cherry": {
                "canonical": "Черрі",
                "gender": "жіночий",
                "forbidden": [
                    "Вишня", "Вішня", "Черри", "Черешня", "Вишенька",
                    "Вишнею", "Вішнею", "Черешнею", "Вишенькою",
                    "Вишень", "Черешень", "Вишеньок"
                ]
            },
            "hope": {
                "canonical": "Гоуп",
                "gender": "жіночий",
                "forbidden": ["Надія", "Надежда", "Сподівання", "Надій"]
            },
            "faith": {
                "canonical": "Фейт",
                "gender": "жіночий",
                "forbidden": ["Віра", "Вера", "Довіра"]
            },
            "robin": {
                "canonical": "Робін",
                "gender": "чоловічий",
                "forbidden": ["Вільшанка", "Малинівка", "Дрізд", "Вільшанок", "Малинівок"]
            },
            "rose": {
                "canonical": "Роуз",
                "gender": "жіночий",
                "forbidden": ["Троянда", "Роза", "Троянд"]
            },
            "flint": {
                "canonical": "Флінт",
                "gender": "чоловічий",
                "forbidden": ["Кремінь", "Кремень"]
            },
            "mary": {
                "canonical": "Марія",
                "gender": "жіночий",
                "forbidden": []
            }
        }

        self.item_canonical_registry: Dict[str, Dict[str, Any]] = {
            "healing potion": {
                "canonical": "зілля зцілення",
                "forbidden": ["напій", "лікувальний напій", "чай", "суп", "компот", "відвар"]
            },
            "mana potion": {
                "canonical": "зілля мани",
                "forbidden": ["напій мани", "напоєм мани", "енергетик", "содова"]
            },
            "stamina elixir": {
                "canonical": "еліксир витривалості",
                "forbidden": ["напій витривалості", "сироп", "мікстура", "компот"]
            },
            "broadsword": {
                "canonical": "палаш",
                "forbidden": ["широкий ніж", "шабля", "лопатка", "широка шабля"]
            },
            "scroll of town portal": {
                "canonical": "сувій міського порталу",
                "forbidden": ["рулон", "прокрутка", "список", "скрол"]
            },
            "chainmail armor": {
                "canonical": "кольчуга",
                "forbidden": ["поштової броні", "ланцюгової пошти", "ланцюгової броні", "панцира з ланцюгів"]
            },
            "wooden staff": {
                "canonical": "дерев'яний посох",
                "forbidden": []
            },
            "silver dagger": {
                "canonical": "срібний кинджал",
                "forbidden": []
            }
        }

    def analyze_book(
        self,
        book: Book,
        segments: Optional[List[TranslationSegment]] = None,
    ) -> BookAnalysisResult:
        """Main analysis entrypoint producing EntityProfiles, EntityMentions, and BookAnalysisResult."""
        # 1. Ensure TranslationSegments exist
        seg_list = segments if segments is not None else self._create_segments_from_book(book)

        # 2. Extract Raw Candidates across segments
        raw_candidates = self._scan_candidate_occurrences(seg_list)

        # 3. Aggregate Statistics & Context Features
        profiles: List[EntityProfile] = []
        for name, occ_list in raw_candidates.items():
            profile = self._build_entity_profile(book.id, name, occ_list)
            if profile:
                profiles.append(profile)

        # 4. Index Mentions per Segment
        mentions: List[EntityMention] = []
        for seg in seg_list:
            seg_mentions = self._index_mentions_for_segment(seg, profiles)
            mentions.extend(seg_mentions)

        # 5. Generate Backwards-Compatible Glossary Items
        glossary_items = [p.to_glossary_item() for p in profiles]

        logger.info(
            f"Book analysis completed: {len(profiles)} profiles ({sum(1 for p in profiles if p.locked)} locked), "
            f"{len(mentions)} mentions across {len(seg_list)} segments."
        )

        return BookAnalysisResult(
            book_id=book.id,
            segments=seg_list,
            entity_profiles=profiles,
            entity_mentions=mentions,
            glossary_items=glossary_items,
            statistics={
                "total_profiles": len(profiles),
                "locked_profiles": sum(1 for p in profiles if p.locked),
                "total_mentions": len(mentions),
                "total_segments": len(seg_list),
            }
        )

    def _create_segments_from_book(self, book: Book) -> List[TranslationSegment]:
        segments: List[TranslationSegment] = []
        order_idx = 0
        for chapter in book.chapters:
            for paragraph in chapter.paragraphs:
                p_text = " ".join(s.original_text for s in paragraph.sentences if s.original_text).strip()
                if not p_text:
                    continue
                seg = TranslationSegment(
                    id=paragraph.id,
                    book_id=book.id,
                    chapter_id=chapter.id,
                    paragraph_id=paragraph.id,
                    order_index=order_idx,
                    source_text=p_text,
                    sentence_ids=[s.id for s in paragraph.sentences],
                    status=SegmentStatus.PENDING,
                )
                segments.append(seg)
                order_idx += 1
        return segments

    def _scan_candidate_occurrences(
        self,
        segments: List[TranslationSegment]
    ) -> Dict[str, List[Dict[str, Any]]]:
        occurrences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for seg in segments:
            text = seg.source_text

            # 1. Bracketed RPG tags
            for m in re.finditer(r'\[(?:Status|Class|Skill Acquired|Combat Log|System Alert)[^\]]*\]', text):
                tag_content = m.group(0)
                class_match = re.search(r'\[Class:\s*([^\]]+)\]', tag_content)
                if class_match:
                    cname = class_match.group(1).strip()
                    occurrences[cname].append({"type": "rpg_class", "seg_id": seg.id, "mid_sentence": True})
                skill_match = re.search(r'\[Skill Acquired:\s*[\'"]([^\'"]+)[\'"]', tag_content)
                if skill_match:
                    sname = skill_match.group(1).strip()
                    occurrences[sname].append({"type": "rpg_skill", "seg_id": seg.id, "mid_sentence": True})

            # 2. Item compounds (healing potion, broadsword, etc.)
            for item_name in self.item_canonical_registry.keys():
                for m in re.finditer(rf'\b{re.escape(item_name)}\b', text, re.IGNORECASE):
                    occurrences[item_name].append({
                        "type": "item",
                        "seg_id": seg.id,
                        "mid_sentence": True,
                        "context_clause": text,
                    })

            # 3. Titles with Proper Names (e.g. Lord Bryan, Princess Elena, Grandmaster Thorne, Dr. Watson)
            title_pattern = rf"\b({'|'.join(re.escape(t) for t in self.title_gender_map.keys())})\.?\s+([A-Z][a-zA-Z0-9'\-]+)\b"
            for tm in re.finditer(title_pattern, text, re.IGNORECASE):
                title_word = tm.group(1).lower().rstrip('.')
                proper_name = tm.group(2)
                compound_name = f"{title_word.capitalize()} {proper_name}"
                inferred_g = self.title_gender_map.get(title_word)

                occurrences[compound_name].append({
                    "type": "title_compound",
                    "seg_id": seg.id,
                    "mid_sentence": True,
                    "inferred_gender": inferred_g,
                    "title": title_word,
                    "base_name": proper_name,
                    "context_clause": text,
                })
                # Also index the base proper name
                occurrences[proper_name].append({
                    "type": "proper_noun",
                    "seg_id": seg.id,
                    "mid_sentence": True,
                    "inferred_gender": inferred_g,
                    "context_clause": text,
                })

            # 4. Sentence & clause tokenization for capitalization disambiguation
            clauses = re.split(r'(?<=[.!?])\s+', text)
            for clause in clauses:
                tokens = clause.strip().split()
                if not tokens:
                    continue

                for i, raw_tok in enumerate(tokens):
                    tok = re.sub(r'^[^\w]+|[^\w]+$', '', raw_tok)
                    tok_clean = re.sub(r"(?:'s|’s)$", "", tok).strip()
                    if not tok_clean or not tok_clean[0].isupper() or len(tok_clean) <= 1:
                        continue
                    if tok_clean.lower() in EXTENDED_STOP_WORDS:
                        continue

                    is_initial = (i == 0)
                    occurrences[tok_clean].append({
                        "type": "proper_noun",
                        "seg_id": seg.id,
                        "mid_sentence": not is_initial,
                        "context_clause": clause,
                    })

        return occurrences

    def _build_entity_profile(
        self,
        book_id: UUID,
        name: str,
        occurrences: List[Dict[str, Any]],
    ) -> Optional[EntityProfile]:
        freq = len(occurrences)
        name_lower = name.lower()

        # Check if known entity in registries
        is_known_homograph = name_lower in self.homograph_registry
        is_known_item = name_lower in self.item_canonical_registry
        is_compound = " " in name

        # Calculate mid-sentence ratio
        mid_count = sum(1 for occ in occurrences if occ.get("mid_sentence", False))
        r_mid = mid_count / freq if freq > 0 else 0.0

        # Contextual feature detection
        has_title = False
        inferred_gender: Optional[str] = None
        has_action_verb = False
        has_rpg_tag = any(occ.get("type") in ("rpg_class", "rpg_skill") for occ in occurrences)
        has_item = any(occ.get("type") == "item" for occ in occurrences) or is_known_item
        aliases_set: Set[str] = set()

        for occ in occurrences:
            if occ.get("inferred_gender"):
                inferred_gender = occ["inferred_gender"]
            if occ.get("type") == "title_compound":
                has_title = True
            if occ.get("base_name") and occ["base_name"] != name:
                aliases_set.add(occ["base_name"])
            if occ.get("compound_name") and occ["compound_name"] != name:
                aliases_set.add(occ["compound_name"])

            clause = occ.get("context_clause", "").lower()
            if clause:
                if any(verb in clause for verb in self.character_verbs):
                    has_action_verb = True
                for title, gender in self.title_gender_map.items():
                    if f"{title} {name_lower}" in clause or f"{title}. {name_lower}" in clause:
                        has_title = True
                        if gender and not inferred_gender:
                            inferred_gender = gender

        has_char_context = has_title or has_action_verb or (r_mid >= 0.40)

        # Occurrence threshold check: filter out low-frequency candidates unless they have strong prior context
        if freq < self.config.min_occurrences and not ((is_known_homograph and has_char_context) or is_known_item or is_compound):
            return None

        # Frequency score component S_freq
        s_freq = min(1.0, math.log2(freq + 1) / math.log2(11)) * 0.40

        # Positional mid-sentence component S_mid
        if r_mid >= 0.75:
            s_mid = 0.25
        elif r_mid >= 0.40:
            s_mid = 0.15
        elif r_mid < 0.10 and not is_compound:
            s_mid = self.config.sentence_start_penalty
        else:
            s_mid = 0.0

        # Compute context boosts
        s_context = 0.0
        if has_title:
            s_context += self.config.title_boost
        if has_rpg_tag:
            s_context += self.config.rpg_tag_boost
        if has_item:
            s_context += self.config.item_pattern_boost
        if has_action_verb:
            s_context += self.config.dialogue_verb_boost
        if is_compound:
            s_context += self.config.multi_word_boost

        confidence = max(0.0, min(1.0, s_freq + s_mid + s_context))

        # Title provides definitive character evidence; ensure auto-lock
        if has_title:
            confidence = max(confidence, 0.90)

        # Determine canonical targets, policies, and forbidden forms
        if is_known_homograph and has_char_context:
            entry = self.homograph_registry[name_lower]
            canonical = entry["canonical"]
            forbidden = list(entry["forbidden"])
            gender = inferred_gender or entry.get("gender")
            etype = EntityType.CHARACTER
            policy = TranslationPolicy.RESTRICTED_VARIANTS
            # Homographs with active context get boosted to guaranteed auto-lock >= 0.90
            confidence = max(confidence, 0.92)
        elif is_known_item:
            entry = self.item_canonical_registry[name_lower]
            canonical = entry["canonical"]
            forbidden = list(entry["forbidden"])
            gender = None
            etype = EntityType.ITEM
            policy = TranslationPolicy.TRANSLATE_MEANING
            confidence = max(confidence, 0.95)
        elif has_rpg_tag:
            canonical = name
            forbidden = []
            gender = None
            etype = EntityType.MAGIC if any(occ.get("type") == "rpg_skill" for occ in occurrences) else EntityType.TERM
            policy = TranslationPolicy.TRANSLATE_MEANING
            confidence = max(confidence, 0.90)
        else:
            canonical = name
            forbidden = []
            gender = inferred_gender
            etype = EntityType.CHARACTER if (has_title or has_action_verb) else EntityType.TERM
            policy = TranslationPolicy.TRANSLITERATE

        # Auto-lock entities when confidence >= threshold
        locked = (confidence >= self.config.auto_lock_threshold)

        return EntityProfile(
            id=uuid4(),
            scope=ScopeLevel.BOOK,
            scope_id=str(book_id),
            source_name=name,
            canonical_target=canonical,
            aliases=list(aliases_set),
            forbidden_target_forms=forbidden,
            entity_type=etype,
            grammatical_gender=gender,
            translation_policy=policy,
            confidence=round(confidence, 3),
            locked=locked,
            frequency=freq,
        )

    def _index_mentions_for_segment(
        self,
        segment: TranslationSegment,
        profiles: List[EntityProfile],
    ) -> List[EntityMention]:
        mentions: List[EntityMention] = []
        text = segment.source_text

        for p in profiles:
            names_to_match = [p.source_name] + p.aliases
            for name in names_to_match:
                pattern = rf"\b{re.escape(name)}(?:'s|’s)?\b"
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    matched = m.group(0)
                    # If the entity name begins with an uppercase letter, do not match lowercase occurrences
                    if name and name[0].isupper() and not matched[0].isupper():
                        continue

                    mentions.append(
                        EntityMention(
                            id=uuid4(),
                            entity_id=p.id,
                            segment_id=segment.id,
                            char_start=m.start(),
                            char_end=m.end(),
                            surface_form=matched,
                            confidence=p.confidence,
                        )
                    )
        return mentions
