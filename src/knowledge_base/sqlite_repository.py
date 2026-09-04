"""
SQLite Knowledge Base Repository implementation for BookTranslator V2.
Supports scoped entities, mention indexing, chunk checkpoints, and backward compatibility.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Union
from uuid import UUID

from src.domain.interfaces.knowledge_base import IKnowledgeBaseRepository
from src.domain.models.chunk import ChunkStatus, TranslationChunk
from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.domain.models.knowledge import (
    Entity,
    EntityMention,
    EntityProfile,
    EntityType,
    GlossaryItem,
    SCOPE_PRIORITY,
    ScopeLevel,
    TranslationPolicy,
)


def _row_to_entity_profile(row: tuple) -> EntityProfile:
    return EntityProfile(
        id=UUID(row[0]),
        entity_type=EntityType(row[1]),
        scope=ScopeLevel(row[2]),
        scope_id=row[3],
        source_name=row[4],
        canonical_target=row[5],
        aliases=json.loads(row[6]) if row[6] else [],
        allowed_target_forms=json.loads(row[7]) if row[7] else [],
        forbidden_target_forms=json.loads(row[8]) if row[8] else [],
        grammatical_gender=row[9],
        translation_policy=row[10] or TranslationPolicy.PRESERVE_CANONICAL.value,
        confidence=float(row[11]) if row[11] is not None else 1.0,
        locked=bool(row[12]),
        description=row[13],
        case_sensitive=bool(row[14]) if row[14] is not None else True,
        frequency=int(row[15]) if row[15] is not None else 1,
        created_at=row[16],
        updated_at=row[17],
    )


def _row_to_entity_mention(row: tuple) -> EntityMention:
    return EntityMention(
        id=UUID(row[0]),
        entity_id=UUID(row[1]),
        segment_id=UUID(row[2]),
        sentence_id=UUID(row[3]) if row[3] else None,
        char_start=int(row[4]),
        char_end=int(row[5]),
        surface_form=row[6],
        confidence=float(row[7]) if row[7] is not None else 1.0,
        metadata=json.loads(row[8]) if row[8] else {},
    )


class SQLiteKnowledgeBaseRepository(IKnowledgeBaseRepository):
    """
    Thread-safe, high-throughput SQLite repository for knowledge base, glossary,
    scoped entity profiles, segment-level mentions, and chunk checkpointing.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all_connections: Set[sqlite3.Connection] = set()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Retrieves or initializes a thread-affine SQLite connection with performance PRAGMAs.
        Ensures complete thread safety and eliminates connection open/close overhead.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self.db_path,
                timeout=10.0,
                check_same_thread=False
            )
            # Apply connection-level performance PRAGMAs
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 10000;")
            conn.execute("PRAGMA cache_size = -64000;")
            conn.execute("PRAGMA temp_store = MEMORY;")
            conn.execute("PRAGMA foreign_keys = ON;")
            self._local.conn = conn
            with self._lock:
                self._all_connections.add(conn)
        return self._local.conn

    def _connect(self) -> sqlite3.Connection:
        """Legacy compatibility method; delegates to _get_connection."""
        return self._get_connection()

    # ==========================================
    # Legacy Entity Methods
    # ==========================================

    def add_entity(self, entity: Entity) -> None:
        """Saves or updates an entity in the entities table."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO entities (id, name, entity_type, aliases, description, canonical_translation, frequency) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(entity.id),
                    entity.name,
                    entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type),
                    json.dumps(entity.aliases) if entity.aliases is not None else None,
                    entity.description,
                    entity.canonical_translation,
                    entity.frequency
                )
            )

    # ==========================================
    # Legacy Glossary Methods
    # ==========================================

    def get_glossary(self) -> List[GlossaryItem]:
        """Retrieves all glossary items including review status and grammatical gender."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT g.source_term, g.target_term, g.entity_type, g.case_sensitive,
                       COALESCE(m.reviewed, 0), m.grammatical_gender
                FROM glossary g
                LEFT JOIN glossary_metadata m ON g.source_term = m.source_term
            """)
            rows = cursor.fetchall()
            return [
                GlossaryItem(
                    source_term=row[0],
                    target_term=row[1],
                    entity_type=row[2],
                    case_sensitive=bool(row[3]),
                    reviewed=bool(row[4]),
                    grammatical_gender=row[5]
                ) for row in rows
            ]
        except Exception:
            cursor.execute("SELECT source_term, target_term, entity_type, case_sensitive FROM glossary")
            rows = cursor.fetchall()
            return [
                GlossaryItem(
                    source_term=row[0],
                    target_term=row[1],
                    entity_type=row[2],
                    case_sensitive=bool(row[3])
                ) for row in rows
            ]

    def add_glossary_item(self, item: GlossaryItem) -> None:
        """Adds or updates a single glossary item."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO glossary (source_term, target_term, entity_type, case_sensitive) "
                "VALUES (?, ?, ?, ?)",
                (
                    item.source_term,
                    item.target_term,
                    item.entity_type.value if hasattr(item.entity_type, "value") else str(item.entity_type),
                    1 if item.case_sensitive else 0
                )
            )
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO glossary_metadata (source_term, reviewed, grammatical_gender) "
                    "VALUES (?, ?, ?)",
                    (
                        item.source_term,
                        1 if getattr(item, "reviewed", False) else 0,
                        getattr(item, "grammatical_gender", None)
                    )
                )
            except Exception:
                pass

    def add_glossary_items(self, items: List[GlossaryItem]) -> None:
        """Batch inserts or updates glossary items."""
        if not items:
            return
        conn = self._get_connection()
        records = [
            (
                item.source_term,
                item.target_term,
                item.entity_type.value if hasattr(item.entity_type, "value") else str(item.entity_type),
                1 if item.case_sensitive else 0
            )
            for item in items
        ]
        meta_records = [
            (
                item.source_term,
                1 if getattr(item, "reviewed", False) else 0,
                getattr(item, "grammatical_gender", None)
            )
            for item in items
        ]
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO glossary (source_term, target_term, entity_type, case_sensitive) "
                "VALUES (?, ?, ?, ?)",
                records
            )
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO glossary_metadata (source_term, reviewed, grammatical_gender) "
                    "VALUES (?, ?, ?)",
                    meta_records
                )
            except Exception:
                pass

    def delete_glossary_item(self, source_term: str) -> None:
        """Deletes a glossary item by its source term."""
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM glossary WHERE source_term = ?", (source_term,))
            try:
                conn.execute("DELETE FROM glossary_metadata WHERE source_term = ?", (source_term,))
            except Exception:
                pass

    def search_translation_memory(self, source_text: str) -> Optional[str]:
        """Searches translation memory for matching source text segment."""
        return None

    # ==========================================
    # Chunk Checkpointing Methods
    # ==========================================

    def save_chunk_state(self, chunk: TranslationChunk) -> None:
        """Saves a single chunk state immediately inside an atomic transaction."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO chunk_checkpoints "
                "(id, book_id, chapter_id, status, token_count, draft_translation, final_translation, error_message, serialized_data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(chunk.id),
                    str(chunk.book_id),
                    str(chunk.chapter_id),
                    chunk.status.value if hasattr(chunk.status, "value") else str(chunk.status),
                    chunk.token_count,
                    chunk.draft_translation,
                    chunk.final_translation,
                    chunk.error_message,
                    chunk.model_dump_json()
                )
            )

    def save_chunk_state_batch(self, chunks: List[TranslationChunk]) -> None:
        """Saves multiple chunk states using executemany in a single atomic transaction."""
        if not chunks:
            return
        conn = self._get_connection()
        records = [
            (
                str(c.id),
                str(c.book_id),
                str(c.chapter_id),
                c.status.value if hasattr(c.status, "value") else str(c.status),
                c.token_count,
                c.draft_translation,
                c.final_translation,
                c.error_message,
                c.model_dump_json()
            )
            for c in chunks
        ]
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunk_checkpoints "
                "(id, book_id, chapter_id, status, token_count, draft_translation, final_translation, error_message, serialized_data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                records
            )

    def save_chunk_states_batch(self, chunks: List[TranslationChunk]) -> None:
        """Alias for save_chunk_state_batch."""
        self.save_chunk_state_batch(chunks)

    def save_chunks_batch(self, chunks: List[TranslationChunk]) -> None:
        """Alias for save_chunk_state_batch."""
        self.save_chunk_state_batch(chunks)

    def load_chunks_by_status(self, book_id: UUID, status: ChunkStatus) -> Iterator[TranslationChunk]:
        """Loads all chunks matching the given book_id and status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        status_val = status.value if hasattr(status, "value") else str(status)
        cursor.execute(
            "SELECT serialized_data FROM chunk_checkpoints WHERE book_id = ? AND status = ?",
            (str(book_id), status_val)
        )
        for row in cursor.fetchall():
            yield TranslationChunk.model_validate_json(row[0])

    def count_chunks_for_book(self, book_id: UUID) -> int:
        """Counts total checkpoints recorded for a book."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunk_checkpoints WHERE book_id = ?", (str(book_id),))
        row = cursor.fetchone()
        return row[0] if row else 0

    def delete_chunks_for_book(self, book_id: UUID) -> None:
        """Deletes all chunks for a specific book."""
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM chunk_checkpoints WHERE book_id = ?", (str(book_id),))

    def count_chunks_by_status(self, book_id: UUID, status: ChunkStatus) -> int:
        """Counts number of chunks for a book in a specific status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        status_val = status.value if hasattr(status, "value") else str(status)
        cursor.execute(
            "SELECT COUNT(*) FROM chunk_checkpoints WHERE book_id = ? AND status = ?",
            (str(book_id), status_val)
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    # ==========================================
    # Scoped Entity Profile Methods (Milestone 2)
    # ==========================================

    def save_entity_profile(self, profile: EntityProfile) -> None:
        """Saves or updates a single EntityProfile."""
        self.save_entity_profiles_batch([profile])

    def add_entity_profile(self, profile: EntityProfile) -> None:
        """Alias for save_entity_profile."""
        self.save_entity_profile(profile)

    def save_entity_profiles(self, profiles: List[EntityProfile]) -> None:
        """Alias for save_entity_profiles_batch."""
        self.save_entity_profiles_batch(profiles)

    def add_entity_profiles(self, profiles: List[EntityProfile]) -> None:
        """Alias for save_entity_profiles_batch."""
        self.save_entity_profiles_batch(profiles)

    def save_entity_profiles_batch(self, profiles: List[EntityProfile]) -> None:
        """Batch inserts or updates EntityProfiles inside an atomic transaction."""
        if not profiles:
            return
        conn = self._get_connection()
        records = [
            (
                str(p.id),
                p.entity_type.value if hasattr(p.entity_type, "value") else str(p.entity_type),
                p.scope.value if hasattr(p.scope, "value") else str(p.scope),
                p.scope_id,
                p.source_name,
                p.canonical_target,
                json.dumps(p.aliases) if p.aliases else None,
                json.dumps(p.allowed_target_forms) if p.allowed_target_forms else None,
                json.dumps(p.forbidden_target_forms) if p.forbidden_target_forms else None,
                p.grammatical_gender,
                p.translation_policy.value if hasattr(p.translation_policy, "value") else str(p.translation_policy),
                p.confidence,
                1 if p.locked else 0,
                p.description,
                1 if p.case_sensitive else 0,
                p.frequency,
                p.created_at,
                p.updated_at,
            )
            for p in profiles
        ]
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO entity_profiles (
                    id, entity_type, scope, scope_id, source_name, canonical_target,
                    aliases, allowed_target_forms, forbidden_target_forms,
                    grammatical_gender, translation_policy, confidence, locked,
                    description, case_sensitive, frequency, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )

    def get_entity_profile(self, profile_id: UUID) -> Optional[EntityProfile]:
        """Retrieves a single EntityProfile by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, entity_type, scope, scope_id, source_name, canonical_target,
                   aliases, allowed_target_forms, forbidden_target_forms,
                   grammatical_gender, translation_policy, confidence, locked,
                   description, case_sensitive, frequency, created_at, updated_at
            FROM entity_profiles WHERE id = ?
            """,
            (str(profile_id),)
        )
        row = cursor.fetchone()
        return _row_to_entity_profile(row) if row else None

    def delete_entity_profile(self, profile_id: UUID) -> None:
        """Deletes an EntityProfile by ID."""
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM entity_profiles WHERE id = ?", (str(profile_id),))

    def get_entity_profiles_scoped(
        self,
        book_id: Optional[str] = None,
        series_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[EntityProfile]:
        """
        Retrieves entity profiles resolved with hierarchical scope precedence:
        BOOK > SERIES > DOMAIN > GLOBAL.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        bid_str = str(book_id) if book_id is not None else None
        sid_str = str(series_id) if series_id is not None else None
        dom_str = str(domain) if domain is not None else None

        cursor.execute(
            """
            SELECT id, entity_type, scope, scope_id, source_name, canonical_target,
                   aliases, allowed_target_forms, forbidden_target_forms,
                   grammatical_gender, translation_policy, confidence, locked,
                   description, case_sensitive, frequency, created_at, updated_at
            FROM entity_profiles
            WHERE (? IS NOT NULL AND scope = 'BOOK' AND scope_id = ?)
               OR (? IS NOT NULL AND scope = 'SERIES' AND scope_id = ?)
               OR (? IS NOT NULL AND scope = 'DOMAIN' AND scope_id = ?)
               OR (scope = 'GLOBAL')
            """,
            (bid_str, bid_str, sid_str, sid_str, dom_str, dom_str)
        )
        rows = cursor.fetchall()
        profiles = [_row_to_entity_profile(r) for r in rows]

        # Apply strict hierarchical conflict resolution: BOOK > SERIES > DOMAIN > GLOBAL
        best_by_name: Dict[str, EntityProfile] = {}
        for p in profiles:
            key = p.source_name.strip().lower()
            if key not in best_by_name:
                best_by_name[key] = p
            else:
                curr = best_by_name[key]
                curr_rank = SCOPE_PRIORITY.get(curr.scope, 0)
                p_rank = SCOPE_PRIORITY.get(p.scope, 0)
                if p_rank > curr_rank:
                    best_by_name[key] = p
                elif p_rank == curr_rank and p.confidence > curr.confidence:
                    best_by_name[key] = p

        return list(best_by_name.values())

    def get_scoped_entities(
        self,
        book_id: Optional[str] = None,
        series_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[EntityProfile]:
        """Alias for get_entity_profiles_scoped."""
        return self.get_entity_profiles_scoped(book_id=book_id, series_id=series_id, domain=domain)

    def get_scoped_glossary(
        self,
        book_id: Optional[str] = None,
        series_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[GlossaryItem]:
        """
        Merges legacy glossary items with resolved scoped EntityProfile items.
        Scoped EntityProfiles take precedence over legacy glossary entries.
        """
        legacy_items = {g.source_term.lower(): g for g in self.get_glossary()}
        scoped_entities = self.get_entity_profiles_scoped(
            book_id=book_id, series_id=series_id, domain=domain
        )
        for ep in scoped_entities:
            legacy_items[ep.source_name.lower()] = ep.to_glossary_item()
        return list(legacy_items.values())

    # ==========================================
    # Segment-Level Mention Indexing (Milestone 2)
    # ==========================================

    def save_entity_mention(self, mention: EntityMention) -> None:
        """Saves a single EntityMention."""
        self.save_entity_mentions_batch([mention])

    def add_entity_mention(self, mention: EntityMention) -> None:
        """Alias for save_entity_mention."""
        self.save_entity_mention(mention)

    def save_entity_mentions(self, mentions: List[EntityMention]) -> None:
        """Alias for save_entity_mentions_batch."""
        self.save_entity_mentions_batch(mentions)

    def add_entity_mentions(self, mentions: List[EntityMention]) -> None:
        """Alias for save_entity_mentions_batch."""
        self.save_entity_mentions_batch(mentions)

    def save_entity_mentions_batch(self, mentions: List[EntityMention]) -> None:
        """Batch saves EntityMentions inside an atomic transaction."""
        if not mentions:
            return
        conn = self._get_connection()
        records = [
            (
                str(m.id),
                str(m.entity_id),
                str(m.segment_id),
                str(m.sentence_id) if m.sentence_id else None,
                m.char_start,
                m.char_end,
                m.surface_form,
                m.confidence,
                json.dumps(m.metadata) if m.metadata else None,
            )
            for m in mentions
        ]
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO entity_mentions (
                    id, entity_id, segment_id, sentence_id,
                    char_start, char_end, surface_form, confidence, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )

    def get_entity_mentions_for_segment(self, segment_id: UUID) -> List[EntityMention]:
        """Retrieves all mentions indexed for a specific translation segment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, entity_id, segment_id, sentence_id,
                   char_start, char_end, surface_form, confidence, metadata
            FROM entity_mentions
            WHERE segment_id = ?
            ORDER BY char_start ASC
            """,
            (str(segment_id),)
        )
        return [_row_to_entity_mention(r) for r in cursor.fetchall()]

    def get_mentions_for_segment(self, segment_id: UUID) -> List[EntityMention]:
        """Alias for get_entity_mentions_for_segment."""
        return self.get_entity_mentions_for_segment(segment_id)

    def get_mentions_by_segment(self, segment_id: UUID) -> List[EntityMention]:
        """Alias for get_entity_mentions_for_segment."""
        return self.get_entity_mentions_for_segment(segment_id)

    def get_entities_for_segment(
        self,
        segment_id: UUID,
        book_id: Optional[str] = None,
        series_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[EntityProfile]:
        """
        Retrieves only the active EntityProfiles present in the specified segment.
        Prevents prompt token bloat while applying scoped resolution.
        """
        mentions = self.get_entity_mentions_for_segment(segment_id)
        if not mentions:
            return []

        entity_ids = list({str(m.entity_id) for m in mentions})
        placeholders = ",".join("?" for _ in entity_ids)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT id, entity_type, scope, scope_id, source_name, canonical_target,
                   aliases, allowed_target_forms, forbidden_target_forms,
                   grammatical_gender, translation_policy, confidence, locked,
                   description, case_sensitive, frequency, created_at, updated_at
            FROM entity_profiles
            WHERE id IN ({placeholders})
            """,
            entity_ids
        )
        profiles = [_row_to_entity_profile(r) for r in cursor.fetchall()]

        # Filter and prioritize scoped profiles
        best_by_name: Dict[str, EntityProfile] = {}
        for p in profiles:
            key = p.source_name.strip().lower()
            if key not in best_by_name:
                best_by_name[key] = p
            else:
                curr = best_by_name[key]
                if SCOPE_PRIORITY.get(p.scope, 0) > SCOPE_PRIORITY.get(curr.scope, 0):
                    best_by_name[key] = p

        return list(best_by_name.values())

    # ==========================================
    # Translation Segments Persistence (Milestone 3 & 4)
    # ==========================================

    def save_segment_state(self, segment: TranslationSegment) -> None:
        """Persists a single TranslationSegment state."""
        self.save_segments_batch([segment])

    def save_segments_batch(self, segments: List[TranslationSegment]) -> None:
        """Batch saves TranslationSegments inside an atomic transaction."""
        if not segments:
            return
        conn = self._get_connection()
        records = [
            (
                str(s.id),
                str(s.book_id),
                str(s.chapter_id),
                str(s.paragraph_id),
                getattr(s, "order_index", 0),
                s.source_text,
                getattr(s, "normalized_text", None),
                getattr(s, "draft_translation", None),
                getattr(s, "refined_translation", None),
                getattr(s, "final_translation", None),
                s.status.value if hasattr(s.status, "value") else str(s.status),
                getattr(s, "repair_attempts", 0),
                getattr(s, "error_message", None),
                s.model_dump_json() if hasattr(s, "model_dump_json") else None,
            )
            for s in segments
        ]
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO translation_segments (
                    id, book_id, chapter_id, paragraph_id, order_index,
                    source_text, normalized_text, draft_translation,
                    refined_translation, final_translation, status,
                    repair_attempts, error_message, serialized_data, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                records
            )

    def load_segments_by_status(
        self, book_id: Union[UUID, str], status: Union[SegmentStatus, str]
    ) -> List[TranslationSegment]:
        """Loads segments for a book filtered by status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        status_val = status.value if hasattr(status, "value") else str(status)
        cursor.execute(
            """
            SELECT serialized_data, id, book_id, chapter_id, paragraph_id,
                   order_index, source_text, normalized_text, draft_translation,
                   refined_translation, final_translation, status, repair_attempts, error_message
            FROM translation_segments
            WHERE book_id = ? AND status = ?
            ORDER BY order_index ASC
            """,
            (str(book_id), status_val)
        )
        results = []
        for r in cursor.fetchall():
            if r[0]:
                try:
                    results.append(TranslationSegment.model_validate_json(r[0]))
                    continue
                except Exception:
                    pass
            results.append(TranslationSegment(
                id=UUID(r[1]),
                book_id=UUID(r[2]),
                chapter_id=UUID(r[3]),
                paragraph_id=UUID(r[4]),
                order_index=r[5],
                source_text=r[6],
                normalized_text=r[7],
                draft_translation=r[8],
                refined_translation=r[9],
                final_translation=r[10],
                status=SegmentStatus(r[11]) if r[11] in SegmentStatus.__members__ else SegmentStatus.PENDING,
                repair_attempts=r[12] or 0,
                error_message=r[13],
            ))
        return results

    # ==========================================
    # Quality Assurance & Audit Persistence (Milestone 4)
    # ==========================================

    def save_qa_report(self, report: Any, book_id: Union[UUID, str]) -> None:
        """Persists a single QAReport record or its constituent violations."""
        self.save_qa_reports_batch([report], book_id=str(book_id))

    def save_qa_reports_batch(self, reports: List[Any], book_id: Union[UUID, str]) -> None:
        """
        Batch saves QA reports and individual violation diagnostics inside an atomic WAL transaction.
        Guarantees that every validated segment has an auditable record in SQLite.
        """
        if not reports:
            return

        book_id_str = str(book_id)
        records = []

        for report in reports:
            seg_id = str(report.segment_id)
            repair_attempts = getattr(report, "repair_attempts", 0)
            status = getattr(report, "status", "ACCEPTED" if getattr(report, "is_valid", True) else "REVIEW_REQUIRED")
            violations = getattr(report, "violations", [])

            if violations:
                for idx, v in enumerate(violations):
                    rec_id = f"{getattr(report, 'id', seg_id)}_{idx}_{repair_attempts}"
                    severity_val = v.severity.value if hasattr(v.severity, "value") else str(v.severity)
                    details = {
                        "source_snippet": getattr(v, "source_snippet", None),
                        "target_snippet": getattr(v, "target_snippet", None),
                        "forbidden_form": getattr(v, "forbidden_form", None),
                        "suggested_fix": getattr(v, "suggested_fix", None),
                        "score": getattr(report, "score", 1.0),
                    }
                    records.append((
                        rec_id,
                        book_id_str,
                        seg_id,
                        v.validator_name,
                        severity_val,
                        v.message,
                        json.dumps(details, ensure_ascii=False),
                        repair_attempts,
                        status
                    ))
            else:
                # Clean pass: persist informational audit milestone
                rec_id = f"{getattr(report, 'id', seg_id)}_clean_{repair_attempts}"
                details = {"score": getattr(report, "score", 1.0)}
                records.append((
                    rec_id,
                    book_id_str,
                    seg_id,
                    "QualityPipeline",
                    "INFO",
                    "Passed all quality validations",
                    json.dumps(details, ensure_ascii=False),
                    repair_attempts,
                    status
                ))

        conn = self._get_connection()
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO quality_reports (
                    id, book_id, segment_id, validator_name, severity,
                    message, details_json, repair_attempts, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )

    def get_qa_reports_for_book(self, book_id: Union[UUID, str]) -> List[Dict[str, Any]]:
        """Retrieves all QA audit records associated with a book, ordered chronologically."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, book_id, segment_id, validator_name, severity,
                   message, details_json, repair_attempts, status, created_at
            FROM quality_reports
            WHERE book_id = ?
            ORDER BY created_at ASC, segment_id ASC
            """,
            (str(book_id),)
        )
        results = []
        for r in cursor.fetchall():
            results.append({
                "id": r[0],
                "book_id": r[1],
                "segment_id": r[2],
                "validator_name": r[3],
                "severity": r[4],
                "message": r[5],
                "details": json.loads(r[6]) if r[6] else {},
                "repair_attempts": r[7],
                "status": r[8],
                "created_at": r[9],
            })
        return results

    def get_qa_reports_for_segment(self, segment_id: Union[UUID, str]) -> List[Dict[str, Any]]:
        """Retrieves all QA audit records associated with a specific translation segment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, book_id, segment_id, validator_name, severity,
                   message, details_json, repair_attempts, status, created_at
            FROM quality_reports
            WHERE segment_id = ?
            ORDER BY created_at ASC
            """,
            (str(segment_id),)
        )
        results = []
        for r in cursor.fetchall():
            results.append({
                "id": r[0],
                "book_id": r[1],
                "segment_id": r[2],
                "validator_name": r[3],
                "severity": r[4],
                "message": r[5],
                "details": json.loads(r[6]) if r[6] else {},
                "repair_attempts": r[7],
                "status": r[8],
                "created_at": r[9],
            })
        return results

    # ==========================================
    # Connection Cleanup
    # ==========================================

    def close(self) -> None:
        """Closes all active database connections across all threads."""
        with self._lock:
            for conn in list(self._all_connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_connections.clear()
        if hasattr(self._local, "conn"):
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
