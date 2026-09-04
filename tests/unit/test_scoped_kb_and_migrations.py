"""
Unit tests for Milestone 2: Scoped Knowledge Base, Domain Models, Migrations, and SQLite Repository.
"""

import sqlite3
import pytest
from pathlib import Path
from uuid import UUID, uuid4

from src.domain.models.knowledge import (
    ScopeLevel,
    TranslationPolicy,
    EntityType,
    EntityProfile,
    EntityMention,
    GlossaryItem,
    SCOPE_PRECEDENCE,
)
from src.knowledge_base.migrations.migration_runner import MigrationRunner, MigrationError
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository


class TestScopeLevelAndPrecedence:
    """Tests for ScopeLevel enum and strict precedence BOOK > SERIES > DOMAIN > GLOBAL."""

    def test_scope_level_precedence_operators(self):
        assert ScopeLevel.BOOK > ScopeLevel.SERIES
        assert ScopeLevel.SERIES > ScopeLevel.DOMAIN
        assert ScopeLevel.DOMAIN > ScopeLevel.GLOBAL

        assert ScopeLevel.BOOK > ScopeLevel.GLOBAL
        assert ScopeLevel.GLOBAL < ScopeLevel.BOOK
        assert ScopeLevel.SERIES >= ScopeLevel.DOMAIN
        assert ScopeLevel.DOMAIN <= ScopeLevel.SERIES

    def test_scope_level_sorting(self):
        unsorted = [ScopeLevel.GLOBAL, ScopeLevel.BOOK, ScopeLevel.DOMAIN, ScopeLevel.SERIES]
        sorted_desc = sorted(unsorted, reverse=True)
        assert sorted_desc == [
            ScopeLevel.BOOK,
            ScopeLevel.SERIES,
            ScopeLevel.DOMAIN,
            ScopeLevel.GLOBAL,
        ]

    def test_scope_level_string_comparison(self):
        assert ScopeLevel.BOOK > "series"
        assert ScopeLevel.BOOK > "SERIES"
        assert ScopeLevel.GLOBAL < "book"
        assert ScopeLevel.BOOK == "BOOK"
        assert ScopeLevel("book") == ScopeLevel.BOOK


class TestEntityProfileModel:
    """Tests for EntityProfile auto-locking, validation, and forbidden variant detection."""

    def test_entity_profile_autolock_at_high_confidence(self):
        # Confidence >= 0.90 triggers auto-lock
        profile = EntityProfile(
            source_name="Cherry",
            canonical_target="Черрі",
            confidence=0.95,
            locked=False,
        )
        assert profile.locked is True

        profile_exact = EntityProfile(
            source_name="Hope",
            canonical_target="Гоуп",
            confidence=0.90,
            locked=False,
        )
        assert profile_exact.locked is True

    def test_entity_profile_preserves_manual_lock_low_confidence(self):
        # Confidence < 0.90 defaults to locked=False
        profile = EntityProfile(
            source_name="Wanderer",
            canonical_target="Мандрівник",
            confidence=0.75,
            locked=False,
        )
        assert profile.locked is False

        # Human-curated lock is preserved even with lower confidence
        curated = EntityProfile(
            source_name="Mysterious Man",
            canonical_target="Таємничий чоловік",
            confidence=0.60,
            locked=True,
        )
        assert curated.locked is True

    def test_entity_profile_confidence_bounds(self):
        with pytest.raises(ValueError):
            EntityProfile(source_name="A", canonical_target="B", confidence=-0.1)

        with pytest.raises(ValueError):
            EntityProfile(source_name="A", canonical_target="B", confidence=1.05)

    def test_entity_profile_canonical_target_in_allowed_forms(self):
        profile = EntityProfile(
            source_name="Bryan",
            canonical_target="Браян",
            allowed_target_forms=["Браяна", "Браянові"],
        )
        assert "Браян" in profile.allowed_target_forms

    def test_entity_profile_legacy_aliases(self):
        profile = EntityProfile(source_name="Bryan", canonical_target="Браян")
        assert profile.name == "Bryan"
        assert profile.canonical_translation == "Браян"

    def test_forbidden_variant_rejection_cherry(self):
        profile = EntityProfile(
            source_name="Cherry",
            canonical_target="Черрі",
            forbidden_target_forms=["Вишня", "Вішня", "Черри", "Черешня", "Вишенька"],
        )

        # 1. Exact tokens
        assert profile.is_variant_forbidden("Вишня") is True
        assert profile.is_variant_forbidden("вишня") is True
        assert profile.is_variant_forbidden("Черри") is True
        assert profile.is_variant_forbidden("Черешня") is True

        # 2. Ukrainian declension inflections (-я, -а)
        assert profile.is_variant_forbidden("Вишнею") is True
        assert profile.is_variant_forbidden("Вишні") is True
        assert profile.is_variant_forbidden("Черешнею") is True
        assert profile.is_variant_forbidden("Вишенькою") is True

        # 3. Sentences with word boundary matching
        assert profile.is_variant_forbidden("Вишня підняла свій посох.") is True
        assert profile.is_variant_forbidden("Старійшина розмовляв із Вишнею про пророцтво.") is True

        # 4. Canonical forms accepted
        assert profile.is_variant_forbidden("Черрі підняла свій посох.") is False
        assert profile.is_variant_forbidden("Старійшина розмовляв із Черрі.") is False

        # 5. Diagnostic mentions
        detected = profile.find_forbidden_mentions("Старійшина розмовляв із Вишнею.")
        assert "Вишнею" in detected

        # 6. Allowed forms check
        assert profile.is_variant_allowed("Черрі") is True
        assert profile.is_variant_allowed("Вишня") is False


class TestEntityMentionModel:
    """Tests for EntityMention offset verification and validation."""

    def test_entity_mention_valid(self):
        eid = uuid4()
        sid = uuid4()
        mention = EntityMention(
            entity_id=eid,
            segment_id=sid,
            char_start=7,
            char_end=13,
            surface_form="Cherry",
        )
        assert mention.length == 6
        assert mention.span == (7, 13)

        source_text = "Hello, Cherry smiled."
        assert mention.verify_surface_form(source_text) is True

    def test_entity_mention_invalid_offsets(self):
        eid = uuid4()
        sid = uuid4()

        # Negative char_start
        with pytest.raises(ValueError):
            EntityMention(
                entity_id=eid, segment_id=sid,
                char_start=-1, char_end=5, surface_form="Hello"
            )

        # Inverted span (end <= start)
        with pytest.raises(ValueError):
            EntityMention(
                entity_id=eid, segment_id=sid,
                char_start=5, char_end=2, surface_form="Hel"
            )

        # Span length mismatch
        with pytest.raises(ValueError):
            EntityMention(
                entity_id=eid, segment_id=sid,
                char_start=0, char_end=5, surface_form="Hi"
            )


class TestGlossaryItemRoundTrip:
    """Tests backward compatibility of GlossaryItem and EntityProfile adapters."""

    def test_glossary_item_to_profile_and_back(self):
        item = GlossaryItem(
            source_term="Lord Bryan",
            target_term="Лорд Браян",
            entity_type=EntityType.CHARACTER,
            case_sensitive=True,
            reviewed=True,
            grammatical_gender="чоловічий",
        )
        profile = item.to_entity_profile(scope=ScopeLevel.BOOK, scope_id="book_123")
        assert profile.source_name == "Lord Bryan"
        assert profile.canonical_target == "Лорд Браян"
        assert profile.scope == ScopeLevel.BOOK
        assert profile.scope_id == "book_123"
        assert profile.locked is True
        assert profile.grammatical_gender == "чоловічий"

        back_item = profile.to_glossary_item()
        assert back_item.source_term == "Lord Bryan"
        assert back_item.target_term == "Лорд Браян"
        assert back_item.reviewed is True
        assert back_item.grammatical_gender == "чоловічий"


class TestDatabaseMigrations:
    """Tests for MigrationRunner, schema_version tracking, and idempotent bootstrapping."""

    def test_migration_runner_applies_all_on_clean_db(self, tmp_path):
        db_path = tmp_path / "fresh_test.db"
        runner = MigrationRunner(db_path)
        applied = runner.apply_all()

        assert applied == [1, 2, 3]
        assert runner.get_current_version() == 3

        # Verify tables created
        with sqlite3.connect(db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            expected = {
                "schema_version",
                "entities",
                "glossary",
                "glossary_metadata",
                "chunk_checkpoints",
                "entity_profiles",
                "entity_mentions",
                "translation_segments",
                "quality_reports",
            }
            assert expected.issubset(tables)

    def test_migration_runner_idempotent_reapplication(self, tmp_path):
        db_path = tmp_path / "idempotent.db"
        runner = MigrationRunner(db_path)
        first_run = runner.apply_all()
        assert first_run == [1, 2, 3]

        second_run = runner.apply_all()
        assert second_run == []
        assert runner.get_current_version() == 3

    def test_init_db_runs_migration_runner(self, tmp_path):
        db_path = tmp_path / "init_db_test.db"
        init_db(db_path)

        runner = MigrationRunner(db_path)
        assert runner.get_current_version() == 3


class TestSQLiteRepositoryScopedKB:
    """Tests for SQLiteKnowledgeBaseRepository scoped entity and mention queries."""

    @pytest.fixture
    def repo(self, tmp_path):
        db_path = tmp_path / "test_repo.db"
        init_db(db_path)
        r = SQLiteKnowledgeBaseRepository(db_path)
        yield r
        r.close()

    def test_entity_profile_crud(self, repo):
        profile = EntityProfile(
            source_name="Cherry",
            canonical_target="Черрі",
            forbidden_target_forms=["Вишня", "Вішня"],
            confidence=0.95,
        )
        repo.save_entity_profile(profile)

        loaded = repo.get_entity_profile(profile.id)
        assert loaded is not None
        assert loaded.source_name == "Cherry"
        assert loaded.canonical_target == "Черрі"
        assert loaded.locked is True
        assert "Вишня" in loaded.forbidden_target_forms

        repo.delete_entity_profile(profile.id)
        assert repo.get_entity_profile(profile.id) is None

    def test_scoped_hierarchy_resolution(self, repo):
        # Insert same term across multiple scopes:
        # Global: "Staff" -> "посох"
        # Series: "Staff" -> "бойовий посох"
        # Book: "Staff" -> "дерев'яний ціпок"
        global_p = EntityProfile(
            scope=ScopeLevel.GLOBAL,
            source_name="Staff",
            canonical_target="посох",
            confidence=0.80,
        )
        series_p = EntityProfile(
            scope=ScopeLevel.SERIES,
            scope_id="fantasy_series",
            source_name="Staff",
            canonical_target="бойовий посох",
            confidence=0.85,
        )
        book_p = EntityProfile(
            scope=ScopeLevel.BOOK,
            scope_id="book_001",
            source_name="Staff",
            canonical_target="дерев'яний ціпок",
            confidence=0.90,
        )
        repo.save_entity_profiles_batch([global_p, series_p, book_p])

        # 1. Query with book_001: BOOK scope must win
        resolved_book = repo.get_entity_profiles_scoped(
            book_id="book_001", series_id="fantasy_series", domain="fantasy"
        )
        assert len(resolved_book) == 1
        assert resolved_book[0].canonical_target == "дерев'яний ціпок"
        assert resolved_book[0].scope == ScopeLevel.BOOK

        # 2. Query with another book in same series: SERIES scope must win
        resolved_series = repo.get_entity_profiles_scoped(
            book_id="book_999", series_id="fantasy_series", domain="fantasy"
        )
        assert len(resolved_series) == 1
        assert resolved_series[0].canonical_target == "бойовий посох"
        assert resolved_series[0].scope == ScopeLevel.SERIES

        # 3. Query with unrelated book and series: GLOBAL scope must win
        resolved_global = repo.get_entity_profiles_scoped(
            book_id="other_book", series_id="other_series"
        )
        assert len(resolved_global) == 1
        assert resolved_global[0].canonical_target == "посох"
        assert resolved_global[0].scope == ScopeLevel.GLOBAL

    def test_segment_mention_indexing_and_retrieval(self, repo):
        p1 = EntityProfile(source_name="Cherry", canonical_target="Черрі", confidence=0.95)
        p2 = EntityProfile(source_name="Bryan", canonical_target="Браян", confidence=0.95)
        repo.save_entity_profiles_batch([p1, p2])

        seg1_id = uuid4()
        seg2_id = uuid4()

        m1 = EntityMention(
            entity_id=p1.id, segment_id=seg1_id,
            char_start=0, char_end=6, surface_form="Cherry"
        )
        m2 = EntityMention(
            entity_id=p2.id, segment_id=seg2_id,
            char_start=10, char_end=15, surface_form="Bryan"
        )
        repo.save_entity_mentions_batch([m1, m2])

        # Segment 1 mentions
        seg1_mentions = repo.get_entity_mentions_for_segment(seg1_id)
        assert len(seg1_mentions) == 1
        assert seg1_mentions[0].surface_form == "Cherry"

        # Localized entity retrieval for segment 1 (should return ONLY Cherry, avoiding prompt bloat)
        seg1_profiles = repo.get_entities_for_segment(seg1_id)
        assert len(seg1_profiles) == 1
        assert seg1_profiles[0].source_name == "Cherry"

    def test_get_scoped_glossary_merges_legacy_and_scoped(self, repo):
        # Pre-seed legacy glossary
        repo.add_glossary_item(GlossaryItem(source_term="sword", target_term="меч"))
        repo.add_glossary_item(GlossaryItem(source_term="Cherry", target_term="вишня"))

        # Scoped profile for Cherry should override legacy "вишня"
        scoped_cherry = EntityProfile(
            scope=ScopeLevel.BOOK,
            scope_id="book_1",
            source_name="Cherry",
            canonical_target="Черрі",
            confidence=0.95,
        )
        repo.save_entity_profile(scoped_cherry)

        merged = repo.get_scoped_glossary(book_id="book_1")
        merged_map = {item.source_term.lower(): item.target_term for item in merged}

        assert merged_map["sword"] == "меч"
        assert merged_map["cherry"] == "Черрі"
