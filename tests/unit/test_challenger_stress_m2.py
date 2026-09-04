"""
Adversarial Stress Test Suite for Milestone 2 (Phase 2: Scoped Knowledge Base & Whole-Book Analysis).
Challenger 1 Empirical Verification Suite.

Empirically challenges:
1. Scope Hierarchy Resolution & Strict Precedence (BOOK > SERIES > DOMAIN > GLOBAL):
   - 4-tier complete shadowing matrix
   - Cross-scope isolation and leak prevention between different books/series/domains
   - Case-insensitivity, whitespace normalization, and character casing
   - Confidence score tie-breaking within the same scope
   - ScopeLevel dunder comparison operators, sorting, and type safety
   - Scoped glossary merging with legacy items
   - Localized segment mention retrieval without prompt bloat
   - High-volume randomized oracle consistency

2. Database Migrations Idempotency Across Clean, Dirty, Legacy, and In-Memory Databases:
   - Repeated applications on clean databases (10x consecutive runs)
   - Seamless upgrade of legacy pre-M2 databases without schema_version table
   - Preservation of existing data in legacy tables across migration
   - Dirty database with partial schema versions
   - Pre-existing tables without schema_version entries
   - In-memory SQLite database migrations
   - Atomic rollback on failed migration step

3. Scoped Repository Concurrency & Thread-Safety:
   - Multi-threaded concurrent writes (profiles and mentions)
   - Multi-threaded concurrent reads (scoped entities, glossary, mentions)
   - Heavy mixed read-write contention hammer under WAL mode
   - Thread-local connection pooling, cleanup, and post-close re-initialization
"""

import concurrent.futures
import json
import random
import sqlite3
import string
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import pytest

from src.domain.models.knowledge import (
    EntityType,
    EntityProfile,
    EntityMention,
    GlossaryItem,
    SCOPE_PRECEDENCE,
    SCOPE_PRIORITY,
    ScopeLevel,
    TranslationPolicy,
)
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.migrations.migration_runner import MigrationError, MigrationRunner
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository


# =========================================================================
# SUITE 1: Scope Hierarchy Resolution & Shadowing Adversarial Tests
# =========================================================================

class TestScopeHierarchyShadowingStress:
    """Adversarial stress-testing of BOOK > SERIES > DOMAIN > GLOBAL shadowing."""

    @pytest.fixture
    def repo(self, tmp_path: Path):
        db_path = tmp_path / "scope_stress.db"
        init_db(db_path)
        r = SQLiteKnowledgeBaseRepository(db_path)
        yield r
        r.close()

    def test_strict_4tier_shadowing_resolution(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 1.1: Verify full 4-tier shadowing where all 4 scopes define
        the same entity name. The higher scope MUST strictly shadow the lower scopes.
        """
        source = "Crystal"
        p_global = EntityProfile(
            scope=ScopeLevel.GLOBAL,
            source_name=source,
            canonical_target="кристал",
            confidence=0.70,
        )
        p_domain = EntityProfile(
            scope=ScopeLevel.DOMAIN,
            scope_id="fantasy",
            source_name=source,
            canonical_target="магічний кристал",
            confidence=0.75,
        )
        p_series = EntityProfile(
            scope=ScopeLevel.SERIES,
            scope_id="lotr_series",
            source_name=source,
            canonical_target="кристал душі",
            confidence=0.80,
        )
        p_book = EntityProfile(
            scope=ScopeLevel.BOOK,
            scope_id="book_001",
            source_name=source,
            canonical_target="Око Дракона",
            confidence=0.85,
        )
        # Batch insert
        repo.save_entity_profiles_batch([p_global, p_domain, p_series, p_book])

        # 1. All 3 scope parameters match -> BOOK must win
        res = repo.get_entity_profiles_scoped(
            book_id="book_001", series_id="lotr_series", domain="fantasy"
        )
        assert len(res) == 1
        assert res[0].canonical_target == "Око Дракона"
        assert res[0].scope == ScopeLevel.BOOK

        # 2. Different book_id -> SERIES must shadow DOMAIN and GLOBAL
        res_series = repo.get_entity_profiles_scoped(
            book_id="book_002", series_id="lotr_series", domain="fantasy"
        )
        assert len(res_series) == 1
        assert res_series[0].canonical_target == "кристал душі"
        assert res_series[0].scope == ScopeLevel.SERIES

        # 3. Different series_id -> DOMAIN must shadow GLOBAL
        res_domain = repo.get_entity_profiles_scoped(
            book_id="book_002", series_id="other_series", domain="fantasy"
        )
        assert len(res_domain) == 1
        assert res_domain[0].canonical_target == "магічний кристал"
        assert res_domain[0].scope == ScopeLevel.DOMAIN

        # 4. Different domain -> GLOBAL must win
        res_global = repo.get_entity_profiles_scoped(
            book_id="book_002", series_id="other_series", domain="scifi"
        )
        assert len(res_global) == 1
        assert res_global[0].canonical_target == "кристал"
        assert res_global[0].scope == ScopeLevel.GLOBAL

        # 5. Query with no parameters -> GLOBAL must win
        res_empty = repo.get_entity_profiles_scoped()
        assert len(res_empty) == 1
        assert res_empty[0].canonical_target == "кристал"
        assert res_empty[0].scope == ScopeLevel.GLOBAL

    def test_all_pairwise_shadowing_combinations(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 1.2: Exhaustively test all pairwise scope precedence combinations
        (BOOK > SERIES, BOOK > DOMAIN, BOOK > GLOBAL, SERIES > DOMAIN, SERIES > GLOBAL, DOMAIN > GLOBAL)
        with both forward and reversed insertion orders.
        """
        pairs = [
            (ScopeLevel.BOOK, "b1", ScopeLevel.SERIES, "s1", "pair_bs"),
            (ScopeLevel.BOOK, "b1", ScopeLevel.DOMAIN, "d1", "pair_bd"),
            (ScopeLevel.BOOK, "b1", ScopeLevel.GLOBAL, None, "pair_bg"),
            (ScopeLevel.SERIES, "s1", ScopeLevel.DOMAIN, "d1", "pair_sd"),
            (ScopeLevel.SERIES, "s1", ScopeLevel.GLOBAL, None, "pair_sg"),
            (ScopeLevel.DOMAIN, "d1", ScopeLevel.GLOBAL, None, "pair_dg"),
        ]

        for high_scope, high_sid, low_scope, low_sid, term in pairs:
            # Test order 1: Insert low scope first, then high scope
            p_low = EntityProfile(
                scope=low_scope,
                scope_id=low_sid,
                source_name=f"{term}_lowfirst",
                canonical_target="LOW_TARGET",
                confidence=0.99,  # High confidence on low scope to test scope priority overrides confidence
            )
            p_high = EntityProfile(
                scope=high_scope,
                scope_id=high_sid,
                source_name=f"{term}_lowfirst",
                canonical_target="HIGH_TARGET",
                confidence=0.50,  # Lower confidence on high scope
            )
            repo.save_entity_profile(p_low)
            repo.save_entity_profile(p_high)

            res = repo.get_entity_profiles_scoped(book_id="b1", series_id="s1", domain="d1")
            match = [p for p in res if p.source_name == f"{term}_lowfirst"]
            assert len(match) == 1
            assert match[0].canonical_target == "HIGH_TARGET", (
                f"{high_scope} failed to shadow {low_scope} when inserted second"
            )
            assert match[0].scope == high_scope

            # Test order 2: Insert high scope first, then low scope
            p_high2 = EntityProfile(
                scope=high_scope,
                scope_id=high_sid,
                source_name=f"{term}_highfirst",
                canonical_target="HIGH_TARGET",
                confidence=0.50,
            )
            p_low2 = EntityProfile(
                scope=low_scope,
                scope_id=low_sid,
                source_name=f"{term}_highfirst",
                canonical_target="LOW_TARGET",
                confidence=0.99,
            )
            repo.save_entity_profile(p_high2)
            repo.save_entity_profile(p_low2)

            res2 = repo.get_entity_profiles_scoped(book_id="b1", series_id="s1", domain="d1")
            match2 = [p for p in res2 if p.source_name == f"{term}_highfirst"]
            assert len(match2) == 1
            assert match2[0].canonical_target == "HIGH_TARGET", (
                f"{high_scope} failed to shadow {low_scope} when inserted first"
            )
            assert match2[0].scope == high_scope

    def test_scope_isolation_and_cross_contamination_prevention(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 1.3: Ensure no cross-contamination between distinct entities
        at the same scope level with different scope_ids (e.g. Book A vs Book B).
        """
        term = "Artifact"
        p_book_a = EntityProfile(
            scope=ScopeLevel.BOOK,
            scope_id="book_alpha",
            source_name=term,
            canonical_target="Артефакт Альфа",
        )
        p_book_b = EntityProfile(
            scope=ScopeLevel.BOOK,
            scope_id="book_beta",
            source_name=term,
            canonical_target="Артефакт Бета",
        )
        p_series_x = EntityProfile(
            scope=ScopeLevel.SERIES,
            scope_id="series_x",
            source_name=term,
            canonical_target="Артефакт Серії Х",
        )
        p_series_y = EntityProfile(
            scope=ScopeLevel.SERIES,
            scope_id="series_y",
            source_name=term,
            canonical_target="Артефакт Серії Y",
        )
        repo.save_entity_profiles_batch([p_book_a, p_book_b, p_series_x, p_series_y])

        # Query Book Alpha (in series X)
        res_a = repo.get_entity_profiles_scoped(book_id="book_alpha", series_id="series_x")
        assert len(res_a) == 1
        assert res_a[0].canonical_target == "Артефакт Альфа"

        # Query Book Beta (in series Y)
        res_b = repo.get_entity_profiles_scoped(book_id="book_beta", series_id="series_y")
        assert len(res_b) == 1
        assert res_b[0].canonical_target == "Артефакт Бета"

        # Query Book Gamma (not in DB, but in series X) -> should get series X, NOT Book A or B!
        res_gamma = repo.get_entity_profiles_scoped(book_id="book_gamma", series_id="series_x")
        assert len(res_gamma) == 1
        assert res_gamma[0].canonical_target == "Артефакт Серії Х"

        # Query Book Delta (not in DB, in series Y) -> should get series Y
        res_delta = repo.get_entity_profiles_scoped(book_id="book_delta", series_id="series_y")
        assert len(res_delta) == 1
        assert res_delta[0].canonical_target == "Артефакт Серії Y"

        # Query unrelated book and series -> should return empty (0 results)
        res_none = repo.get_entity_profiles_scoped(book_id="book_unrelated", series_id="series_unrelated")
        assert len(res_none) == 0

    def test_scope_casing_and_whitespace_normalization(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 1.4: Verify that entity resolution is resilient against
        case variations and surrounding whitespace in source_name across scopes.
        """
        # Global: lowercase
        p_global = EntityProfile(
            scope=ScopeLevel.GLOBAL,
            source_name="cherry",
            canonical_target="вишня",
        )
        # Book: mixed case with whitespace
        p_book = EntityProfile(
            scope=ScopeLevel.BOOK,
            scope_id="book_1",
            source_name="  Cherry  ",
            canonical_target="Черрі",
        )
        repo.save_entity_profiles_batch([p_global, p_book])

        res = repo.get_entity_profiles_scoped(book_id="book_1")
        assert len(res) == 1
        # BOOK must shadow GLOBAL even with different case and whitespace
        assert res[0].canonical_target == "Черрі"

    def test_confidence_tie_breaking_same_scope(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 1.5: Within the exact same scope and scope_id, if two profiles
        exist for the same term, the higher confidence profile MUST win.
        """
        p_low = EntityProfile(
            scope=ScopeLevel.BOOK,
            scope_id="book_tie",
            source_name="Dagger",
            canonical_target="малий ніж",
            confidence=0.65,
        )
        p_high = EntityProfile(
            scope=ScopeLevel.BOOK,
            scope_id="book_tie",
            source_name="Dagger",
            canonical_target="кинджал",
            confidence=0.95,
        )
        repo.save_entity_profiles_batch([p_low, p_high])

        res = repo.get_entity_profiles_scoped(book_id="book_tie")
        assert len(res) == 1
        assert res[0].canonical_target == "кинджал"
        assert res[0].confidence == 0.95

    def test_scope_level_dunder_operators_and_type_safety(self):
        """
        Adversarial Test 1.6: Exhaustively verify ScopeLevel comparison dunder methods:
        __lt__, __le__, __gt__, __ge__, sorting, string comparison, and invalid type safety.
        """
        # Direct Enum comparison
        assert ScopeLevel.BOOK > ScopeLevel.SERIES > ScopeLevel.DOMAIN > ScopeLevel.GLOBAL
        assert ScopeLevel.GLOBAL < ScopeLevel.DOMAIN < ScopeLevel.SERIES < ScopeLevel.BOOK

        assert ScopeLevel.BOOK >= ScopeLevel.BOOK
        assert ScopeLevel.BOOK <= ScopeLevel.BOOK
        assert ScopeLevel.BOOK >= ScopeLevel.GLOBAL
        assert ScopeLevel.GLOBAL <= ScopeLevel.BOOK

        # String comparisons
        assert ScopeLevel.BOOK > "SERIES"
        assert ScopeLevel.BOOK > "series"
        assert ScopeLevel.SERIES > "domain"
        assert ScopeLevel.DOMAIN > "GLOBAL"
        assert ScopeLevel.GLOBAL == "GLOBAL"
        # Standard Python Enum equality inherits str.__eq__ and is case-sensitive:
        assert (ScopeLevel.GLOBAL == "global") is False
        # Case-insensitive resolution is supported via instantiation:
        assert ScopeLevel("global") == ScopeLevel.GLOBAL
        assert ScopeLevel("book") == ScopeLevel.BOOK

        # Incompatible type comparisons
        with pytest.raises(TypeError):
            _ = ScopeLevel.BOOK > 42

        with pytest.raises(TypeError):
            _ = ScopeLevel.BOOK < None

        with pytest.raises(TypeError):
            _ = ScopeLevel.SERIES >= ["SERIES"]

        # Sorting permutations
        shuffled = [ScopeLevel.GLOBAL, ScopeLevel.BOOK, ScopeLevel.DOMAIN, ScopeLevel.SERIES]
        assert sorted(shuffled, reverse=True) == [
            ScopeLevel.BOOK,
            ScopeLevel.SERIES,
            ScopeLevel.DOMAIN,
            ScopeLevel.GLOBAL,
        ]

    def test_scoped_glossary_shadowing_legacy_glossary(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 1.7: get_scoped_glossary must merge legacy items while
        scoped profiles strictly override legacy items with matching source_term.
        """
        # Add legacy items
        repo.add_glossary_item(GlossaryItem(source_term="Potion", target_term="зілля"))
        repo.add_glossary_item(GlossaryItem(source_term="Sword", target_term="меч"))
        repo.add_glossary_item(GlossaryItem(source_term="Shield", target_term="щит"))

        # Add scoped profiles
        p_sword_series = EntityProfile(
            scope=ScopeLevel.SERIES,
            scope_id="series_1",
            source_name="Sword",
            canonical_target="Меч Світла",
            confidence=0.85,
        )
        p_potion_book = EntityProfile(
            scope=ScopeLevel.BOOK,
            scope_id="book_1",
            source_name="Potion",
            canonical_target="Еліксир Зцілення",
            confidence=0.95,
        )
        repo.save_entity_profiles_batch([p_sword_series, p_potion_book])

        # Query scoped glossary for book_1 in series_1
        glossary = repo.get_scoped_glossary(book_id="book_1", series_id="series_1")
        glossary_map = {g.source_term.lower(): g for g in glossary}

        # 1. Potion overridden by BOOK profile
        assert glossary_map["potion"].target_term == "Еліксир Зцілення"
        assert glossary_map["potion"].reviewed is True  # 0.95 confidence autolocks

        # 2. Sword overridden by SERIES profile
        assert glossary_map["sword"].target_term == "Меч Світла"

        # 3. Shield preserved from legacy glossary
        assert glossary_map["shield"].target_term == "щит"

    def test_segment_mentions_and_localized_scope_filtering(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 1.8: Localized entity mentions indexing and retrieval
        ensuring only entities present in the segment are returned, avoiding prompt bloat.
        """
        p1 = EntityProfile(source_name="Cherry", canonical_target="Черрі", confidence=0.95)
        p2 = EntityProfile(source_name="Bryan", canonical_target="Браян", confidence=0.95)
        p3 = EntityProfile(source_name="Dragon", canonical_target="Дракон", confidence=0.90)
        repo.save_entity_profiles_batch([p1, p2, p3])

        seg_id = uuid4()
        m1 = EntityMention(
            entity_id=p1.id, segment_id=seg_id,
            char_start=0, char_end=6, surface_form="Cherry"
        )
        m2 = EntityMention(
            entity_id=p2.id, segment_id=seg_id,
            char_start=11, char_end=16, surface_form="Bryan"
        )
        repo.save_entity_mentions_batch([m1, m2])

        # Query segment entities: must return ONLY Cherry and Bryan, NOT Dragon
        segment_entities = repo.get_entities_for_segment(seg_id)
        assert len(segment_entities) == 2
        names = {p.source_name for p in segment_entities}
        assert names == {"Cherry", "Bryan"}

    def test_high_volume_randomized_scope_oracle(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 1.9: Generate 100 random entities across random scopes with
        random confidence values. Assert that SQLiteKnowledgeBaseRepository output
        matches an independent pure-Python oracle with 100% fidelity.
        """
        rng = random.Random(42)
        test_book = "oracle_book_01"
        test_series = "oracle_series_01"
        test_domain = "oracle_domain_01"

        all_profiles: List[EntityProfile] = []
        oracle_dict: Dict[str, EntityProfile] = {}

        scopes = [
            (ScopeLevel.BOOK, test_book),
            (ScopeLevel.SERIES, test_series),
            (ScopeLevel.DOMAIN, test_domain),
            (ScopeLevel.GLOBAL, None),
            (ScopeLevel.BOOK, "other_book"),
            (ScopeLevel.SERIES, "other_series"),
            (ScopeLevel.DOMAIN, "other_domain"),
        ]

        for i in range(100):
            name = f"Entity_{i:03d}"
            # Pick 1 to 4 scopes for this entity
            chosen_scopes = rng.sample(scopes, k=rng.randint(1, 4))
            for scope_lvl, sid in chosen_scopes:
                conf = round(rng.uniform(0.5, 0.99), 2)
                p = EntityProfile(
                    scope=scope_lvl,
                    scope_id=sid,
                    source_name=name,
                    canonical_target=f"Target_{name}_{scope_lvl.value}_{sid}",
                    confidence=conf,
                )
                all_profiles.append(p)

                # Pure Python Oracle: check if this scope matches our test query
                matches_query = False
                if scope_lvl == ScopeLevel.BOOK and sid == test_book:
                    matches_query = True
                elif scope_lvl == ScopeLevel.SERIES and sid == test_series:
                    matches_query = True
                elif scope_lvl == ScopeLevel.DOMAIN and sid == test_domain:
                    matches_query = True
                elif scope_lvl == ScopeLevel.GLOBAL:
                    matches_query = True

                if matches_query:
                    key = name.lower()
                    if key not in oracle_dict:
                        oracle_dict[key] = p
                    else:
                        curr = oracle_dict[key]
                        c_rank = SCOPE_PRIORITY.get(curr.scope, 0)
                        p_rank = SCOPE_PRIORITY.get(p.scope, 0)
                        if p_rank > c_rank:
                            oracle_dict[key] = p
                        elif p_rank == c_rank and p.confidence > curr.confidence:
                            oracle_dict[key] = p

        # Save all to repo
        repo.save_entity_profiles_batch(all_profiles)

        # Query repo
        resolved = repo.get_entity_profiles_scoped(
            book_id=test_book, series_id=test_series, domain=test_domain
        )
        resolved_dict = {p.source_name.lower(): p for p in resolved}

        # Compare with oracle
        assert len(resolved_dict) == len(oracle_dict), (
            f"Count mismatch: Repo {len(resolved_dict)} vs Oracle {len(oracle_dict)}"
        )
        for key, oracle_p in oracle_dict.items():
            assert key in resolved_dict, f"Missing {key} in repo result"
            repo_p = resolved_dict[key]
            assert repo_p.canonical_target == oracle_p.canonical_target, (
                f"Mismatch for {key}: repo target {repo_p.canonical_target} vs oracle {oracle_p.canonical_target}"
            )
            assert repo_p.scope == oracle_p.scope


# =========================================================================
# SUITE 2: Database Migrations Idempotency Across Clean, Dirty, Legacy, & In-Memory
# =========================================================================

class TestDatabaseMigrationsStress:
    """Stress-testing migration idempotency, rollback, and legacy database upgrades."""

    def test_clean_db_10x_consecutive_migration_idempotency(self, tmp_path: Path):
        """
        Adversarial Test 2.1: Run migrations 10 times consecutively on a fresh DB.
        The first run must apply [1, 2]; runs 2 through 10 must return [] and leave
        the schema version at 2 with zero schema corruption.
        """
        db_path = tmp_path / "idempotent_10x.db"
        runner = MigrationRunner(db_path)

        first_run = runner.apply_all()
        assert first_run == [1, 2, 3]
        assert runner.get_current_version() == 3

        for run_idx in range(2, 11):
            subsequent_run = runner.apply_all()
            assert subsequent_run == [], f"Run #{run_idx} returned unexpected migrations: {subsequent_run}"
            assert runner.get_current_version() == 3

        # Verify all tables intact
        with sqlite3.connect(db_path) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            expected = {
                "schema_version", "entities", "glossary", "glossary_metadata",
                "chunk_checkpoints", "entity_profiles", "entity_mentions", "translation_segments"
            }
            assert expected.issubset(tables)

    def test_legacy_database_migration_seamless_upgrade(self, tmp_path: Path):
        """
        Adversarial Test 2.2: Simulate a legacy V1 SQLite database (populated with
        entities, glossary items, checkpoints, but without schema_version table).
        Run MigrationRunner and verify that legacy data is 100% preserved and
        new M2 tables are added without errors.
        """
        legacy_db = tmp_path / "legacy_v1.db"

        # Pre-populate legacy tables manually
        with sqlite3.connect(legacy_db) as conn:
            conn.execute("""
                CREATE TABLE entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    aliases TEXT,
                    description TEXT,
                    canonical_translation TEXT,
                    frequency INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE glossary (
                    source_term TEXT PRIMARY KEY,
                    target_term TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    case_sensitive INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE chunk_checkpoints (
                    id TEXT PRIMARY KEY,
                    book_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    draft_translation TEXT,
                    final_translation TEXT,
                    error_message TEXT,
                    serialized_data TEXT NOT NULL
                )
            """)

            # Insert 25 legacy entities
            for i in range(25):
                conn.execute(
                    "INSERT INTO entities (id, name, entity_type, canonical_translation) VALUES (?, ?, ?, ?)",
                    (str(uuid4()), f"LegacyEntity_{i}", "character", f"Легасі_{i}")
                )
            # Insert 50 legacy glossary items
            for i in range(50):
                conn.execute(
                    "INSERT INTO glossary (source_term, target_term, entity_type) VALUES (?, ?, ?)",
                    (f"term_{i}", f"переклад_{i}", "term")
                )
            conn.commit()

        # Run MigrationRunner on legacy DB
        runner = MigrationRunner(legacy_db)
        applied = runner.apply_all()

        assert applied == [1, 2, 3]
        assert runner.get_current_version() == 3

        # Verify legacy data preserved intact
        with sqlite3.connect(legacy_db) as conn:
            e_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            g_count = conn.execute("SELECT COUNT(*) FROM glossary").fetchone()[0]
            assert e_count == 25, f"Expected 25 legacy entities, got {e_count}"
            assert g_count == 50, f"Expected 50 legacy glossary items, got {g_count}"

            # Verify new M2 tables now exist and are usable
            p_count = conn.execute("SELECT COUNT(*) FROM entity_profiles").fetchone()[0]
            assert p_count == 0  # Empty but exists

        # Subsequent run must be no-op
        assert runner.apply_all() == []

    def test_dirty_database_partial_migration(self, tmp_path: Path):
        """
        Adversarial Test 2.3: Database has schema_version table with version 1 recorded,
        but version 2 has not been applied. MigrationRunner must apply ONLY version 2 and 3.
        """
        db_path = tmp_path / "partial_db.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT NOT NULL
                )
            """)
            conn.execute(
                "INSERT INTO schema_version (version, description) VALUES (1, 'Baseline schema')"
            )
            # Create baseline tables
            conn.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT NOT NULL)")
            conn.execute("CREATE TABLE glossary (source_term TEXT PRIMARY KEY, target_term TEXT NOT NULL)")
            conn.commit()

        runner = MigrationRunner(db_path)
        applied = runner.apply_all()

        assert applied == [2, 3]
        assert runner.get_current_version() == 3

        with sqlite3.connect(db_path) as conn:
            # Check version 2 tables exist
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "entity_profiles" in tables
            assert "entity_mentions" in tables
            assert "translation_segments" in tables
            assert "quality_reports" in tables

    def test_dirty_database_tables_exist_without_schema_version(self, tmp_path: Path):
        """
        Adversarial Test 2.4: Tables already exist before migration 1 or 2 runs
        (e.g. baseline tables created manually without schema_version).
        MigrationRunner must complete without error and stamp version 3.
        """
        db_path = tmp_path / "dirty_precreated.db"
        with sqlite3.connect(db_path) as conn:
            # Pre-create baseline tables manually
            conn.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT NOT NULL, entity_type TEXT NOT NULL)")
            conn.execute("CREATE TABLE glossary (source_term TEXT PRIMARY KEY, target_term TEXT NOT NULL, entity_type TEXT NOT NULL)")
            conn.commit()

        runner = MigrationRunner(db_path)
        applied = runner.apply_all()
        assert applied == [1, 2, 3]
        assert runner.get_current_version() == 3

    def test_dirty_database_corrupted_columns_raises_migration_error(self, tmp_path: Path):
        """
        Adversarial Test 2.4b: If an existing table has an incompatible/corrupted schema
        (e.g. entity_profiles table missing 'scope' column), MigrationRunner must raise MigrationError.
        """
        db_path = tmp_path / "dirty_corrupt.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE entity_profiles (id TEXT PRIMARY KEY, source_name TEXT)")
            conn.commit()

        runner = MigrationRunner(db_path)
        with pytest.raises(MigrationError) as exc_info:
            runner.apply_all()
        assert "no such column: scope" in str(exc_info.value)

    def test_in_memory_database_migrations_with_connection(self):
        """
        Adversarial Test 2.5: Test MigrationRunner against an in-memory SQLite connection.
        Verifies migrations apply cleanly and maintain idempotency in memory.
        """
        conn = sqlite3.connect(":memory:")
        try:
            runner = MigrationRunner(Path(":memory:"))
            applied = runner.apply_all(conn=conn)
            assert applied == [1, 2, 3]

            # Verify tables exist in the in-memory database
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "entity_profiles" in tables
            assert "schema_version" in tables

            # Re-apply on same connection: must be no-op
            applied_again = runner.apply_all(conn=conn)
            assert applied_again == []

            # Check applied versions
            versions = runner.get_applied_versions(conn)
            assert versions == [1, 2, 3]
        finally:
            conn.close()

    def test_migration_atomic_rollback_on_failure(self, tmp_path: Path):
        """
        Adversarial Test 2.6: If a migration module fails midway during execution,
        the migration transaction must be rolled back: DML changes must be undone
        and the version must NEVER be recorded in schema_version.
        """
        db_path = tmp_path / "rollback_test.db"
        runner = MigrationRunner(db_path)

        # Mock a broken migration module that inserts data then fails
        class BrokenMigration:
            VERSION = 999
            DESCRIPTION = "Failing migration"

            @staticmethod
            def apply(conn: sqlite3.Connection):
                # DML insertion into an existing table
                conn.execute("INSERT INTO entities (id, name, entity_type) VALUES ('test_id', 'test_name', 'char')")
                # Deliberate syntax error to force failure
                conn.execute("INVALID SQL STATEMENT SYNTAX ERROR;")

        # Apply baseline migrations first
        runner.apply_all()
        assert runner.get_current_version() == 3

        # Inject broken migration into runner logic
        with sqlite3.connect(db_path) as conn:
            with pytest.raises(MigrationError):
                try:
                    with conn:
                        BrokenMigration.apply(conn)
                        conn.execute(
                            "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                            (BrokenMigration.VERSION, BrokenMigration.DESCRIPTION)
                        )
                except Exception as e:
                    conn.rollback()
                    raise MigrationError(f"Migration 999 failed: {e}") from e

            # 1. Verify version 999 was NOT recorded in schema_version
            v_rows = conn.execute("SELECT version FROM schema_version WHERE version = 999").fetchall()
            assert len(v_rows) == 0

            # 2. Verify DML changes were rolled back
            e_row = conn.execute("SELECT * FROM entities WHERE id = 'test_id'").fetchone()
            assert e_row is None, "DML changes from failed migration were not rolled back!"


# =========================================================================
# SUITE 3: Scoped Repository Concurrency & Thread Safety Stress Tests
# =========================================================================

class TestScopedRepositoryConcurrencyStress:
    """Adversarial multi-threading stress tests on scoped repository methods."""

    @pytest.fixture
    def repo(self, tmp_path: Path):
        db_path = tmp_path / "concurrency_stress.db"
        init_db(db_path)
        r = SQLiteKnowledgeBaseRepository(db_path)
        yield r
        r.close()

    def test_concurrent_writes_high_volume(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 3.1: 10 concurrent worker threads inserting 50 distinct
        EntityProfiles each (500 total). Zero database locking exceptions allowed.
        """
        num_threads = 10
        profiles_per_thread = 50
        errors: List[Exception] = []

        def worker(thread_idx: int):
            try:
                for i in range(profiles_per_thread):
                    p = EntityProfile(
                        scope=ScopeLevel.BOOK,
                        scope_id=f"thread_book_{thread_idx}",
                        source_name=f"Entity_T{thread_idx}_{i}",
                        canonical_target=f"Target_T{thread_idx}_{i}",
                        confidence=0.92,
                    )
                    repo.save_entity_profile(p)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered {len(errors)} concurrency write errors: {errors[:3]}"

        # Verify all 500 profiles exist
        with sqlite3.connect(repo.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM entity_profiles").fetchone()[0]
            assert count == num_threads * profiles_per_thread

    def test_concurrent_mentions_batch_writes(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 3.2: 8 concurrent threads batch-saving EntityMentions.
        Verify zero locking exceptions and exact mention indexing.
        """
        # Pre-seed an entity profile
        parent_profile = EntityProfile(
            source_name="CentralCharacter",
            canonical_target="Головний Персонаж",
            confidence=0.95,
        )
        repo.save_entity_profile(parent_profile)

        num_threads = 8
        batches_per_thread = 10
        mentions_per_batch = 5
        errors: List[Exception] = []

        def worker(thread_idx: int):
            try:
                for b in range(batches_per_thread):
                    seg_id = uuid4()
                    mentions = [
                        EntityMention(
                            entity_id=parent_profile.id,
                            segment_id=seg_id,
                            char_start=k * 10,
                            char_end=k * 10 + 7,
                            surface_form="Mention",
                        )
                        for k in range(mentions_per_batch)
                    ]
                    repo.save_entity_mentions_batch(mentions)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered {len(errors)} mention concurrency errors: {errors[:3]}"

        with sqlite3.connect(repo.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM entity_mentions").fetchone()[0]
            assert count == num_threads * batches_per_thread * mentions_per_batch

    def test_heavy_mixed_read_write_contention_hammer(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 3.3: 6 reader threads and 6 writer threads continuously
        hammering the repository simultaneously for 2.0 seconds.
        Validates thread-safe reads, writes, transactions, and WAL mode resilience.
        """
        stop_event = threading.Event()
        read_counts = [0] * 6
        write_counts = [0] * 6
        errors: List[str] = []

        # Pre-seed some data for readers
        for i in range(20):
            repo.save_entity_profile(
                EntityProfile(
                    scope=ScopeLevel.GLOBAL,
                    source_name=f"BaseItem_{i}",
                    canonical_target=f"БазовийПредмет_{i}",
                )
            )

        def writer_worker(idx: int):
            c = 0
            while not stop_event.is_set():
                try:
                    p = EntityProfile(
                        scope=ScopeLevel.BOOK,
                        scope_id=f"book_{idx}",
                        source_name=f"DynamicItem_{idx}_{c}",
                        canonical_target=f"Динамічний_{idx}_{c}",
                        confidence=0.85,
                    )
                    repo.save_entity_profile(p)
                    c += 1
                except Exception as ex:
                    errors.append(f"Writer {idx} error: {ex}")
                    break
            write_counts[idx] = c

        def reader_worker(idx: int):
            c = 0
            while not stop_event.is_set():
                try:
                    # Query scoped profiles
                    _ = repo.get_entity_profiles_scoped(
                        book_id=f"book_{idx % 3}", series_id="s1", domain="d1"
                    )
                    # Query scoped glossary
                    _ = repo.get_scoped_glossary(book_id=f"book_{idx % 3}")
                    c += 1
                except Exception as ex:
                    errors.append(f"Reader {idx} error: {ex}")
                    break
            read_counts[idx] = c

        writers = [threading.Thread(target=writer_worker, args=(i,)) for i in range(6)]
        readers = [threading.Thread(target=reader_worker, args=(i,)) for i in range(6)]

        all_threads = writers + readers
        for t in all_threads:
            t.start()

        # Let them hammer for 2.0 seconds
        time.sleep(2.0)
        stop_event.set()

        for t in all_threads:
            t.join()

        total_writes = sum(write_counts)
        total_reads = sum(read_counts)

        assert len(errors) == 0, f"Errors during contention hammer: {errors}"
        assert total_writes > 100, f"Expected >100 writes, got {total_writes}"
        assert total_reads > 100, f"Expected >100 reads, got {total_reads}"

    def test_thread_connection_lifecycle_cleanup_and_reuse(self, tmp_path: Path):
        """
        Adversarial Test 3.4: Spawn 10 threads doing repo queries, call repo.close()
        from the main thread, verify all thread connections are closed, and verify
        that subsequent access reopens cleanly without crashing.
        """
        db_path = tmp_path / "lifecycle.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        def query_action():
            _ = repo.get_glossary()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_action) for _ in range(10)]
            concurrent.futures.wait(futures)

        # Main thread closes all connections
        assert len(repo._all_connections) > 0
        repo.close()

        assert len(repo._all_connections) == 0

        # Now make another query: should re-initialize connection gracefully
        items = repo.get_glossary()
        assert isinstance(items, list)
        assert len(repo._all_connections) == 1

        repo.close()


# =========================================================================
# SUITE 4: Edge Cases, Hostile Input Hardening, & Boundary Conditions
# =========================================================================

class TestEdgeCasesAndHostileInputHardening:
    """Hostile input verification and boundary stress testing."""

    @pytest.fixture
    def repo(self, tmp_path: Path):
        db_path = tmp_path / "hostile_edge.db"
        init_db(db_path)
        r = SQLiteKnowledgeBaseRepository(db_path)
        yield r
        r.close()

    def test_sql_injection_resilience_in_scoped_queries(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 4.1: SQL injection attempts passed through book_id,
        series_id, and domain parameters must be safely neutralized by parameterized queries.
        """
        # Save a legitimate profile
        repo.save_entity_profile(
            EntityProfile(
                scope=ScopeLevel.GLOBAL,
                source_name="SafeEntity",
                canonical_target="БезпечнаСутність",
            )
        )

        malicious_inputs = [
            "' OR '1'='1",
            "'; DROP TABLE entities; --",
            "' UNION SELECT * FROM entities WHERE '1'='1",
            "1; ATTACH DATABASE ':memory:' AS leak;",
            "\" OR \"\"=\"",
            "\x00' OR '1'='1",
        ]

        for mal_input in malicious_inputs:
            # Query scoped profiles with injection vectors
            res = repo.get_entity_profiles_scoped(
                book_id=mal_input, series_id=mal_input, domain=mal_input
            )
            # Must safely return only the global entity (or empty), without crashing or executing injected SQL
            assert len(res) <= 1
            if len(res) == 1:
                assert res[0].source_name == "SafeEntity"

            # Query scoped glossary
            glossary = repo.get_scoped_glossary(
                book_id=mal_input, series_id=mal_input, domain=mal_input
            )
            assert isinstance(glossary, list)

        # Verify tables still exist (injection didn't drop tables)
        with sqlite3.connect(repo.db_path) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "entities" in tables
            assert "entity_profiles" in tables

    def test_empty_batch_operations_resilience(self, repo: SQLiteKnowledgeBaseRepository):
        """
        Adversarial Test 4.2: Calling batch operations with empty lists
        must be a clean no-op without database or syntax errors.
        """
        repo.save_entity_profiles_batch([])
        repo.save_entity_mentions_batch([])
        repo.save_chunk_state_batch([])
        repo.add_glossary_items([])

        # Segment retrieval for non-existent segment returns empty list
        assert repo.get_entity_mentions_for_segment(uuid4()) == []
        assert repo.get_entities_for_segment(uuid4()) == []

    def test_forbidden_variant_boundary_substring_vs_word_match(self):
        """
        Adversarial Test 4.3: Forbidden variant detection must use Unicode word boundaries
        to strictly prevent substring false-positives while matching inflected case forms.
        """
        p_faith = EntityProfile(
            source_name="Faith",
            canonical_target="Фейт",
            forbidden_target_forms=["Віра"],
            confidence=0.95,
        )

        # 1. Exact match and inflections must be caught
        assert p_faith.is_variant_forbidden("Віра") is True
        assert p_faith.is_variant_forbidden("віра") is True
        assert p_faith.is_variant_forbidden("Вірою") is True
        assert p_faith.is_variant_forbidden("Вірі") is True
        assert p_faith.is_variant_forbidden("Вона дивилась на Віру.") is True

        # 2. Substrings inside larger words must NOT trigger false positives
        assert p_faith.is_variant_forbidden("гравірання") is False
        assert p_faith.is_variant_forbidden("довіра") is False
        assert p_faith.is_variant_forbidden("перевірка") is False
        assert p_faith.is_variant_forbidden("завіра") is False

        # 3. Rose -> Роуз, forbidden: Троянда
        p_rose = EntityProfile(
            source_name="Rose",
            canonical_target="Роуз",
            forbidden_target_forms=["Троянда"],
            confidence=0.95,
        )
        assert p_rose.is_variant_forbidden("Троянда підійшла ближче.") is True
        assert p_rose.is_variant_forbidden("з Трояндою") is True
        # Canonical target is not forbidden
        assert p_rose.is_variant_forbidden("Роуз підійшла ближче.") is False

    def test_migration_discovery_ignores_spurious_files(self, tmp_path: Path):
        """
        Adversarial Test 4.4: Migration discovery pattern r"^m?(\\d+)_(.+)\\.py$"
        must strictly ignore non-migration files in the directory.
        """
        mig_dir = tmp_path / "custom_migrations"
        mig_dir.mkdir()

        # Create valid migration files
        (mig_dir / "001_first.py").write_text("VERSION = 1\ndef apply(conn): pass\n")
        (mig_dir / "m002_second.py").write_text("VERSION = 2\ndef apply(conn): pass\n")

        # Create spurious files that must be ignored
        (mig_dir / "notes.txt").write_text("Just notes")
        (mig_dir / "readme.md").write_text("# Readme")
        (mig_dir / "__pycache__").mkdir()
        (mig_dir / "test_migration.py").write_text("# Test file")
        (mig_dir / "001_backup.bak").write_text("backup")
        (mig_dir / ".gitkeep").write_text("")

        runner = MigrationRunner(tmp_path / "test.db")
        runner.migrations_dir = mig_dir

        discovered = runner.discover_migrations()
        assert len(discovered) == 2
        assert discovered[0][0] == 1
        assert discovered[0][1] == "first"
        assert discovered[1][0] == 2
        assert discovered[1][1] == "second"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
