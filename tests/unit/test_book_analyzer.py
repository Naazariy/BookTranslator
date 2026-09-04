"""
Unit tests for Milestone 2: BookAnalyzer, Pre-Translation Analysis, and Mention Indexing.
"""

import pytest
from uuid import uuid4

from src.analysis.book_analyzer import BookAnalyzer, BookAnalyzerConfig, BookAnalysisResult
from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.knowledge import EntityType, TranslationPolicy


def make_test_book() -> Book:
    """Constructs a sample Book DOM with diverse literary prose, RPG tags, items, and dialogue."""
    book = Book(id=uuid4(), title="Chronicles of Eldoria", source_language="en", target_language="uk")

    ch1 = Chapter(id=uuid4(), title="Chapter 1: The Gathering", order_index=0)
    p1 = Paragraph(
        id=uuid4(),
        sentences=[
            Sentence(id=uuid4(), original_text="Lord Bryan gazed across the courtyard.", order_index=0),
            Sentence(id=uuid4(), original_text="Beside him, Princess Elena held a glowing wooden staff.", order_index=1),
        ],
        order_index=0,
    )
    p2 = Paragraph(
        id=uuid4(),
        sentences=[
            Sentence(id=uuid4(), original_text="Cherry smiled warmly at the party.", order_index=0),
            Sentence(id=uuid4(), original_text="The elder spoke with Cherry about the ancient prophecy.", order_index=1),
        ],
        order_index=1,
    )
    p3 = Paragraph(
        id=uuid4(),
        sentences=[
            Sentence(id=uuid4(), original_text="Suddenly the dark sky rumbled.", order_index=0),
            Sentence(id=uuid4(), original_text="However, no rain fell upon the dry stone.", order_index=1),
            Sentence(id=uuid4(), original_text="Bryan drew his broadsword without hesitation.", order_index=2),
            Sentence(id=uuid4(), original_text="He quickly consumed a healing potion.", order_index=3),
        ],
        order_index=2,
    )
    p4 = Paragraph(
        id=uuid4(),
        sentences=[
            Sentence(id=uuid4(), original_text="A notification flashed before his eyes: [Class: Shadowblade].", order_index=0),
            Sentence(id=uuid4(), original_text="Another alert arrived: [Skill Acquired: 'Shadow Step'].", order_index=1),
        ],
        order_index=3,
    )
    ch1.paragraphs = [p1, p2, p3, p4]
    book.chapters = [ch1]
    return book


class TestBookAnalyzer:
    """Comprehensive test suite for BookAnalyzer extraction and indexing."""

    @pytest.fixture
    def analyzer(self):
        return BookAnalyzer()

    def test_analyze_book_produces_valid_result(self, analyzer):
        book = make_test_book()
        result = analyzer.analyze_book(book)

        assert isinstance(result, BookAnalysisResult)
        assert result.book_id == book.id
        assert len(result.segments) == 4
        assert len(result.entity_profiles) > 0
        assert len(result.entity_mentions) > 0
        assert len(result.glossary_items) == len(result.entity_profiles)
        assert result.statistics["total_profiles"] == len(result.entity_profiles)

    def test_title_parsing_and_gender_resolution(self, analyzer):
        book = make_test_book()
        result = analyzer.analyze_book(book)

        profiles_by_name = {p.source_name.lower(): p for p in result.entity_profiles}

        # Lord Bryan / Bryan
        bryan = profiles_by_name.get("bryan") or profiles_by_name.get("lord bryan")
        assert bryan is not None
        assert bryan.grammatical_gender == "чоловічий"
        assert bryan.entity_type == EntityType.CHARACTER
        assert bryan.locked is True

        # Princess Elena
        elena = profiles_by_name.get("elena") or profiles_by_name.get("princess elena")
        assert elena is not None
        assert elena.grammatical_gender == "жіночий"
        assert elena.entity_type == EntityType.CHARACTER
        assert elena.locked is True

    def test_homograph_character_cherry(self, analyzer):
        book = make_test_book()
        result = analyzer.analyze_book(book)

        profiles_by_name = {p.source_name.lower(): p for p in result.entity_profiles}
        cherry = profiles_by_name.get("cherry")
        assert cherry is not None
        assert cherry.canonical_target == "Черрі"
        assert cherry.locked is True
        assert cherry.grammatical_gender == "жіночий"
        assert cherry.translation_policy == TranslationPolicy.RESTRICTED_VARIANTS.value

        # Verify forbidden variants populated
        for forbidden in ["Вишня", "Вішня", "Черри", "Черешня"]:
            assert forbidden in cherry.forbidden_target_forms

    def test_sentence_initial_disambiguation(self, analyzer):
        book = make_test_book()
        result = analyzer.analyze_book(book)

        profile_names = [p.source_name.lower() for p in result.entity_profiles]

        # Sentence starters ("Suddenly", "However") must NOT be registered as entities
        assert "suddenly" not in profile_names
        assert "however" not in profile_names

    def test_rpg_tags_and_items_extraction(self, analyzer):
        book = make_test_book()
        result = analyzer.analyze_book(book)

        profiles_by_name = {p.source_name.lower(): p for p in result.entity_profiles}

        # healing potion
        potion = profiles_by_name.get("healing potion")
        assert potion is not None
        assert potion.canonical_target == "зілля зцілення"
        assert potion.entity_type == EntityType.ITEM
        assert "напій" in potion.forbidden_target_forms
        assert potion.locked is True

        # broadsword
        broadsword = profiles_by_name.get("broadsword")
        assert broadsword is not None
        assert broadsword.canonical_target == "палаш"
        assert broadsword.entity_type == EntityType.ITEM

        # Shadowblade
        shadowblade = profiles_by_name.get("shadowblade")
        assert shadowblade is not None
        assert shadowblade.entity_type == EntityType.TERM

        # Shadow Step
        step = profiles_by_name.get("shadow step")
        assert step is not None
        assert step.entity_type == EntityType.MAGIC

    def test_segment_mention_offsets_fidelity(self, analyzer):
        book = make_test_book()
        result = analyzer.analyze_book(book)

        segments_by_id = {s.id: s for s in result.segments}

        # Every indexed mention must strictly verify against the segment text
        assert len(result.entity_mentions) > 0
        for mention in result.entity_mentions:
            seg = segments_by_id[mention.segment_id]
            assert mention.verify_surface_form(seg.source_text) is True
            assert seg.source_text[mention.char_start:mention.char_end] == mention.surface_form

    def test_homograph_hope_faith_robin_flint(self, analyzer):
        """Verify additional homographs from the homograph registry."""
        book = Book(id=uuid4(), title="Homographs Test", source_language="en", target_language="uk")
        ch = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        p = Paragraph(
            id=uuid4(),
            sentences=[
                Sentence(id=uuid4(), original_text="Hope looked at Robin with a smile.", order_index=0),
                Sentence(id=uuid4(), original_text="Faith and Flint stood guard at the gate.", order_index=1),
                Sentence(id=uuid4(), original_text="Rose adjusted her spectacles calmly.", order_index=2),
            ],
            order_index=0,
        )
        ch.paragraphs = [p]
        book.chapters = [ch]

        result = analyzer.analyze_book(book)
        profiles = {p.source_name.lower(): p for p in result.entity_profiles}

        # Hope -> Гоуп (forbidden: Надія)
        assert "hope" in profiles
        assert profiles["hope"].canonical_target == "Гоуп"
        assert "Надія" in profiles["hope"].forbidden_target_forms
        assert profiles["hope"].locked is True

        # Robin -> Робін (forbidden: Вільшанка)
        assert "robin" in profiles
        assert profiles["robin"].canonical_target == "Робін"
        assert "Вільшанка" in profiles["robin"].forbidden_target_forms
        assert profiles["robin"].locked is True

        # Faith -> Фейт (forbidden: Віра)
        assert "faith" in profiles
        assert profiles["faith"].canonical_target == "Фейт"
        assert "Віра" in profiles["faith"].forbidden_target_forms
        assert profiles["faith"].locked is True

        # Flint -> Флінт (forbidden: Кремінь)
        assert "flint" in profiles
        assert profiles["flint"].canonical_target == "Флінт"
        assert "Кремінь" in profiles["flint"].forbidden_target_forms
        assert profiles["flint"].locked is True

        # Rose -> Роуз (forbidden: Троянда)
        assert "rose" in profiles
        assert profiles["rose"].canonical_target == "Роуз"
        assert "Троянда" in profiles["rose"].forbidden_target_forms
        assert profiles["rose"].locked is True
