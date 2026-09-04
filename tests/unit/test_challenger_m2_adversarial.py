"""
Adversarial Stress Test Suite for Milestone 2:
Scoped Knowledge Base, Morphology, Unicode Boundaries & Whole-Book Analysis.

Challenger 2 Empirical Verification Suite.
"""

import time
import pytest
from uuid import uuid4
from typing import List, Dict, Any

from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.knowledge import (
    EntityProfile,
    EntityType,
    ScopeLevel,
    TranslationPolicy,
)
from src.analysis.book_analyzer import BookAnalyzer, BookAnalyzerConfig


# =========================================================================
# SUITE 1: Ukrainian Oblique Nominal Declensions on is_variant_forbidden()
# =========================================================================

class TestUkrainianObliqueDeclensions:
    """
    Stress-tests nominal declension coverage across Ukrainian cases
    for character name homographs with forbidden target forms.
    """

    @pytest.fixture
    def homograph_profiles(self) -> Dict[str, EntityProfile]:
        analyzer = BookAnalyzer()
        profiles = {}
        for name, data in analyzer.homograph_registry.items():
            profiles[name] = EntityProfile(
                source_name=name,
                canonical_target=data["canonical"],
                forbidden_target_forms=data["forbidden"],
                grammatical_gender=data.get("gender"),
                confidence=0.95,
            )
        return profiles

    def test_dispatch_mandated_oblique_cases_cherry(self, homograph_profiles):
        """
        Tests the 6 Cherry-related forms mandated by Challenger dispatch:
        Вишнею, Вишні, Вишень, Вишенькою, Вішнею, Черешнею.
        """
        cherry = homograph_profiles["cherry"]

        # Instrumental case of Вишня
        assert cherry.is_variant_forbidden("Вишнею") is True, "Вишнею must be forbidden"
        # Genitive/Dative/Locative case of Вишня
        assert cherry.is_variant_forbidden("Вишні") is True, "Вишні must be forbidden"
        # Instrumental case of diminutive Вишенька
        assert cherry.is_variant_forbidden("Вишенькою") is True, "Вишенькою must be forbidden"
        # Instrumental case of archaic/dialectal Вішня
        assert cherry.is_variant_forbidden("Вішнею") is True, "Вішнею must be forbidden"
        # Instrumental case of Черешня
        assert cherry.is_variant_forbidden("Черешнею") is True, "Черешнею must be forbidden"

        # Genitive plural of Вишня with fleeting vowel 'е' (Вишень)
        # EMPIRICAL CHALLENGE: In Ukrainian 1st declension, -шн stems insert fleeting 'е':
        # вишня -> вишень. _generate_ukrainian_inflections drops 'я' to get 'Вишн' and adds 'ь' -> 'Вишнь'.
        # Thus 'Вишень' is NOT generated and fails matching!
        is_vishen_forbidden = cherry.is_variant_forbidden("Вишень")
        assert is_vishen_forbidden is True, (
            "BUG FOUND: 'Вишень' (genitive plural of 'Вишня') is NOT detected as forbidden! "
            "Inflection generator generated 'Вишнь' instead of inserting fleeting vowel 'е'."
        )

    def test_dispatch_mandated_oblique_cases_other_homographs(self, homograph_profiles):
        """
        Tests the 4 other homograph forms mandated by Challenger dispatch:
        Надією, Вірою, Трояндою, Кременем.
        """
        hope = homograph_profiles["hope"]
        faith = homograph_profiles["faith"]
        rose = homograph_profiles["rose"]
        flint = homograph_profiles["flint"]

        # Instrumental case of Надія
        assert hope.is_variant_forbidden("Надією") is True, "Надією must be forbidden for Hope"
        # Instrumental case of Віра
        assert faith.is_variant_forbidden("Вірою") is True, "Вірою must be forbidden for Faith"
        # Instrumental case of Троянда
        assert rose.is_variant_forbidden("Трояндою") is True, "Трояндою must be forbidden for Rose"
        # Instrumental case of Кремінь / Кремень
        assert flint.is_variant_forbidden("Кременем") is True, "Кременем must be forbidden for Flint"

    def test_additional_genitive_plural_gaps(self, homograph_profiles):
        """
        Tests additional genitive plural and inflected forms:
        - Черешень (genitive plural of Черешня)
        - Вишеньок (genitive plural of Вишенька)
        - Надій (genitive plural of Надія)
        """
        cherry = homograph_profiles["cherry"]
        hope = homograph_profiles["hope"]

        # Черешень (stem черешн + fleeting е -> черешень)
        assert cherry.is_variant_forbidden("Черешень") is True, (
            "BUG FOUND: 'Черешень' (genitive plural of Черешня) is NOT detected as forbidden."
        )
        # Вишеньок (stem вишеньк + fleeting о -> вишеньок)
        assert cherry.is_variant_forbidden("Вишеньок") is True, (
            "BUG FOUND: 'Вишеньок' (genitive plural of Вишенька) is NOT detected as forbidden."
        )
        # Надій (stem наді + й -> надій)
        assert hope.is_variant_forbidden("Надій") is True, (
            "BUG FOUND: 'Надій' (genitive plural of Надія) is NOT detected as forbidden."
        )

    def test_oblique_forms_inside_sentences(self, homograph_profiles):
        """Tests detection of oblique forms embedded in Ukrainian running prose."""
        cherry = homograph_profiles["cherry"]
        hope = homograph_profiles["hope"]
        rose = homograph_profiles["rose"]
        flint = homograph_profiles["flint"]

        assert cherry.is_variant_forbidden("Він пішов у сад разом із Вишнею.") is True
        assert cherry.is_variant_forbidden("Вони підійшли ближче до Вишні.") is True
        assert hope.is_variant_forbidden("Він жив однією лише Надією.") is True
        assert rose.is_variant_forbidden("Він захоплювався прекрасною Трояндою.") is True
        assert flint.is_variant_forbidden("Він розпалив вогнище старим Кременем.") is True

        # When Вишень is in running text:
        assert cherry.is_variant_forbidden("У кошику не залишилося стиглих Вишень.") is True, (
            "BUG FOUND: 'Вишень' in running prose is not forbidden."
        )


# =========================================================================
# SUITE 2: Unicode Boundaries & False Positive Resistance
# =========================================================================

class TestUnicodeBoundariesAndFalsePositives:
    """
    Verifies that unrelated Ukrainian words containing forbidden stems
    as prefixes or substrings are NOT erroneously flagged (Zero False Positives).
    """

    @pytest.fixture
    def homograph_profiles(self) -> Dict[str, EntityProfile]:
        analyzer = BookAnalyzer()
        profiles = {}
        for name, data in analyzer.homograph_registry.items():
            profiles[name] = EntityProfile(
                source_name=name,
                canonical_target=data["canonical"],
                forbidden_target_forms=data["forbidden"],
                confidence=0.95,
            )
        return profiles

    def test_unrelated_derived_words_not_flagged(self, homograph_profiles):
        """
        Derived words and adjectives sharing roots MUST NOT be forbidden:
        - вишняк (cherry orchard)
        - вишневий (cherry adjective)
        - вишнівка (cherry liqueur)
        - черешневий (sweet cherry adjective)
        - надійний (reliable)
        - надійність (reliability)
        - безнадійний (hopeless)
        - вірогідний (plausible)
        - недовіра (distrust)
        - трояндовий (rose adjective)
        - кременевий (flint adjective)
        - кремінний (flint adjective)
        """
        cherry = homograph_profiles["cherry"]
        hope = homograph_profiles["hope"]
        faith = homograph_profiles["faith"]
        rose = homograph_profiles["rose"]
        flint = homograph_profiles["flint"]

        assert cherry.is_variant_forbidden("вишняк") is False
        assert cherry.is_variant_forbidden("вишневий") is False
        assert cherry.is_variant_forbidden("вишнівка") is False
        assert cherry.is_variant_forbidden("черешневий") is False

        assert hope.is_variant_forbidden("надійний") is False
        assert hope.is_variant_forbidden("надійність") is False
        assert hope.is_variant_forbidden("безнадійний") is False

        assert faith.is_variant_forbidden("вірогідний") is False
        assert faith.is_variant_forbidden("недовіра") is False

        assert rose.is_variant_forbidden("трояндовий") is False
        assert flint.is_variant_forbidden("кременевий") is False
        assert flint.is_variant_forbidden("кремінний") is False

    def test_unrelated_sentences_not_flagged(self, homograph_profiles):
        """Tests that full sentences with legitimate derived words do not trigger false positives."""
        cherry = homograph_profiles["cherry"]
        hope = homograph_profiles["hope"]
        faith = homograph_profiles["faith"]
        rose = homograph_profiles["rose"]
        flint = homograph_profiles["flint"]

        assert cherry.is_variant_forbidden("Біля хати ріс густий вишняк.") is False
        assert cherry.is_variant_forbidden("Вона пила солодкий вишневий сік.") is False
        assert hope.is_variant_forbidden("Це був надійний та перевірений метод.") is False
        assert faith.is_variant_forbidden("Цей висновок є цілком вірогідним.") is False
        assert rose.is_variant_forbidden("У дворі цвів трояндовий кущ.") is False
        assert flint.is_variant_forbidden("Археологи знайшли давнє кременеве рубило.") is False

    def test_hyphenated_and_compound_boundaries(self, homograph_profiles):
        """Tests behavior on hyphenated compound adjectives and words."""
        cherry = homograph_profiles["cherry"]

        # Adjectives containing 'вишневий' should not trigger
        assert cherry.is_variant_forbidden("темно-вишневий") is False
        assert cherry.is_variant_forbidden("вишнево-червоний") is False

        # But a compound containing the literal forbidden noun should trigger
        assert cherry.is_variant_forbidden("вишня-черешня") is True

    def test_punctuation_and_quote_boundaries(self, homograph_profiles):
        """Tests that punctuation marks around forbidden words do not inhibit detection."""
        cherry = homograph_profiles["cherry"]

        assert cherry.is_variant_forbidden("«Вишня»") is True
        assert cherry.is_variant_forbidden("„Вишня“") is True
        assert cherry.is_variant_forbidden('"Вишня"') is True
        assert cherry.is_variant_forbidden("(Вишня!)") is True
        assert cherry.is_variant_forbidden("Вишня...") is True
        assert cherry.is_variant_forbidden("— Вишня, — сказав він.") is True

    def test_homonym_verb_collision_imperative_vir(self, homograph_profiles):
        """
        EMPIRICAL CHALLENGE:
        For Faith, forbidden form is 'Віра'.
        _generate_ukrainian_inflections('Віра') with clean[:-1] + '' generates 'Вір'.
        In Ukrainian, 'Вір' is ALSO the imperative form of the verb 'вірити' ('Believe!').
        Checks if imperative verb 'Вір мені!' gets erroneously flagged as forbidden.
        """
        faith = homograph_profiles["faith"]
        # 'Вір мені!' means 'Believe me!'
        is_imperative_flagged = faith.is_variant_forbidden("Вір мені!")
        # Document whether this homonym collides
        assert is_imperative_flagged is False, (
            "VULNERABILITY DETECTED: 'Вір мені!' (imperative verb 'Believe!') is falsely flagged "
            "as forbidden character variant for Faith due to inflection 'Вір'."
        )


# =========================================================================
# SUITE 3: BookAnalyzer Stress Testing, Ambiguous Names & Sentence Starters
# =========================================================================

class TestBookAnalyzerStressAndSentenceStarters:
    """
    Stress-tests BookAnalyzer candidate extraction on sentence starters,
    ambiguous homograph names, high volume, and mention index accuracy.
    """

    def test_sentence_starter_homograph_overboost_bug(self):
        """
        EMPIRICAL CHALLENGE:
        When a homograph word (e.g. 'Rose' or 'Cherry') appears ONLY at the start
        of sentences as an ordinary common noun (e.g. 'Rose petals covered the path.'),
        does BookAnalyzer mistakenly auto-lock it as a CHARACTER with confidence >= 0.90?

        In book_analyzer.py line 425:
        `confidence = max(confidence, 0.92)`
        unconditionally overrides the sentence start penalty (-0.40) and absence of character verbs!
        """
        book = Book(
            title="Botany Guide",
            chapters=[
                Chapter(
                    title="Chapter 1: Flora",
                    order_index=0,
                    paragraphs=[
                        Paragraph(
                            sentences=[
                                Sentence(original_text="Rose petals covered the damp stone path.", position=0),
                                Sentence(original_text="Cherry orchards stretched across the southern hills.", position=1),
                            ]
                        )
                    ],
                )
            ],
        )

        analyzer = BookAnalyzer()
        result = analyzer.analyze_book(book)

        profiles_by_name = {p.source_name: p for p in result.entity_profiles}

        # Check 'Rose' profile
        if "Rose" in profiles_by_name:
            rose_profile = profiles_by_name["Rose"]
            # A common noun at sentence start with no character context should NOT be auto-locked as a Character!
            assert not (rose_profile.entity_type == EntityType.CHARACTER and rose_profile.locked), (
                f"BUG FOUND: Common noun 'Rose' at sentence start was auto-locked as CHARACTER "
                f"(confidence={rose_profile.confidence}, locked={rose_profile.locked})! "
                f"Line 425 of book_analyzer.py forces confidence=0.92 unconditionally."
            )

        # Check 'Cherry' profile
        if "Cherry" in profiles_by_name:
            cherry_profile = profiles_by_name["Cherry"]
            assert not (cherry_profile.entity_type == EntityType.CHARACTER and cherry_profile.locked), (
                f"BUG FOUND: Common noun 'Cherry' at sentence start was auto-locked as CHARACTER "
                f"(confidence={cherry_profile.confidence}, locked={cherry_profile.locked})! "
                f"Line 425 of book_analyzer.py forces confidence=0.92 unconditionally."
            )

    def test_sentence_starter_capitalization_penalty_on_general_nouns(self):
        """
        Tests that common capitalized sentence starters (e.g. 'Summer') that appear
        repeatedly ONLY at the start of sentences are correctly penalized.
        """
        book = Book(
            title="Seasons Chronicle",
            chapters=[
                Chapter(
                    title="Chapter 1",
                    order_index=0,
                    paragraphs=[
                        Paragraph(sentences=[Sentence(original_text="Summer arrived early that year.", position=0)]),
                        Paragraph(sentences=[Sentence(original_text="Summer brought unbearable drought.", position=0)]),
                        Paragraph(sentences=[Sentence(original_text="Summer ended with sudden rain.", position=0)]),
                    ],
                )
            ],
        )

        analyzer = BookAnalyzer()
        result = analyzer.analyze_book(book)
        profiles_by_name = {p.source_name: p for p in result.entity_profiles}

        if "Summer" in profiles_by_name:
            summer_profile = profiles_by_name["Summer"]
            # Due to r_mid == 0.0, s_mid should apply sentence_start_penalty (-0.40)
            assert summer_profile.locked is False, "Repeated sentence starter 'Summer' must NOT be locked."
            assert summer_profile.confidence < 0.50, (
                f"Expected low confidence for sentence starter, got {summer_profile.confidence}"
            )

    def test_term_entity_mention_case_insensitivity_leak(self):
        """
        EMPIRICAL CHALLENGE:
        In _index_mentions_for_segment:
        `if p.entity_type == EntityType.CHARACTER and not matched[0].isupper(): continue`
        Only skips lowercase matches for CHARACTER entities.
        For TERM entities, re.IGNORECASE matches lowercase common words anywhere in text!
        """
        book = Book(
            title="Seasons Chronicle",
            chapters=[
                Chapter(
                    title="Chapter 1",
                    order_index=0,
                    paragraphs=[
                        Paragraph(sentences=[Sentence(original_text="Summer was very hot.", position=0)]),
                        Paragraph(sentences=[Sentence(original_text="Summer lasted three months.", position=0)]),
                        Paragraph(sentences=[Sentence(original_text="The swimming pool was open in summer.", position=0)]),
                    ],
                )
            ],
        )

        analyzer = BookAnalyzer()
        result = analyzer.analyze_book(book)

        # Look at mentions for the third paragraph
        seg3 = result.segments[2]
        seg3_mentions = [m for m in result.entity_mentions if m.segment_id == seg3.id]

        lowercase_mentions = [m for m in seg3_mentions if m.surface_form == "summer"]
        assert len(lowercase_mentions) == 0, (
            "VULNERABILITY DETECTED: Common noun 'summer' was indexed as entity mention "
            "for segment 'The swimming pool was open in summer.'"
        )

    def test_high_volume_book_stress_scalability(self):
        """
        Stress-tests BookAnalyzer throughput and memory on a large synthetic corpus:
        500 paragraphs, 2,500 sentences, multiple entity types.
        """
        chapters = []
        for c in range(5):
            paragraphs = []
            for p in range(100):
                sentences = [
                    Sentence(original_text=f"Lord Bryan whispered secretly to Princess Elena in section {p}.", position=0),
                    Sentence(original_text="Cherry and Rose walked towards the healing potion apothecary.", position=1),
                    Sentence(original_text="The knight drew a broadsword while wearing heavy chainmail armor.", position=2),
                    Sentence(original_text=f"[Status: Level {p}] [Class: Shadowblade] [Skill Acquired: 'Shadow Step']", position=3),
                    Sentence(original_text="Dr. Watson smiled warmly at Flint, who held a mana potion.", position=4),
                ]
                paragraphs.append(Paragraph(sentences=sentences))
            chapters.append(Chapter(title=f"Chapter {c+1}", order_index=c, paragraphs=paragraphs))

        book = Book(title="Stress Test Volume", chapters=chapters)

        t0 = time.time()
        result = BookAnalyzer().analyze_book(book)
        elapsed = time.time() - t0

        # Assert throughput: 500 paragraphs / 2500 sentences should complete in under 5.0 seconds
        assert elapsed < 5.0, f"Analysis took {elapsed:.2f}s, exceeding 5.0s threshold!"
        assert len(result.segments) == 500
        assert len(result.entity_profiles) > 0
        assert len(result.entity_mentions) > 0

    def test_mention_span_exactness_across_book(self):
        """
        Verifies that every single EntityMention.char_start and char_end exactly
        slices the surface_form from segment.source_text without off-by-one errors.
        """
        chapters = [
            Chapter(
                title="Chapter 1",
                order_index=0,
                paragraphs=[
                    Paragraph(
                        sentences=[
                            Sentence(original_text="Lord Bryan greeted Princess Elena at the castle gate.", position=0),
                            Sentence(original_text="Princess Elena held a healing potion tightly.", position=1),
                            Sentence(original_text="Dr. Watson and Flint spoke with Grandmaster Thorne.", position=2),
                        ]
                    )
                ],
            )
        ]
        book = Book(title="Span Verification", chapters=chapters)
        result = BookAnalyzer().analyze_book(book)

        seg_map = {s.id: s for s in result.segments}
        for mention in result.entity_mentions:
            seg = seg_map[mention.segment_id]
            assert mention.verify_surface_form(seg.source_text) is True, (
                f"Span mismatch in segment {seg.id}: "
                f"surface_form='{mention.surface_form}' != slice='{seg.source_text[mention.char_start:mention.char_end]}'"
            )
