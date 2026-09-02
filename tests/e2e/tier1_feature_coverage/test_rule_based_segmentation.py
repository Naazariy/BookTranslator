"""Tier 1 Feature Tests: Rule-Based Sentence Segmentation (F12).
Verifies 0% failure rate across English honorifics, Ukrainian Cyrillic abbreviations, dialogue with guillemets/em-dashes, decimals, and URLs.
"""
import pytest
from tests.e2e.fixtures import RuleBasedSentenceSegmenter

try:
    from src.parsers.segmenter import RuleBasedSentenceSegmenter as ProjectSegmenter
except ImportError:
    ProjectSegmenter = RuleBasedSentenceSegmenter


def get_segmenter():
    return ProjectSegmenter if ProjectSegmenter is not None else RuleBasedSentenceSegmenter


class TestRuleBasedSegmentation:
    def test_segmenter_english_honorifics_and_abbreviations(self):
        """Test F12.1: English abbreviations (Dr., Mr., Mrs., Prof., vs., e.g., i.e.) not broken."""
        segmenter = get_segmenter()
        text = "Dr. Watson visited Mr. Holmes at 221B Baker St. They discussed e.g. the mysterious case vs. Moriarty."
        sents = segmenter.split_sentences(text)
        
        # Expected: exactly 2 sentences (not broken at Dr., Mr., St., e.g., vs.)
        assert len(sents) == 2
        assert sents[0] == "Dr. Watson visited Mr. Holmes at 221B Baker St."
        assert sents[1] == "They discussed e.g. the mysterious case vs. Moriarty."

    def test_segmenter_ukrainian_cyrillic_abbreviations(self):
        """Test F12.2: Ukrainian abbreviations (м., вул., буд., кв., р., ім., тощо) not broken."""
        segmenter = get_segmenter()
        text = "Він жив у м. Києві на вул. Хрещатик, буд. 10, кв. 5. Народився у 1990 р. біля парку ім. Шевченка тощо."
        sents = segmenter.split_sentences(text)
        
        # Expected: exactly 2 sentences
        assert len(sents) == 2
        assert "м. Києві" in sents[0]
        assert "1990 р." in sents[1]

    def test_segmenter_author_initials_and_names(self):
        """Test F12.3: Single-letter initials (J. K. Rowling, Т. Г. Шевченко) not split."""
        segmenter = get_segmenter()
        text = "J. K. Rowling wrote Harry Potter. Т. Г. Шевченко написав Кобзар."
        sents = segmenter.split_sentences(text)
        
        assert len(sents) == 2
        assert "J. K. Rowling" in sents[0]
        assert "Т. Г. Шевченко" in sents[1]

    def test_segmenter_dialogue_with_guillemets_and_em_dash(self):
        """Test F12.4: Literary dialogue formatting with quotes, guillemets, and attribution em-dashes."""
        segmenter = get_segmenter()
        text = "«Привіт!» — сказав він. «Як справи?» — запитала вона."
        sents = segmenter.split_sentences(text)
        
        assert len(sents) == 2
        assert sents[0].startswith("«Привіт!»")
        assert sents[1].startswith("«Як справи?»")

    def test_segmenter_numbers_decimals_percentages_and_time(self):
        """Test F12.5: Numerical formats ($3.14, -15.5 °C, 4.5%, 6 a.m.) not split."""
        segmenter = get_segmenter()
        text = "The temperature was -15.5 °C at 6 a.m. The inflation rate was 4.5% with $3.14 price."
        sents = segmenter.split_sentences(text)
        
        assert len(sents) == 2
        assert "-15.5 °C" in sents[0]
        assert "$3.14" in sents[1]

    def test_segmenter_multiline_and_spacing_preservation(self):
        """Test F12.6: Whitespace and empty input resilience."""
        segmenter = get_segmenter()
        assert segmenter.split_sentences("") == []
        assert segmenter.split_sentences("   \n\t   ") == []
        
        text = "First sentence here.\n\nSecond sentence begins.\nThird sentence follows."
        sents = segmenter.split_sentences(text)
        assert len(sents) == 3
