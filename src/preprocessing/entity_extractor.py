"""
Heuristic Entity Extractor for BookTranslator.

Extracts recurring proper names and named entities from English prose
using frequency analysis, capitalization patterns, and English stop-word filtering.
"""
from typing import List, Set, Optional, Dict
from collections import Counter
import re

from src.domain.models.document import Book, Sentence

DEFAULT_STOP_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "because",
    "when", "while", "where", "after", "before", "as", "in", "on", "at",
    "to", "for", "with", "by", "from", "about", "into", "through", "during",
    "above", "below", "between", "under", "there", "here", "this", "that",
    "these", "those", "it", "its", "he", "his", "him", "she", "her", "hers",
    "they", "them", "their", "theirs", "we", "us", "our", "ours", "you",
    "your", "yours", "i", "me", "my", "mine", "what", "which", "who",
    "whom", "whose", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "than", "too", "very", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must", "just", "now", "one", "two",
    "three", "first", "second", "yes", "chapter", "book", "part", "section",
    "title", "author", "page", "paragraph", "sentence", "uk", "en", "none",
    "however", "although", "though", "meanwhile", "suddenly", "later", "finally",
    "already", "still", "again", "never", "always", "sometimes", "often", "soon",
    "thus", "therefore", "since", "until", "upon", "without", "within", "against",
    "among", "along", "across", "behind", "beyond", "towards", "toward", "instead",
    "perhaps", "maybe", "indeed", "almost", "nearly", "either", "neither", "also",
    "even", "much", "many", "well", "oh", "ah", "hey", "hello", "hi", "ok", "okay",
    "somewhere", "anyway", "anyhow", "anywhere", "everywhere", "nowhere", "besides",
    "furthermore", "moreover", "otherwise", "likewise", "nevertheless", "nonetheless",
    "yesterday", "tomorrow", "today", "tonight",
    "mr", "mrs", "ms", "miss", "dr", "prof", "st", "sir", "madam", "lady", "lord"
}


class HeuristicEntityExtractor:
    """
    Extracts recurring proper names / named entities from English text
    using statistical heuristics and stop-word filtering.
    """
    def __init__(
        self,
        min_occurrences: int = 3,
        stop_words: Optional[Set[str]] = None
    ):
        self.min_occurrences = min_occurrences
        self.stop_words = {w.lower() for w in (stop_words or DEFAULT_STOP_WORDS)}

    def extract_from_text(self, text: str) -> List[str]:
        """
        Extracts entities from raw string text that appear at least `min_occurrences` times.
        """
        if not text:
            return []

        counter = Counter()

        # 1. Multi-word proper names (e.g., "Sherlock Holmes", "Lord Voldemort")
        multi_words = re.findall(r'\b[A-Z][a-zA-Z0-9\'-]+(?:\s+[A-Z][a-zA-Z0-9\'-]+)+\b', text)
        for phrase in multi_words:
            phrase_clean = re.sub(r"(?:'s|’s|['’\-])$", "", phrase).strip()
            parts = phrase_clean.split()
            if len(parts) >= 2 and any(p.lower() not in self.stop_words for p in parts):
                counter[phrase_clean] += 1

        # 2. Single-word proper names (e.g., "Eldoria", "Gandalf")
        words = re.findall(r'\b[A-Z][a-zA-Z0-9\'-]*\b', text)
        for w in words:
            w_clean = re.sub(r"(?:'s|’s|['’\-])$", "", w).strip()
            if len(w_clean) > 1 and w_clean.lower() not in self.stop_words:
                counter[w_clean] += 1

        extracted = [
            entity for entity, count in counter.items()
            if count >= self.min_occurrences
        ]
        return sorted(extracted)

    def extract_entities(self, book: Book) -> List[str]:
        """
        Extracts recurring entities across all chapters and paragraphs in a Book DOM.
        """
        sentences_text = []
        for chapter in book.chapters:
            for paragraph in chapter.paragraphs:
                for sentence in paragraph.sentences:
                    if sentence.original_text:
                        sentences_text.append(sentence.original_text)

        full_text = " ".join(sentences_text)
        return self.extract_from_text(full_text)
