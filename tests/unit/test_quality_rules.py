"""
Unit tests for QualityPipeline / QualityChecker validation rules:
- Residual control tokens (<|...|>)
- Consecutive duplicate sentences (SequenceMatcher > 0.8)
- Sentence count drop (< 0.7 * expected)
"""
from uuid import uuid4
import pytest

from src.domain.models.document import Sentence
from src.domain.models.chunk import TranslationChunk, ChunkStatus, ChunkContext
from src.domain.models.quality import IssueSeverity
from src.quality.pipeline import QualityPipeline, QualityChecker


class TestQualityCheckerRules:
    """Tests new validation rules in QualityChecker / QualityPipeline."""

    @pytest.fixture
    def checker(self):
        return QualityChecker()

    def test_clean_translation_passes_validation(self, checker):
        s1 = Sentence(id=uuid4(), original_text="The sky was blue.", order_index=0)
        s2 = Sentence(id=uuid4(), original_text="The grass was green.", order_index=1)
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_indices=[0],
            source_sentences=[s1, s2],
            target_sentence_ids=[s1.id, s2.id],
            token_count=10,
            status=ChunkStatus.PENDING
        )
        translated_text = "Небо було блакитним. Трава була зеленою."
        report = checker.validate(chunk, translated_text)

        assert report.is_passed is True
        assert len(report.issues) == 0
        assert report.translation_score == 1.0

    def test_detects_residual_control_tokens_as_critical(self, checker):
        s1 = Sentence(id=uuid4(), original_text="Hello world.", order_index=0)
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_indices=[0],
            source_sentences=[s1],
            target_sentence_ids=[s1.id],
            token_count=5,
            status=ChunkStatus.PENDING
        )
        translated_text = "Привіт, світе!<|im_end|>"
        report = checker.validate(chunk, translated_text)

        assert report.is_passed is False
        rule_names = [issue.rule_name for issue in report.issues]
        assert "ResidualControlToken" in rule_names
        ctrl_issue = next(i for i in report.issues if i.rule_name == "ResidualControlToken")
        assert ctrl_issue.severity == IssueSeverity.CRITICAL

    def test_detects_consecutive_duplicate_sentences(self, checker):
        s1 = Sentence(id=uuid4(), original_text="He walked home.", order_index=0)
        s2 = Sentence(id=uuid4(), original_text="He entered the house.", order_index=1)
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_indices=[0],
            source_sentences=[s1, s2],
            target_sentence_ids=[s1.id, s2.id],
            token_count=10,
            status=ChunkStatus.PENDING
        )
        # Consecutive near-duplicate sentences
        translated_text = "Він повільно пішов додому ввечері. Він повільно пішов додому увечері."
        report = checker.validate(chunk, translated_text)

        rule_names = [issue.rule_name for issue in report.issues]
        assert "ConsecutiveDuplicateSentences" in rule_names
        dup_issue = next(i for i in report.issues if i.rule_name == "ConsecutiveDuplicateSentences")
        assert dup_issue.severity == IssueSeverity.WARNING

    def test_detects_sentence_count_drop(self, checker):
        sents = [Sentence(id=uuid4(), original_text=f"Sentence {i}.", order_index=i) for i in range(5)]
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_indices=[0],
            source_sentences=sents,
            target_sentence_ids=[s.id for s in sents],
            token_count=25,
            status=ChunkStatus.PENDING
        )
        # Expected 5 sentences, but translation only produced 1 sentence (< 0.7 * 5 = 3.5)
        translated_text = "Лише одне речення для всього довгого абзацу."
        report = checker.validate(chunk, translated_text)

        assert report.is_passed is False
        rule_names = [issue.rule_name for issue in report.issues]
        assert "SentenceCountDrop" in rule_names
        drop_issue = next(i for i in report.issues if i.rule_name == "SentenceCountDrop")
        assert drop_issue.severity == IssueSeverity.CRITICAL

    def test_single_sentence_chunk_no_false_positive_drop(self, checker):
        s1 = Sentence(id=uuid4(), original_text="Single sentence.", order_index=0)
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_indices=[0],
            source_sentences=[s1],
            target_sentence_ids=[s1.id],
            token_count=5,
            status=ChunkStatus.PENDING
        )
        translated_text = "Одне речення."
        report = checker.validate(chunk, translated_text)
        assert report.is_passed is True
        assert not any(i.rule_name == "SentenceCountDrop" for i in report.issues)

    def test_handles_non_chunk_objects_gracefully(self, checker):
        # When chunk is None or arbitrary object
        report = checker.validate(None, "Звичайний переклад тексту.")
        assert report.is_passed is True
        assert len(report.issues) == 0

    def test_context_sentences_excluded_from_expected_count(self, checker):
        # Chunk with 2 context sentences and 1 target sentence (target_sentence_ids not set)
        ctx1 = Sentence(id=uuid4(), original_text="Context 1.", order_index=0)
        ctx2 = Sentence(id=uuid4(), original_text="Context 2.", order_index=1)
        tgt1 = Sentence(id=uuid4(), original_text="Target 1.", order_index=2)
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_indices=[0],
            source_sentences=[ctx1, ctx2, tgt1],
            context_sentence_ids=[ctx1.id, ctx2.id],
            target_sentence_ids=[],
            token_count=15,
            status=ChunkStatus.PENDING
        )
        translated_text = "Одне цільове речення."
        report = checker.validate(chunk, translated_text)
        assert report.is_passed is True
        assert not any(i.rule_name == "SentenceCountDrop" for i in report.issues)

