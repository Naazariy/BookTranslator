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

    def test_detects_gender_agreement_mismatch_warning(self, checker):
        from src.domain.models.knowledge import GlossaryItem, EntityType

        maria = GlossaryItem(
            source_term="Mary",
            target_term="Марія",
            entity_type=EntityType.CHARACTER,
            grammatical_gender="жіночий"
        )
        john = GlossaryItem(
            source_term="John",
            target_term="Джон",
            entity_type=EntityType.CHARACTER,
            grammatical_gender="чоловічий"
        )
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            source_sentences=[Sentence(id=uuid4(), original_text="Mary said. John went.", order_index=0)],
            token_count=10,
            status=ChunkStatus.PENDING,
            context=ChunkContext(active_glossary=[maria, john])
        )

        # 1. Mismatched: Maria + masculine verb (сказав), John + feminine verb (пішла)
        mismatched_text = "Марія сказав правду. Джон пішла додому."
        report = checker.validate(chunk, mismatched_text)
        rule_names = [i.rule_name for i in report.issues]
        assert "GenderAgreementMismatch" in rule_names
        mismatch_issues = [i for i in report.issues if i.rule_name == "GenderAgreementMismatch"]
        assert len(mismatch_issues) == 2
        assert all(i.severity == IssueSeverity.WARNING for i in mismatch_issues)
        assert report.is_passed is True

        # 2. Correct gender agreement: Maria + feminine verb (сказала), John + masculine verb (пішов)
        correct_text = "Марія сказала правду. Джон пішов додому."
        report_ok = checker.validate(chunk, correct_text)
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_ok.issues)

        # 3. False-positive resistance: oblique object forms, nouns ending in -ла/-ів, multi-clause
        fp_resistant_text = "Марія покликала Джона. Джон знав правила і не знайшов слів. Марія бачила, що Джон сказав правду."
        report_fp = checker.validate(chunk, fp_resistant_text)
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_fp.issues)

        # 4. Inverted reporting verb mismatch
        dialogue_mismatch = "— Так, — сказала тихо Джон."
        report_dial = checker.validate(chunk, dialogue_mismatch)
        assert any(i.rule_name == "GenderAgreementMismatch" for i in report_dial.issues)

        # 5. Direct objects with indeclinable names (Jane, Harry)
        jane = GlossaryItem(source_term="Jane", target_term="Джейн", grammatical_gender="жіночий")
        harry = GlossaryItem(source_term="Harry", target_term="Гаррі", grammatical_gender="чоловічий")
        chunk_indec = TranslationChunk(
            id=uuid4(), book_id=uuid4(), chapter_id=uuid4(),
            source_sentences=[], token_count=10,
            context=ChunkContext(active_glossary=[jane, harry])
        )
        report_indec = checker.validate(chunk_indec, "Джон покликав Джейн. Марія покликала Гаррі.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_indec.issues)

        # 6. Hard-stem feminine genitive modifier (Анна -> Анни)
        anna = GlossaryItem(source_term="Anna", target_term="Анна", grammatical_gender="жіночий")
        chunk_anna = TranslationChunk(
            id=uuid4(), book_id=uuid4(), chapter_id=uuid4(),
            source_sentences=[], token_count=10,
            context=ChunkContext(active_glossary=[anna])
        )
        report_anna = checker.validate(chunk_anna, "Брат Анни сказав правду.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_anna.issues)

        # 7. Typographic apostrophe handling
        report_apo_ok = checker.validate(chunk, "Марія зв’язала вузол.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_apo_ok.issues)

        report_apo_bad = checker.validate(chunk, "Марія зв’язав вузол.")
        assert any(i.rule_name == "GenderAgreementMismatch" for i in report_apo_bad.issues)

        # 8. Neuter verb mismatches and neuter character agreement
        monster = GlossaryItem(source_term="Monster", target_term="Чудовисько", grammatical_gender="середній")
        chunk_neuter = TranslationChunk(
            id=uuid4(), book_id=uuid4(), chapter_id=uuid4(),
            source_sentences=[], token_count=10,
            context=ChunkContext(active_glossary=[maria, monster])
        )
        # Feminine character with neuter verb
        report_fem_neuter = checker.validate(chunk_neuter, "Марія пішло геть.")
        assert any(i.rule_name == "GenderAgreementMismatch" for i in report_fem_neuter.issues)

        # Neuter character with neuter verb
        report_neuter_ok = checker.validate(chunk_neuter, "Чудовисько загарчало.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_neuter_ok.issues)

        # 9. Copula verb 'був' mismatch with feminine character
        anna_item = GlossaryItem(source_term="Anna", target_term="Анна", grammatical_gender="жіночий")
        chunk_anna_copula = TranslationChunk(
            id=uuid4(), book_id=uuid4(), chapter_id=uuid4(),
            source_sentences=[], token_count=10,
            context=ChunkContext(active_glossary=[anna_item])
        )
        report_copula_bad = checker.validate(chunk_anna_copula, "Анна був у саду.")
        assert any(i.rule_name == "GenderAgreementMismatch" for i in report_copula_bad.issues)

        report_copula_ok = checker.validate(chunk_anna_copula, "Анна була у саду.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_copula_ok.issues)

        # 10. Common noun subjects in SVO clauses (no false positive on direct object)
        jane_item = GlossaryItem(source_term="Jane", target_term="Джейн", grammatical_gender="жіночий")
        chunk_svo = TranslationChunk(
            id=uuid4(), book_id=uuid4(), chapter_id=uuid4(),
            source_sentences=[], token_count=10,
            context=ChunkContext(active_glossary=[jane_item])
        )
        report_svo_boy = checker.validate(chunk_svo, "Раптом хлопець покликав Джейн.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_svo_boy.issues)

        report_svo_teacher = checker.validate(chunk_svo, "Учора вчитель похвалив Джейн.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_svo_teacher.issues)

        # 11. Dialogue vocative address preceding dialogue attribution clause
        john_item = GlossaryItem(source_term="John", target_term="Джон", grammatical_gender="чоловічий")
        chunk_dialogue = TranslationChunk(
            id=uuid4(), book_id=uuid4(), chapter_id=uuid4(),
            source_sentences=[], token_count=10,
            context=ChunkContext(active_glossary=[jane_item, john_item])
        )
        report_diag_ok = checker.validate(chunk_dialogue, "— Джейн! — покликав Джон.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_diag_ok.issues)

        report_diag_bad = checker.validate(chunk_dialogue, "— Джейн! — покликала Джон.")
        assert any(i.rule_name == "GenderAgreementMismatch" for i in report_diag_bad.issues)

        # 12. Adverb sequence 'ще не'
        report_adv_ok = checker.validate(chunk_anna_copula, "Анна ще не сказала правду.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_adv_ok.issues)

        report_adv_bad = checker.validate(chunk_anna_copula, "Анна ще не сказав правду.")
        assert any(i.rule_name == "GenderAgreementMismatch" for i in report_adv_bad.issues)

        # 13. Oblique case for masculine names ending in -о (Павло -> Павла)
        paul_item = GlossaryItem(source_term="Paul", target_term="Павло", grammatical_gender="чоловічий")
        chunk_paul = TranslationChunk(
            id=uuid4(), book_id=uuid4(), chapter_id=uuid4(),
            source_sentences=[], token_count=10,
            context=ChunkContext(active_glossary=[paul_item])
        )
        report_paul_gen = checker.validate(chunk_paul, "Брат Павла прийшов додому.")
        assert not any(i.rule_name == "GenderAgreementMismatch" for i in report_paul_gen.issues)



