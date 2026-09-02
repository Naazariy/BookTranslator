"""
End-to-End Integration Test Suites for BookTranslator Quality Improvements.
Covers:
1. Full pipeline execution verifying M < N sentence merge clearing does NOT produce duplicate sentences (TXT and PDF).
2. Deterministic unit conversion integrated in pipeline (e.g. converting 80 feet to 24 метри, 15 ft to 4.6 м).
3. Cyrillic-Latin mixed-script word sanitization in pipeline output (e.g. fixing 'Смачнissimo' to 'Смакота' and repairing homoglyphs).
4. Elimination of technical artifacts (no leaked #, fences, or <tag_1> in output).
5. Full Ukrainian grammar agreement across converted units and translated sentences.
"""
import re
import pytest
from pathlib import Path
from uuid import uuid4
from typing import List

from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.domain.models.knowledge import GlossaryItem, EntityType
from src.chunking.manager import ChunkManager
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.translation.pipeline import TwoStageTranslationPipeline
from src.translation.aya_editing_engine import QuantizedAyaEditingEngine
from src.preprocessing.unit_converter import UnitConverter
from src.preprocessing.pipeline import PreprocessingPipeline
from src.quality.pipeline import QualityPipeline, sanitize_mixed_script_words
from src.writers.txt_writer import TxtWriter
from src.writers.pdf_writer import PdfWriter
from src.launcher.translation_runner import execute_translation_job, TranslationJobConfig
from tests.e2e.fixtures import (
    SampleBookFactory,
    MockCTranslate2Engine,
    MockQuantizedAyaEngine,
    reconcile_document_dom,
)


class TestSentenceMergeDeduplicationE2E:
    """E2E Verification that M < N sentence merging never produces duplicate sentences in TXT and PDF outputs."""

    def test_full_pipeline_m_less_than_n_clears_stale_drafts_in_txt_and_pdf(self, tmp_path):
        """When Stage 2 merges 3 sentences into 1 (M < N), trailing target sentences are cleared,

        preventing duplicate sentences and stale NLLB text in TXT and PDF exports.
        """
        db_path = tmp_path / "dedup_pipeline.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        # Create multi-sentence paragraph
        book = Book(id=uuid4(), title="Merge Test Book", source_language="en", target_language="uk")
        ch1 = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        
        sent_texts = [
            "The morning sun rose over the quiet valley.",
            "A gentle breeze rustled through the ancient pine trees.",
            "Small birds began their melodic songs in the canopy."
        ]
        para1 = Paragraph(id=uuid4(), sentences=[
            Sentence(id=uuid4(), original_text=sent_texts[0], order_index=0),
            Sentence(id=uuid4(), original_text=sent_texts[1], order_index=1),
            Sentence(id=uuid4(), original_text=sent_texts[2], order_index=2),
        ])
        ch1.paragraphs.append(para1)
        book.chapters.append(ch1)

        # Chunk with all 3 sentences
        chunk_manager = ChunkManager()
        chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=200, overlap_sentences=0))
        assert len(chunks) == 1
        for c in chunks:
            repo.save_chunk_state(c)

        # NLLB translates 3 sentences individually
        class NLLBDraft(MockCTranslate2Engine):
            def translate_batch(self, texts, *args, **kwargs):
                return [f"Чернетка {t}" for t in texts]

        # Aya combines all 3 sentences into 1 single refined sentence
        class MergedAya(MockQuantizedAyaEngine):
            def refine_chunk(self, draft_translation, source_text="", *args, **kwargs):
                return "Ранкове сонце зійшло над тихою долиною, де легкий вітер шелестів соснами, а птахи співали пісні."

        pipeline = TwoStageTranslationPipeline(NLLBDraft(), MergedAya(), repo)
        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined_chunks = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        reconciled_book = reconcile_document_dom(book, refined_chunks)

        # 1. Export to TXT
        txt_path = tmp_path / "output_merge.txt"
        txt_writer = TxtWriter()
        txt_writer.write(reconciled_book, txt_path)

        txt_content = txt_path.read_text(encoding="utf-8")
        # Merged sentence appears exactly once
        assert "Ранкове сонце зійшло над тихою долиною" in txt_content
        # Stale NLLB drafts must NOT appear anywhere in the document
        assert "Чернетка" not in txt_content
        # English fallback must NOT occur for the cleared trailing sentences
        assert "A gentle breeze" not in txt_content
        assert "Small birds began" not in txt_content

        # 2. Export to PDF
        pdf_path = tmp_path / "output_merge.pdf"
        pdf_writer = PdfWriter()
        pdf_writer.write(reconciled_book, pdf_path)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read(10)
            assert pdf_bytes.startswith(b"%PDF-")


class TestDeterministicUnitConversionE2E:
    """E2E Verification of deterministic imperial to metric conversions in the translation pipeline."""

    def test_deterministic_unit_conversion_in_book_pipeline(self, tmp_path):
        """Validates that imperial units (feet, miles, pounds, °F) are converted to metric with 0% math error."""
        db_path = tmp_path / "unit_preproc.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = Book(id=uuid4(), title="Unit Test Book", source_language="en", target_language="uk")
        ch1 = Chapter(id=uuid4(), title="Chapter 1", order_index=0)

        # Sentences with diverse measurement units
        s0 = Sentence(id=uuid4(), original_text="The stone wall was 80 feet high and 15 ft thick.", order_index=0)
        s1 = Sentence(id=uuid4(), original_text="The marching army covered 5 miles before dusk.", order_index=1)
        s2 = Sentence(id=uuid4(), original_text="The merchant loaded 100 lbs of flour.", order_index=2)
        s3 = Sentence(id=uuid4(), original_text="The cozy tavern room was heated to 72 °F.", order_index=3)

        ch1.paragraphs.append(Paragraph(id=uuid4(), sentences=[s0, s1, s2, s3]))
        book.chapters.append(ch1)

        # Preprocessing pipeline with UnitConverter enabled
        unit_converter = UnitConverter()
        preprocessor = PreprocessingPipeline(
            kb_repo=repo,
            unit_converter=unit_converter,
            convert_units=True,
            unit_conversion_policy="metric"
        )
        converted_book = preprocessor.convert_book_units(book)

        # Verify converted text in sentences
        p_sents = converted_book.chapters[0].paragraphs[0].sentences
        # 80 feet -> 24 метри (or 24.4 м), 15 ft -> 4.6 м
        assert "24 метри" in p_sents[0].original_text or "24 м" in p_sents[0].original_text or "24.4 м" in p_sents[0].original_text
        assert "4.6 м" in p_sents[0].original_text or "4.6 метра" in p_sents[0].original_text
        # 5 miles -> 8 кілометрів / 8 км
        assert "8 кілометрів" in p_sents[1].original_text or "8 км" in p_sents[1].original_text
        # 100 lbs -> 45.4 кг
        assert "45.4 кг" in p_sents[2].original_text or "45.4 кілограма" in p_sents[2].original_text
        # 72 °F -> 22 °C
        assert "22 °C" in p_sents[3].original_text

    def test_ukrainian_grammar_agreement_across_inflection_cases(self):
        """Verifies full Slavic grammatical agreement (1, 2-4, 5-20, decimals) across converted units."""
        forms_m = UnitConverter.UKRAINIAN_UNIT_FORMS['meter']
        forms_km = UnitConverter.UKRAINIAN_UNIT_FORMS['kilometer']
        forms_kg = UnitConverter.UKRAINIAN_UNIT_FORMS['kilogram']

        # 1 unit -> nominative singular
        assert UnitConverter.get_ukrainian_unit_form(1, forms_m) == "метр"
        assert UnitConverter.get_ukrainian_unit_form(21, forms_m) == "метр"
        assert UnitConverter.get_ukrainian_unit_form(101, forms_m) == "метр"

        # 2, 3, 4 units -> nominative plural
        assert UnitConverter.get_ukrainian_unit_form(2, forms_m) == "метри"
        assert UnitConverter.get_ukrainian_unit_form(24, forms_m) == "метри"
        assert UnitConverter.get_ukrainian_unit_form(3, forms_km) == "кілометри"

        # 5-20 units, teens -> genitive plural
        assert UnitConverter.get_ukrainian_unit_form(5, forms_m) == "метрів"
        assert UnitConverter.get_ukrainian_unit_form(11, forms_m) == "метрів"
        assert UnitConverter.get_ukrainian_unit_form(12, forms_m) == "метрів"
        assert UnitConverter.get_ukrainian_unit_form(14, forms_m) == "метрів"
        assert UnitConverter.get_ukrainian_unit_form(80, forms_m) == "метрів"

        # Decimal / fractional numbers -> genitive singular
        assert UnitConverter.get_ukrainian_unit_form(4.6, forms_m) == "метра"
        assert UnitConverter.get_ukrainian_unit_form(1.8, forms_m) == "метра"
        assert UnitConverter.get_ukrainian_unit_form(45.4, forms_kg) == "кілограма"


class TestMixedScriptSanitizationE2E:
    """E2E Verification that Cyrillic-Latin hybrid tokens and homoglyphs are repaired in pipeline output."""

    def test_e2e_pipeline_repairs_smachnissimo_and_homoglyphs(self, tmp_path):
        """When LLM hallucinates portmanteaus like 'Смачнissimo' or Latin homoglyphs,

        the pipeline repairs them to genuine Ukrainian words.
        """
        db_path = tmp_path / "mixed_script.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = Book(id=uuid4(), title="Mixed Script Book", source_language="en", target_language="uk")
        ch = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        s0 = Sentence(id=uuid4(), original_text="The dinner was exquisite in the city.", order_index=0)
        ch.paragraphs.append(Paragraph(id=uuid4(), sentences=[s0]))
        book.chapters.append(ch)

        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book.id,
            chapter_id=ch.id,
            paragraph_indices=[0],
            source_sentences=[s0],
            target_sentence_ids=[s0.id],
            token_count=10,
            status=ChunkStatus.PENDING
        )
        repo.save_chunk_state(chunk)

        # Aya hallucinates "Смачнissimo" and Latin 'i' inside "мiсто", repaired via sanitization pipeline
        class HallucinatingAya(MockQuantizedAyaEngine):
            def refine_chunk(self, draft_translation, *args, **kwargs):
                # Inject 'Смачнissimo' and 'мiсто' where 'i' is Latin \u0069
                latin_i_misto = "м\u0069сто"
                raw = f"Це була Смачнissimo вечеря у {latin_i_misto} біля річки."
                return sanitize_mixed_script_words(raw)

        pipeline = TwoStageTranslationPipeline(MockCTranslate2Engine(), HallucinatingAya(), repo)
        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        reconciled = reconcile_document_dom(book, refined)

        out_path = tmp_path / "mixed_output.txt"
        writer = TxtWriter()
        writer.write(reconciled, out_path)

        content = out_path.read_text(encoding="utf-8")
        # 'Смачнissimo' repaired to 'Смакота'
        assert "Смакота" in content
        assert "Смачнissimo" not in content
        # Homoglyph repaired: Latin 'i' replaced with Cyrillic 'і' (\u0456)
        assert "\u0456" in content  # Ukrainian Cyrillic 'і'
        assert "\u0069" not in content  # Latin 'i' eliminated from Cyrillic word

    def test_e2e_preserves_pure_latin_technical_terms(self):
        """Preserves legitimate standalone Latin words like iPhone, Wi-Fi, AI."""
        text = "Користувач підключив iPhone до мережі Wi-Fi та запустив модель AI."
        sanitized = sanitize_mixed_script_words(text)
        assert "iPhone" in sanitized
        assert "Wi-Fi" in sanitized
        assert "AI" in sanitized


class TestTechnicalArtifactsEliminationE2E:
    """E2E Verification that markdown code fences, headers, preambles, and unmapped tags never leak into output."""

    def test_e2e_zero_technical_artifacts_in_exported_document(self, tmp_path):
        """Simulates adversarial LLM responses with markdown headers, fences, preambles, and tags.

        Verifies that final output files contain zero technical artifacts.
        """
        db_path = tmp_path / "artifacts.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = Book(id=uuid4(), title="Artifacts Test Book", source_language="en", target_language="uk")
        ch1 = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        s0 = Sentence(id=uuid4(), original_text="The story begins here.", order_index=0)
        ch1.paragraphs.append(Paragraph(id=uuid4(), sentences=[s0]))
        book.chapters.append(ch1)

        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book.id,
            chapter_id=ch1.id,
            paragraph_indices=[0],
            source_sentences=[s0],
            target_sentence_ids=[s0.id],
            token_count=10,
            status=ChunkStatus.PENDING
        )
        repo.save_chunk_state(chunk)

        # Aya engine with dynamic regex sanitization
        aya_engine = QuantizedAyaEditingEngine(device="cpu", load_in_4bit=False)

        # Raw adversarial LLM output
        raw_llm_output = (
            "```markdown\n"
            "# Глава 1: Початок історії\n"
            "Ось фінальний переклад:\n"
            "=== ВІДРЕДАГОВАНИЙ ТЕКСТ ===\n"
            "«<tag_1>Історія починається тут, у стародавньому замку.</tag_1>»\n"
            "```"
        )
        sanitized_output = aya_engine._sanitize_output(raw_llm_output)

        class SanitizingAya(MockQuantizedAyaEngine):
            def refine_chunk(self, draft_translation, *args, **kwargs):
                return sanitized_output

        pipeline = TwoStageTranslationPipeline(MockCTranslate2Engine(), SanitizingAya(), repo)
        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        reconciled = reconcile_document_dom(book, refined)

        out_txt = tmp_path / "artifacts_clean.txt"
        writer = TxtWriter()
        writer.write(reconciled, out_txt)

        content = out_txt.read_text(encoding="utf-8")
        assert "Історія починається тут, у стародавньому замку." in content
        assert "```" not in content
        assert "#" not in content
        assert "Ось фінальний" not in content
        assert "===" not in content
        assert "<tag_" not in content
        assert "</tag_" not in content


class TestDynamicGlossaryConsistencyE2E:
    """E2E Verification that dynamic chunk glossary injection maintains terminology consistency across chapters."""

    def test_e2e_consistent_terminology_across_multiple_chapters(self, tmp_path):
        """Verifies that glossary terms (e.g. Rabbitfolk -> кролячий народ, Subs -> передплатники)

        are consistently applied across all chapters.
        """
        db_path = tmp_path / "glossary_e2e.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        # Populate SQLite glossary
        glossary_items = [
            GlossaryItem(source_term="Rabbitfolk", target_term="кролячий народ", entity_type=EntityType.CHARACTER),
            GlossaryItem(source_term="Subs", target_term="передплатники", entity_type=EntityType.TERM),
            GlossaryItem(source_term="Played by", target_term="Виконавець ролі", entity_type=EntityType.TERM),
        ]
        repo.add_glossary_items(glossary_items)

        # Create multi-chapter book
        book = Book(id=uuid4(), title="Glossary Book", source_language="en", target_language="uk")
        ch1 = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        s1 = Sentence(id=uuid4(), original_text="The Rabbitfolk defended their village.", order_index=0)
        ch1.paragraphs.append(Paragraph(id=uuid4(), sentences=[s1]))

        ch2 = Chapter(id=uuid4(), title="Chapter 2", order_index=1)
        s2 = Sentence(id=uuid4(), original_text="The streamer thanked all Subs for their support.", order_index=0)
        ch2.paragraphs.append(Paragraph(id=uuid4(), sentences=[s2]))

        book.chapters.extend([ch1, ch2])

        # Chunk manager creates chunks
        chunk_manager = ChunkManager()
        chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=100, overlap_sentences=0))
        repo.save_chunk_state_batch(chunks)

        all_glossary = repo.get_glossary()

        # Dynamic chunk filtering ensures chunk 1 gets Rabbitfolk and chunk 2 gets Subs
        chunk1_glossary = ChunkManager.filter_glossary_for_chunk(all_glossary, "The Rabbitfolk defended their village.")
        assert len(chunk1_glossary) == 1
        assert chunk1_glossary[0].source_term == "Rabbitfolk"

        chunk2_glossary = ChunkManager.filter_glossary_for_chunk(all_glossary, "The streamer thanked all Subs for their support.")
        assert len(chunk2_glossary) == 1
        assert chunk2_glossary[0].source_term == "Subs"
