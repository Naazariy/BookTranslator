"""Regression tests verifying sentence overlap deduplication in BookTranslator.
Ensures that each sentence is translated and assembled into the final document exactly once,
with distinct target_sentence_ids vs context_sentence_ids, per-sentence translation mapping,
and order_index based assembly without text duplication at chunk boundaries.
"""
from pathlib import Path
from uuid import uuid4
import pytest

from src.domain.models.document import Book, Chapter, Paragraph, Sentence
from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.chunking.manager import ChunkManager
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.translation.pipeline import TwoStageTranslationPipeline
from src.writers.txt_writer import TxtWriter
from src.parsers.txt_parser import TxtParser
from src.launcher.translation_runner import execute_translation_job, TranslationJobConfig
from tests.e2e.fixtures import (
    SampleBookFactory,
    MockCTranslate2Engine,
    MockQuantizedAyaEngine,
    reconcile_document_dom,
)


class TestSentenceOverlapDeduplication:
    def test_chunk_manager_distinct_target_and_context_roles(self):
        """Verify ChunkManager partitions target_sentence_ids and keeps overlap in context_sentence_ids."""
        book = SampleBookFactory.create_sample_book(title="Overlap Test Book")
        all_sentence_ids = [
            s.id
            for ch in book.chapters
            for p in ch.paragraphs
            for s in p.sentences
        ]
        assert len(all_sentence_ids) > 0

        chunk_manager = ChunkManager()
        chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=100, overlap_sentences=2))

        assert len(chunks) > 1, "Should create multiple chunks to test overlap"

        seen_target_ids = []
        for i, chunk in enumerate(chunks):
            # Target and context sets must be disjoint within every chunk
            target_set = set(chunk.target_sentence_ids)
            context_set = set(chunk.context_sentence_ids)
            assert target_set.isdisjoint(context_set), f"Chunk {i} has overlapping target and context IDs"

            # All source sentences in chunk must match target_sentence_ids + context_sentence_ids
            source_ids = [s.id for s in chunk.source_sentences]
            assert set(source_ids) == target_set.union(context_set)

            # Target sentences must be non-empty
            assert len(chunk.target_sentence_ids) > 0

            # Target sentences must never have appeared in target_sentence_ids of any prior chunk
            for t_id in chunk.target_sentence_ids:
                assert t_id not in seen_target_ids, f"Sentence {t_id} is duplicated in target_sentence_ids of chunk {i}"
                seen_target_ids.append(t_id)

        # Every sentence in the book must appear in target_sentence_ids EXACTLY ONCE
        assert set(seen_target_ids) == set(all_sentence_ids)
        assert len(seen_target_ids) == len(all_sentence_ids)

    def test_nllb_stage_translates_only_targets_and_ignores_context(self, temp_work_dir):
        """Verify NLLB stage writes translations per sentence_id for targets and ignores context."""
        db_path = temp_work_dir / "nllb_overlap.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = SampleBookFactory.create_sample_book()
        chunk_manager = ChunkManager()
        chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=100, overlap_sentences=2))
        for c in chunks:
            repo.save_chunk_state(c)

        translated_sentences = []
        class TrackingNLLB(MockCTranslate2Engine):
            def translate_batch(self, texts, *args, **kwargs):
                translated_sentences.extend(texts)
                return super().translate_batch(texts, *args, **kwargs)

        nllb = TrackingNLLB()
        aya = MockQuantizedAyaEngine()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        pipeline.execute_nllb_stage(book.id)

        # Total sentences translated by NLLB must equal the total unique target sentences in the book
        total_unique_sentences = sum(len(p.sentences) for ch in book.chapters for p in ch.paragraphs)
        assert len(translated_sentences) == total_unique_sentences

        draft_chunks = list(repo.load_chunks_by_status(book.id, ChunkStatus.DRAFT_COMPLETED))
        for chunk in draft_chunks:
            target_ids = set(chunk.target_sentence_ids)
            for s in chunk.source_sentences:
                if s.id in target_ids:
                    assert s.translated_text is not None and s.translated_text != ""

    def test_aya_stage_refines_targets_and_ignores_context(self, temp_work_dir):
        """Verify Aya stage refines translations per sentence_id for targets without contaminating context."""
        db_path = temp_work_dir / "aya_overlap.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = SampleBookFactory.create_sample_book()
        chunk_manager = ChunkManager()
        chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=100, overlap_sentences=2))
        for c in chunks:
            repo.save_chunk_state(c)

        nllb = MockCTranslate2Engine()
        aya = MockQuantizedAyaEngine()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined_chunks = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        assert len(refined_chunks) == len(chunks)

        for chunk in refined_chunks:
            target_ids = set(chunk.target_sentence_ids)
            for s in chunk.source_sentences:
                if s.id in target_ids:
                    assert s.translated_text is not None and s.translated_text != ""

    def test_full_pipeline_no_duplicate_sentences_at_overlap_boundaries(self, temp_work_dir):
        """Regression test verifying no duplicates in final document text at overlap boundaries."""
        db_path = temp_work_dir / "full_dedup.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        # Create a book with multiple paragraphs and distinct sentences
        book = Book(id=uuid4(), title="Distinct Book", source_language="en", target_language="uk")
        ch1 = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        
        sentences_text = [
            "The first sentence introduces the morning sun.",
            "The second sentence describes the gentle breeze.",
            "The third sentence mentions the singing birds.",
            "The fourth sentence notes the quiet street.",
            "The fifth sentence observes the distant mountains.",
            "The sixth sentence ends the peaceful morning.",
        ]
        
        para1 = Paragraph(id=uuid4(), sentences=[
            Sentence(id=uuid4(), original_text=sentences_text[0], order_index=0),
            Sentence(id=uuid4(), original_text=sentences_text[1], order_index=1),
            Sentence(id=uuid4(), original_text=sentences_text[2], order_index=2),
        ])
        para2 = Paragraph(id=uuid4(), sentences=[
            Sentence(id=uuid4(), original_text=sentences_text[3], order_index=0),
            Sentence(id=uuid4(), original_text=sentences_text[4], order_index=1),
            Sentence(id=uuid4(), original_text=sentences_text[5], order_index=2),
        ])
        ch1.paragraphs.extend([para1, para2])
        book.chapters.append(ch1)

        # Chunk with overlap=2 sentences per chunk
        chunk_manager = ChunkManager()
        chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=60, overlap_sentences=2))
        assert len(chunks) >= 2, "Must produce multiple chunks to test overlap boundary"

        for c in chunks:
            repo.save_chunk_state(c)

        nllb = MockCTranslate2Engine()
        aya = MockQuantizedAyaEngine()
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        reconciled_book = reconcile_document_dom(book, refined)

        out_path = temp_work_dir / "output_dedup.txt"
        writer = TxtWriter()
        writer.write(reconciled_book, out_path)

        output_content = out_path.read_text(encoding="utf-8")

        # Each refined sentence must appear EXACTLY ONCE in the output document
        for s_orig in sentences_text:
            count = output_content.count(s_orig)
            assert count == 1, f"Sentence '{s_orig}' appeared {count} times in final document (expected exactly 1)"

    def test_assembly_writer_strictly_follows_order_index(self, temp_work_dir):
        """Verify that assembly writers sort sentences by order_index to maintain narrative sequence."""
        book = Book(id=uuid4(), title="Order Book", source_language="en", target_language="uk")
        ch = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        
        # Insert sentences out of order in the paragraph list
        s0 = Sentence(id=uuid4(), original_text="First.", translated_text="Перше.", order_index=0)
        s1 = Sentence(id=uuid4(), original_text="Second.", translated_text="Друге.", order_index=1)
        s2 = Sentence(id=uuid4(), original_text="Third.", translated_text="Третє.", order_index=2)
        
        # Scramble order in paragraph.sentences
        para = Paragraph(id=uuid4(), sentences=[s2, s0, s1])
        ch.paragraphs.append(para)
        book.chapters.append(ch)

        out_path = temp_work_dir / "order_test.txt"
        writer = TxtWriter()
        writer.write(book, out_path)

        content = out_path.read_text(encoding="utf-8")
        assert "Перше. Друге. Третє." in content

    def test_multi_chapter_overlap_isolation(self):
        """Verify that overlap sentences from chapter N never leak into chapter N+1."""
        book = Book(id=uuid4(), title="Multi Chapter Book", source_language="en", target_language="uk")
        ch1 = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        ch1.paragraphs.append(Paragraph(id=uuid4(), sentences=[
            Sentence(id=uuid4(), original_text=f"This is chapter one sentence number {i} with sufficient text.", order_index=i) for i in range(15)
        ]))
        ch2 = Chapter(id=uuid4(), title="Chapter 2", order_index=1)
        ch2.paragraphs.append(Paragraph(id=uuid4(), sentences=[
            Sentence(id=uuid4(), original_text=f"This is chapter two sentence number {i} with sufficient text.", order_index=i) for i in range(15)
        ]))
        book.chapters.extend([ch1, ch2])

        chunk_manager = ChunkManager()
        chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=60, overlap_sentences=2))

        ch1_chunks = [c for c in chunks if c.chapter_id == ch1.id]
        ch2_chunks = [c for c in chunks if c.chapter_id == ch2.id]

        assert len(ch1_chunks) >= 2
        assert len(ch2_chunks) >= 2

        # The first chunk of chapter 2 must NOT have context from chapter 1
        assert len(ch2_chunks[0].context_sentence_ids) == 0
        assert len(ch2_chunks[0].context.previous_sentences) == 0

        # Subsequent chunks within chapters should have overlap context
        assert len(ch1_chunks[1].context_sentence_ids) == 2
        assert len(ch2_chunks[1].context_sentence_ids) == 2

        # Ensure all target sentence IDs across all chunks are strictly disjoint
        all_target_ids = [s_id for c in chunks for s_id in c.target_sentence_ids]
        assert len(all_target_ids) == len(set(all_target_ids)) == 30

    def test_stage2_aya_hallucination_fewer_sentences_fallback(self, temp_work_dir):
        """Verify Stage 2 preserves NLLB draft translation when LLM returns fewer sentence splits than targets."""
        db_path = temp_work_dir / "fewer_sentences.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = Book(id=uuid4(), title="Hallucination Book", source_language="en", target_language="uk")
        ch = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        s0 = Sentence(id=uuid4(), original_text="Sentence zero.", order_index=0)
        s1 = Sentence(id=uuid4(), original_text="Sentence one.", order_index=1)
        s2 = Sentence(id=uuid4(), original_text="Sentence two.", order_index=2)
        ch.paragraphs.append(Paragraph(id=uuid4(), sentences=[s0, s1, s2]))
        book.chapters.append(ch)

        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book.id,
            chapter_id=ch.id,
            paragraph_indices=[0],
            source_sentences=[s0, s1, s2],
            context_sentence_ids=[],
            target_sentence_ids=[s0.id, s1.id, s2.id],
            token_count=30,
            status=ChunkStatus.PENDING
        )
        repo.save_chunk_state(chunk)

        # Stage 1: NLLB translates all 3 sentences
        class FixedNLLB(MockCTranslate2Engine):
            def translate_batch(self, texts, *args, **kwargs):
                return [f"Чернетка {t}" for t in texts]

        # Stage 2: Aya returns only 1 merged sentence without splitting
        class MergedAya(MockQuantizedAyaEngine):
            def refine_chunk(self, draft_translation, *args, **kwargs):
                return "Одне відредаговане речення на всі три вхідні."

        pipeline = TwoStageTranslationPipeline(FixedNLLB(), MergedAya(), repo)
        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        assert len(refined) == 1
        target_sents = refined[0].target_sentences
        assert len(target_sents) == 3

        # First sentence gets the refined text
        assert target_sents[0].translated_text == "Одне відредаговане речення на всі три вхідні."
        # Sentences 1 and 2 are cleared to avoid duplication
        assert target_sents[1].translated_text == ""
        assert target_sents[2].translated_text == ""

    def test_stage2_aya_hallucination_more_sentences_handling(self, temp_work_dir):
        """Verify Stage 2 aggregates excess sentences into the last target sentence when LLM splits aggressively."""
        db_path = temp_work_dir / "more_sentences.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = Book(id=uuid4(), title="Split Book", source_language="en", target_language="uk")
        ch = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        s0 = Sentence(id=uuid4(), original_text="First sentence.", order_index=0)
        s1 = Sentence(id=uuid4(), original_text="Second sentence.", order_index=1)
        ch.paragraphs.append(Paragraph(id=uuid4(), sentences=[s0, s1]))
        book.chapters.append(ch)

        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book.id,
            chapter_id=ch.id,
            paragraph_indices=[0],
            source_sentences=[s0, s1],
            context_sentence_ids=[],
            target_sentence_ids=[s0.id, s1.id],
            token_count=20,
            status=ChunkStatus.PENDING
        )
        repo.save_chunk_state(chunk)

        class MultiSplitAya(MockQuantizedAyaEngine):
            def refine_chunk(self, draft_translation, *args, **kwargs):
                return "Це перше речення. Це друге речення. Це третє речення. Це четверте речення."

        pipeline = TwoStageTranslationPipeline(MockCTranslate2Engine(), MultiSplitAya(), repo)
        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        assert len(refined) == 1
        target_sents = refined[0].target_sentences
        assert target_sents[0].translated_text == "Це перше речення."
        assert target_sents[1].translated_text == "Це друге речення. Це третє речення. Це четверте речення."

    def test_stage2_aya_empty_output_preserves_draft(self, temp_work_dir):
        """Verify Stage 2 fully preserves draft translations if Aya produces empty or whitespace string."""
        db_path = temp_work_dir / "empty_aya.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = Book(id=uuid4(), title="Empty Output Book", source_language="en", target_language="uk")
        ch = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        s0 = Sentence(id=uuid4(), original_text="Single sentence.", order_index=0)
        ch.paragraphs.append(Paragraph(id=uuid4(), sentences=[s0]))
        book.chapters.append(ch)

        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book.id,
            chapter_id=ch.id,
            paragraph_indices=[0],
            source_sentences=[s0],
            context_sentence_ids=[],
            target_sentence_ids=[s0.id],
            token_count=10,
            status=ChunkStatus.PENDING
        )
        repo.save_chunk_state(chunk)

        class EmptyAya(MockQuantizedAyaEngine):
            def refine_chunk(self, draft_translation, *args, **kwargs):
                return "   \n\n   "

        pipeline = TwoStageTranslationPipeline(MockCTranslate2Engine(), EmptyAya(), repo)
        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        assert len(refined) == 1
        assert refined[0].target_sentences[0].translated_text is not None
        assert refined[0].target_sentences[0].translated_text != ""

    def test_writer_fallback_on_empty_and_whitespace_translated_text(self, temp_work_dir):
        """Verify TxtWriter falls back to original_text ONLY when translated_text is None, and skips empty string."""
        book = Book(id=uuid4(), title="Fallback Book", source_language="en", target_language="uk")
        ch = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        s0 = Sentence(id=uuid4(), original_text="Valid English 1.", translated_text="Переклад 1.", order_index=0)
        s1 = Sentence(id=uuid4(), original_text="Valid English 2.", translated_text="", order_index=1)
        s2 = Sentence(id=uuid4(), original_text="Valid English 3.", translated_text=None, order_index=2)
        ch.paragraphs.append(Paragraph(id=uuid4(), sentences=[s0, s1, s2]))
        book.chapters.append(ch)

        out_path = temp_work_dir / "writer_fallback.txt"
        writer = TxtWriter()
        writer.write(book, out_path)

        content = out_path.read_text(encoding="utf-8")
        assert "Переклад 1." in content
        assert "Valid English 2." not in content
        assert "Valid English 3." in content

    def test_zero_overlap_and_large_overlap_invariance(self):
        """Verify ChunkManager guarantees disjoint target sets with overlap=0 and large overlap."""
        book = SampleBookFactory.create_sample_book(title="Overlap Variations")
        all_ids = [s.id for ch in book.chapters for p in ch.paragraphs for s in p.sentences]

        chunk_manager = ChunkManager()

        # Zero overlap
        chunks_zero = list(chunk_manager.create_chunks_stream(book, max_tokens=100, overlap_sentences=0))
        target_ids_zero = [t for c in chunks_zero for t in c.target_sentence_ids]
        assert set(target_ids_zero) == set(all_ids)
        assert len(target_ids_zero) == len(all_ids)
        assert all(len(c.context_sentence_ids) == 0 for c in chunks_zero)

        # Large overlap (e.g. 10 sentences)
        chunks_large = list(chunk_manager.create_chunks_stream(book, max_tokens=100, overlap_sentences=10))
        target_ids_large = [t for c in chunks_large for t in c.target_sentence_ids]
        assert set(target_ids_large) == set(all_ids)
        assert len(target_ids_large) == len(all_ids)
        for c in chunks_large:
            assert set(c.target_sentence_ids).isdisjoint(set(c.context_sentence_ids))

    def test_single_sentence_chunk_and_multi_split_dialogue_handling(self, temp_work_dir):
        """Verify that when 1 target sentence is refined into multiple dialogue sentences with colons/quotes, all text is preserved."""
        db_path = temp_work_dir / "dialogue_test.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = Book(id=uuid4(), title="Dialogue Book", source_language="en", target_language="uk")
        ch = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        s0 = Sentence(id=uuid4(), original_text='He shouted: "Stop right now!" and paused...', order_index=0)
        ch.paragraphs.append(Paragraph(id=uuid4(), sentences=[s0]))
        book.chapters.append(ch)

        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book.id,
            chapter_id=ch.id,
            paragraph_indices=[0],
            source_sentences=[s0],
            context_sentence_ids=[],
            target_sentence_ids=[s0.id],
            token_count=15,
            status=ChunkStatus.PENDING
        )
        repo.save_chunk_state(chunk)

        class DialogueAya(MockQuantizedAyaEngine):
            def refine_chunk(self, draft_translation, *args, **kwargs):
                return 'Він вигукнув: "Зупиніться негайно!" Потім він зробив паузу...'

        pipeline = TwoStageTranslationPipeline(MockCTranslate2Engine(), DialogueAya(), repo)
        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        assert len(refined) == 1
        assert len(refined[0].target_sentences) == 1
        result_text = refined[0].target_sentences[0].translated_text
        # Must preserve BOTH sentences and dialogue punctuation
        assert 'Він вигукнув: "Зупиніться негайно!"' in result_text
        assert "Потім він зробив паузу..." in result_text

    def test_single_sentence_paragraphs_and_empty_paragraphs_reconciliation(self, temp_work_dir):
        """Verify DOM reconciliation and writer on books with single-sentence paragraphs and empty chapters."""
        book = Book(id=uuid4(), title="Short Paragraphs Book", source_language="en", target_language="uk")
        
        # Chapter 0: empty
        ch0 = Chapter(id=uuid4(), title="Empty Chapter", order_index=0)
        
        # Chapter 1: multiple 1-sentence paragraphs
        ch1 = Chapter(id=uuid4(), title="Dialogue Chapter", order_index=1)
        for i in range(10):
            para = Paragraph(id=uuid4(), sentences=[
                Sentence(id=uuid4(), original_text=f"Dialogue line {i}.", order_index=0)
            ], is_dialogue=True)
            ch1.paragraphs.append(para)
            
        book.chapters.extend([ch0, ch1])

        chunk_manager = ChunkManager()
        chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=20, overlap_sentences=1))
        
        assert len(chunks) >= 2
        for c in chunks:
            assert c.chapter_id == ch1.id

        class EchoNLLB(MockCTranslate2Engine):
            def translate_batch(self, texts, *args, **kwargs):
                return [f"Рядок {t.split()[-1]}" for t in texts]

        class DirectAya(MockQuantizedAyaEngine):
            def refine_chunk(self, draft_translation, *args, **kwargs):
                return draft_translation

        db_path = temp_work_dir / "para_test.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        repo.save_chunk_state_batch(chunks)

        pipeline = TwoStageTranslationPipeline(EchoNLLB(), DirectAya(), repo)
        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        reconcile_document_dom(book, refined)

        out_path = temp_work_dir / "short_paras.txt"
        writer = TxtWriter()
        writer.write(book, out_path)

        content = out_path.read_text(encoding="utf-8")
        assert "Рядок 0." in content
        assert "Рядок 9." in content
        # Ensure no duplicates across all 10 dialogue lines
        for i in range(10):
            assert content.count(f"Рядок {i}.") == 1

    def test_streaming_two_stage_translation_overlap_deduplication(self, temp_work_dir):
        """Verify execute_two_stage_translation streaming method maintains strict sentence deduplication."""
        book = SampleBookFactory.create_sample_book(title="Streaming Book")
        chunk_manager = ChunkManager()
        chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=50, overlap_sentences=2))
        assert len(chunks) >= 3

        nllb = MockCTranslate2Engine()
        aya = MockQuantizedAyaEngine()
        db_path = temp_work_dir / "stream_dedup.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        pipeline = TwoStageTranslationPipeline(nllb, aya, repo)

        refined_chunks = list(pipeline.execute_two_stage_translation(chunks))
        assert len(refined_chunks) == len(chunks)

        reconciled_book = reconcile_document_dom(book, refined_chunks)
        out_path = temp_work_dir / "streaming_output.txt"
        writer = TxtWriter()
        writer.write(reconciled_book, out_path)

        content = out_path.read_text(encoding="utf-8")
        # Each sentence's translated text (or original fallback) should appear exactly once
        for ch in reconciled_book.chapters:
            for p in ch.paragraphs:
                for s in p.sentences:
                    expected = s.translated_text.strip() if (s.translated_text and s.translated_text.strip()) else s.original_text.strip()
                    assert content.count(expected) == 1, f"Sentence translation '{expected}' appeared {content.count(expected)} times (expected 1)"

    def test_context_sentence_ids_dynamic_reconstruction_without_prepopulated_context(self, temp_work_dir):
        """Verify pipeline dynamically builds ChunkContext from context_sentence_ids when previous_sentences is empty."""
        db_path = temp_work_dir / "context_recon.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)

        book = Book(id=uuid4(), title="Dynamic Context Book", source_language="en", target_language="uk")
        ch = Chapter(id=uuid4(), title="Chapter 1", order_index=0)
        s0 = Sentence(id=uuid4(), original_text="Context sentence alpha.", order_index=0)
        s1 = Sentence(id=uuid4(), original_text="Target sentence beta.", order_index=1)
        ch.paragraphs.append(Paragraph(id=uuid4(), sentences=[s0, s1]))
        book.chapters.append(ch)

        # Chunk with context_sentence_ids set but previous_sentences left empty
        chunk = TranslationChunk(
            id=uuid4(),
            book_id=book.id,
            chapter_id=ch.id,
            paragraph_indices=[0],
            source_sentences=[s0, s1],
            context_sentence_ids=[s0.id],
            target_sentence_ids=[s1.id],
            token_count=20,
            context=ChunkContext(previous_sentences=[]),
            status=ChunkStatus.PENDING
        )
        repo.save_chunk_state(chunk)

        captured_context = []
        class ContextTrackingAya(MockQuantizedAyaEngine):
            def refine_chunk(self, draft_translation, source_text="", context=None, glossary=None, *args, **kwargs):
                if context and context.previous_sentences:
                    captured_context.extend(context.previous_sentences)
                return "Відредаговане речення бета."

        pipeline = TwoStageTranslationPipeline(MockCTranslate2Engine(), ContextTrackingAya(), repo)
        pipeline.execute_nllb_stage(book.id)
        pipeline.execute_aya_stage(book.id)

        # Confirm context was dynamically reconstructed from context_sentence_ids
        assert len(captured_context) == 1
        assert captured_context[0] == "Context sentence alpha."

        refined = list(repo.load_chunks_by_status(book.id, ChunkStatus.REFINED))
        assert len(refined) == 1
        # Context sentence must NOT have been modified or translated
        ctx_s = [s for s in refined[0].source_sentences if s.id == s0.id][0]
        assert ctx_s.translated_text is None

    def test_varied_token_budgets_and_overlap_parameters_boundary_stress(self):
        """Stress-test chunking across diverse combinations of max_tokens and overlap_sentences."""
        book = SampleBookFactory.create_sample_book(title="Stress Chunking Book")
        all_sentence_ids = [
            s.id
            for ch in book.chapters
            for p in ch.paragraphs
            for s in p.sentences
        ]

        test_configs = [
            (10, 1),
            (25, 2),
            (50, 3),
            (100, 5),
            (500, 0),
            (1000, 10),
        ]

        chunk_manager = ChunkManager()
        for max_tokens, overlap in test_configs:
            chunks = list(chunk_manager.create_chunks_stream(book, max_tokens=max_tokens, overlap_sentences=overlap))
            
            seen_target_ids = []
            for chunk in chunks:
                target_set = set(chunk.target_sentence_ids)
                context_set = set(chunk.context_sentence_ids)
                
                # Invariant 1: Disjoint target and context within each chunk
                assert target_set.isdisjoint(context_set), f"Overlap in config ({max_tokens}, {overlap})"
                
                # Invariant 2: Non-empty target sentences
                assert len(chunk.target_sentence_ids) > 0
                
                # Invariant 3: Global uniqueness of target sentences
                for t_id in chunk.target_sentence_ids:
                    assert t_id not in seen_target_ids
                    seen_target_ids.append(t_id)

            # Invariant 4: Every sentence in book is covered exactly once
            assert len(seen_target_ids) == len(all_sentence_ids)
            assert set(seen_target_ids) == set(all_sentence_ids)

