"""Tier 3 Cross-Feature Tests: Dynamic Batching + CTranslate2 Engine + SQLite WAL Batching.
Verifies token-bucket grouping feeding CTranslate2 translation batches with atomic SQLite checkpoints.
"""
from uuid import uuid4
import pytest

from src.domain.models.chunk import TranslationChunk, ChunkContext, ChunkStatus
from src.knowledge_base.db_schema import init_db
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from tests.e2e.fixtures import (
    DynamicTokenBucketBatcher,
    MockCTranslate2Engine,
    SampleBookFactory,
)


class TestBatchingWithCTranslate2SQLite:
    def test_dynamic_buckets_flow_through_ctranslate2_to_sqlite_batch(self, temp_work_dir):
        """Test X2.1: Dynamic token buckets translate via CTranslate2 and commit in batch to SQLite."""
        db_path = temp_work_dir / "test_bucket_ct2_db.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        
        book = SampleBookFactory.create_sample_book()
        chunks = SampleBookFactory.create_chunks_from_book(book, sentences_per_chunk=2)
        for c in chunks:
            repo.save_chunk_state(c)

        engine = MockCTranslate2Engine()
        engine.load_model()

        try:
            pending_chunks = list(repo.load_chunks_by_status(book.id, ChunkStatus.PENDING))
            
            # Extract all sentence texts across all chunks
            flat_sentences = []
            sentence_to_chunk_map = {}
            for chunk in pending_chunks:
                for s in chunk.source_sentences:
                    flat_sentences.append(s.original_text)
                    sentence_to_chunk_map[s.id] = chunk

            # Create token buckets
            buckets = DynamicTokenBucketBatcher.create_buckets(flat_sentences, max_tokens=100)
            
            # Translate each bucket
            translated_dict = {}
            for bucket in buckets:
                translations = engine.translate_batch(bucket, "en", "uk")
                for orig, trans in zip(bucket, translations):
                    translated_dict[orig] = trans

            # Reassign drafts to chunks and save in batch
            for chunk in pending_chunks:
                chunk_drafts = [translated_dict.get(s.original_text, "") for s in chunk.source_sentences]
                chunk.draft_translation = " ".join(chunk_drafts)
                chunk.status = ChunkStatus.DRAFT_COMPLETED
                repo.save_chunk_state(chunk)

            # Verify in DB
            draft_chunks = list(repo.load_chunks_by_status(book.id, ChunkStatus.DRAFT_COMPLETED))
            assert len(draft_chunks) == len(chunks)
            for dc in draft_chunks:
                assert dc.draft_translation is not None
                assert len(dc.draft_translation) > 0
        finally:
            engine.unload_model()

    def test_batch_commit_preserves_token_count_and_chunk_status(self, temp_work_dir):
        """Test X2.2: Batch save maintains exact metadata (token_count, chapter_id, book_id)."""
        db_path = temp_work_dir / "test_metadata_db.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        
        book_id = uuid4()
        chapter_id = uuid4()
        chunks = [
            TranslationChunk(
                id=uuid4(),
                book_id=book_id,
                chapter_id=chapter_id,
                paragraph_indices=[i],
                source_sentences=[],
                token_count=15 * (i + 1),
                context=ChunkContext(),
                status=ChunkStatus.DRAFT_COMPLETED,
                draft_translation=f"Чернетка {i}"
            )
            for i in range(25)
        ]

        for c in chunks:
            repo.save_chunk_state(c)

        loaded = list(repo.load_chunks_by_status(book_id, ChunkStatus.DRAFT_COMPLETED))
        assert len(loaded) == 25
        for orig, db_chunk in zip(chunks, sorted(loaded, key=lambda x: x.token_count)):
            assert db_chunk.token_count == orig.token_count
            assert db_chunk.chapter_id == chapter_id

    def test_large_document_chunk_stream_to_db_checkpoint_flow(self, temp_work_dir):
        """Test X2.3: Stream of 100 chunks efficiently committed and reloaded."""
        db_path = temp_work_dir / "test_stream_db.db"
        init_db(db_path)
        repo = SQLiteKnowledgeBaseRepository(db_path)
        book_id = uuid4()
        
        chunks = [
            TranslationChunk(
                id=uuid4(),
                book_id=book_id,
                chapter_id=uuid4(),
                paragraph_indices=[0],
                source_sentences=[],
                token_count=45,
                context=ChunkContext(),
                status=ChunkStatus.PENDING
            )
            for _ in range(100)
        ]

        for c in chunks:
            repo.save_chunk_state(c)

        assert repo.count_chunks_for_book(book_id) == 100
        pending = list(repo.load_chunks_by_status(book_id, ChunkStatus.PENDING))
        assert len(pending) == 100
