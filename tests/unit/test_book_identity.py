"""
Unit tests for deterministic Book.id (sha256), deterministic job_id & fingerprint,
and incompatible resume detection.
"""
import hashlib
from uuid import UUID
from pathlib import Path
import pytest

from src.config.settings import Settings
from src.launcher.translation_runner import (
    compute_job_fingerprint,
    _verify_or_save_job_fingerprint,
    count_existing_chunks,
    IncompatibleJobConfigurationError,
)
from src.knowledge_base.db_schema import init_db


class TestBookIdentityAndFingerprint:
    """Tests verifying content-hashed Book.id and deterministic configuration fingerprints."""

    def test_book_id_identical_for_same_content_across_different_paths(self, tmp_path):
        file1 = tmp_path / "original_book.txt"
        file2 = tmp_path / "renamed_book.txt"
        content = b"Chapter 1\nIt was the best of times, it was the worst of times."
        file1.write_bytes(content)
        file2.write_bytes(content)

        id1 = UUID(hex=hashlib.sha256(file1.read_bytes()).hexdigest()[:32])
        id2 = UUID(hex=hashlib.sha256(file2.read_bytes()).hexdigest()[:32])

        assert id1 == id2
        assert id1 == UUID(hex=hashlib.sha256(content).hexdigest()[:32])

    def test_book_id_differs_for_different_content(self, tmp_path):
        file1 = tmp_path / "book_a.txt"
        file2 = tmp_path / "book_b.txt"
        file1.write_bytes(b"Content A")
        file2.write_bytes(b"Content B")

        id1 = UUID(hex=hashlib.sha256(file1.read_bytes()).hexdigest()[:32])
        id2 = UUID(hex=hashlib.sha256(file2.read_bytes()).hexdigest()[:32])

        assert id1 != id2

    def test_compute_job_fingerprint_deterministic(self, tmp_path):
        content = b"Sample book text for fingerprinting."
        test_settings = Settings(
            aya_temperature=0.0,
            aya_top_p=1.0,
            aya_repetition_penalty=1.02,
        )

        job_id1, fp1, payload1 = compute_job_fingerprint(content, test_settings)
        job_id2, fp2, payload2 = compute_job_fingerprint(content, test_settings)

        assert job_id1 == job_id2
        assert fp1 == fp2
        assert payload1 == payload2
        assert job_id1.startswith(hashlib.sha256(content).hexdigest()[:32])

    def test_fingerprint_changes_when_configuration_changes(self):
        content = b"Sample book text for fingerprinting."
        settings_deterministic = Settings(aya_temperature=0.0, aya_top_p=1.0)
        settings_stochastic = Settings(aya_temperature=0.7, aya_top_p=0.9)

        _, fp1, _ = compute_job_fingerprint(content, settings_deterministic)
        _, fp2, _ = compute_job_fingerprint(content, settings_stochastic)

        assert fp1 != fp2

    def test_incompatible_resume_detected_and_rejected(self, tmp_path):
        db_path = tmp_path / "test_fingerprints.sqlite"
        init_db(db_path)

        content = b"Persistent book bytes."
        book_id = UUID(hex=hashlib.sha256(content).hexdigest()[:32])

        settings_v1 = Settings(aya_temperature=0.0, aya_top_p=1.0)
        job_id1, fp1, payload1 = compute_job_fingerprint(content, settings_v1)

        # Initial recording
        _verify_or_save_job_fingerprint(
            db_path=db_path,
            book_id=book_id,
            job_id=job_id1,
            fingerprint=fp1,
            payload=payload1,
            clear_cache=False,
        )

        # Resume with identical settings -> succeeds
        _verify_or_save_job_fingerprint(
            db_path=db_path,
            book_id=book_id,
            job_id=job_id1,
            fingerprint=fp1,
            payload=payload1,
            clear_cache=False,
        )

        # Resume with changed settings -> raises IncompatibleJobConfigurationError
        settings_v2 = Settings(aya_temperature=0.5, aya_top_p=0.8)
        job_id2, fp2, payload2 = compute_job_fingerprint(content, settings_v2)

        with pytest.raises(IncompatibleJobConfigurationError) as exc_info:
            _verify_or_save_job_fingerprint(
                db_path=db_path,
                book_id=book_id,
                job_id=job_id2,
                fingerprint=fp2,
                payload=payload2,
                clear_cache=False,
            )

        assert "Incompatible resume" in str(exc_info.value)
        assert "fingerprint mismatch" in str(exc_info.value)

        # Resume with changed settings when clear_cache=True -> succeeds
        _verify_or_save_job_fingerprint(
            db_path=db_path,
            book_id=book_id,
            job_id=job_id2,
            fingerprint=fp2,
            payload=payload2,
            clear_cache=True,
        )
