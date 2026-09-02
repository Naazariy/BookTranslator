"""Pytest configuration and common fixtures."""
import gc
import os
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e.fixtures import (
    MockCTranslate2Engine,
    MockQuantizedAyaEngine,
    SampleBookFactory,
    UkrainianLiteraryCorpus,
)


@pytest.fixture
def temp_work_dir():
    """Provides an isolated temporary directory for test file operations."""
    tmp = tempfile.TemporaryDirectory()
    try:
        yield Path(tmp.name)
    finally:
        gc.collect()
        try:
            tmp.cleanup()
        except Exception:
            pass


@pytest.fixture
def mock_ctranslate2_engine():
    """Provides a reference CTranslate2 NLLB engine."""
    return MockCTranslate2Engine()


@pytest.fixture
def mock_aya_engine():
    """Provides a reference Quantized Aya editing engine."""
    return MockQuantizedAyaEngine()


@pytest.fixture
def sample_book():
    """Provides a sample Book DOM."""
    return SampleBookFactory.create_sample_book()


@pytest.fixture
def ukrainian_corpus():
    """Provides realistic Ukrainian literary sentences and edge cases."""
    return UkrainianLiteraryCorpus
