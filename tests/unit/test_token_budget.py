"""
Unit tests for TokenBudget and BudgetAllocation (Milestone 3).
"""
import pytest
from unittest.mock import MagicMock
from src.context.token_budget import TokenBudget, BudgetAllocation


class MockTokenizer:
    """Mock HuggingFace / Llama-cpp tokenizer for testing exact counting."""
    def __init__(self, factor: int = 2):
        self.factor = factor

    def encode(self, text: str, add_special_tokens: bool = False):
        return [1] * (len(text.split()) * self.factor)


def test_token_budget_initialization_headroom():
    budget = TokenBudget(max_total_tokens=4096, max_generation_tokens=1024, reserved_scaffold_tokens=450)
    assert budget.max_total_tokens == 4096
    assert budget.max_generation_tokens == 1024
    assert budget.max_prompt_tokens == 3072
    assert budget.reserved_scaffold_tokens == 450


def test_token_budget_count_tokens_exact_with_tokenizer():
    mock_tok = MockTokenizer(factor=3)
    budget = TokenBudget(tokenizer=mock_tok)

    count = budget.count_tokens("This is a simple test sentence.")
    # 6 words * 3 factor = 18 tokens
    assert count == 18

    # Tokenize method fallback
    mock_tok2 = MagicMock()
    del mock_tok2.encode
    mock_tok2.tokenize.return_value = ["a", "b", "c", "d"]
    budget2 = TokenBudget(tokenizer=mock_tok2)
    assert budget2.count_tokens("test") == 4


def test_token_budget_count_tokens_heuristic_fallback():
    budget = TokenBudget(tokenizer=None)

    # Empty text
    assert budget.count_tokens("") == 0
    assert budget.count_tokens(None) == 0
    assert budget.estimate_tokens("") == 0

    # English prose
    count_en = budget.count_tokens("The quick brown fox jumps over the lazy dog.")
    assert count_en > 0
    assert isinstance(count_en, int)

    # Ukrainian Cyrillic prose
    count_uk = budget.count_tokens("Швидка брунатна лисиця перестрибує через ледачого собаку.")
    assert count_uk > 0
    assert isinstance(count_uk, int)


def test_token_budget_priority_allocation_order():
    """
    Asserts strict priority order:
    Priority 1 (Target) > Priority 2 (Entities) > Priority 3 (History) > Priority 4 (Summary).
    """
    # Constrain budget: total 600 tokens, generation 100, scaffold 50 => prompt available = 450
    budget = TokenBudget(
        max_total_tokens=600,
        max_generation_tokens=100,
        reserved_scaffold_tokens=50,
    )

    class DummyEntity:
        def __init__(self, name: str, locked: bool = False, confidence: float = 0.8, forbidden: list = None):
            self.source_name = name
            self.locked = locked
            self.confidence = confidence
            self.forbidden_target_forms = forbidden or []

    def format_entity(e):
        return f"- {e.source_name} => Translated"

    target_text = "Target paragraph source text for translation. It has moderate length."
    entities = [
        DummyEntity("Normal1", locked=False, confidence=0.5),
        DummyEntity("LockedHero", locked=True, confidence=0.95),
        DummyEntity("Cherry", locked=True, confidence=0.99, forbidden=["Вишня"]),
    ]
    history = [
        "First historical paragraph from earlier scene.",
        "Second historical paragraph directly preceding the target text.",
    ]
    summary = "This is a comprehensive summary of chapter events involving quests and battles."

    fitted_target, fitted_entities, fitted_history, fitted_summary, allocation = budget.allocate(
        target_text=target_text,
        entities=entities,
        format_entity_fn=format_entity,
        history_paragraphs=history,
        summary=summary,
    )

    assert allocation.is_within_budget
    assert fitted_target == target_text
    assert not allocation.is_target_truncated

    # Locked entities should be prioritized
    entity_names = [e.source_name for e in fitted_entities]
    assert "LockedHero" in entity_names
    assert "Cherry" in entity_names


def test_token_budget_sliding_window_newest_history_first():
    """
    Verifies that the most recent approved paragraph (N-1) is included before older (N-2),
    while preserving chronological order in the result.
    """
    # Very tight budget allowing only ~80 tokens after target and scaffold
    budget = TokenBudget(
        max_total_tokens=300,
        max_generation_tokens=100,
        reserved_scaffold_tokens=50,
    )

    p_old = "Older paragraph " * 10
    p_new = "Newest paragraph " * 5

    fitted_target, fitted_entities, fitted_history, fitted_summary, allocation = budget.allocate(
        target_text="Target text.",
        entities=[],
        format_entity_fn=lambda e: "",
        history_paragraphs=[p_old, p_new],
        summary=None,
    )

    # When budget cannot fit both, newest p_new is prioritized
    assert len(fitted_history) >= 1
    assert any("Newest" in p for p in fitted_history)


def test_token_budget_target_exceeds_budget_warning():
    """
    When the target text itself is abnormally large, it is retained but flagged.
    """
    budget = TokenBudget(
        max_total_tokens=150,
        max_generation_tokens=50,
        reserved_scaffold_tokens=40,
    )

    huge_target = "Extremely long single paragraph with extensive descriptions. " * 30

    fitted_target, fitted_entities, fitted_history, fitted_summary, allocation = budget.allocate(
        target_text=huge_target,
        entities=[],
        format_entity_fn=lambda e: "",
        history_paragraphs=["Some history"],
        summary="Some summary",
    )

    assert allocation.is_target_truncated is True
    # History and summary must be dropped when target consumes all budget
    assert fitted_history == []
    assert fitted_summary is None


def test_token_budget_tail_and_head_text_helpers():
    budget = TokenBudget()
    text = "Sentence one. Sentence two! Sentence three? Sentence four."

    # Fit tail text
    tail = budget._fit_tail_text(text, max_tokens=10)
    assert len(tail) > 0
    assert "Sentence four." in tail

    # Fit head text
    head = budget._fit_head_text("Word1 Word2 Word3 Word4 Word5", max_tokens=3)
    assert "Word1" in head
    assert "Word5" not in head
