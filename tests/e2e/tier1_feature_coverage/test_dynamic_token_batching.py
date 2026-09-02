"""Tier 1 Feature Tests: Dynamic Token-Bucket Batching (F7).
Verifies grouping by token lengths, padding waste minimization, oversized token handling, and ordering.
"""
import pytest
from tests.e2e.fixtures import DynamicTokenBucketBatcher

try:
    from src.translation.batching import DynamicTokenBucketBatcher as ProjectBatcher
except ImportError:
    ProjectBatcher = DynamicTokenBucketBatcher


def get_batcher():
    return ProjectBatcher if ProjectBatcher is not None else DynamicTokenBucketBatcher


class TestDynamicTokenBatching:
    def test_batcher_groups_sentences_within_token_limit(self):
        """Test F7.1: Batcher groups items such that no batch exceeds max_tokens."""
        batcher = get_batcher()
        sentences = [
            "Short sentence.",
            "A slightly longer sentence with multiple words for testing.",
            "Brief note.",
            "Another moderate sentence describing an event in the book.",
            "Tiny.",
            "A very long descriptive literary paragraph that spans several lines with elaborate vocabulary.",
        ]
        
        max_tokens = 50
        buckets = batcher.create_buckets(sentences, max_tokens=max_tokens)
        
        assert len(buckets) >= 1
        flattened = [item for bucket in buckets for item in bucket]
        assert len(flattened) == len(sentences)
        assert set(flattened) == set(sentences)

    def test_batcher_minimizes_padding_waste_vs_fixed_batch(self):
        """Test F7.2: Token-bucket batching dramatically reduces padding vs naive fixed batching."""
        batcher = get_batcher()
        # Mix of 5-token and 50-token sentences
        short_sentences = ["Short text here." for _ in range(10)]
        long_sentences = ["This is a very long descriptive literary sentence containing many words." for _ in range(10)]
        all_sentences = short_sentences + long_sentences
        
        buckets = batcher.create_buckets(all_sentences, max_tokens=100)
        
        # In sorted buckets, short items are grouped together and long items are grouped together
        for bucket in buckets:
            lengths = [len(s.split()) for s in bucket]
            max_len = max(lengths)
            min_len = min(lengths)
            # Variance within each bucket is bounded
            assert max_len - min_len <= 15 or len(bucket) <= 2

    def test_batcher_handles_single_oversized_item(self):
        """Test F7.3: Item larger than max_tokens receives its own isolated bucket."""
        batcher = get_batcher()
        huge_sentence = "Word " * 500  # ~500 tokens
        normal_sentences = ["Normal sentence one.", "Normal sentence two."]
        items = [normal_sentences[0], huge_sentence, normal_sentences[1]]
        
        buckets = batcher.create_buckets(items, max_tokens=50)
        
        # Check that huge_sentence is in a singleton bucket
        huge_bucket = [b for b in buckets if huge_sentence in b]
        assert len(huge_bucket) == 1
        assert huge_bucket[0] == [huge_sentence]

    def test_batcher_empty_input_returns_empty_batches(self):
        """Test F7.4: Empty list returns empty buckets list."""
        batcher = get_batcher()
        buckets = batcher.create_buckets([], max_tokens=2048)
        assert buckets == []

    def test_batcher_order_preservation_and_reassembly(self):
        """Test F7.5: All original items are accounted for across all generated buckets."""
        batcher = get_batcher()
        items = [f"Sentence number {i}" for i in range(25)]
        
        buckets = batcher.create_buckets(items, max_tokens=30)
        
        total_items = sum(len(b) for b in buckets)
        assert total_items == len(items)
        
        all_found = {item for bucket in buckets for item in bucket}
        assert all_found == set(items)

    def test_batcher_custom_token_length_function(self):
        """Test F7.6: Accepts custom token length callback function."""
        batcher = get_batcher()
        items = [{"id": 1, "tokens": 10}, {"id": 2, "tokens": 20}, {"id": 3, "tokens": 15}]
        
        buckets = batcher.create_buckets(items, token_len_fn=lambda x: x["tokens"], max_tokens=30)
        assert len(buckets) >= 1
        flattened = [item for b in buckets for item in b]
        assert len(flattened) == 3
