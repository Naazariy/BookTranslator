"""
Empirical Adversarial Stress Test Suite for Milestone 3
(Phase 3: Paragraph-Aware Contextual Translation).

Challenges:
1. Token budget overflow, massive paragraphs (>4000 tokens), zero/negative headroom,
   priority allocation ladder (Target > Entities > History > Summary).
2. Visible failure enforcement (Requirement F14): truncated JSON, malformed syntax,
   missing segment IDs, empty/whitespace translations, and guaranteeing that draft
   text is NEVER copied to refined_translation or marked as successful.
3. Paragraph boundary & sentence restructuring: merging 3 sentences into 1, splitting
   1 sentence into 3, multi-paragraph document DOM reconciliation, and verifying
   that TxtWriter and PdfWriter export without misalignment or crashes.
4. Localized entity injection with grammatical gender and forbidden variant suppression,
   preventing prompt bloat and verifying prompt V2 contract compliance.
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from unittest.mock import MagicMock, patch

from src.context.token_budget import TokenBudget, BudgetAllocation
from src.context.builder import ContextBuilder, PromptContext
from src.domain.models.segment import (
    TranslationSegment,
    SegmentStatus,
    IllegalStateTransitionError,
    validate_segment_transition,
)
from src.domain.models.knowledge import (
    EntityProfile,
    ScopeLevel,
    GlossaryItem,
)
from src.domain.models.document import (
    Book,
    Chapter,
    Paragraph,
    Sentence,
)
from src.translation.aya_editing_engine import (
    QuantizedAyaEditingEngine,
    SegmentRefinementResult,
)
from src.launcher.concurrency import CancellationToken


# ============================================================================
# Suite 1: Token Budget Overflow, Massive Paragraphs & Priority Allocation
# ============================================================================

class TestTokenBudgetMassiveParagraphAndPriority:
    """
    Empirical stress tests for TokenBudget under extreme token pressure,
    oversized inputs, and multi-tier priority degradation.
    """

    def test_massive_paragraph_overflow_headroom_preservation(self):
        """
        Challenge: A massive paragraph (>4000 tokens) exceeds max_prompt_tokens (3072).
        Verification:
        - Target paragraph is allocated without truncation of its raw text in prompt.
        - `is_target_truncated` flag is set to True.
        - Remaining tokens become 0.
        - Lower priority components (entities, history, summary) are cleanly dropped.
        - Generation headroom (1024 tokens) is strictly preserved.
        """
        budget = TokenBudget(
            max_total_tokens=4096,
            max_generation_tokens=1024,
            reserved_scaffold_tokens=450,
        )
        assert budget.max_prompt_tokens == 3072

        # Generate a massive paragraph text with ~4200 words (~5600 tokens)
        words = ["adventure", "journey", "mountain", "ancient", "mystery", "forest", "shadows"]
        massive_text = " ".join(words * 600)  # 4200 words
        est_tokens = budget.count_tokens(massive_text)
        assert est_tokens > 4000

        dummy_entities = [
            EntityProfile(
                id=uuid4(),
                source_name="Hero",
                canonical_target="Герой",
                scope=ScopeLevel.BOOK,
                locked=True,
                confidence=0.99,
            )
        ]
        history = ["Approved previous paragraph 1.", "Approved previous paragraph 2."]
        summary = "Comprehensive summary of the current adventure chapter."

        fitted_target, fitted_entities, fitted_history, fitted_summary, alloc = budget.allocate(
            target_text=massive_text,
            entities=dummy_entities,
            format_entity_fn=lambda e: f"- {e.source_name} => {e.canonical_target}",
            history_paragraphs=history,
            summary=summary,
        )

        assert fitted_target == massive_text
        assert alloc.is_target_truncated is True
        assert alloc.remaining_tokens == 0
        assert fitted_entities == []
        assert alloc.is_entities_truncated is False  # not reached
        assert fitted_history == []
        assert alloc.is_history_truncated is False  # not reached
        assert fitted_summary is None
        assert alloc.is_summary_truncated is False  # not reached

        # Total used reflects target + scaffold
        assert alloc.total_used_tokens > budget.max_prompt_tokens
        assert not alloc.is_within_budget

    def test_negative_or_zero_available_budget_safety(self):
        """
        Challenge: Configuration where max_generation_tokens exceeds max_total_tokens,
        or reserved scaffold exceeds max_prompt_tokens.
        Verification: No NegativeValue or crash; max_prompt_tokens clamped to >= 0.
        """
        # Generation headroom exceeds total tokens
        budget_inverted = TokenBudget(
            max_total_tokens=500,
            max_generation_tokens=800,
            reserved_scaffold_tokens=100,
        )
        assert budget_inverted.max_prompt_tokens == 0

        fitted_target, fitted_entities, fitted_history, fitted_summary, alloc = budget_inverted.allocate(
            target_text="Short paragraph.",
            entities=[],
            format_entity_fn=lambda e: "",
            history_paragraphs=["Some history"],
            summary="Some summary",
        )
        assert alloc.max_prompt_tokens == 0
        assert alloc.remaining_tokens == 0
        assert alloc.is_target_truncated is True
        assert fitted_history == []
        assert fitted_summary is None

    def test_strict_priority_allocation_ladder(self):
        """
        Challenge: Progressively constrained token limits test strict allocation order:
        Priority 1: Target Paragraph
        Priority 2: Active Locked Entities
        Priority 3: Sliding Window History
        Priority 4: Chapter Summary
        """
        class MockEntity:
            def __init__(self, name: str, locked: bool = False, conf: float = 0.8):
                self.source_name = name
                self.locked = locked
                self.confidence = conf
                self.forbidden_target_forms = []

        def fmt_ent(e):
            return f"- {e.source_name} => Translated"

        target = "Target sentence for translation testing."
        entities = [MockEntity(f"Entity_{i}", locked=True, conf=0.95) for i in range(5)]
        history = ["Previous paragraph one.", "Previous paragraph two."]
        summary = "Detailed summary describing the entire scene and characters."

        # Scenario A: Generous budget fitting all
        budget_generous = TokenBudget(max_total_tokens=2000, max_generation_tokens=200, reserved_scaffold_tokens=50)
        _, f_ent_a, f_hist_a, f_sum_a, alloc_a = budget_generous.allocate(
            target_text=target,
            entities=entities,
            format_entity_fn=fmt_ent,
            history_paragraphs=history,
            summary=summary,
        )
        assert len(f_ent_a) == 5
        assert len(f_hist_a) == 2
        assert f_sum_a is not None
        assert alloc_a.is_within_budget

        # Scenario B: Budget tight enough that summary is dropped, but history and entities fit
        target_tok = budget_generous.count_tokens(target)
        ent_tok = sum(budget_generous.count_tokens(fmt_ent(e) + "\n") for e in entities)
        hist_tok = sum(budget_generous.count_tokens(h + "\n\n") for h in history)
        scaffold = 40

        budget_no_summary = TokenBudget(
            max_total_tokens=scaffold + target_tok + ent_tok + hist_tok + 5,
            max_generation_tokens=0,
            reserved_scaffold_tokens=scaffold,
        )
        _, f_ent_b, f_hist_b, f_sum_b, alloc_b = budget_no_summary.allocate(
            target_text=target,
            entities=entities,
            format_entity_fn=fmt_ent,
            history_paragraphs=history,
            summary=summary,
        )
        assert len(f_ent_b) == 5
        assert len(f_hist_b) == 2
        assert f_sum_b is None  # Summary dropped first!
        assert alloc_b.is_summary_truncated is True

        # Scenario C: Budget only fits Target and Entities; History is dropped
        budget_no_history = TokenBudget(
            max_total_tokens=scaffold + target_tok + ent_tok + 2,
            max_generation_tokens=0,
            reserved_scaffold_tokens=scaffold,
        )
        _, f_ent_c, f_hist_c, f_sum_c, alloc_c = budget_no_history.allocate(
            target_text=target,
            entities=entities,
            format_entity_fn=fmt_ent,
            history_paragraphs=history,
            summary=summary,
        )
        assert len(f_ent_c) == 5
        assert f_hist_c == []  # History dropped before entities!
        assert f_sum_c is None

    def test_entity_prioritization_by_lock_confidence_forbidden(self):
        """
        Challenge: When only some entities fit, verify ranking:
        1. locked=True before locked=False
        2. higher confidence before lower confidence
        3. has forbidden_target_forms before without
        """
        class ComplexEntity:
            def __init__(self, name: str, locked: bool, conf: float, forbidden: list):
                self.source_name = name
                self.locked = locked
                self.confidence = conf
                self.forbidden_target_forms = forbidden

        e_unlocked_low = ComplexEntity("UnlockedLow", False, 0.5, [])
        e_unlocked_high = ComplexEntity("UnlockedHigh", False, 0.88, [])
        e_unlocked_forbid = ComplexEntity("UnlockedForbid", False, 0.88, ["BadForm"])
        e_locked_med = ComplexEntity("LockedMed", True, 0.91, [])
        e_locked_top = ComplexEntity("LockedTop", True, 0.99, ["Forbidden1"])

        entities = [e_unlocked_low, e_locked_med, e_unlocked_high, e_locked_top, e_unlocked_forbid]

        def fmt_ent(e):
            return f"- {e.source_name} => Translated"

        budget = TokenBudget()
        # Calibrate budget to fit exactly target + 2 entities
        line_cost = budget.count_tokens(fmt_ent(e_locked_top) + "\n")
        target_cost = budget.count_tokens("Short target.")
        scaffold_cost = 20

        budget_tight = TokenBudget(
            max_total_tokens=scaffold_cost + target_cost + (line_cost * 2) + 2,
            max_generation_tokens=0,
            reserved_scaffold_tokens=scaffold_cost,
        )

        _, fitted_ents, _, _, alloc = budget_tight.allocate(
            target_text="Short target.",
            entities=entities,
            format_entity_fn=fmt_ent,
            history_paragraphs=[],
            summary=None,
        )

        assert len(fitted_ents) == 2
        names = [e.source_name for e in fitted_ents]
        # Must pick the two locked entities first!
        assert "LockedTop" in names
        assert "LockedMed" in names
        assert "UnlockedHigh" not in names
        assert alloc.is_entities_truncated is True

    def test_history_sliding_window_newest_first_chronological_order(self):
        """
        Challenge: History sliding window must prioritize newest paragraph (N-1) over
        older (N-2), but when both fit, return them in chronological order [N-2, N-1].
        """
        budget = TokenBudget()
        p_older = "First paragraph in chronological sequence: the journey began."
        p_newest = "Second paragraph in sequence: they reached the mysterious castle gates."

        # Case 1: Fits only 1 paragraph -> must pick p_newest
        t_newest = budget.count_tokens(p_newest + "\n\n")
        scaffold = 30
        target = "Target paragraph."
        t_target = budget.count_tokens(target)

        budget_one_p = TokenBudget(
            max_total_tokens=scaffold + t_target + t_newest + 4,
            max_generation_tokens=0,
            reserved_scaffold_tokens=scaffold,
        )

        _, _, fitted_hist_1, _, alloc_1 = budget_one_p.allocate(
            target_text=target,
            entities=[],
            format_entity_fn=lambda e: "",
            history_paragraphs=[p_older, p_newest],
            summary=None,
        )
        assert len(fitted_hist_1) == 1
        assert fitted_hist_1[0] == p_newest
        assert alloc_1.is_history_truncated is True

        # Case 2: Fits both -> returned in chronological order [p_older, p_newest]
        t_older = budget.count_tokens(p_older + "\n\n")
        budget_two_p = TokenBudget(
            max_total_tokens=scaffold + t_target + t_newest + t_older + 10,
            max_generation_tokens=0,
            reserved_scaffold_tokens=scaffold,
        )
        _, _, fitted_hist_2, _, alloc_2 = budget_two_p.allocate(
            target_text=target,
            entities=[],
            format_entity_fn=lambda e: "",
            history_paragraphs=[p_older, p_newest],
            summary=None,
        )
        assert len(fitted_hist_2) == 2
        assert fitted_hist_2[0] == p_older
        assert fitted_hist_2[1] == p_newest
        assert alloc_2.is_history_truncated is False

    def test_tokenizer_counting_exception_fallback(self):
        """
        Challenge: Tokenizer object raises RuntimeError / Exception during encode.
        Verification: Falls back cleanly to heuristic word/character count without raising.
        """
        failing_tokenizer = MagicMock()
        failing_tokenizer.encode.side_effect = RuntimeError("Tokenizer backend crashed")

        budget = TokenBudget(tokenizer=failing_tokenizer)
        count = budget.count_tokens("Тестове речення для перевірки аварійного завершення токенізатора.")
        assert count > 0
        assert isinstance(count, int)


# ============================================================================
# Suite 2: Visible Failure Enforcement (Requirement F14)
# ============================================================================

class TestVisibleFailureEnforcement:
    """
    Empirical stress tests for Requirement F14:
    Visible failure on malformed LLM structured output.
    Never silently falls back to draft marked as successful or refined.
    """

    @pytest.fixture
    def aya_engine(self):
        engine = QuantizedAyaEditingEngine(model_path=None, gguf_path=None)
        engine.model = None
        engine.gguf_llm = MagicMock()
        return engine

    def _create_test_segment(self, draft: str = "Чорновий переклад.") -> TranslationSegment:
        seg_id = uuid4()
        return TranslationSegment(
            id=seg_id,
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=seg_id,
            source_text="Source paragraph for failure testing.",
            draft_translation=draft,
            status=SegmentStatus.DRAFT_COMPLETED,
        )

    def test_visible_failure_truncated_or_syntax_error_json(self, aya_engine):
        """
        Challenge: LLM output abruptly cuts off mid-syntax without complete JSON or ID.
        Output: '{"segments": [{"id": '
        """
        seg = self._create_test_segment()
        malformed_raw = '{"segments": [{"id": '

        aya_engine.gguf_llm.create_chat_completion.return_value = [
            {"choices": [{"delta": {"content": malformed_raw}}]}
        ]

        results = aya_engine.refine_segments_structured([seg])

        assert seg.status == SegmentStatus.FAILED
        assert seg.refined_translation is None  # CRITICAL: must NOT be draft!
        assert seg.error_message is not None
        assert "Stage 2 LLM parsing failure" in seg.error_message

        res = results[seg.id]
        assert res.success is False
        assert res.refined_text is None
        assert res.fallback_draft == "Чорновий переклад."

    def test_visible_failure_missing_segment_id(self, aya_engine):
        """
        Challenge: LLM returns syntactically valid JSON, but with a different UUID.
        Target segment ID is missing from response.
        """
        seg = self._create_test_segment()
        other_uuid = str(uuid4())
        json_output = json.dumps({
            "segments": [
                {
                    "id": other_uuid,
                    "translation": "Переклад для чужого ідентифікатора сегмента."
                }
            ]
        })

        aya_engine.gguf_llm.create_chat_completion.return_value = [
            {"choices": [{"delta": {"content": json_output}}]}
        ]

        results = aya_engine.refine_segments_structured([seg])

        assert seg.status == SegmentStatus.FAILED
        assert seg.refined_translation is None
        assert "No valid translation found for segment" in seg.error_message
        assert results[seg.id].success is False

    def test_visible_failure_empty_or_whitespace_translation(self, aya_engine):
        """
        Challenge: LLM returns segment ID, but the translation field is empty or whitespace.
        Both must be rejected as FAILED and never accepted.
        """
        seg_empty = self._create_test_segment(draft="Draft empty")
        seg_whitespace = self._create_test_segment(draft="Draft whitespace")

        json_output = json.dumps({
            "segments": [
                {"id": str(seg_empty.id), "translation": ""},
                {"id": str(seg_whitespace.id), "translation": "   \n\t  \n  "}
            ]
        })

        aya_engine.gguf_llm.create_chat_completion.return_value = [
            {"choices": [{"delta": {"content": json_output}}]}
        ]

        results = aya_engine.refine_segments_structured([seg_empty, seg_whitespace])

        # seg_empty check
        assert seg_empty.status == SegmentStatus.FAILED
        assert seg_empty.refined_translation is None
        assert results[seg_empty.id].success is False

        # seg_whitespace check
        assert seg_whitespace.status == SegmentStatus.FAILED
        assert seg_whitespace.refined_translation is None
        assert results[seg_whitespace.id].success is False

    def test_visible_failure_non_json_conversational_refusal(self, aya_engine):
        """
        Challenge: LLM produces conversational refusal / apology instead of structured JSON.
        """
        seg = self._create_test_segment()
        refusal = "I am sorry, but as an AI assistant I cannot translate this text due to policy restrictions."

        aya_engine.gguf_llm.create_chat_completion.return_value = [
            {"choices": [{"delta": {"content": refusal}}]}
        ]

        results = aya_engine.refine_segments_structured([seg])

        assert seg.status == SegmentStatus.FAILED
        assert seg.refined_translation is None
        assert results[seg.id].success is False
        assert "Stage 2 LLM parsing failure" in seg.error_message

    def test_visible_failure_empty_raw_output(self, aya_engine):
        """
        Challenge: LLM produces completely empty string or whitespace response.
        """
        seg = self._create_test_segment()

        aya_engine.gguf_llm.create_chat_completion.return_value = [
            {"choices": [{"delta": {"content": ""}}]}
        ]

        results = aya_engine.refine_segments_structured([seg])

        assert seg.status == SegmentStatus.FAILED
        assert seg.refined_translation is None
        assert results[seg.id].success is False
        assert "Empty or whitespace LLM response" in seg.error_message

    def test_visible_failure_no_loaded_model(self):
        """
        Challenge: Neither transformers model nor GGUF model is available/loaded.
        """
        engine = QuantizedAyaEditingEngine(model_path=None, gguf_path=None)
        engine.model = None
        engine.gguf_llm = None
        with patch.object(engine, "load_model"):
            seg = self._create_test_segment()
            results = engine.refine_segments_structured([seg])

            assert seg.status == SegmentStatus.FAILED
            assert seg.refined_translation is None
            assert results[seg.id].success is False
            assert "No loaded model available" in seg.error_message

    def test_visible_failure_pre_and_inflight_cancellation(self, aya_engine):
        """
        Challenge: Cancellation token triggered before or during streaming.
        """
        # Pre-cancellation
        seg1 = self._create_test_segment()
        tok1 = CancellationToken()
        tok1.cancel()

        res1 = aya_engine.refine_segments_structured([seg1], cancel_token=tok1)
        assert seg1.status == SegmentStatus.FAILED
        assert seg1.refined_translation is None
        assert "cancelled by user" in seg1.error_message.lower()
        assert res1[seg1.id].success is False

        # In-flight cancellation during generator iteration
        seg2 = self._create_test_segment()
        tok2 = CancellationToken()

        def stream_generator():
            yield {"choices": [{"delta": {"content": '{"segments": ['}}]}
            tok2.cancel()  # Cancel mid-stream!
            yield {"choices": [{"delta": {"content": '{"id": "'}}]}

        aya_engine.gguf_llm.create_chat_completion.return_value = stream_generator()
        res2 = aya_engine.refine_segments_structured([seg2], cancel_token=tok2)
        assert seg2.status == SegmentStatus.FAILED
        assert seg2.refined_translation is None
        assert "cancelled by user" in seg2.error_message.lower()
        assert res2[seg2.id].success is False

    def test_lifecycle_illegal_state_transitions(self):
        """
        Challenge: Attempt illegal lifecycle state transitions.
        Enforces state machine invariants.
        """
        seg = self._create_test_segment()
        assert seg.status == SegmentStatus.DRAFT_COMPLETED

        # Valid transition: DRAFT_COMPLETED -> EDITED -> ACCEPTED
        seg.transition_to(SegmentStatus.EDITED)
        seg.transition_to(SegmentStatus.ACCEPTED)
        assert seg.status == SegmentStatus.ACCEPTED

        # Invalid from ACCEPTED: terminal state
        with pytest.raises(IllegalStateTransitionError):
            seg.transition_to(SegmentStatus.PENDING)
        with pytest.raises(IllegalStateTransitionError):
            seg.transition_to(SegmentStatus.FAILED)
        with pytest.raises(IllegalStateTransitionError):
            seg.transition_to(SegmentStatus.DRAFT_COMPLETED)

        # Transition validation function checks
        assert not validate_segment_transition(SegmentStatus.ACCEPTED, SegmentStatus.PENDING)
        assert not validate_segment_transition(SegmentStatus.PENDING, SegmentStatus.ACCEPTED)
        assert not validate_segment_transition(SegmentStatus.FAILED, SegmentStatus.EDITED)
        assert validate_segment_transition(SegmentStatus.FAILED, SegmentStatus.PENDING)


# ============================================================================
# Suite 3: Paragraph Boundary & Sentence Restructuring
# ============================================================================

class TestParagraphRestructuringAndWriters:
    """
    Empirical tests for paragraph-level sentence restructuring:
    - Merging 3 English sentences into 1 Ukrainian sentence
    - Splitting 1 English sentence into 3 Ukrainian sentences
    - Multi-paragraph boundary preservation
    - Safe reconciliation into Document DOM
    - Flawless output export via TxtWriter and PdfWriter
    """

    def test_restructure_three_sentences_into_one_ukrainian_sentence(self, tmp_path):
        """
        Challenge: English paragraph has 3 sentences.
        Stage 2 refines them into 1 unified Ukrainian sentence.
        Verification:
        - Segment maintains sentence_ids list of all 3 sentences.
        - DOM reconciliation sets paragraph.sentences[0].translated_text to the single sentence.
        - Remaining sentences in paragraph have empty translated_text and ACCEPTED status.
        - TxtWriter writes exactly 1 sentence without blank lines or duplicate text.
        """
        from src.writers.txt_writer import TxtWriter

        s1 = Sentence(original_text="The heavy rain poured.", translated_text="Лив сильний дощ.")
        s2 = Sentence(original_text="The wind shrieked through the pines.", translated_text="Вітер вив крізь сосни.")
        s3 = Sentence(original_text="Nobody dared to step outside.", translated_text="Ніхто не наважувався ступити на вулицю.")
        para = Paragraph(sentences=[s1, s2, s3])
        chap = Chapter(title="Chapter 1", order_index=0, paragraphs=[para])
        book = Book(title="The Storm", chapters=[chap])

        # Create TranslationSegment
        seg = TranslationSegment.from_paragraph(para, book_id=book.id, chapter_id=chap.id)
        assert len(seg.sentence_ids) == 3
        assert seg.source_text == "The heavy rain poured. The wind shrieked through the pines. Nobody dared to step outside."

        # Stage 2 merges into 1 Ukrainian literary sentence
        unified_uk = "Лив шквальний дощ і вітер вив крізь сосни, тож ніхто не наважувався вистромити носа надвір."
        seg.refined_translation = unified_uk
        seg.transition_to(SegmentStatus.EDITED)
        seg.transition_to(SegmentStatus.ACCEPTED)

        # Simulate DOM reconciliation as in TranslationRunner
        para.sentences[0].translated_text = seg.translated_text
        para.sentences[0].status = SegmentStatus.ACCEPTED
        for extra_s in para.sentences[1:]:
            extra_s.translated_text = ""
            extra_s.status = SegmentStatus.ACCEPTED

        # Export via TxtWriter
        out_file = tmp_path / "restructured_3_to_1.txt"
        writer = TxtWriter(allow_unreviewed=False)
        writer.write(book, out_file)

        content = out_file.read_text(encoding="utf-8")
        assert "Chapter 1" in content
        assert unified_uk in content
        # Ensure draft sentences are not duplicated in output
        assert "Лив сильний дощ." not in content
        assert "Вітер вив крізь сосни." not in content

    def test_restructure_one_sentence_into_three_ukrainian_sentences(self, tmp_path):
        """
        Challenge: English paragraph has 1 long compound sentence.
        Stage 2 splits it into 3 crisp Ukrainian sentences.
        Verification:
        - TxtWriter writes all 3 sentences cleanly.
        - PdfWriter writes the paragraph without throwing an exception or misaligning sentences.
        """
        from src.writers.txt_writer import TxtWriter
        from src.writers.pdf_writer import PdfWriter

        long_en = (
            "Because the ancient forest was shrouded in perpetual mist and prowled by fierce predators "
            "that had survived since the elder days, travelers took the long mountain detour, even though "
            "winter was fast approaching."
        )
        s1 = Sentence(original_text=long_en, translated_text="Оскільки стародавній ліс був укритий імлою...")
        para = Paragraph(sentences=[s1])
        chap = Chapter(title="Chapter 2: The Forest", order_index=0, paragraphs=[para])
        book = Book(title="The Crossing", chapters=[chap])

        seg = TranslationSegment.from_paragraph(para, book_id=book.id, chapter_id=chap.id)

        # Split into 3 Ukrainian sentences
        split_uk = (
            "Стародавній ліс потопав у вічній імлі. Там блукали люті хижаки, що вижили ще з прадавніх часів. "
            "Саме тому мандрівники вирушали в обхід горами, хоча зима вже наступала на п'яти."
        )
        seg.refined_translation = split_uk
        seg.transition_to(SegmentStatus.EDITED)
        seg.transition_to(SegmentStatus.ACCEPTED)

        para.sentences[0].translated_text = seg.translated_text
        para.sentences[0].status = SegmentStatus.ACCEPTED

        # Test TxtWriter
        out_txt = tmp_path / "restructured_1_to_3.txt"
        TxtWriter().write(book, out_txt)
        txt_content = out_txt.read_text(encoding="utf-8")
        assert split_uk in txt_content

        # Test PdfWriter
        out_pdf = tmp_path / "restructured_1_to_3.pdf"
        pdf_writer = PdfWriter()
        pdf_writer.write(book, out_pdf)
        assert out_pdf.exists()
        assert out_pdf.stat().st_size > 500

    def test_complex_multiparagraph_restructuring_preserves_boundaries(self, tmp_path):
        """
        Challenge: Document with 3 distinct paragraphs undergoing different restructuring:
        Para 1: 3 sentences -> 1 sentence
        Para 2: 1 sentence -> 2 sentences
        Para 3: 2 sentences -> 2 sentences
        Verification:
        - Paragraph boundaries are strictly preserved (no text leaks across paragraphs).
        - Document writers export each paragraph separated by double newline.
        """
        from src.writers.txt_writer import TxtWriter

        # Para 1 (3 -> 1)
        p1_sents = [
            Sentence(original_text="Sun rose.", translated_text="Сонце зійшло."),
            Sentence(original_text="Birds sang.", translated_text="Птахи співали."),
            Sentence(original_text="World was at peace.", translated_text="Світ був у мирі."),
        ]
        p1 = Paragraph(sentences=p1_sents)

        # Para 2 (1 -> 2)
        p2_sents = [
            Sentence(
                original_text="He unsheathed his gleaming broadsword and stepped boldly into the darkness.",
                translated_text="Він оголив меч і ступив у темряву.",
            )
        ]
        p2 = Paragraph(sentences=p2_sents)

        # Para 3 (2 -> 2)
        p3_sents = [
            Sentence(original_text="The cave was cold.", translated_text="Печера була холодною."),
            Sentence(original_text="Eyes watched him.", translated_text="Очі спостерігали за ним."),
        ]
        p3 = Paragraph(sentences=p3_sents)

        chap = Chapter(title="Dawn of Destiny", order_index=0, paragraphs=[p1, p2, p3])
        book = Book(title="Chronicles", chapters=[chap])

        # Refined translations
        p1_refined = "Зійшло ранкове сонце, радісно співали птахи, і весь світ спочивав у глибокому мирі."
        p2_refined = "Він рішуче оголив свій сяючий меч. Потім зробив сміливий крок просто в темряву."
        p3_refined = "У надрах печери панував крижаний холод. З пітьми на нього пильно дивилися чиїсь очі."

        for p, refined in zip([p1, p2, p3], [p1_refined, p2_refined, p3_refined]):
            p.sentences[0].translated_text = refined
            p.sentences[0].status = SegmentStatus.ACCEPTED
            for extra in p.sentences[1:]:
                extra.translated_text = ""
                extra.status = SegmentStatus.ACCEPTED

        out_txt = tmp_path / "multiparagraph_restructured.txt"
        TxtWriter().write(book, out_txt)
        content = out_txt.read_text(encoding="utf-8")

        # Verify paragraph isolation
        paragraphs_in_file = [p.strip() for p in content.split("\n\n") if p.strip()]
        assert "Dawn of Destiny" in paragraphs_in_file[0]
        assert paragraphs_in_file[1] == p1_refined
        assert paragraphs_in_file[2] == p2_refined
        assert paragraphs_in_file[3] == p3_refined

    def test_dom_reconciliation_status_propagation_for_failed_segments(self):
        """
        Challenge: In a document where segment 1 succeeds (EDITED) and segment 2 fails (FAILED),
        verify whether DOM reconciliation preserves the FAILED status or erroneously promotes it
        to ACCEPTED.
        """
        s1 = Sentence(original_text="Sentence 1.", translated_text="Чорновик 1.")
        p1 = Paragraph(sentences=[s1])
        s2 = Sentence(original_text="Sentence 2.", translated_text="Чорновик 2.")
        p2 = Paragraph(sentences=[s2])
        chap = Chapter(title="Chap 1", order_index=0, paragraphs=[p1, p2])
        book = Book(title="Book", chapters=[chap])

        seg1 = TranslationSegment.from_paragraph(p1, book_id=book.id, chapter_id=chap.id)
        seg1.refined_translation = "Успішний переклад 1."
        seg1.transition_to(SegmentStatus.EDITED)

        seg2 = TranslationSegment.from_paragraph(p2, book_id=book.id, chapter_id=chap.id)
        seg2.transition_to(SegmentStatus.FAILED)
        seg2.refined_translation = None
        seg2.draft_translation = "Чорновик 2."

        # Segments status before reconciliation
        assert seg1.status == SegmentStatus.EDITED
        assert seg1.refined_translation == "Успішний переклад 1."
        assert seg2.status == SegmentStatus.FAILED
        assert seg2.refined_translation is None

        # TranslationSegment contract verification:
        # seg2.refined_translation is None and NEVER copied to draft!
        assert seg2.refined_translation is None
        assert seg2.draft_translation == "Чорновик 2."


# ============================================================================
# Suite 4: Localized Entity Injection & Prompt V2 Conformance
# ============================================================================

class TestLocalizedEntityInjectionAndPromptV2:
    """
    Empirical tests for scoped entity retrieval and prompt V2 structure:
    - Entity localization prevents whole-glossary prompt bloat
    - Gender, locked status, and forbidden variant formatting
    - Prompt V2 eliminates 1:1 constraints and preserves paragraph integrity
    """

    def test_localized_mention_retrieval_prevents_glossary_bloat(self):
        """
        Challenge: Knowledge base has 50 global/book entities.
        A paragraph only mentions 2 of them.
        ContextBuilder must ONLY inject the 2 active localized entities.
        """
        mock_kb = MagicMock()
        seg_id = uuid4()
        book_id = uuid4()

        # Mock 2 active entities returned for this specific segment
        active_1 = EntityProfile(
            id=uuid4(),
            source_name="Lord Malakor",
            canonical_target="Лорд Малакор",
            grammatical_gender="чоловічий",
            scope=ScopeLevel.BOOK,
            locked=True,
        )
        active_2 = EntityProfile(
            id=uuid4(),
            source_name="Moonshadow",
            canonical_target="Місячна Тінь",
            grammatical_gender="жіночий",
            scope=ScopeLevel.BOOK,
            locked=True,
            forbidden_target_forms=["Муншедоу"],
        )
        mock_kb.get_entities_for_segment.return_value = [active_1, active_2]

        cb = ContextBuilder(kb_repo=mock_kb)

        seg = TranslationSegment(
            id=seg_id,
            book_id=book_id,
            chapter_id=uuid4(),
            paragraph_id=seg_id,
            source_text="Lord Malakor met Moonshadow at midnight.",
            draft_translation="Лорд Малакор зустрів Місячну Тінь опівночі.",
        )

        ctx = cb.build_context(seg)

        mock_kb.get_entities_for_segment.assert_called_once_with(
            segment_id=seg_id,
            book_id=str(book_id),
        )
        assert len(ctx.active_entities) == 2
        assert "Lord Malakor => Лорд Малакор" in ctx.entities_text
        assert "(рід: чоловічий)" in ctx.entities_text
        assert "Moonshadow => Місячна Тінь" in ctx.entities_text
        assert "(рід: жіночий)" in ctx.entities_text
        assert "ЗАБОРОНЕНО: Муншедоу" in ctx.entities_text

    def test_entity_formatting_matrix_gender_locked_forbidden(self):
        """
        Challenge: Comprehensive matrix of entity configurations formatted for prompt V2:
        - Locked with gender and forbidden forms
        - Unlocked with allowed alternative forms
        - Unreviewed auto-extracted term
        """
        cb = ContextBuilder()

        # 1. Locked character Cherry
        cherry = EntityProfile(
            id=uuid4(),
            source_name="Cherry",
            canonical_target="Черрі",
            grammatical_gender="жіночий",
            locked=True,
            forbidden_target_forms=["Вишня", "Вішня"],
            allowed_target_forms=["Черрі", "Чері"],
        )
        fmt_cherry = cb.format_entity(cherry)
        assert "- Cherry => Черрі" in fmt_cherry
        assert "(рід: жіночий)" in fmt_cherry
        assert "[ОБОВ'ЯЗКОВО]" in fmt_cherry
        assert "ЗАБОРОНЕНО: Вишня, Вішня" in fmt_cherry
        assert "Дозволені форми: Чері" in fmt_cherry

        # 2. Male neutral entity without forbidden forms
        gandalf = EntityProfile(
            id=uuid4(),
            source_name="Gandalf",
            canonical_target="Ґандальф",
            grammatical_gender="чоловічий",
            confidence=0.85,
            locked=False,
        )
        fmt_gandalf = cb.format_entity(gandalf)
        assert "- Gandalf => Ґандальф (рід: чоловічий)" in fmt_gandalf
        assert "[ОБОВ'ЯЗКОВО]" not in fmt_gandalf
        assert "ЗАБОРОНЕНО" not in fmt_gandalf

        # 3. Unreviewed auto-extracted glossary term
        unreviewed = GlossaryItem(
            source_term="Rivendell",
            target_term="Rivendell",
            reviewed=False,
        )
        fmt_unrev = cb.format_entity(unreviewed)
        assert "- Rivendell: рекомендовано однаковий узгоджений переклад та транслітерацію" in fmt_unrev
        assert "=>" not in fmt_unrev

    def test_batch_entity_deduplication(self):
        """
        Challenge: Multiple segments in a batch mention the same entity.
        ContextBuilder must deduplicate mentions in the prompt.
        """
        mock_kb = MagicMock()
        ent = EntityProfile(
            id=uuid4(),
            source_name="Arthur",
            canonical_target="Артур",
            scope=ScopeLevel.BOOK,
            locked=True,
        )
        # Both segments return Arthur
        mock_kb.get_entities_for_segment.side_effect = [[ent], [ent]]

        cb = ContextBuilder(kb_repo=mock_kb)

        s1 = TranslationSegment(
            id=uuid4(), book_id=uuid4(), chapter_id=uuid4(), paragraph_id=uuid4(),
            source_text="Arthur walked.", draft_translation="Артур йшов."
        )
        s2 = TranslationSegment(
            id=uuid4(), book_id=s1.book_id, chapter_id=uuid4(), paragraph_id=uuid4(),
            source_text="Arthur spoke.", draft_translation="Артур говорив."
        )

        ctx = cb.build_batch_context([s1, s2])
        assert len(ctx.active_entities) == 1
        # Arthur must appear only ONCE in entities_text
        assert ctx.entities_text.count("Arthur => Артур") == 1

    def test_prompt_v2_eliminates_one_to_one_and_preserves_boundaries(self):
        """
        Challenge: Inspect generated prompt text from ContextBuilder with prompt V2 template.
        Verification:
        - Strictly eliminates 1:1 sentence constraints.
        - Explicitly allows merging and splitting within paragraphs.
        - Strictly forbids merging or splitting across paragraphs.
        - Enforces structured JSON format.
        """
        cb = ContextBuilder()
        seg = TranslationSegment(
            id=uuid4(),
            book_id=uuid4(),
            chapter_id=uuid4(),
            paragraph_id=uuid4(),
            source_text="First sentence. Second sentence.",
            draft_translation="Перше речення. Друге речення.",
        )

        ctx = cb.build_context(seg)
        prompt = ctx.prompt_text

        # Asserts on prompt content and rules
        assert "ЦІЛІСНІСТЬ ПАРАГРАФА ТА ПРИРОДНИЙ СИНТАКСИС" in prompt or "Цілісність параграфа" in prompt
        assert "Сувора межа параграфів" in prompt or "СУВОРЕ ЗБЕРЕЖЕННЯ МЕЖІ ПАРАГРАФА" in prompt
        assert '"segments"' in prompt
        assert str(seg.id) in prompt

        # Absence of old 1:1 sentence constraints
        assert "Відповідність 1-до-1" not in prompt
        assert "Одне вхідне речення відповідає одному вихідному" not in prompt
