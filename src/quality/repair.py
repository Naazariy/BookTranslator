"""
src/quality/repair.py

Bounded Repair Engine for BookTranslator V2.
Orchestrates targeted LLM repair passes for translation segments failing QA validation.
Enforces a strict upper bound of 2 retry attempts and explicit SegmentStatus lifecycle transitions.
"""

from __future__ import annotations
import json
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from uuid import UUID

from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.quality.models import QAReport, QAViolation, QASeverity

logger = logging.getLogger(__name__)


class BoundedRepairEngine:
    """
    Orchestrates targeted repair loops for translation segments.
    Retries at most max_retries (default: 2) times using targeted QA violation diagnostics.
    Transitions segment status:
      - Clean / Fixed -> SegmentStatus.ACCEPTED (final_translation populated)
      - Still failing after max_retries -> SegmentStatus.REVIEW_REQUIRED
    """

    def __init__(
        self,
        editing_engine: Optional[Any] = None,
        quality_pipeline: Optional[Any] = None,
        max_retries: int = 2,
    ):
        self.editing_engine = editing_engine
        self.quality_pipeline = quality_pipeline
        self.max_retries = max_retries

    def synthesize_repair_prompt(
        self,
        segment: TranslationSegment,
        report: QAReport,
        current_translation: str,
        context: Optional[Any] = None,
    ) -> str:
        """
        Synthesizes a targeted remediation prompt emphasizing exact diagnostics,
        forbidden forms, and required fixes while preserving paragraph context.
        """
        violations_lines: List[str] = []
        for i, v in enumerate(report.violations, 1):
            line = f"{i}. [Перевірка: {v.validator_name}] {v.message}"
            if v.target_snippet:
                line += f'\n   - Проблемний фрагмент: "{v.target_snippet}"'
            if v.forbidden_form:
                line += f'\n   - ЗАБОРОНЕНО вживати: "{v.forbidden_form}"'
            if v.suggested_fix:
                line += f'\n   - Обов\'язковий відповідник: "{v.suggested_fix}"'
            violations_lines.append(line)

        violations_block = "\n".join(violations_lines) if violations_lines else "Помилки перекладу."

        prompt = (
            "# ІНСТРУКЦІЯ ДЛЯ ЛІТЕРАТУРНОГО КОРЕКТОРА (ЕТАП ВИПРАВЛЕННЯ ПОМИЛОК)\n\n"
            "Ти — провідний літературний редактор та коректор українського перекладу художньої літератури.\n"
            "Твоє завдання — ТОЧКОВО ВИПРАВИТИ конкретні порушення якості в наданому перекладі параграфа.\n\n"
            "## ПРАВИЛА ВИПРАВЛЕННЯ:\n"
            "1. **Точкове усунення помилок**: Виправ ВИКЛЮЧНО вказані нижче зауваження. Не переписуй вдалі речення чи зміст без потреби.\n"
            "2. **Категорична заборона**: Суворо заборонено вживати будь-які форми зі списку «ЗАБОРОНЕНО».\n"
            "3. **Обов'язкові форми**: Використовуй канонічні відповідники імен персонажів, термінів та зазначений граматичний рід.\n"
            "4. **Незмінність змісту**: Англійський оригінал наведено виключно для звірки смислу — не пропускай речень.\n"
            "5. **Суворий формат JSON**: Надай відповідь СУВОРО у форматі JSON без markdown-огорож чи пояснень:\n"
            "{\n"
            '  "segments": [\n'
            "    {\n"
            f'      "id": "{str(segment.id)}",\n'
            '      "translation": "<виправлений текст українською мовою>"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "---\n"
            "## ОРИГІНАЛЬНИЙ АНГЛІЙСЬКИЙ ТЕКСТ (НЕДОТОРКАНИЙ):\n"
            f"{segment.source_text}\n\n"
            "## ПОТОЧНИЙ ПЕРЕКЛАД З ПОМИЛКАМИ:\n"
            f"{current_translation}\n\n"
            "## ВИЯВЛЕНІ ПОРУШЕННЯ ЯКОСТІ:\n"
            f"{violations_block}\n"
            "---\n"
        )
        return prompt

    def repair_segment(
        self,
        segment: TranslationSegment,
        report: QAReport,
        context: Optional[Any] = None,
        editing_engine: Optional[Any] = None,
        quality_pipeline: Optional[Any] = None,
        cancel_token: Optional[Any] = None,
    ) -> Tuple[TranslationSegment, QAReport]:
        """
        Executes bounded repair loop for a single TranslationSegment.
        Attempts at most self.max_retries repairs.
        Transitions:
          - If repaired and passes QA: SegmentStatus.ACCEPTED, final_translation set
          - If still failing after max_retries: SegmentStatus.REVIEW_REQUIRED
        """
        engine = editing_engine or self.editing_engine
        qp = quality_pipeline or self.quality_pipeline

        if engine is None or qp is None:
            logger.warning(
                f"Repair engine or quality pipeline missing for segment {segment.id}; marking REVIEW_REQUIRED."
            )
            if segment.status != SegmentStatus.REVIEW_REQUIRED:
                try:
                    segment.transition_to(SegmentStatus.REVIEW_REQUIRED)
                except Exception:
                    segment.status = SegmentStatus.REVIEW_REQUIRED
            report.status = SegmentStatus.REVIEW_REQUIRED.value
            return segment, report

        current_candidate = (
            segment.final_translation or
            segment.refined_translation or
            segment.draft_translation or
            ""
        )
        current_report = report

        # Ensure segment is in VALIDATING state
        if segment.status != SegmentStatus.VALIDATING:
            try:
                segment.transition_to(SegmentStatus.VALIDATING)
            except Exception:
                segment.status = SegmentStatus.VALIDATING

        for attempt in range(1, self.max_retries + 1):
            if cancel_token and getattr(cancel_token, "is_cancelled", False):
                logger.info(f"Repair aborted due to cancellation on attempt {attempt}")
                break

            segment.repair_attempts = attempt
            current_report.repair_attempts = attempt
            logger.info(
                f"Executing repair attempt {attempt}/{self.max_retries} for segment {segment.id}"
            )

            repair_prompt = self.synthesize_repair_prompt(
                segment=segment,
                report=current_report,
                current_translation=current_candidate,
                context=context,
            )

            # Generate repaired candidate via editing engine
            repaired_text = self._request_llm_repair(
                engine=engine,
                segment=segment,
                repair_prompt=repair_prompt,
                cancel_token=cancel_token,
            )

            if not repaired_text or not repaired_text.strip():
                logger.warning(
                    f"Repair attempt {attempt} returned empty translation for segment {segment.id}"
                )
                continue

            current_candidate = repaired_text.strip()

            # Temporarily attach candidate to test through QualityPipeline
            test_seg = segment.model_copy()
            test_seg.refined_translation = current_candidate
            test_seg.final_translation = current_candidate

            new_report = qp.validate_segment(test_seg, context=context)
            new_report.repair_attempts = attempt

            if new_report.is_valid:
                logger.info(f"Segment {segment.id} successfully repaired on attempt {attempt}")
                segment.final_translation = current_candidate
                segment.refined_translation = current_candidate
                segment.transition_to(SegmentStatus.ACCEPTED)
                segment.error_message = None
                new_report.status = SegmentStatus.ACCEPTED.value
                return segment, new_report

            current_report = new_report

        # If loop exhausts without passing QA
        logger.warning(
            f"Segment {segment.id} failed repair after {segment.repair_attempts} attempts. "
            f"Transitioning to REVIEW_REQUIRED."
        )
        segment.final_translation = current_candidate
        segment.transition_to(SegmentStatus.REVIEW_REQUIRED)
        violations_summary = "; ".join(v.message for v in current_report.violations)
        segment.error_message = (
            f"Quality validation failed after {segment.repair_attempts} repair attempts. "
            f"Violations: {violations_summary}"
        )
        current_report.status = SegmentStatus.REVIEW_REQUIRED.value
        return segment, current_report

    def _request_llm_repair(
        self,
        engine: Any,
        segment: TranslationSegment,
        repair_prompt: str,
        cancel_token: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Dispatches repair prompt to the editing engine and extracts repaired translation.
        """
        # If engine is a callable (mock / lambda in tests)
        if callable(engine):
            try:
                res = engine(segment, repair_prompt)
                if isinstance(res, str):
                    return res
                if isinstance(res, dict):
                    return res.get(segment.id) or res.get(str(segment.id)) or res.get("translation")
            except Exception as e:
                logger.error(f"Callable repair engine error: {e}")
                return None

        # If engine has refine_segments_structured
        if hasattr(engine, "refine_segments_structured"):
            try:
                from src.context.builder import PromptContext
                ctx = PromptContext(
                    segment_id=segment.id,
                    prompt_text=repair_prompt,
                    target_source=segment.source_text,
                    target_draft=segment.draft_translation or "",
                    previous_context="",
                    entities_text="",
                )
                res_dict = engine.refine_segments_structured(
                    segments=[segment],
                    prompt_context=ctx,
                    cancel_token=cancel_token,
                )
                res = res_dict.get(segment.id) or res_dict.get(str(segment.id))
                if res and hasattr(res, "refined_text") and res.refined_text:
                    return res.refined_text
                elif res and isinstance(res, str) and res.strip():
                    return res.strip()
            except Exception as e:
                logger.error(f"Engine refine_segments_structured error: {e}")

        # If engine has refine_chunk_structured or refine_chunk
        if hasattr(engine, "refine_chunk_structured"):
            try:
                res = engine.refine_chunk_structured(
                    chunk_text=repair_prompt,
                    context=None,
                    glossary=None,
                    cancel_token=cancel_token,
                )
                if hasattr(res, "refined_text") and res.refined_text:
                    return res.refined_text
                elif isinstance(res, str):
                    return res
            except Exception as e:
                logger.error(f"Engine refine_chunk_structured error: {e}")

        return None
