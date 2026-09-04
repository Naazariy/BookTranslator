"""
ContextBuilder for Literary Translation (Milestone 3).
Dynamically assembles contextual prompt components for paragraph-level literary translation:
integrates source paragraph, NLLB draft, approved Ukrainian history, scoped entities,
and chapter summary into a typed PromptContext governed by TokenBudget.
"""
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from pathlib import Path
import logging
from pydantic import BaseModel, Field

from src.domain.models.segment import TranslationSegment, SegmentStatus
from src.domain.models.knowledge import EntityProfile, ScopeLevel, GlossaryItem
from src.domain.interfaces.knowledge_base import IKnowledgeBaseRepository
from src.context.token_budget import TokenBudget, BudgetAllocation

logger = logging.getLogger(__name__)


class PromptContext(BaseModel):
    """
    Complete assembled context for Stage 2 LLM generation adhering to TokenBudget.
    """
    segment_id: UUID
    segment_ids: List[UUID] = Field(default_factory=list)
    prompt_text: str
    target_source: str
    target_draft: str
    previous_context: str
    entities_text: str
    chapter_summary: Optional[str] = None
    active_entities: List[Any] = Field(default_factory=list)
    token_breakdown: Dict[str, int] = Field(default_factory=dict)
    budget_allocation: Optional[Any] = None


class ContextBuilder:
    """
    Dynamically assembles contextual prompt components for paragraph-level literary translation.
    Integrates source paragraph, NLLB draft, approved history, scoped entities, and chapter summary.
    """
    DEFAULT_PROMPT_PATH = Path("data/prompts/editing_prompt_v2.md")

    def __init__(
        self,
        kb_repo: Optional[IKnowledgeBaseRepository] = None,
        token_budget: Optional[TokenBudget] = None,
        prompt_path: Optional[Path] = None,
        prompt_template: Optional[str] = None,
    ):
        self.kb_repo = kb_repo
        self.token_budget = token_budget or TokenBudget()
        self.prompt_path = prompt_path or self.DEFAULT_PROMPT_PATH
        self._cached_prompt_template: str = ""

        if prompt_template:
            self._cached_prompt_template = prompt_template
        else:
            self._load_prompt_template()

    def _load_prompt_template(self) -> None:
        """Loads prompt template from disk or uses high-fidelity literary default."""
        target_path = None
        if self.prompt_path:
            p = Path(self.prompt_path)
            if p.exists() and p.is_file():
                target_path = p
            else:
                project_root = Path(__file__).resolve().parent.parent.parent
                if (project_root / p).exists():
                    target_path = project_root / p

        if target_path and target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    self._cached_prompt_template = f.read()
                logger.info(f"Loaded ContextBuilder prompt template from {target_path}")
                return
            except Exception as e:
                logger.warning(f"Could not read prompt template from {target_path}: {e}")

        self._cached_prompt_template = self._get_default_fallback_template()

    @staticmethod
    def _get_default_fallback_template() -> str:
        return (
            "# Інструкція для Літературного Редактора (Aya LLM) — Версія 2 (Параграфи)\n\n"
            "Ти — провідний художній перекладач та літературний редактор художньої прози з англійської на українську мову.\n"
            "Твоє завдання — здійснити стилістично довершений літературний переклад повного параграфа художнього тексту.\n\n"
            "## ПРАВИЛА ЛІТЕРАТУРНОГО РЕДАГУВАННЯ:\n"
            "1. **Цілісність параграфа та природний синтаксис**: Параграф є неподільною змістовою та ритмічною одиницею. "
            "Тобі дозволено та рекомендовано природно об'єднувати короткі рубані речення або розбивати громіздкі складні речення всередині параграфа, "
            "використовувати питомий український порядок слів та мелодійні звороти.\n"
            "2. **Сувора межа параграфів**: Заборонено об'єднувати різні параграфи або розбивати один параграф на кілька записів. "
            "Один вхідний параграф відповідає рівно одному запису перекладу.\n"
            "3. **Дотримання глосарія та роду**: Всі терміни та імена зі списку «ГЛОСАРІЙ ТА СУТНОСТІ» є обов'язковими. "
            "Суворо дотримуйся зазначеного граматичного роду персонажів (дієслова минулого часу та прикметники). "
            "Категорично заборонено використовувати варіанти з позначкою ЗАБОРОНЕНО.\n"
            "4. **Формат виводу**: Надай відповідь СУВОРО у форматі JSON без markdown-огорож, коментарів чи преамбул:\n"
            '{"segments": [{"id": "<id>", "translation": "<переклад параграфа>"}]}\n\n'
            "=== ПОПЕРЕДНІЙ КОНТЕКСТ ===\n{context_previous}\n\n"
            "=== ГЛОСАРІЙ ТА СУТНОСТІ ===\n{glossary_terms}\n\n"
            "=== КОРОТКИЙ ЗМІСТ ГЛАВИ ===\n{chapter_summary}\n\n"
            "=== ПАРАГРАФ ДЛЯ РЕДАГУВАННЯ ===\n{target_segments_block}\n\n"
            "=== ВІДРЕДАГОВАНИЙ JSON ===\n"
        )

    def format_entity(self, entity: Any) -> str:
        """Formats an EntityProfile or GlossaryItem into strict, readable prompt instructions."""
        source = getattr(entity, "source_name", getattr(entity, "source_term", ""))
        canonical = getattr(entity, "canonical_target", getattr(entity, "target_term", ""))

        reviewed = getattr(entity, "reviewed", True)
        if not reviewed and source == canonical:
            # Auto-extracted unreviewed term without explicit target
            return f"- {source}: рекомендовано однаковий узгоджений переклад та транслітерацію"

        parts = [f"- {source} => {canonical}"]
        gender = getattr(entity, "grammatical_gender", None)
        if gender:
            parts.append(f"(рід: {gender})")
        if getattr(entity, "locked", False):
            parts.append("[ОБОВ'ЯЗКОВО]")
        forbidden = getattr(entity, "forbidden_target_forms", None)
        if forbidden:
            parts.append(f"| ЗАБОРОНЕНО: {', '.join(forbidden)}")
        allowed = getattr(entity, "allowed_target_forms", None)
        if allowed and len(allowed) > 1:
            other_forms = [f for f in allowed if f != canonical]
            if other_forms:
                parts.append(f"| Дозволені форми: {', '.join(other_forms)}")
        return " ".join(parts)

    def extract_approved_history(
        self,
        prev_segments: Optional[List[TranslationSegment]],
        max_paragraphs: int = 2
    ) -> List[str]:
        """Extracts the last 1–2 approved Ukrainian paragraph translations."""
        if not prev_segments:
            return []

        approved: List[str] = []
        for seg in reversed(prev_segments):
            is_approved = (
                seg.status in (SegmentStatus.ACCEPTED, SegmentStatus.EDITED) or
                bool(seg.final_translation or seg.refined_translation)
            )
            if is_approved:
                text = (seg.final_translation or seg.refined_translation or seg.draft_translation or "").strip()
                if text:
                    approved.insert(0, text)
                if len(approved) >= max_paragraphs:
                    break

        return approved

    def format_target_block(self, segments: List[TranslationSegment]) -> str:
        """Formats target paragraph block with ID, English source, and NLLB draft."""
        blocks: List[str] = []
        for s in segments:
            source = (s.normalized_text or s.source_text or "").strip()
            draft = (s.draft_translation or s.draft_text or "").strip()
            block = (
                f"[ID: {s.id}]\n"
                f"ОРИГІНАЛ (EN):\n{source}\n\n"
                f"ЧОРНОВИЙ ПЕРЕКЛАД NLLB (UK, потребує редагування):\n{draft}"
            )
            blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    def build_context(
        self,
        segment: TranslationSegment,
        prev_segments: Optional[List[TranslationSegment]] = None,
        entities: Optional[List[Any]] = None,
        summary: Optional[str] = None,
        token_budget: Optional[TokenBudget] = None,
    ) -> PromptContext:
        """Assembles PromptContext for a single TranslationSegment."""
        return self.build_batch_context(
            segments=[segment],
            prev_segments=prev_segments,
            entities=entities,
            summary=summary,
            token_budget=token_budget,
        )

    def build_batch_context(
        self,
        segments: List[TranslationSegment],
        prev_segments: Optional[List[TranslationSegment]] = None,
        entities: Optional[List[Any]] = None,
        summary: Optional[str] = None,
        token_budget: Optional[TokenBudget] = None,
    ) -> PromptContext:
        """
        Assembles PromptContext for a batch of TranslationSegments adhering to TokenBudget.
        """
        if not segments:
            raise ValueError("Cannot build PromptContext for an empty list of segments.")

        budget = token_budget or self.token_budget
        primary_seg = segments[0]

        # 1. Resolve Active Entities
        active_entities: List[Any] = []
        if entities is not None:
            active_entities = entities
        elif self.kb_repo is not None and hasattr(self.kb_repo, "get_entities_for_segment"):
            collected: Dict[str, Any] = {}
            for s in segments:
                s_entities = self.kb_repo.get_entities_for_segment(
                    segment_id=s.id,
                    book_id=str(s.book_id) if s.book_id else None
                )
                for e in s_entities:
                    name = getattr(e, "source_name", getattr(e, "source_term", "")).strip().lower()
                    if name and name not in collected:
                        collected[name] = e
            active_entities = list(collected.values())
        elif self.kb_repo is not None and hasattr(self.kb_repo, "get_glossary"):
            active_entities = self.kb_repo.get_glossary()

        # 2. Extract Approved History (1–2 paragraphs)
        raw_history = self.extract_approved_history(prev_segments, max_paragraphs=2)

        # 3. Prepare Target Text
        target_block_raw = self.format_target_block(segments)

        # 4. Allocate Budget
        scaffold_stub = self._cached_prompt_template.format(
            context_previous="",
            glossary_terms="",
            chapter_summary="",
            target_segments_block="",
        )
        fitted_target, fitted_entities, fitted_history, fitted_summary, allocation = budget.allocate(
            target_text=target_block_raw,
            entities=active_entities,
            format_entity_fn=self.format_entity,
            history_paragraphs=raw_history,
            summary=summary,
            scaffold_text=scaffold_stub,
        )

        # 5. Format Substituted Sections
        if fitted_history:
            history_lines = []
            for i, p in enumerate(fitted_history, 1):
                history_lines.append(f"[Параграф {i}]:\n{p}")
            history_str = "\n\n".join(history_lines)
        else:
            history_str = "Відсутній (це початковий параграф глави)."

        if fitted_entities:
            entity_lines = [self.format_entity(e) for e in fitted_entities]
            entities_str = "\n".join(entity_lines)
        else:
            entities_str = "Відсутній (специфічних термінів чи персонажів у цьому фрагменті не виявлено)."

        summary_str = fitted_summary if fitted_summary else "Відсутній."

        # 6. Build Final Prompt
        prompt_text = self._cached_prompt_template.format(
            context_previous=history_str,
            glossary_terms=entities_str,
            chapter_summary=summary_str,
            target_segments_block=fitted_target,
        )

        breakdown = {
            "total_used": allocation.total_used_tokens,
            "max_allowed": allocation.max_prompt_tokens,
            "target": allocation.target_tokens,
            "entities": allocation.entities_tokens,
            "history": allocation.history_tokens,
            "summary": allocation.summary_tokens,
            "scaffold": allocation.scaffold_tokens,
        }

        return PromptContext(
            segment_id=primary_seg.id,
            segment_ids=[s.id for s in segments],
            prompt_text=prompt_text,
            target_source=" ".join(s.source_text for s in segments),
            target_draft=" ".join(s.draft_translation or s.draft_text or "" for s in segments),
            previous_context=history_str,
            entities_text=entities_str,
            chapter_summary=fitted_summary,
            active_entities=fitted_entities,
            token_breakdown=breakdown,
            budget_allocation=allocation,
        )
