"""
Quantized Aya Editing Engine for Stage 2 Literary Refinement.
Utilizes BitsAndBytes 4-bit NF4 double quantization with GGUF / CPU fallback,
in-memory prompt template caching, dynamic token ceiling, and cooperative cancellation.
"""
from typing import List, Optional, Any, Dict
from pathlib import Path
import logging
import gc
import re

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteriaList
    )
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    StoppingCriteriaList = None

from src.domain.interfaces.translation import IEditingEngine
from src.domain.models.chunk import ChunkContext
from src.domain.models.knowledge import GlossaryItem
from src.translation.stopping_criteria import CancellationTokenStoppingCriteria
from src.config.settings import settings

logger = logging.getLogger(__name__)


from src.quality.pipeline import (
    LATIN_TO_CYRILLIC_HOMOGLYPHS,
    KNOWN_PORTMANTEAU_FIXES,
    sanitize_mixed_script_words,
)



def _ensure_llama_cpp():
    """Dynamically resolves and imports Llama from llama_cpp, probing site-packages if needed."""
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError:
        import site
        import sys
        import os
        candidates = []
        user_appdata = os.environ.get("LOCALAPPDATA", "")
        if user_appdata:
            candidates.append(Path(user_appdata) / "Python" / "pythoncore-3.14-64" / "Lib" / "site-packages")
            candidates.append(Path(user_appdata) / "Programs" / "Python" / "Python314" / "Lib" / "site-packages")
            candidates.append(Path(user_appdata) / "Programs" / "Python" / "Python312" / "Lib" / "site-packages")

        if hasattr(site, "getsitepackages"):
            try:
                for sp in site.getsitepackages():
                    candidates.append(Path(sp))
            except Exception:
                pass

        for cand in candidates:
            if cand.exists() and str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
                try:
                    from llama_cpp import Llama
                    return Llama
                except ImportError:
                    pass
        return None


def _is_cancelled(cancel_token: Optional[Any]) -> bool:
    if cancel_token is None:
        return False
    if hasattr(cancel_token, "is_cancelled"):
        checker = getattr(cancel_token, "is_cancelled")
        return checker() if callable(checker) else bool(checker)
    if hasattr(cancel_token, "is_set"):
        return cancel_token.is_set()
    if hasattr(cancel_token, "cancelled"):
        return bool(cancel_token.cancelled)
    return False


class QuantizedAyaEditingEngine(IEditingEngine):
    """
    Literary editing engine supporting BitsAndBytes 4-bit (NF4) quantization,
    GGUF fallback, in-memory prompt template caching, and cooperative cancellation.
    """
    def __init__(
        self,
        model_name: str = "CohereForAI/aya-23-8B",
        device: str = "auto",
        prompt_path: Optional[Path] = None,
        load_in_4bit: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = 42
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.prompt_path = prompt_path or getattr(settings, "prompt_file_path", Path("data/prompts/editing_prompt.md"))
        self.load_in_4bit = load_in_4bit and (self.device == "cuda")
        self.temperature = temperature if temperature is not None else getattr(settings, "aya_temperature", 0.2)
        self.top_p = top_p if top_p is not None else getattr(settings, "aya_top_p", 0.9)
        self.repetition_penalty = repetition_penalty if repetition_penalty is not None else getattr(settings, "aya_repetition_penalty", 1.03)
        self.seed = seed if seed is not None else getattr(settings, "aya_seed", 42)

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.gguf_llm: Optional[Any] = None
        self.cached_prompt_template: str = ""

        self._load_cached_prompt()

    def _resolve_device(self, requested: str) -> str:
        if requested in ("cuda", "cpu"):
            if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
                logger.warning("CUDA requested for Aya but unavailable. Falling back to CPU.")
                return "cpu"
            return requested
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_cached_prompt(self) -> None:
        """Loads prompt template from disk into memory once to eliminate per-chunk disk I/O."""
        project_root = Path(__file__).resolve().parent.parent.parent
        target_path = None
        if self.prompt_path:
            p = Path(self.prompt_path)
            if p.exists() and p.is_file():
                target_path = p
            elif not p.is_absolute() and (project_root / p).exists():
                target_path = project_root / p

        if target_path is not None:
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    self.cached_prompt_template = f.read()
                logger.info(f"Loaded prompt template into memory from '{target_path}'.")
                return
            except Exception as e:
                logger.warning(f"Failed to read prompt file from '{target_path}': {e}")

        # High-quality literary editing fallback template
        self.cached_prompt_template = (
            "# Інструкція для Літературного Редактора (Aya LLM)\n\n"
            "Ти — провідний літературний редактор та художній перекладач художньої прози з англійської на українську мову.\n"
            "Тобі надано:\n"
            "1. Попередній контекст ({context_previous}).\n"
            "2. Обов'язковий глосарій термінів та сутностей ({glossary_terms}).\n"
            "3. Список речень для редагування з оригінальним текстом та чорновим машинним перекладом:\n"
            "{target_sentences_block}\n\n"
            "Твоє завдання — відредагувати чорновий переклад кожного речення, створивши довершений, стилістично багатий, природний український літературний текст.\n\n"
            "## СУВОРІ ПРАВИЛА РЕДАГУВАННЯ:\n"
            "1. **Точність змісту**: Зберігай 100% змісту оригіналу. Заборонено пропускати речення або додавати інформацію, якої немає в оригіналі.\n"
            "2. **Відповідність 1-до-1**: Одне вхідне речення відповідає одному відредагованому реченню. Не об'єднуй і не розбивай речення.\n"
            "3. **Дотримання глосарія**: Якщо термін чи ім'я є в списку «ГЛОСАРІЙ» — використовуй ВИКЛЮЧНО вказаний переклад без самовільних змін. Формат обов'язкових записів: `- {{source}} => {{target}}` (з можливим зазначенням роду). Для імен зі списку рекомендацій дотримуйся єдиного перекладу/транслітерації без нав'язування заглушки.\n"
            "4. **Одиниці виміру**: Дотримуйся точних метричних еквівалентів (80 feet = 24 метри / 24,4 м, 15 feet = 4,6 м, 100 miles = 161 км).\n"
            "5. **Чистота мови та заборони**:\n"
            "   - Заборонено змішувати латиницю та кирилицю всередині слів (наприклад, \"Смачнissimo\").\n"
            "   - Заборонено додавати іноземні афікси чи суфікси (-issimo, -able тощо) до українських коренів.\n"
            "   - Використовуй виключно чисту, автентичну українську лексику та усталені літературні норми.\n"
            "6. **Теги форматування**: Збережи всі спеціальні теги форматування у вихідному тексті на їхніх точних семантичних місцях. Не вигадуй нових тегів.\n"
            "7. **Формат виводу**: Надай відповідь СУВОРО у форматі JSON без жодних markdown-огорож (без ```json ... ```), без привітань, пояснень та додаткового тексту. Ключі — порядкові номери речень як рядки, значення — фінальний український переклад:\n"
            "{{\"1\": \"відредаговане речення 1\", \"2\": \"відредаговане речення 2\"}}\n\n"
            "=== ПОПЕРЕДНІЙ КОНТЕКСТ ===\n"
            "{context_previous}\n\n"
            "=== ГЛОСАРІЙ ТА СУТНОСТІ ===\n"
            "{glossary_terms}\n\n"
            "=== РЕЧЕННЯ ДЛЯ РЕДАГУВАННЯ ===\n"
            "{target_sentences_block}\n\n"
            "=== ВІДРЕДАГОВАНИЙ JSON ===\n"
        )

    def reload_prompt(self) -> None:
        """Forces reloading the prompt template from disk."""
        self._load_cached_prompt()

    @property
    def is_loaded(self) -> bool:
        """Returns True if GGUF LLM or PyTorch model is loaded in memory."""
        return self.model is not None or self.gguf_llm is not None

    def _find_gguf_path(self) -> Optional[Path]:
        """Resolves model_name to a local GGUF file path if available."""
        # 1. Direct file path
        p = Path(self.model_name)
        if p.exists() and p.is_file():
            return p.resolve()

        project_root = Path(__file__).resolve().parent.parent.parent

        # Check direct relative to project root
        p_root = project_root / self.model_name
        if p_root.exists() and p_root.is_file():
            return p_root.resolve()

        # 2. Check candidate directories for GGUF files (anchored both locally and to project root)
        raw_candidate_dirs = [
            Path("models/gguf"),
            Path("models"),
            Path("hf_cache"),
            Path("data/models"),
            Path("models/ctranslate2"),
            Path("scratch"),
        ]

        candidate_dirs = []
        for cd in raw_candidate_dirs:
            if cd not in candidate_dirs:
                candidate_dirs.append(cd)
            p_cd = project_root / cd
            if p_cd not in candidate_dirs:
                candidate_dirs.append(p_cd)

        clean_name = str(self.model_name).lower().replace("/", "_").replace("\\", "_")

        # Check direct filename match in candidate directories
        base_name = Path(self.model_name).name
        for cand_dir in candidate_dirs:
            if not cand_dir.exists():
                continue
            for name_variant in [base_name, f"{base_name}.gguf"]:
                cand_file = cand_dir / name_variant
                if cand_file.exists() and cand_file.is_file():
                    return cand_file.resolve()

        # Build candidate patterns based on model family and requested quantization
        preferred_matches = []
        if "aya-expanse" in clean_name or "aya_expanse" in clean_name:
            all_expanse_variants = [
                "aya-expanse-8b-Q4_K_M.gguf",
                "aya-expanse-8b-q8_0.gguf",
                "aya-expanse-8b-Q4_0.gguf",
                "aya-expanse-8b-IQ4_XS.gguf",
                "aya-expanse-8b-IQ2_M.gguf",
                "aya_expanse_8b_q8_0.gguf",
            ]
            if "q4_k_m" in clean_name or "q4_k" in clean_name:
                preferred_matches = ["aya-expanse-8b-Q4_K_M.gguf"] + [v for v in all_expanse_variants if "q4_k_m" not in v.lower()]
            elif "q8" in clean_name:
                preferred_matches = ["aya-expanse-8b-q8_0.gguf", "aya_expanse_8b_q8_0.gguf"] + [v for v in all_expanse_variants if "q8" not in v.lower()]
            elif "q4_0" in clean_name:
                preferred_matches = ["aya-expanse-8b-Q4_0.gguf"] + [v for v in all_expanse_variants if "q4_0" not in v.lower()]
            else:
                preferred_matches = all_expanse_variants

        elif "aya-23" in clean_name or "aya_23" in clean_name:
            all_aya23_variants = [
                "aya-23-8b-Q4_K_M.gguf",
                "aya-23-8b-q8_0.gguf",
                "aya-23-8b-Q4_0.gguf",
                "aya_23_8b_q8_0.gguf",
            ]
            if "q4" in clean_name:
                preferred_matches = ["aya-23-8b-Q4_K_M.gguf", "aya-23-8b-Q4_0.gguf"] + [v for v in all_aya23_variants if "q4" not in v.lower()]
            elif "q8" in clean_name:
                preferred_matches = ["aya-23-8b-q8_0.gguf", "aya_23_8b_q8_0.gguf"] + [v for v in all_aya23_variants if "q8" not in v.lower()]
            else:
                preferred_matches = all_aya23_variants

        for cand_dir in candidate_dirs:
            if not cand_dir.exists():
                continue
            for pref in preferred_matches:
                target = cand_dir / pref
                if target.exists() and target.is_file():
                    return target.resolve()

            # Scoped recursive search only for matching model family
            try:
                for gguf_file in cand_dir.rglob("*.gguf"):
                    if not gguf_file.is_file():
                        continue
                    fname_lower = gguf_file.name.lower()
                    if ("aya-expanse" in clean_name or "aya_expanse" in clean_name) and ("aya-expanse" in fname_lower or "aya_expanse" in fname_lower):
                        return gguf_file.resolve()
                    if ("aya-23" in clean_name or "aya_23" in clean_name) and ("aya-23" in fname_lower or "aya_23" in fname_lower):
                        return gguf_file.resolve()
            except Exception:
                pass

        return None

    def load_model(self) -> None:
        if self.model is not None or self.gguf_llm is not None:
            return

        effective_seed = self.seed if self.seed is not None else 42
        if torch is not None and self.seed is not None:
            torch.manual_seed(effective_seed)

        # 1. Check if model can be loaded via GGUF (llama-cpp-python)
        gguf_path = self._find_gguf_path()
        if gguf_path is not None:
            try:
                LlamaClass = _ensure_llama_cpp()
                if LlamaClass is None:
                    raise ImportError("llama_cpp module could not be imported")

                logger.info(f"Loading GGUF model from '{gguf_path}' (device='{self.device}')...")
                n_gpu_layers = -1 if self.device == "cuda" else 0
                try:
                    self.gguf_llm = LlamaClass(
                        model_path=str(gguf_path.resolve()),
                        n_ctx=4096,
                        n_gpu_layers=n_gpu_layers,
                        seed=effective_seed,
                        verbose=False
                    )
                except Exception as gpu_err:
                    if n_gpu_layers != 0:
                        logger.warning(f"GGUF GPU acceleration failed ({gpu_err}); falling back to GGUF CPU mode.")
                        self.gguf_llm = LlamaClass(
                            model_path=str(gguf_path.resolve()),
                            n_ctx=4096,
                            n_gpu_layers=0,
                            seed=effective_seed,
                            verbose=False
                        )
                    else:
                        raise gpu_err
                logger.info(f"GGUF model successfully loaded from '{gguf_path}'.")
                return
            except Exception as e:
                logger.warning(f"Failed to load GGUF model from '{gguf_path}': {e}. Falling back to Hugging Face loader.")

        # 2. Hugging Face Model Loading Path
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers and torch are required to load QuantizedAyaEditingEngine.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=getattr(settings, "offline_mode", False),
            use_fast=False
        )

        logger.info(
            f"Loading Aya Model '{self.model_name}' (4-bit={self.load_in_4bit}, device='{self.device}')..."
        )

        if self.load_in_4bit and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                compute_dtype = torch.bfloat16 if (torch is not None and torch.cuda.is_bf16_supported()) else torch.float16
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map={"": self.device},
                    low_cpu_mem_usage=True,
                    local_files_only=getattr(settings, "offline_mode", False)
                )
                logger.info("Aya 4-bit (NF4 Double Quant) model successfully loaded into VRAM (~4.8GB).")
                return
            except Exception as e:
                logger.error(f"BitsAndBytes 4-bit loading failed: {e}.")
                raise RuntimeError(f"Failed to load model in 4-bit mode. Check if bitsandbytes is installed correctly and your GPU is supported. Details: {e}")

        # Standard precision / CPU fallback (Only if 4-bit was NOT requested or device is CPU)
        if self.device == "cuda" and torch is not None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32 if torch is not None else None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=getattr(settings, "offline_mode", False)
        )
        if torch is not None and hasattr(self.model, "to") and self.device != "auto":
            self.model = self.model.to(self.device)
        logger.info("Aya model successfully loaded in standard mode.")

    def unload_model(self) -> None:
        logger.info("Unloading Aya Editing Engine from memory...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.gguf_llm is not None:
            del self.gguf_llm
            self.gguf_llm = None

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Aya Editing Engine unloaded successfully.")

    def _sanitize_output(self, text: str) -> str:
        """
        Strips markdown code fences, prompt headers, conversational preambles,
        markdown headers, model control tokens, inline bold/italic formatting,
        unmapped inline tags, mixed-script artifacts, and wrapping quotes.
        Uses multi-pass normalization until convergence.
        """
        if not text:
            return ""

        cleaned = re.sub(r"^[\s\u200b\u200c\u200d\ufeff]+|[\s\u200b\u200c\u200d\ufeff]+$", "", text)

        # Pass 0: Extract after explicit section headers if echoed
        section_markers = [
            "=== ВІДРЕДАГОВАНИЙ JSON ===:",
            "=== ВІДРЕДАГОВАНИЙ JSON ===",
            "=== REFINED TRANSLATION ===:",
            "=== REFINED TRANSLATION ===",
            "=== ВІДРЕДАГОВАНИЙ ТЕКСТ ===:",
            "=== ВІДРЕДАГОВАНИЙ ТЕКСТ ===",
            "=== ФІНАЛЬНИЙ ПЕРЕКЛАД ===:",
            "=== ФІНАЛЬНИЙ ПЕРЕКЛАД ===",
            "## Фінальний покращений переклад:",
            "### Фінальний покращений переклад:",
            "# Фінальний покращений переклад:",
            "**Фінальний переклад:**",
            "**Відредагований переклад:**",
            "Фінальний покращений переклад:",
            "Відредагований літературний переклад:",
            "REFINED TRANSLATION:",
            "Refined translation:",
        ]
        for marker in section_markers:
            if marker in cleaned:
                candidate = cleaned.rpartition(marker)[2]
                candidate = re.sub(r"^[\s\u200b\u200c\u200d\ufeff]+|[\s\u200b\u200c\u200d\ufeff]+$", "", candidate)
                if candidate.startswith(":"):
                    candidate = re.sub(r"^[\s\u200b\u200c\u200d\ufeff]+|[\s\u200b\u200c\u200d\ufeff]+$", "", candidate[1:])
                if candidate:
                    cleaned = candidate
                    break

        # Multi-pass loop (up to 5 passes) until convergence
        for _ in range(5):
            prev = cleaned

            # 1. Strip model control tokens matching <|...|> (ChatML, Gemma, Llama, etc.)
            cleaned = re.sub(r"<\|[^|>\n]{1,40}\|>", "", cleaned)

            # 2. Strip markdown code fences (```markdown ... ``` or ```json ... ``` or ``` ...)
            cleaned = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

            # 3. Strip echoed prompt section headers (e.g. === REFINED TRANSLATION === or === ВІДРЕДАГОВАНИЙ ТЕКСТ ===)
            cleaned = re.sub(r"^={3,}\s*[^\n]+?\s*={3,}\s*:?\s*\n?", "", cleaned, flags=re.MULTILINE)

            # 4. Strip markdown headings generated by LLM (e.g., # Глава 1: ..., ## Розділ ...)
            cleaned = re.sub(r"^#{1,6}\s+[^\n]+\n*", "", cleaned, flags=re.MULTILINE)

            # 5. Strip conversational preambles
            cleaned = re.sub(
                r"^(?:\*{1,2})?(?:Ось\s+(?:фінальний|виправлений|відредагований|готовий|покращений)\s+переклад|Відредагований\s+(?:літературний\s+)?текст|Відредагований\s+(?:літературний\s+)?переклад|Фінальний\s+(?:покращений\s+)?переклад|Покращений\s+переклад|Переклад|Ukrainian\s+translation|Here\s+is\s+the\s+refined\s+translation|Here\s+is\s+the\s+translation)(?:\*{1,2})?\s*:\s*(?:\*{1,2})?\s*\n*",
                "",
                cleaned,
                flags=re.IGNORECASE
            )

            # 6. Strip unmapped inline tag artifacts (e.g., <tag_1>, </tag_1>, <tag_12>)
            cleaned = re.sub(r"</?tag_\d+>", "", cleaned)

            # 7. Strip inline bold and italic markdown emphasis (**bold**, *italic*)
            cleaned = re.sub(r"\*\*([^*]+?)\*\*", r"\1", cleaned)
            cleaned = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", cleaned)

            # 8. Sanitize mixed Cyrillic-Latin words / homoglyphs
            cleaned = sanitize_mixed_script_words(cleaned)

            # 9. Strip outer wrapping quotes if entire text is enclosed
            if len(cleaned) >= 2:
                if (cleaned.startswith('"') and cleaned.endswith('"')) or \
                   (cleaned.startswith('«') and cleaned.endswith('»')) or \
                   (cleaned.startswith('“') and cleaned.endswith('”')) or \
                   (cleaned.startswith("'") and cleaned.endswith("'")):
                    if cleaned.count('"') == 2 or cleaned.count('«') == 1 or cleaned.count('“') == 1 or cleaned.count("'") == 2:
                        cleaned = cleaned[1:-1]
                        cleaned = re.sub(r"^[\s\u200b\u200c\u200d\ufeff]+|[\s\u200b\u200c\u200d\ufeff]+$", "", cleaned)

            cleaned = re.sub(r"^[\s\u200b\u200c\u200d\ufeff]+|[\s\u200b\u200c\u200d\ufeff]+$", "", cleaned)
            if cleaned == prev:
                break

        return cleaned

    def _parse_json_response(
        self,
        raw: str,
        expected_count: int,
        fallback_drafts: Optional[Dict[int, str]] = None
    ) -> Dict[int, str]:
        """
        Safely parses LLM JSON response mapping sentence indices 1..N to refined strings.
        In case of invalid JSON, markdown fence wrapping, missing keys, or syntax errors,
        safely falls back to NLLB draft translations without crashing.
        """
        import json

        result: Dict[int, str] = {}
        for i in range(1, expected_count + 1):
            fb = fallback_drafts.get(i, "") if fallback_drafts else ""
            result[i] = fb

        if not raw or not raw.strip():
            return result

        cleaned_raw = raw.strip()
        cleaned_raw = re.sub(r"<\|[^|>\n]{1,40}\|>", "", cleaned_raw).strip()

        # If markdown code fence is present, extract interior JSON block
        fence_match = re.search(r"```(?:json|markdown)?\s*(\{.*?\})\s*```", cleaned_raw, re.DOTALL | re.IGNORECASE)
        if fence_match:
            target_str = fence_match.group(1).strip()
        else:
            target_str = re.sub(r"^```(?:json|markdown)?\s*\n?", "", cleaned_raw, flags=re.IGNORECASE)
            target_str = re.sub(r"\n?```\s*$", "", target_str).strip()

        parsed_data = None

        # Strategy 1: Direct json.loads with strict=False (allows literal newlines/controls in strings)
        try:
            parsed_data = json.loads(target_str, strict=False)
        except Exception as e:
            logger.debug(f"Strategy 1 json.loads failed ({e}); transitioning to Strategy 2.")

        # Strategy 2: Extract outermost {...} or [...]
        if parsed_data is None:
            brace_match = re.search(r'\{.*\}', target_str, re.DOTALL)
            if brace_match:
                candidate = brace_match.group(0)
                try:
                    parsed_data = json.loads(candidate, strict=False)
                except Exception:
                    fixed_candidate = re.sub(r',\s*\}', '}', candidate)
                    try:
                        parsed_data = json.loads(fixed_candidate, strict=False)
                    except Exception:
                        pass
            if parsed_data is None:
                bracket_match = re.search(r'\[.*\]', target_str, re.DOTALL)
                if bracket_match:
                    candidate = bracket_match.group(0)
                    try:
                        parsed_data = json.loads(candidate, strict=False)
                    except Exception:
                        fixed_candidate = re.sub(r',\s*\]', ']', candidate)
                        try:
                            parsed_data = json.loads(fixed_candidate, strict=False)
                        except Exception:
                            pass

        # Strategy 3: Key-value pattern extraction resilient to unescaped quotes & dialogue
        if parsed_data is None or not isinstance(parsed_data, (dict, list)):
            logger.debug("Strategy 1 and Strategy 2 failed; transitioning to Strategy 3 (regex key-value extraction).")
            extracted = {}
            pattern = r'(?:^|[\n,{\[])\s*[\'"`]?(?:sentence_|sent_|s)?(\d+)[\'"`]?\s*[:.]\s*(.*?)(?=(?:[\n,]\s*[\'"`]?(?:sentence_|sent_|s)?\d+[\'"`]?\s*[:.]|\s*\}|\s*\]|\s*$))'
            matches = re.findall(pattern, target_str, re.DOTALL)
            for k_str, v_raw in matches:
                try:
                    k_num = int(k_str)
                    v = v_raw.strip()
                    if v.endswith(','):
                        v = v[:-1].strip()

                    # Strip trailing backslashes if truncated
                    while v.endswith('\\'):
                        v = v[:-1]

                    # Normalize escaped quotes at boundaries
                    if v.startswith('\\"'):
                        v = '"' + v[2:]
                    elif v.startswith('\\'):
                        v = v[1:]

                    if v.endswith('\\"'):
                        v = v[:-2] + '"'

                    # Unescape escaped characters inside
                    v = v.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')

                    while v.endswith('\\'):
                        v = v[:-1]

                    # Quote resolution for JSON value delimiters
                    if v.startswith('"') or v.startswith("'"):
                        q = v[0]
                        q_count = v.count(q)
                        ends_with_q = v.endswith(q)

                        if ends_with_q and q_count % 2 == 0:
                            v = v[1:-1]
                        elif not ends_with_q or q_count % 2 == 1:
                            v = v[1:]

                    # Remove residual leading backslash or unclosed quote
                    while v.startswith('\\"') or v.startswith('\\'):
                        if v.startswith('\\"'):
                            v = v[2:]
                        else:
                            v = v[1:]
                    if v.startswith('"') and not v.endswith('"'):
                        v = v[1:]

                    v = v.strip()

                    extracted[k_num] = v
                except Exception:
                    continue

            if extracted:
                parsed_data = extracted

        # Unwrap root wrapper object if present (e.g., {"translations": {...}} or {"results": [...]})
        if isinstance(parsed_data, dict):
            for wrapper_key in ("translations", "sentences", "results", "result", "data", "output", "refined", "items"):
                if wrapper_key in parsed_data and isinstance(parsed_data[wrapper_key], (dict, list)):
                    parsed_data = parsed_data[wrapper_key]
                    break

        # Convert list format to 1-indexed dict (e.g. ["sent 1", "sent 2"] -> {1: "sent 1", 2: "sent 2"})
        if isinstance(parsed_data, list):
            parsed_data = {i + 1: item for i, item in enumerate(parsed_data)}

        # Normalize 0-indexed dict if LLM produced 0..N-1 instead of 1..N
        if isinstance(parsed_data, dict):
            has_zero = 0 in parsed_data or "0" in parsed_data or "sentence_0" in parsed_data
            has_max = expected_count in parsed_data or str(expected_count) in parsed_data or f"sentence_{expected_count}" in parsed_data
            if has_zero and not has_max and expected_count > 1:
                shifted = {}
                for k, v in parsed_data.items():
                    try:
                        idx_int = int(re.sub(r'\D', '', str(k)))
                        shifted[idx_int + 1] = v
                    except Exception:
                        shifted[k] = v
                parsed_data = shifted

            # Populate result from parsed_data
            for i in range(1, expected_count + 1):
                val = None
                for key_variant in (i, str(i), f"sentence_{i}", f"sent_{i}", f"s{i}"):
                    if key_variant in parsed_data:
                        val = parsed_data[key_variant]
                        break

                if val is None and expected_count == 1:
                    for single_key in ("translation", "text", "refined", "uk", "output", "result"):
                        if single_key in parsed_data:
                            val = parsed_data[single_key]
                            break

                if isinstance(val, dict):
                    val = (
                        val.get("uk") or
                        val.get("translation") or
                        val.get("text") or
                        val.get("translated_text") or
                        val.get("refined") or
                        str(val)
                    )

                if val is not None and isinstance(val, str) and val.strip():
                    sanitized_val = self._sanitize_output(val)
                    if sanitized_val and sanitized_val.strip():
                        result[i] = sanitized_val

        return result

    def refine_chunk_structured(
        self,
        target_sentences: List[Any],
        context_sentences: Optional[List[Any]] = None,
        glossary: Optional[List[GlossaryItem]] = None,
        cancel_token: Optional[Any] = None
    ) -> Dict[int, str]:
        """
        Executes Stage 2 literary refinement using structured JSON format.
        Takes target sentences, context sentences, and glossary, and returns
        a dictionary mapping 1-based index (1..N) to refined translation strings.
        """
        if not target_sentences:
            return {}

        expected_count = len(target_sentences)
        fallback_drafts: Dict[int, str] = {}
        target_lines: List[str] = []

        for i, s in enumerate(target_sentences, 1):
            if hasattr(s, "original_text"):
                en_text = s.original_text or ""
                uk_draft = s.translated_text or ""
            elif isinstance(s, dict):
                en_text = s.get("original_text", s.get("en", ""))
                uk_draft = s.get("translated_text", s.get("draft", s.get("uk", "")))
            elif isinstance(s, (list, tuple)) and len(s) >= 2:
                en_text = str(s[0])
                uk_draft = str(s[1])
            else:
                en_text = str(s)
                uk_draft = ""

            fallback_drafts[i] = uk_draft
            target_lines.append(f"{i}. EN: {en_text}\n   UK (чорновик): {uk_draft}")

        if _is_cancelled(cancel_token):
            logger.info("refine_chunk_structured cancelled before execution; returning fallbacks.")
            return fallback_drafts

        if not self.model and not self.gguf_llm:
            self.load_model()

        # Format context string
        if isinstance(context_sentences, ChunkContext):
            ctx_list = context_sentences.previous_sentences or []
        elif isinstance(context_sentences, list):
            ctx_list = [
                s.original_text if hasattr(s, "original_text") else str(s)
                for s in context_sentences
            ]
        else:
            ctx_list = []
        context_str = " ".join(ctx_list) if ctx_list else "Відсутній"

        # Format glossary string
        if glossary:
            reviewed_items = [item for item in glossary if getattr(item, "reviewed", False)]
            unreviewed_items = [item for item in glossary if not getattr(item, "reviewed", False)]

            glossary_lines = []
            if reviewed_items:
                for item in reviewed_items:
                    line = f"- {item.source_term} => {item.target_term}"
                    gender = getattr(item, "grammatical_gender", None)
                    if gender:
                        line += f", граматичний рід: {gender}"
                    glossary_lines.append(line)

            if unreviewed_items:
                glossary_lines.append("Рекомендація однакового перекладу та транслітерації повторюваних імен:")
                for item in unreviewed_items:
                    glossary_lines.append(f"- {item.source_term}")

            glossary_str = "\n".join(glossary_lines) if glossary_lines else "Відсутній (термінів у цьому фрагменті не виявлено)"
        else:
            glossary_str = "Відсутній (термінів у цьому фрагменті не виявлено)"

        target_block_str = "\n".join(target_lines)

        # Format prompt template safely
        source_text_all = " ".join(
            s.original_text if hasattr(s, "original_text") else str(s) for s in target_sentences
        )
        draft_text_all = " ".join(fallback_drafts.values())
        try:
            prompt = self.cached_prompt_template.format(
                context_previous=context_str,
                glossary_terms=glossary_str,
                target_sentences_block=target_block_str
            )
        except KeyError:
            try:
                prompt = self.cached_prompt_template.format(
                    context_previous=context_str,
                    glossary_terms=glossary_str,
                    source_text=source_text_all,
                    draft_text=draft_text_all,
                    target_sentences_block=target_block_str
                )
            except Exception:
                prompt = self.cached_prompt_template
                prompt = prompt.replace("{context_previous}", context_str)
                prompt = prompt.replace("{glossary_terms}", glossary_str)
                prompt = prompt.replace("{target_sentences_block}", target_block_str)
                prompt = prompt.replace("{source_text}", source_text_all)
                prompt = prompt.replace("{draft_text}", draft_text_all)
        except Exception:
            prompt = self.cached_prompt_template
            prompt = prompt.replace("{context_previous}", context_str)
            prompt = prompt.replace("{glossary_terms}", glossary_str)
            prompt = prompt.replace("{target_sentences_block}", target_block_str)
            prompt = prompt.replace("{source_text}", source_text_all)
            prompt = prompt.replace("{draft_text}", draft_text_all)

        stop_tokens = [
            "<|endoftext|>",
            "<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|>",
            "<|END_OF_USER_TOKEN|>",
            "<|END_OF_ASSISTANT_TOKEN|>",
            "<|im_end|>",
            "<|im_start|>",
            "</s>"
        ]

        raw_generated = ""

        # 1. GGUF Inference Path
        if self.gguf_llm is not None:
            try:
                if _is_cancelled(cancel_token):
                    return fallback_drafts

                max_tokens = min(max(256, int(len(prompt.split()) * 2) + 128), 4096)
                tokens = []
                try:
                    stream_resp = self.gguf_llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        repeat_penalty=self.repetition_penalty,
                        stop=stop_tokens,
                        stream=True
                    )
                    for chunk_resp in stream_resp:
                        if _is_cancelled(cancel_token):
                            logger.info("GGUF chat generation cancelled during streaming.")
                            return fallback_drafts
                        delta = chunk_resp.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            tokens.append(delta)
                    raw_generated = "".join(tokens)
                except Exception:
                    tokens.clear()
                    try:
                        stream_resp = self.gguf_llm.create_completion(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            repeat_penalty=self.repetition_penalty,
                            stop=stop_tokens,
                            stream=True
                        )
                        for chunk_resp in stream_resp:
                            if _is_cancelled(cancel_token):
                                return fallback_drafts
                            text_delta = chunk_resp.get("choices", [{}])[0].get("text", "")
                            if text_delta:
                                tokens.append(text_delta)
                        raw_generated = "".join(tokens)
                    except Exception:
                        resp = self.gguf_llm.create_completion(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            repeat_penalty=self.repetition_penalty,
                            stop=stop_tokens
                        )
                        raw_generated = resp["choices"][0]["text"]

            except Exception as e:
                logger.error(f"GGUF generation error in refine_chunk_structured: {e}")
                return fallback_drafts

        # 2. PyTorch / Transformers Inference Path
        elif self.model is not None:
            try:
                target_device = getattr(self.model, "device", self.device)
                messages = [{"role": "user", "content": prompt}]

                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                    try:
                        inputs = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            return_dict=True
                        )
                    except Exception:
                        inputs = self.tokenizer(prompt, return_tensors="pt")
                else:
                    inputs = self.tokenizer(prompt, return_tensors="pt")

                if hasattr(inputs, "items"):
                    inputs_dict = {k: v.to(target_device) for k, v in inputs.items()}
                    input_length = inputs_dict["input_ids"].shape[-1]
                else:
                    inputs_dict = {"input_ids": inputs.to(target_device)}
                    input_length = inputs.shape[-1]

                dynamic_max_new_tokens = min(max(256, int(input_length * 1.5) + 128), 4096)
                criteria_list = StoppingCriteriaList()
                if cancel_token is not None:
                    criteria_list.append(CancellationTokenStoppingCriteria(cancel_token))

                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs_dict,
                        max_new_tokens=dynamic_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=(self.temperature > 0.0),
                        stopping_criteria=criteria_list,
                        pad_token_id=pad_token_id
                    )

                if _is_cancelled(cancel_token):
                    return fallback_drafts

                raw_generated = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

            except Exception as e:
                logger.error(f"Transformers generation error in refine_chunk_structured: {e}")
                return fallback_drafts

        else:
            return fallback_drafts

        return self._parse_json_response(raw_generated, expected_count, fallback_drafts)

    def refine_chunk(
        self,
        draft_translation: str,
        source_text: Any = "",
        context: Optional[ChunkContext] = None,
        glossary: Optional[List[GlossaryItem]] = None,
        cancel_token: Optional[Any] = None,
        cancel_event: Optional[Any] = None
    ) -> str:
        """
        Deprecated: Refines a single draft string. Kept for backwards compatibility.
        Use refine_chunk_structured instead.
        """
        import warnings
        warnings.warn(
            "refine_chunk is deprecated; use refine_chunk_structured instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if isinstance(source_text, ChunkContext):
            if isinstance(context, list):
                actual_glossary = context
                actual_cancel = glossary if (cancel_token is None and cancel_event is None) else cancel_token
            else:
                actual_glossary = None
                actual_cancel = context if (cancel_token is None and cancel_event is None) else cancel_token
            context = source_text
            source_text = ""
            glossary = actual_glossary
            cancel_token = actual_cancel
        elif isinstance(context, list) and glossary is None:
            glossary = context
            context = None

        token = cancel_token if cancel_token is not None else cancel_event

        if not draft_translation or not draft_translation.strip():
            return draft_translation

        if _is_cancelled(token):
            logger.info("refine_chunk cancelled before execution; returning draft.")
            return draft_translation

        from src.domain.models.document import Sentence
        s = Sentence(
            original_text=str(source_text) if source_text else draft_translation,
            translated_text=draft_translation
        )
        ctx_sents = context.previous_sentences if context else None
        res_map = self.refine_chunk_structured(
            target_sentences=[s],
            context_sentences=ctx_sents,
            glossary=glossary,
            cancel_token=token
        )
        return res_map.get(1, draft_translation)

    def refine_batch(
        self,
        drafts: List[str],
        contexts: Optional[List[ChunkContext]] = None,
        glossary: Optional[List[GlossaryItem]] = None,
        source_texts: Optional[List[str]] = None,
        cancel_token: Optional[Any] = None
    ) -> List[str]:
        """Processes multiple draft chunks sequentially with cooperative cancellation."""
        results = []
        contexts = contexts or []
        for i, draft in enumerate(drafts):
            if _is_cancelled(cancel_token):
                logger.info("refine_batch interrupted by cancellation token.")
                results.extend(drafts[i:])
                break
            ctx = contexts[i] if i < len(contexts) else None
            src = source_texts[i] if (source_texts and i < len(source_texts)) else ""
            refined = self.refine_chunk(
                draft_translation=draft,
                source_text=src,
                context=ctx,
                glossary=glossary,
                cancel_token=cancel_token
            )
            results.append(refined)
        return results


# Alias for backward compatibility across container and caller modules
AyaEditingEngine = QuantizedAyaEditingEngine


