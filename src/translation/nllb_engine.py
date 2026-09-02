"""
NLLB Machine Translation Engine backed by CTranslate2.
Supports INT8/FP16 quantized acceleration, automated JIT conversion, and Flores-200 language mapping.
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import gc

try:
    import torch
except ImportError:
    torch = None

from src.domain.interfaces.translation import ITranslationEngine
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Complete Flores-200 Language Mapping Dictionary
FLORES_200_LANG_MAP: Dict[str, str] = {
    "en": "eng_Latn",
    "uk": "ukr_Cyrl",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "pl": "pol_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "bg": "bul_Cyrl",
    "ru": "rus_Cyrl",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "fi": "fin_Latn",
    "da": "dan_Latn",
    "no": "nob_Latn",
    "el": "ell_Grek",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "ko": "kor_Hang",
    "hu": "hun_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
}


def get_flores_code(lang_code: str) -> str:
    """Normalizes an ISO 639-1/2 or BCP-47 language code into Flores-200 standard."""
    if not lang_code:
        return "ukr_Cyrl"
    clean_code = lang_code.strip().lower()
    return FLORES_200_LANG_MAP.get(clean_code, lang_code.strip())


class CTranslate2NLLBEngine(ITranslationEngine):
    """
    High-performance NLLB translation engine utilizing CTranslate2 for INT8/FP16 inference.
    Features automated JIT model conversion, Flores-200 code resolution, and clean VRAM lifecycle.
    """
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        compute_type: str = "auto",
        ct2_models_dir: Optional[Path] = None
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.compute_type = self._resolve_compute_type(compute_type, self.device)
        self.ct2_models_dir = ct2_models_dir or Path("models/ctranslate2")
        self.translator: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._pytorch_fallback_model: Optional[Any] = None

    def _resolve_device(self, requested: str) -> str:
        if requested in ("cuda", "cpu"):
            if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
                logger.warning("CUDA requested but unavailable. Falling back to CPU.")
                return "cpu"
            return requested
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_compute_type(self, requested: str, device: str) -> str:
        if requested != "auto":
            return requested
        if device == "cuda" and torch is not None and torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                # INT8 with FP16 activations supported on Turing (7.0) and newer
                return "int8_float16" if major >= 7 else "float16"
            except Exception:
                return "float16"
        return "int8"

    def _get_converted_path(self) -> Path:
        sanitized_name = self.model_name.replace("/", "--").replace("\\", "--")
        return self.ct2_models_dir / f"{sanitized_name}-{self.compute_type}"

    def _ensure_converted_model(self) -> Optional[Path]:
        converted_dir = self._get_converted_path()
        model_bin = converted_dir / "model.bin"

        if model_bin.exists():
            return converted_dir

        try:
            import ctranslate2
            logger.info(
                f"Converting HuggingFace model '{self.model_name}' to CTranslate2 format at '{converted_dir}'..."
            )
            converter = ctranslate2.converters.TransformersConverter(
                model_name_or_path=self.model_name,
                load_as_float16=(self.device == "cuda"),
                low_cpu_mem_usage=True
            )
            converted_dir.mkdir(parents=True, exist_ok=True)
            quant_spec = "int8" if "int8" in self.compute_type else ("float16" if self.device == "cuda" else "default")
            converter.convert(
                output_dir=str(converted_dir),
                quantization=quant_spec,
                force=True
            )
            logger.info("CTranslate2 model conversion completed successfully.")
            return converted_dir
        except Exception as e:
            logger.warning(
                f"Automatic CTranslate2 conversion failed or ctranslate2 unavailable: {e}. "
                "Will attempt standard PyTorch execution as fallback."
            )
            return None

    def load_model(self) -> None:
        if self.translator is not None or self._pytorch_fallback_model is not None:
            return

        from transformers import AutoTokenizer

        logger.info(f"Loading NLLB Tokenizer for '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=getattr(settings, "offline_mode", False)
        )

        model_path = self._ensure_converted_model()

        if model_path is not None:
            try:
                import ctranslate2
                logger.info(
                    f"Initializing CTranslate2 Translator from '{model_path}' on device='{self.device}', "
                    f"compute_type='{self.compute_type}'..."
                )
                inter_threads = 2 if self.device == "cpu" else 1
                intra_threads = 4 if self.device == "cpu" else 0

                self.translator = ctranslate2.Translator(
                    model_path=str(model_path),
                    device=self.device,
                    device_index=0 if self.device == "cuda" else 0,
                    compute_type=self.compute_type,
                    inter_threads=inter_threads,
                    intra_threads=intra_threads
                )
                logger.info("CTranslate2 NLLB engine successfully loaded.")
                return
            except Exception as e:
                logger.warning(f"Failed to load CTranslate2 translator: {e}. Falling back to PyTorch.")

        # PyTorch fallback path
        from transformers import AutoModelForSeq2SeqLM
        logger.info(f"Loading PyTorch Seq2Seq model for '{self.model_name}' (device='{self.device}')...")
        dtype = torch.float16 if (self.device == "cuda" and torch is not None) else torch.float32
        self._pytorch_fallback_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=getattr(settings, "offline_mode", False)
        )
        if torch is not None and hasattr(self._pytorch_fallback_model, "to"):
            self._pytorch_fallback_model = self._pytorch_fallback_model.to(self.device)
        logger.info("PyTorch fallback NLLB engine successfully loaded.")

    def unload_model(self) -> None:
        logger.info("Unloading NLLB engine from memory...")
        if self.translator is not None:
            del self.translator
            self.translator = None
        if self._pytorch_fallback_model is not None:
            del self._pytorch_fallback_model
            self._pytorch_fallback_model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("NLLB engine unloaded successfully.")

    @property
    def is_loaded(self) -> bool:
        """Returns True if CTranslate2 translator or PyTorch fallback model is loaded in memory."""
        return self.translator is not None or self._pytorch_fallback_model is not None

    def translate_batch(self, texts: List[str], source_lang: str = "en", target_lang: str = "uk") -> List[str]:
        if not texts:
            return []

        if not any(t.strip() for t in texts):
            return ["" for _ in texts]

        if not self.is_loaded:
            raise RuntimeError("Model is not loaded into memory. Call load_model() first.")

        src_flores = get_flores_code(source_lang)
        tgt_flores = get_flores_code(target_lang)

        if self.tokenizer is not None:
            self.tokenizer.src_lang = src_flores

        # 1. CTranslate2 Execution Path
        if self.translator is not None and self.tokenizer is not None:
            non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
            if not non_empty_indices:
                return ["" for _ in texts]

            source_tokens = []
            for i in non_empty_indices:
                text = texts[i]
                token_ids = self.tokenizer.encode(text)
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                source_tokens.append(tokens)

            target_prefix = [[tgt_flores]] * len(non_empty_indices)

            results = self.translator.translate_batch(
                source_tokens,
                target_prefix=target_prefix,
                beam_size=2,
                max_decoding_length=256,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )

            decoded_non_empty = []
            for result in results:
                output_tokens = result.hypotheses[0]
                if output_tokens and output_tokens[0] == tgt_flores:
                    output_tokens = output_tokens[1:]
                token_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
                decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                decoded_non_empty.append(decoded)

            translated_texts = ["" for _ in texts]
            for idx, trans in zip(non_empty_indices, decoded_non_empty):
                translated_texts[idx] = trans

            return translated_texts

        # 2. PyTorch Fallback Execution Path
        if self._pytorch_fallback_model is not None and self.tokenizer is not None:
            non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
            if not non_empty_indices:
                return ["" for _ in texts]

            non_empty_texts = [texts[i] for i in non_empty_indices]
            inputs = self.tokenizer(non_empty_texts, return_tensors="pt", padding=True, truncation=True)
            if torch is not None and hasattr(inputs, "to"):
                inputs = inputs.to(self.device)

            with torch.no_grad():
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_flores)
                generated_tokens = self._pytorch_fallback_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=256
                )
            decoded_non_empty = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            results = ["" for _ in texts]
            for idx, trans in zip(non_empty_indices, decoded_non_empty):
                results[idx] = trans.strip()
            return results

        raise RuntimeError("No active translation backend available in CTranslate2NLLBEngine.")


# Alias for backward compatibility across container and caller modules
NLLBMachineTranslationEngine = CTranslate2NLLBEngine
