from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(PROJECT_ROOT / "hf_cache")


class Settings(BaseSettings):
    # API Tokens
    hf_token: Optional[str] = None

    # App Paths
    db_path: Path = Path("data/kb_storage/knowledge_base.sqlite")
    prompt_file_path: Path = Path("data/prompts/editing_prompt.md")
    
    # Model Configurations
    nllb_model_name: str = "facebook/nllb-200-distilled-600M"
    aya_model_name: str = "CohereLabs/tiny-aya-global"
    device: str = "cuda"
    offline_mode: bool = True
    aya_temperature: float = 0.2
    aya_top_p: float = 0.9
    aya_repetition_penalty: float = 1.03
    aya_seed: Optional[int] = 42
    unit_conversion_policy: str = "metric"
    
    # Chunking Configurations
    max_tokens_per_chunk: int = 512
    overlap_sentences: int = 2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def save_to_env(self, key: str, value: str):
        import dotenv
        env_path = Path(".env")
        if not env_path.exists():
            env_path.touch()
        dotenv.set_key(str(env_path), key, value)
        
settings = Settings()
