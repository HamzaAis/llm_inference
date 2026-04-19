from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "llm-inferance"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    database_url: str = Field(
        default=f"sqlite+aiosqlite:///{(PROJECT_ROOT / 'data' / 'llm_inferance.db').as_posix()}"
    )
    sync_database_url: str = Field(
        default=f"sqlite:///{(PROJECT_ROOT / 'data' / 'llm_inferance.db').as_posix()}"
    )

    data_dir: Path = PROJECT_ROOT / "data"
    files_dir: Path = PROJECT_ROOT / "files"
    hf_cache_dir: Path = PROJECT_ROOT / "hf_cache"

    model_name: str = "mistralai/Ministral-3-3B-Instruct-2512-ONNX"
    model_provider: str = "cuda"  # cpu, cuda
    model_verbose: bool = False
    model_precision: str = "q4f16"

    vl_system_prompt: str = (
        "You extract fields from images. Return COMPACT minified JSON on a single line, "
        "no whitespace or newlines. Use null for missing values. "
        "Copy text exactly as shown. Do not invent fields or repeat patterns."
    )

    text_temperature: float = 1.0
    text_top_p: float = 1.0
    text_top_k: int = 20
    text_min_p: float = 0.0
    text_presence_penalty: float = 2.0
    text_repeat_penalty: float = 1.0

    vl_temperature: float = 0.0  # Greedy decoding for factual extraction
    vl_top_p: float = 1.0
    vl_top_k: int = 1
    vl_min_p: float = 0.0
    vl_presence_penalty: float = 2.0
    vl_repeat_penalty: float = 2.0  # Strong penalty for repetition

    default_max_new_tokens: int = 512
    max_new_tokens_ceiling: int = 4096
    max_image_mb: int = 10
    image_max_side: int = 512
    allowed_image_mimes: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/webp",
    )

    rate_limit_per_minute: int = 30
    cors_allow_origins: tuple[str, ...] = ("*",)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
