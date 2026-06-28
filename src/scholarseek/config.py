from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .qwen import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding existing env vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class ApiConfig:
    qwen_api_key: Optional[str]
    qwen_model: str
    qwen_base_url: str
    semantic_scholar_api_key: Optional[str]
    openalex_email: Optional[str]
    openalex_api_key: Optional[str]
    sources: str
    reranker_path: Optional[str]


def get_api_config(load_env_file: bool = True) -> ApiConfig:
    if load_env_file:
        load_dotenv()
    return ApiConfig(
        qwen_api_key=_clean_optional(os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")),
        qwen_model=os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL),
        qwen_base_url=os.getenv("QWEN_BASE_URL") or os.getenv("DASHSCOPE_API_BASE") or DEFAULT_QWEN_BASE_URL,
        semantic_scholar_api_key=_clean_optional(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
        openalex_email=_clean_optional(os.getenv("OPENALEX_EMAIL")),
        openalex_api_key=_clean_optional(os.getenv("OPENALEX_API_KEY")),
        sources=os.getenv("SCHOLARSEEK_SOURCES", "openalex,semantic-scholar,arxiv"),
        reranker_path=_clean_optional(os.getenv("SCHOLARSEEK_RERANKER_PATH")),
    )


def mask_secret(value: Optional[str]) -> str:
    if not value:
        return "missing"
    if len(value) <= 8:
        return "configured"
    return f"{value[:4]}...{value[-4:]}"


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    if lowered.startswith("replace_with") or lowered in {"your-key", "your_api_key", "your_email@example.com"}:
        return None
    return stripped
