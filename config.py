from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
ROOT_DIR = Path(__file__).parent.resolve()
load_dotenv(ROOT_DIR / ".env")


class AppConfig:
    """Centralized configuration for the LangGraph Research Assistant."""

    # Project Paths
    PROJECT_ROOT: Path = ROOT_DIR
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CACHE_DIR: Path = DATA_DIR / "cache"
    PDF_CACHE_DIR: Path = CACHE_DIR / "pdfs"
    TEXT_CACHE_DIR: Path = CACHE_DIR / "text"
    GITHUB_CACHE_DIR: Path = CACHE_DIR / "github"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"

    # LLM Settings
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "gemini").lower()
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-3.1-flash-lite")
    TEMPERATURE: float = 0.0

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # ``OPENAI_API_BASE`` is retained for backwards compatibility with the
    # original OpenAI-compatible configuration.  New Z.AI deployments should
    # use the explicit ``ZAI_*`` variables below.
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ZAI_API_KEY: str = os.getenv("ZAI_API_KEY", "")
    ZAI_API_BASE: str = os.getenv(
        "ZAI_API_BASE", "https://api.z.ai/api/paas/v4/"
    )
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

    # ArXiv Tool Settings
    ARXIV_USER_AGENT: str = os.getenv(
        "ARXIV_USER_AGENT", "AI-Research-Agent/4.0 (contact: local@lab.ai)"
    )
    ARXIV_API_URL: str = "https://export.arxiv.org/api/query"
    ARXIV_MIN_INTERVAL_SEC: float = 3.0

    # Workflow & Search Tuning
    MAX_RETRIES: int = 2
    MAX_SEARCH_RESULTS: int = 5
    TOP_K_PAPERS: int = 3
    
    # PDF Parsing limits
    PDF_MAX_PAGES: int = 8
    PDF_MAX_CHARS: int = 15000


    def __init__(self) -> None:
        # Create directories if they do not exist
        self.PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.GITHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


config = AppConfig()


def resolved_provider(provider: str | None = None) -> str:
    """Return the canonical provider name used by the runtime.

    ``openai`` was historically used as an alias for OpenAI-compatible
    endpoints in this project.  Keep that setup working when its base URL is
    explicitly Z.AI, while making ``zai`` the canonical provider for new
    configuration.  This helper does not expose or mutate any credentials.
    """

    chosen = (provider or config.DEFAULT_PROVIDER or "gemini").strip().lower()
    if chosen in {"z.ai", "glm"}:
        return "zai"
    if chosen == "openai" and "api.z.ai" in (config.OPENAI_API_BASE or "").lower():
        return "zai"
    return chosen


def provider_is_configured(provider: str | None = None) -> bool:
    """Report whether a provider has a usable credential without returning it."""

    chosen = resolved_provider(provider)
    if chosen == "gemini":
        return bool(config.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY", ""))
    if chosen == "openai":
        return bool(config.OPENAI_API_KEY)
    if chosen == "openrouter":
        return bool(config.OPENROUTER_API_KEY)
    if chosen == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    if chosen == "zai":
        # OPENAI_API_KEY is an intentionally temporary compatibility fallback
        # only when the legacy base URL clearly targets Z.AI.
        legacy_key = config.OPENAI_API_KEY if "api.z.ai" in (config.OPENAI_API_BASE or "").lower() else ""
        return bool(config.ZAI_API_KEY or legacy_key)
    return False
