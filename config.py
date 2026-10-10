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
    SUMMARY_CACHE_DIR: Path = CACHE_DIR / "summaries"
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
    # Batched direct-input metadata lookup (read_paper): one short request, no
    # retry, and a bounded grace wait after the PDFs finish.  A miss only
    # degrades metadata fields; it never blocks the node for long.
    ARXIV_METADATA_TIMEOUT_SEC: float = float(os.getenv("ARXIV_METADATA_TIMEOUT_SEC", "6"))
    ARXIV_METADATA_MAX_ATTEMPTS: int = int(os.getenv("ARXIV_METADATA_MAX_ATTEMPTS", "1"))
    ARXIV_METADATA_GRACE_SEC: float = float(os.getenv("ARXIV_METADATA_GRACE_SEC", "3"))

    # Concurrency caps.  Each stage uses its own pool so slow GitHub work
    # never queues behind LLM calls (and the reverse).
    READ_PAPER_MAX_WORKERS: int = int(os.getenv("READ_PAPER_MAX_WORKERS", "8"))
    GITHUB_MAX_CONCURRENCY: int = int(os.getenv("GITHUB_MAX_CONCURRENCY", "4"))
    LLM_MAX_CONCURRENCY: int = int(os.getenv("LLM_MAX_CONCURRENCY", "6"))

    # final_report: write each paper section and the overview in separate
    # concurrent LLM calls.  False restores the legacy single synthesis call.
    REPORT_PARALLEL_SECTIONS: bool = os.getenv("REPORT_PARALLEL_SECTIONS", "true").strip().lower() not in {
        "0", "false", "no", "off", ""
    }

    # Workflow & Search Tuning
    MAX_RETRIES: int = 2
    MAX_SEARCH_RESULTS: int = 5
    TOP_K_PAPERS: int = 3
    
    # PDF Parsing limits
    PDF_MAX_PAGES: int = 8
    PDF_MAX_CHARS: int = 15000

    # HTML-first paper reading (ar5iv -> arxiv.org/html -> PDF).
    # arxiv.org/robots.txt sets ``Crawl-delay: 15``; it covers /html and /pdf,
    # so both share one limiter.  ar5iv.labs.arxiv.org sets no crawl delay.
    ARXIV_HOST_MIN_INTERVAL_SEC: float = float(os.getenv("ARXIV_HOST_MIN_INTERVAL_SEC", "15"))
    AR5IV_MIN_INTERVAL_SEC: float = float(os.getenv("AR5IV_MIN_INTERVAL_SEC", "3"))
    # Per-section character caps for sections taken from HTML (the PDF path
    # keeps its own 4,000 / 2,000 caps).
    HTML_EXPERIMENTS_MAX_CHARS: int = int(os.getenv("HTML_EXPERIMENTS_MAX_CHARS", "12000"))
    HTML_METHODOLOGY_MAX_CHARS: int = int(os.getenv("HTML_METHODOLOGY_MAX_CHARS", "8000"))
    HTML_SETUP_MAX_CHARS: int = int(os.getenv("HTML_SETUP_MAX_CHARS", "6000"))
    HTML_ABSTRACT_MAX_CHARS: int = int(os.getenv("HTML_ABSTRACT_MAX_CHARS", "3000"))
    HTML_LIMITATIONS_MAX_CHARS: int = int(os.getenv("HTML_LIMITATIONS_MAX_CHARS", "3000"))


    def __init__(self) -> None:
        # Create directories if they do not exist
        self.PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.GITHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
