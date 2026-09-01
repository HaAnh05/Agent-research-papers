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
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"

    # LLM Settings
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "gemini").lower()
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-3.1-flash-lite")
    TEMPERATURE: float = 0.0

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
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
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


config = AppConfig()
