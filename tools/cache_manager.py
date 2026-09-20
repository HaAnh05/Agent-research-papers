from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from config import config


def canonicalize_paper_key(input_identifier: str) -> str:
    """Returns a deterministic unique key: e.g., 'arxiv_1706.03762' or 'local_attention_a1b2c3'."""
    raw = (input_identifier or "").strip()
    if not raw:
        return "empty_key"

    # 1. Match ArXiv ID pattern: e.g. 1706.03762, 2312.00752v1, abs/1706.03762
    arxiv_match = re.search(r"(\d{4}\.\d{4,5})", raw)
    if arxiv_match:
        return f"arxiv_{arxiv_match.group(1)}"

    # 2. Local File Path
    p = Path(raw)
    if p.exists() and p.is_file():
        stat_info = f"{p.name}_{p.stat().st_size}"
        h = hashlib.md5(stat_info.encode("utf-8")).hexdigest()[:8]
        return f"local_{p.stem}_{h}"

    # 3. Query string or URL fallback hash
    clean_str = re.sub(r"[^A-Za-z0-9]", "_", raw.lower())[:30]
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"id_{clean_str}_{h}"


def get_cached_pdf_path(paper_key: str) -> Optional[Path]:
    """Check if PDF is already cached locally."""
    pdf_path = config.PDF_CACHE_DIR / f"{paper_key}.pdf"
    return pdf_path if pdf_path.exists() and pdf_path.stat().st_size > 1024 else None


def get_cached_text_data(paper_key: str) -> Optional[Dict[str, Any]]:
    """Check if extracted text/sections are cached in JSON."""
    json_path = config.TEXT_CACHE_DIR / f"{paper_key}.json"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_cached_text_data(paper_key: str, data: Dict[str, Any]) -> None:
    """Save extracted text and sections to local JSON cache."""
    try:
        json_path = config.TEXT_CACHE_DIR / f"{paper_key}.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Warning: Failed to save text cache for {paper_key}: {exc}")


def prompt_fingerprint(*parts: str) -> str:
    """Hash LLM inputs (templates + content + model) into a short cache key.

    The fingerprint covers the exact text sent to the provider, so identical
    inputs reuse identical outputs while any prompt, model, or source change
    transparently misses the cache.  No TTL is needed: invalidation is exact.
    """

    digest = hashlib.sha256()
    for part in parts:
        digest.update((part or "").encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def summary_cache_path(name: str) -> Path:
    """Resolve a summaries-cache file for PMRL/report payloads."""

    safe = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:80] or "entry"
    return config.SUMMARY_CACHE_DIR / f"{safe}.json"


def load_json_cache(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON cache entry; return None on any miss or corruption (fail-open)."""

    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def save_json_cache(path: Path, data: Dict[str, Any]) -> None:
    """Atomically persist a JSON cache entry; failures never break the run."""

    try:
        tmp_path = path.with_suffix(path.suffix + ".part")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        print(f"Warning: Failed to save cache {path.name}: {exc}")
