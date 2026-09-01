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
