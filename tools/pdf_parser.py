from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import requests

from config import config
from state import PaperItem
from tools.cache_manager import (
    canonicalize_paper_key,
    get_cached_pdf_path,
    get_cached_text_data,
    save_cached_text_data,
)

SECTION_PATTERNS = {
    "abstract": r"(?i)(?:abstract|tóm tắt)\s*[\r\n]+(.*?)(?=(?:1\.?\s+|i\.?\s+)?introduction|giới thiệu|\Z)",
    "methodology": r"(?i)(?:method|methodology|proposed method|architecture|model|approach|framework|phương pháp)\s*[\r\n]+(.*?)(?=(?:experiments|results|evaluations|thực nghiệm)|\Z)",
    "experiments": r"(?i)(?:experiments|results|evaluations|benchmarks|kết quả)\s*[\r\n]+(.*?)(?=(?:limitations|discussion|related work|conclusion)|\Z)",
    "limitations": r"(?i)(?:limitations|discussion|future work|giới hạn)\s*[\r\n]+(.*?)(?=(?:references|acknowledgments|tài liệu tham khảo)|\Z)",
}


def _download_pdf(pdf_url: str, output_path: Path) -> Path:
    """Download PDF from URL with streaming."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        pdf_url,
        headers={"User-Agent": config.ARXIV_USER_AGENT},
        timeout=20,
        stream=True,
    )
    resp.raise_for_status()
    with output_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return output_path


def extract_ordered_blocks_from_pdf(pdf_path: Path, max_pages: int = config.PDF_MAX_PAGES) -> str:
    """Extract text from PDF preserving two-column academic reading order."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        page_texts = []

        for page_idx in range(min(len(doc), max_pages)):
            page = doc[page_idx]
            blocks = page.get_text("blocks")
            # Filter regular text blocks (type 0)
            text_blocks = [b for b in blocks if b[6] == 0]

            # Detect two-column layout by splitting across middle X coordinate
            mid_x = page.rect.width / 2.0
            left_col = sorted([b for b in text_blocks if b[0] < mid_x], key=lambda b: b[1])
            right_col = sorted([b for b in text_blocks if b[0] >= mid_x], key=lambda b: b[1])

            ordered_page = "\n\n".join(b[4].strip() for b in left_col + right_col if b[4].strip())
            page_texts.append(ordered_page)

        return "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

    except ImportError:
        # Fallback to standard pypdf
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages_to_read = min(len(reader.pages), max_pages)
        parts = [reader.pages[i].extract_text() or "" for i in range(pages_to_read)]
        return "\n\n".join(parts)


def parse_structured_sections(raw_text: str) -> Dict[str, str]:
    """Parse text into academic sections via Regex with fallback chunking."""
    structured: Dict[str, str] = {}
    for section, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            max_chars = 2000 if section in ["abstract", "limitations"] else 4000
            structured[section] = " ".join(text.split())[:max_chars]
        else:
            structured[section] = ""

    # Fallback if section headers were not matched
    if not structured.get("methodology") and not structured.get("experiments"):
        structured["abstract"] = raw_text[:1500]
        structured["methodology"] = raw_text[1500:6500]
        structured["experiments"] = raw_text[6500:11500]
        structured["limitations"] = raw_text[11500:14000]

    return structured


def sync_extract_pdf_content(paper: PaperItem) -> PaperItem:
    """Synchronous worker function to download (if needed), parse, and extract sections."""
    paper_key = canonicalize_paper_key(paper.arxiv_id or paper.paper_id or paper.title)

    # 1. Check if structured data is already cached
    cached_data = get_cached_text_data(paper_key)
    if cached_data and cached_data.get("sections"):
        paper.sections = cached_data["sections"]
        paper.extracted_text = cached_data.get("extracted_text", "")
        paper.local_pdf_path = cached_data.get("local_pdf_path")
        return paper

    # 2. Determine PDF Path (Cached, Local, or Download)
    pdf_path = get_cached_pdf_path(paper_key)
    if not pdf_path or not pdf_path.exists():
        if paper.source_type == "local_pdf" and paper.local_pdf_path:
            local_p = Path(paper.local_pdf_path)
            if local_p.exists():
                pdf_path = local_p
            else:
                raise FileNotFoundError(f"Local PDF file not found: {paper.local_pdf_path}")
        elif paper.pdf_url:
            target_path = config.PDF_CACHE_DIR / f"{paper_key}.pdf"
            pdf_path = _download_pdf(paper.pdf_url, target_path)
        elif paper.arxiv_id:
            target_path = config.PDF_CACHE_DIR / f"{paper_key}.pdf"
            pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
            pdf_path = _download_pdf(pdf_url, target_path)
        else:
            raise ValueError(f"No PDF source available for paper: {paper.title}")

    # 3. Extract text and parse sections
    raw_ordered_text = extract_ordered_blocks_from_pdf(pdf_path, max_pages=config.PDF_MAX_PAGES)
    sections = parse_structured_sections(raw_ordered_text)

    # 4. Update PaperItem
    paper.local_pdf_path = str(pdf_path)
    paper.extracted_text = raw_ordered_text[:config.PDF_MAX_CHARS]
    paper.sections = sections

    # 5. Save to Cache
    save_cached_text_data(
        paper_key,
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "local_pdf_path": str(pdf_path),
            "sections": sections,
            "extracted_text": paper.extracted_text,
        },
    )

    return paper
