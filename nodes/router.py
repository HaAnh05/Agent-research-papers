from __future__ import annotations

import re
import time
from typing import Any, Dict, Literal
from pydantic import BaseModel, Field

from config import config
from prompts.router_prompt import ROUTER_SYSTEM, ROUTER_USER
from state import ResearchState, PaperItem
from tools.cache_manager import canonicalize_paper_key
from tools.arxiv_search import get_cached_arxiv_metadata
from tools.llm_provider import get_llm, invoke_structured_output


class RouterDecision(BaseModel):
    intent: Literal["direct_read", "search", "followup"] = Field(
        description="One of: 'direct_read', 'search', 'followup'"
    )
    search_query: str = Field(default="", description="Cleaned core search query if searching")


def _safe_paper_label(item: str, index: int | None = None) -> str:
    """Create a display label without copying a local/server path."""

    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", item or "")
    if match:
        return f"ArXiv {match.group(1)}"
    if str(item or "").lower().endswith(".pdf"):
        return "Local PDF" if index is None else f"Paper {index} (Local PDF)"
    return "Paper" if index is None else f"Paper {index}"


_ARXIV_URL_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_BARE_ID_RE = re.compile(r"(?<![\w./])\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}(?:v\d+)?(?![\w.])")


def _arxiv_ids_in_text(text: str) -> list[str]:
    """arXiv ids named in a message (URL or bare ``YYMM.NNNNN``), in order, without duplicates."""

    found: list[str] = []
    for match in sorted(
        [(m.start(), m.group(1)) for m in _ARXIV_URL_ID_RE.finditer(text or "")]
        + [(m.start(), m.group(0)) for m in _ARXIV_BARE_ID_RE.finditer(text or "")]
    ):
        arxiv_id = re.sub(r"v\d+$", "", match[1])
        if arxiv_id not in found:
            found.append(arxiv_id)
    return found[:5]


def _thread_report_text(context: Any) -> str:
    """One compact line per paper of the thread's finished report, or ``none``."""

    if not isinstance(context, dict) or not context.get("reportId"):
        return "none"
    lines = ["available"]
    for index, paper in enumerate((context.get("papers") or [])[:8], 1):
        if isinstance(paper, dict):
            lines.append(f"{index}. {paper.get('title') or 'Untitled'} (arXiv {paper.get('arxivId') or 'N/A'})")
    return "\n".join(lines)


def router_node(state: ResearchState) -> Dict[str, Any]:
    """Xác định intent: đọc trực tiếp (1 hoặc nhiều paper) hoặc tìm kiếm bài báo."""
    started = time.perf_counter()
    user_query = state.get("user_query", "").strip()
    raw_inputs = list(state.get("raw_inputs", []) or [])
    # Rule shortcut (no LLM): an arXiv id or link in the message is a request to
    # read that paper, even when the thread already has a report.
    if not raw_inputs:
        raw_inputs = _arxiv_ids_in_text(user_query)
    conversation_context = (state.get("conversation_context") or "").strip()

    # Keep operational logs free of query contents and local/server paths.
    logs = [
        f"[router] Nhận request ({len(raw_inputs)} paper input(s), "
        f"context={'yes' if conversation_context else 'no'})"
    ]

    # 1. Kiểm tra quy tắc cố định trước để xử lý nhanh và ổn định.
    if len(raw_inputs) == 1:
        item = raw_inputs[0].strip()
        arxiv_match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", item)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None
        cached_metadata = get_cached_arxiv_metadata(arxiv_id) if arxiv_id else None

        paper = PaperItem(
            paper_id=canonicalize_paper_key(item),
            title=(cached_metadata or {}).get("title") or _safe_paper_label(item),
            summary=(cached_metadata or {}).get("summary", ""),
            authors=list((cached_metadata or {}).get("authors") or []),
            published=(cached_metadata or {}).get("published", ""),
            subjects=list((cached_metadata or {}).get("subjects") or []),
            source_type="local_pdf" if item.lower().endswith(".pdf") and not item.startswith("http") else "arxiv",
            arxiv_id=arxiv_id,
            arxiv_abs_url=(cached_metadata or {}).get("abs_url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
            arxiv_html_url=(cached_metadata or {}).get("html_url") or (f"https://arxiv.org/html/{arxiv_id}" if arxiv_id else None),
            local_pdf_path=item if item.lower().endswith(".pdf") and not item.startswith("http") else None,
            pdf_url=(cached_metadata or {}).get("pdf_url") or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else (item if item.startswith("http") else None)),
            metadata_status=(cached_metadata or {}).get("metadata_status", "unknown"),
        )
        logs.append("[router] Có 1 input -> direct_read")
        return {
            "intent": "direct_read",
            "selected_papers": [paper],
            "trace_logs": logs,
            "node_timings": {"router": {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": 0}},
        }
        
    elif len(raw_inputs) >= 2:
        selected = []
        for idx, item in enumerate(raw_inputs, 1):
            item_str = item.strip()
            arxiv_match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", item_str)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None
            cached_metadata = get_cached_arxiv_metadata(arxiv_id) if arxiv_id else None
            
            paper = PaperItem(
                paper_id=canonicalize_paper_key(item_str),
                title=(cached_metadata or {}).get("title") or _safe_paper_label(item_str, idx),
                summary=(cached_metadata or {}).get("summary", ""),
                authors=list((cached_metadata or {}).get("authors") or []),
                published=(cached_metadata or {}).get("published", ""),
                subjects=list((cached_metadata or {}).get("subjects") or []),
                source_type="local_pdf" if item_str.lower().endswith(".pdf") and not item_str.startswith("http") else "arxiv",
                arxiv_id=arxiv_id,
                arxiv_abs_url=(cached_metadata or {}).get("abs_url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
                arxiv_html_url=(cached_metadata or {}).get("html_url") or (f"https://arxiv.org/html/{arxiv_id}" if arxiv_id else None),
                local_pdf_path=item_str if item_str.lower().endswith(".pdf") and not item_str.startswith("http") else None,
                pdf_url=(cached_metadata or {}).get("pdf_url") or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else (item_str if item_str.startswith("http") else None)),
                metadata_status=(cached_metadata or {}).get("metadata_status", "unknown"),
            )
            selected.append(paper)
            
        logs.append(f"[router] Có {len(selected)} input -> direct_read")
        return {
            "intent": "direct_read",
            "selected_papers": selected,
            "trace_logs": logs,
            "node_timings": {"router": {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": 0}},
        }

    # 2. Với query tự nhiên, dùng LLM để phân loại ý định.
    try:
        llm = get_llm()
        llm_started = time.perf_counter()
        prompt_text = ROUTER_USER.format(
            user_query=user_query,
            raw_inputs=(f"{len(raw_inputs)} validated paper input(s)" if raw_inputs else "none"),
            conversation_context=conversation_context[:6000] or "(empty)",
            thread_report=_thread_report_text(state.get("followup_context")),
        )
        decision: RouterDecision = invoke_structured_output(
            prompt_text,
            RouterDecision,
            llm=llm,
            provider=config.DEFAULT_PROVIDER,
            system=ROUTER_SYSTEM,
        )
        llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
        
        chosen_intent = decision.intent
        # A direct workflow requires a concrete ArXiv/PDF input. Natural-language
        # paper titles still need an ArXiv search before there is anything for
        # the PDF reader to consume.
        if chosen_intent == "direct_read" and not raw_inputs:
            logs.append(f"[router] Bỏ direct intent không có paper input -> search")
            chosen_intent = "search"
        # A follow-up needs a finished report in the thread.
        if chosen_intent == "followup" and _thread_report_text(state.get("followup_context")) == "none":
            logs.append("[router] Bỏ followup vì thread chưa có báo cáo -> search")
            chosen_intent = "search"
        search_q = decision.search_query.strip() or user_query
        
        logs.append(f"[router] intent='{chosen_intent}'")
        return {
            "intent": chosen_intent,
            "search_query": search_q,
            "trace_logs": logs,
            "node_timings": {"router": {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": llm_ms}},
        }
    except Exception:
        # Nếu LLM lỗi, quay về tìm kiếm theo query gốc.
        # Do not copy provider exception text into append-only trace logs.
        logs.append("[router] LLM không khả dụng -> fallback search")
        return {
            "intent": "search",
            "search_query": user_query,
            "trace_logs": logs,
            "node_timings": {"router": {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": 0}},
        }
