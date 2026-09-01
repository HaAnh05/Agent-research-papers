from __future__ import annotations

import json
import re
from typing import Any, Dict
from pydantic import BaseModel, Field

from config import config
from prompts.router_prompt import ROUTER_SYSTEM_PROMPT
from state import ResearchState, PaperItem
from tools.cache_manager import canonicalize_paper_key
from tools.llm_provider import get_llm


class RouterDecision(BaseModel):
    intent: str = Field(description="One of: 'direct_read', 'direct_compare', 'search'")
    reasoning: str = Field(description="Explanation of classification decision")
    search_query: str = Field(default="", description="Cleaned core search query if searching")


def _looks_like_explicit_compare_query(query: str) -> bool:
    """Only trigger on very clear comparison requests like 'compare A and B' or 'so sánh A và B'."""
    text = (query or "").strip()
    if not text:
        return False

    lowered = text.lower()
    compare_markers = ["compare", "comparison", "vs", "versus", "so sánh", "so sanh", "đối chiếu", "đánh giá"]
    if not any(marker in lowered for marker in compare_markers):
        return False

    named_terms = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", text)
    if len(named_terms) >= 2:
        return True

    token_pairs = re.findall(r"\b[A-Z][A-Za-z0-9]+\b", text)
    return len(token_pairs) >= 2


def _build_safe_compare_search_query(query: str) -> str:
    """Keep exact paper names together and avoid turning them into generic MoE keywords."""
    text = (query or "").strip()
    named_terms = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", text)
    if len(named_terms) >= 2:
        return " ".join(named_terms[:2])

    split_terms = re.findall(r"\b[A-Z][A-Za-z0-9]+\b", text)
    return " ".join(split_terms[:2]) if len(split_terms) >= 2 else text


def router_node(state: ResearchState) -> Dict[str, Any]:
    """Xác định intent: đọc trực tiếp, so sánh, hoặc tìm kiếm."""
    user_query = state.get("user_query", "").strip()
    raw_inputs = state.get("raw_inputs", [])

    # Kiểm tra nhanh trước: nếu đã có input rõ ràng thì skip LLM.
    logs = [f"[router] query='{user_query}', inputs={raw_inputs}"]

    # 1. Kiểm tra quy tắc cố định trước để xử lý nhanh và ổn định.
    if len(raw_inputs) == 1:
        item = raw_inputs[0].strip()
        arxiv_match = re.search(r"(\d{4}\.\d{4,5})", item)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None
        
        paper = PaperItem(
            paper_id=canonicalize_paper_key(item),
            title=f"Target Paper {item}",
            source_type="local_pdf" if item.endswith(".pdf") and not item.startswith("http") else "arxiv",
            arxiv_id=arxiv_id,
            local_pdf_path=item if item.endswith(".pdf") and not item.startswith("http") else None,
            pdf_url=item if item.startswith("http") else (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None)
        )
        logs.append("[router] Có 1 input -> direct_read")
        return {
            "intent": "direct_read",
            "selected_papers": [paper],
            "trace_logs": logs
        }
        
    elif len(raw_inputs) >= 2:
        selected = []
        for idx, item in enumerate(raw_inputs, 1):
            item_str = item.strip()
            arxiv_match = re.search(r"(\d{4}\.\d{4,5})", item_str)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None
            
            paper = PaperItem(
                paper_id=canonicalize_paper_key(item_str),
                title=f"Paper {idx}: {item_str}",
                source_type="local_pdf" if item_str.endswith(".pdf") and not item_str.startswith("http") else "arxiv",
                arxiv_id=arxiv_id,
                local_pdf_path=item_str if item_str.endswith(".pdf") and not item_str.startswith("http") else None,
                pdf_url=item_str if item_str.startswith("http") else (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None)
            )
            selected.append(paper)
            
        logs.append(f"[router] Có {len(selected)} input -> direct_compare")
        return {
            "intent": "direct_compare",
            "selected_papers": selected,
            "trace_logs": logs
        }

    # 2. Nếu là câu so sánh rõ ràng, giữ tên paper nguyên vẹn thay vì rút thành keyword quá rộng.
    if _looks_like_explicit_compare_query(user_query):
        safe_query = _build_safe_compare_search_query(user_query)
        logs.append(f"[router] Exact compare pattern -> search_query='{safe_query}'")
        return {
            "intent": "search",
            "search_query": safe_query,
            "trace_logs": logs,
        }

    # 3. Với query tự nhiên, dùng LLM để phân loại ý định.
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(RouterDecision)
        prompt_text = ROUTER_SYSTEM_PROMPT.format(
            user_query=user_query,
            raw_inputs=str(raw_inputs)
        )
        decision: RouterDecision = structured_llm.invoke(prompt_text)
        
        chosen_intent = decision.intent if decision.intent in ["direct_read", "direct_compare", "search"] else "search"
        search_q = decision.search_query.strip() or user_query
        
        logs.append(f"[router] intent='{chosen_intent}', search_query='{search_q}'")
        return {
            "intent": chosen_intent,
            "search_query": search_q,
            "trace_logs": logs
        }
    except Exception as exc:
        # Nếu LLM lỗi, quay về tìm kiếm theo query gốc.
        logs.append(f"[router] LLM lỗi ({exc}) -> fallback search")
        return {
            "intent": "search",
            "search_query": user_query,
            "trace_logs": logs
        }
