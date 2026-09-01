from __future__ import annotations

from typing import Any, Dict, List
from state import ResearchState, PaperItem
from tools.github_enricher import search_github_code_multitier
from tools.bibtex_generator import generate_bibtex


def web_enrich_node(state: ResearchState) -> Dict[str, Any]:
    """Enriches each selected paper with GitHub code repositories and formatted BibTeX citation."""
    papers = state.get("selected_papers", [])
    logs = [f"[web_enrich] Bổ sung GitHub và BibTeX cho {len(papers)} bài"]

    enriched_papers: List[PaperItem] = []

    for p in papers:
        # 1. Multi-tier GitHub search
        repos = search_github_code_multitier(
            title=p.title,
            arxiv_id=p.arxiv_id or "",
            authors=p.authors,
            max_results=3,
        )
        p.github_repos = repos

        # 2. BibTeX Generation
        p.bibtex = generate_bibtex(
            title=p.title,
            authors=p.authors,
            published=p.published,
            arxiv_id=p.arxiv_id,
        )

        repo_summary = f"{len(repos)} repos (Top: {repos[0].name} ⭐{repos[0].stars})" if repos else "Chưa có repo public"
        logs.append(f"[web_enrich] {p.paper_id}: {repo_summary}")
        enriched_papers.append(p)

    return {
        "selected_papers": enriched_papers,
        "trace_logs": logs
    }
