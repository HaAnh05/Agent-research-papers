from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from nodes.write_notes import analyze_paper_notes
from state import PaperItem, ResearchState
from tools.bibtex_generator import generate_bibtex
from tools.github_enricher import search_github_code_multitier
from tools.llm_provider import get_llm


def _github_one(paper: PaperItem) -> Tuple[List[Any], Dict[str, Any], bool]:
    """Run one isolated GitHub job and keep network failure local."""

    github_timing: Dict[str, Any] = {}
    try:
        repos = search_github_code_multitier(
            title=paper.title,
            arxiv_id=paper.arxiv_id or "",
            authors=paper.authors,
            max_results=3,
            timing=github_timing,
        )
    except Exception:
        repos = []
        github_timing.update({"durationMs": 0, "cache": "error", "tiers": {}})
    tier_failed = any(
        str(value.get("status") or "") not in {"ok"}
        for value in (github_timing.get("tiers") or {}).values()
        if isinstance(value, dict)
    )
    return repos, github_timing, bool(tier_failed or github_timing.get("cache") == "error")


def web_enrich_node(state: ResearchState) -> Dict[str, Any]:
    """Run per-paper GitHub enrichment and PMRL analysis with two workers.

    The graph still flows through ``web_enrich -> write_notes``.  Doing both
    independent per-paper operations in this bounded pool removes the full
    GitHub-before-PMRL barrier while the following node finalizes the prepared
    notes and keeps its existing public activity step.
    """

    papers = list(state.get("selected_papers", []) or [])
    started = time.perf_counter()
    logs = [f"[web_enrich] Bổ sung GitHub, BibTeX và PMRL cho {len(papers)} bài"]
    if not papers:
        return {
            "selected_papers": [],
            "trace_logs": logs,
            "node_timings": {"web_enrich": {"durationMs": 0, "papers": {}}},
        }

    llm: Any | None = None
    try:
        llm = get_llm()
    except Exception:
        logs.append("[web_enrich] Provider unavailable; PMRL fallback kept per paper")

    # GitHub and PMRL are separate jobs.  The two-worker cap applies to the
    # combined queue, so a slow GitHub tier does not serialize every PMRL call.
    jobs: Dict[Tuple[int, str], Any] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, paper in enumerate(papers):
            jobs[(index, "github")] = executor.submit(_github_one, paper)
            jobs[(index, "pmrl")] = executor.submit(analyze_paper_notes, paper, llm=llm)
        completed = {}
        for index, paper in enumerate(papers):
            github_repos, github_timing, github_error = jobs[(index, "github")].result()
            notes_paper, notes_status, pmrl_timing = jobs[(index, "pmrl")].result()
            result = notes_paper
            result.github_repos = github_repos
            result.bibtex = generate_bibtex(
                title=result.title,
                authors=result.authors,
                published=result.published,
                arxiv_id=result.arxiv_id,
            )
            result.processing_metadata = dict(result.processing_metadata or {})
            result.processing_metadata["pmrl_precomputed"] = True
            result.processing_metadata["github"] = github_timing
            result.processing_metadata["web_enrich"] = {
                "githubError": github_error,
                "pmrlStatus": notes_status,
            }
            completed[index] = (result, f"{notes_status}:github_error" if github_error else notes_status, {
                "durationMs": max(
                    int(github_timing.get("durationMs") or 0),
                    int(pmrl_timing.get("durationMs") or 0),
                ),
                "github": github_timing,
                "pmrl": pmrl_timing,
            })

    enriched: List[PaperItem] = []
    paper_timings: Dict[str, Any] = {}
    errors: List[str] = []
    for index, original in enumerate(papers):
        paper, status, timing = completed[index]
        enriched.append(paper)
        paper_timings[paper.paper_id] = timing
        repos = paper.github_repos
        repo_summary = f"{len(repos)} repos (Top: {repos[0].name} ⭐{repos[0].stars})" if repos else "Chưa có repo public"
        logs.append(f"[web_enrich] {paper.paper_id}: {repo_summary}")
        if paper.notes_quality == "invalid":
            logs.append(f"[web_enrich] {paper.paper_id}: PMRL fallback")
            errors.append(f"PMRL fallback for {paper.paper_id}")
        if "github_error" in status:
            logs.append(f"[web_enrich] {paper.paper_id}: GitHub unavailable")
            errors.append(f"GitHub enrichment unavailable for {paper.paper_id}")

    timing = {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "papers": paper_timings,
        "workerCount": min(2, len(papers)),
    }
    return {
        "selected_papers": enriched,
        "trace_logs": logs,
        "error_logs": errors,
        "node_timings": {"web_enrich": timing},
    }
