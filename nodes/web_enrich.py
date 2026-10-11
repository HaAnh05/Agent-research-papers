from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from config import config
from state import PaperItem, ResearchState
from tools.bibtex_generator import generate_bibtex
from tools.github_enricher import search_github_code_multitier


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


def _html_repo_urls(paper: PaperItem) -> List[str]:
    """Primary GitHub repositories already found in the paper's own HTML (no search needed)."""

    urls: List[str] = []
    for link in paper.repo_links:
        if link.kind == "github" and link.is_primary_candidate and link.url not in urls:
            urls.append(link.url)
    quality = paper.source_quality if isinstance(paper.source_quality, dict) else {}
    for candidate in quality.get("codeUrls") or quality.get("code_urls") or []:
        if isinstance(candidate, str) and candidate not in urls:
            urls.append(candidate)
    return urls


def _enrich_one(paper: PaperItem) -> Dict[str, Any]:
    """Repository lookup and BibTeX for one paper; returns only this branch's fields."""

    html_urls = _html_repo_urls(paper)
    if html_urls:
        repos: List[Any] = []
        github_timing: Dict[str, Any] = {"durationMs": 0, "cache": "skipped", "tiers": {}}
        github_error = False
        repo_source = "html"
        repo_count = len(html_urls)
    else:
        repos, github_timing, github_error = _github_one(paper)
        repo_source = "github_search" if repos else "none"
        repo_count = len(repos)
    bibtex = generate_bibtex(
        title=paper.title,
        authors=paper.authors,
        published=paper.published,
        arxiv_id=paper.arxiv_id,
    )
    return {
        "github_repos": repos,
        "bibtex": bibtex,
        "repo_source": repo_source,
        "repo_count": repo_count,
        "processing_metadata": {
            "github": github_timing,
            "web_enrich": {"githubError": github_error, "repoSource": repo_source},
        },
    }


def web_enrich_node(state: ResearchState) -> Dict[str, Any]:
    """Per-paper repository lookup and BibTeX (no LLM).

    Runs in parallel with ``write_notes`` and ``citation_graph``.  A paper whose
    HTML already exposes a primary GitHub link skips the GitHub search.  The
    node does not return ``selected_papers``; results go to
    ``paper_enrichment`` (keyed by paper id) and ``final_report`` joins them.
    """

    papers = list(state.get("selected_papers", []) or [])
    started = time.perf_counter()
    logs = [f"[web_enrich] Tìm mã nguồn và BibTeX cho {len(papers)} bài"]
    if not papers:
        return {
            "paper_enrichment": {},
            "trace_logs": logs,
            "node_timings": {"web_enrich": {"durationMs": 0, "papers": {}}},
        }

    workers = max(1, min(len(papers), int(config.GITHUB_MAX_CONCURRENCY)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_enrich_one, papers))

    enrichment: Dict[str, Dict[str, Any]] = {}
    paper_timings: Dict[str, Any] = {}
    errors: List[str] = []
    for paper, result in zip(papers, results):
        enrichment[paper.paper_id] = result
        meta = result["processing_metadata"]
        paper_timings[paper.paper_id] = {
            "durationMs": int(meta["github"].get("durationMs") or 0),
            "github": meta["github"],
        }
        repos = result["github_repos"]
        if result["repo_source"] == "html":
            repo_summary = f"{result['repo_count']} repo từ HTML của bài"
        elif repos:
            repo_summary = f"{len(repos)} repos (Top: {repos[0].name} ⭐{repos[0].stars})"
        else:
            repo_summary = "Chưa có repo public"
        logs.append(f"[web_enrich] {paper.paper_id}: {repo_summary}")
        if meta["web_enrich"]["githubError"]:
            logs.append(f"[web_enrich] {paper.paper_id}: GitHub unavailable")
            errors.append(f"GitHub enrichment unavailable for {paper.paper_id}")

    timing = {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "papers": paper_timings,
        "workerCount": workers,
    }
    return {
        "paper_enrichment": enrichment,
        "trace_logs": logs,
        "error_logs": errors,
        "node_timings": {"web_enrich": timing},
    }
