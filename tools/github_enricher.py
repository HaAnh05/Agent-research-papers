from __future__ import annotations

import os
import re
from typing import List

import requests

from config import config
from state import GitHubRepoInfo

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"


def search_github_code_multitier(
    title: str,
    arxiv_id: str = "",
    authors: List[str] | None = None,
    max_results: int = 3,
) -> List[GitHubRepoInfo]:
    """Search GitHub for official or community code implementations using a 3-tier fallback."""
    headers = {"User-Agent": "Research-Agent/4.0"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"token {config.GITHUB_TOKEN}"

    queries: List[str] = []

    # Tier 1: ArXiv ID (Highest precision when authors mention it in description/README)
    if arxiv_id:
        clean_arxiv = re.sub(r"v\d+$", "", arxiv_id.strip())
        queries.append(f'"{clean_arxiv}"')

    # Tier 2: Cleaned Paper Title Keywords
    clean_title = re.sub(r"[^\w\s-]", " ", title or "").strip()
    title_words = [w for w in clean_title.split() if len(w) > 2][:6]
    if title_words:
        queries.append(f'{" ".join(title_words)} paper')

    # Tier 3: First Author's Last Name + 3 core keywords
    if authors and len(authors) > 0 and title_words:
        first_author_last = authors[0].split()[-1]
        queries.append(f"{first_author_last} {' '.join(title_words[:3])}")

    found_repos: List[GitHubRepoInfo] = []
    seen_urls = set()

    for q in queries:
        try:
            resp = requests.get(
                GITHUB_SEARCH_URL,
                params={"q": q, "sort": "stars", "order": "desc", "per_page": max_results},
                headers=headers,
                timeout=6,
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                for item in items:
                    url = item.get("html_url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    desc = (item.get("description") or "").lower()
                    is_official = (
                        "official" in desc
                        or (arxiv_id and arxiv_id in desc)
                        or (clean_title and clean_title.lower() in desc)
                    )
                    found_repos.append(
                        GitHubRepoInfo(
                            name=item.get("full_name", ""),
                            url=url,
                            stars=item.get("stargazers_count", 0),
                            framework=item.get("language") or "Python",
                            is_official=is_official,
                            description=(item.get("description") or "")[:200],
                        )
                    )
                if len(found_repos) >= max_results:
                    break
        except Exception:
            continue

    return found_repos[:max_results]
