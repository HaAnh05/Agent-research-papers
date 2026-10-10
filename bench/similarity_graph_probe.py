"""Live probe: build the similarity graph for ONE arXiv paper (real HTML, index and OpenAlex).

Usage:
    .venv/bin/python -I bench/similarity_graph_probe.py <arxiv_id> [--out file.json]

Needs network (arxiv.org/ar5iv HTML, OpenAlex) and ``OPENALEX_API_KEY`` (or
``OPEN_ALEX_API_KEY``) in the environment/.env.  No LLM call.  OpenAlex responses
are cached under ``data/cache/openalex/``, so a second run is fast.  Prints stats,
the top 15 nodes, and the prior / derivative lists; ``--out`` writes the full
``similarity`` dict as JSON (the API key is never written).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # works under ``python -I``, which drops the script dir
    sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402
from nodes.similarity_graph import build_similarity_graph  # noqa: E402
from state import PaperItem  # noqa: E402
from tools.html_source import clean_arxiv_id, fetch_html_extraction  # noqa: E402
from tools.openalex import OpenAlexClient  # noqa: E402
from tools.ref_resolver import RefResolver  # noqa: E402


def _line(node: Dict[str, Any]) -> str:
    sim = node.get("similarity")
    sim_txt = f"{sim:.3f}" if isinstance(sim, (int, float)) else "  -  "
    support = f" x{node['support']}" if "support" in node else ""
    return (f"  {sim_txt} {str(node.get('year') or '----'):>4} cit={str(node.get('cited_by_count')):>6} "
            f"{(node.get('first_author') or '-')[:18]:<18} {str(node.get('arxiv_id') or '-'):<12} "
            f"{node['title'][:70]!r}{support}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("arxiv_id")
    parser.add_argument("--out", type=Path, default=None, help="write the full similarity dict as JSON")
    args = parser.parse_args()
    arxiv_id = clean_arxiv_id(args.arxiv_id)
    base = re.sub(r"v\d+$", "", arxiv_id)

    client = OpenAlexClient()
    if not client.has_key:
        print("OPENALEX_API_KEY is not set: the graph will be empty (status no_key).", file=sys.stderr)

    attempts: list = []
    extraction, source = fetch_html_extraction(arxiv_id, attempts)
    print(f"html: source={source} attempts={attempts}")
    references = extraction.references if extraction else []
    paper = PaperItem(
        paper_id=f"arxiv_{base}", title=(extraction.title if extraction else "") or base,
        arxiv_id=base, references=references,
    )
    resolver = RefResolver(config.CITATION_INDEX_PATH)
    print(f"index: {config.CITATION_INDEX_PATH} ({'found' if resolver.index_available else 'MISSING'}); "
          f"references from HTML: {len(references)}")
    try:
        graph = build_similarity_graph([paper], client, resolver)
    finally:
        resolver.close()

    print(f"stats: {json.dumps(graph['stats'])}")
    print(f"nodes={len(graph['nodes'])} edges={len(graph['edges'])}")
    ranked = sorted(graph["nodes"], key=lambda n: (n["role"] != "origin", -(n["similarity"] or 0)))
    print("top 15 nodes (similarity, year, citations, first author, arXiv id, title):")
    for node in ranked[:15]:
        print(("* " if node["role"] == "origin" else "  ") + _line(node).lstrip(" "))
    print("prior works:")
    for node in graph["prior_works"]:
        print(_line(node))
    print("derivative works:")
    for node in graph["derivative_works"]:
        print(_line(node))
    if args.out:
        args.out.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
