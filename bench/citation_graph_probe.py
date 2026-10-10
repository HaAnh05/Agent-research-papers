"""Offline probe: resolve bibliographies of saved LaTeXML HTML files and build the citation graph.

Usage:
    .venv/bin/python -I bench/citation_graph_probe.py [DIRECTORY]

``DIRECTORY`` defaults to ``data/arxiv/html_probe`` (files named ``ar5iv_<id>.html``
or ``arxiv_<id>.html``; one paper per id, ar5iv preferred).  The probe uses the real
local index (``config.CITATION_INDEX_PATH``, opened read-only) and NO OpenAlex, no
network, no LLM.  It prints references resolved by method per paper, then the
foundational list for all files as one selected set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # works under ``python -I``, which drops the script dir
    sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402
from nodes.citation_graph import build_citation_graph  # noqa: E402
from state import PaperItem  # noqa: E402
from tools.html_extractor import extract_html, looks_like_latexml  # noqa: E402
from tools.ref_resolver import METHODS, RefResolver  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "arxiv" / "html_probe"


def load_papers(directory: Path) -> list[PaperItem]:
    chosen: dict[str, Path] = {}
    for path in sorted(directory.glob("*.html")):          # ar5iv_ sorts before arxiv_? no: "ar5iv" < "arxiv"
        arxiv_id = path.stem.split("_", 1)[1]
        chosen.setdefault(arxiv_id, path)
    papers: list[PaperItem] = []
    for arxiv_id, path in chosen.items():
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not looks_like_latexml(raw):
            continue
        result = extract_html(raw)
        papers.append(PaperItem(
            paper_id=f"arxiv_{arxiv_id}", title=result.title or arxiv_id, arxiv_id=arxiv_id,
            references=result.references,
        ))
    return papers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", default=str(DEFAULT_DIR), type=Path)
    args = parser.parse_args()
    papers = load_papers(args.directory)
    if not papers:
        print(f"No LaTeXML .html files in {args.directory}", file=sys.stderr)
        return 1

    index = config.CITATION_INDEX_PATH
    print(f"index: {index} ({'found' if index.is_file() else 'MISSING: explicit ids only'})")
    for paper in papers:
        resolver = RefResolver(index)
        graph = build_citation_graph([paper], resolver, None)
        stats = graph["stats"]
        methods = ", ".join(f"{m}={stats['by_method'][m]}" for m in METHODS)
        print(f"{paper.paper_id}: refs={stats['refs_total']} resolved={stats['refs_resolved']} "
              f"({methods}; ambiguous={stats['refs_ambiguous']})")
        print("  title: " + paper.title[:80])
        for node in [n for n in graph["nodes"] if n["role"] == "foundational"][:3]:
            print(f"  first resolved ref: {node['arxiv_id']} {node['title'][:70]!r} ({node['year']})")
        resolver.close()

    resolver = RefResolver(index)
    graph = build_citation_graph(papers, resolver, None, top_n=config.CITATION_TOP_N)
    stats = graph["stats"]
    print(f"\nall {len(papers)} files as one selected set: refs={stats['refs_total']} resolved={stats['refs_resolved']} "
          f"openalex={stats['openalex']}")
    methods = ", ".join(f"{m}={stats['by_method'][m]}" for m in METHODS)
    print(f"  by method: {methods}; ambiguous={stats['refs_ambiguous']}")
    foundational = [n for n in graph["nodes"] if n["role"] == "foundational"]
    print(f"  foundational (cited by >= 2 selected papers): {len(foundational)}")
    for node in foundational:
        print(f"    - [{node['arxiv_id']}] {node['title'][:80]!r} ({node['year']}) cited by {node['support']}/{len(papers)}")
    print(f"  nodes={len(graph['nodes'])} edges={len(graph['edges'])}")
    resolver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
