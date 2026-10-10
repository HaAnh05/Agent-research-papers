"""Offline probe: parse saved LaTeXML HTML files and print a compact summary.

Usage:
    .venv/bin/python -I bench/html_extract_probe.py [DIRECTORY]

``DIRECTORY`` defaults to ``data/arxiv/html_probe`` (git-ignored local samples,
named ``ar5iv_<id>.html`` or ``arxiv_<id>.html``).  The script never uses the
network or an LLM.  Files are only read, never executed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # works under ``python -I``, which drops the script dir
    sys.path.insert(0, str(ROOT))

from tools.html_extractor import HARDWARE_RE, extract_html, looks_like_latexml  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "arxiv" / "html_probe"


def _source_of(name: str) -> str:
    if name.startswith("ar5iv_"):
        return "ar5iv"
    if name.startswith("arxiv_"):
        return "arxiv_html"
    return "unknown"


def _hardware_mentions(sections: dict[str, str], limit: int = 3) -> list[str]:
    found: list[str] = []
    for name in ("setup", "experiments"):
        for match in HARDWARE_RE.finditer(sections.get(name, "")):
            text = sections[name]
            snippet = " ".join(text[max(0, match.start() - 40): match.end() + 30].split())
            entry = f"{name}: ...{snippet}..."
            if entry not in found and not any(snippet[20:50] in other for other in found):
                found.append(entry)
            if len(found) >= limit:
                return found
    return found


def summarize(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not looks_like_latexml(raw):
        return f"{path.name}\n  source={_source_of(path.name)}  NOT LaTeXML (would fall through to the next source)"
    result = extract_html(raw)
    sections = result.sections
    with_arxiv = sum(1 for ref in result.references if ref.arxiv_id)
    with_doi = sum(1 for ref in result.references if ref.doi)
    primary = [link for link in result.repo_links if link.is_primary_candidate]
    lines = [
        f"{path.name}",
        f"  source={_source_of(path.name)}  title={result.title[:70]!r}",
        "  sections: " + " ".join(f"{key}={len(sections.get(key, ''))}" for key in
                                  ("abstract", "methodology", "setup", "experiments", "limitations")),
        f"  tables={len(result.tables)}  references={len(result.references)} "
        f"(with arXiv id: {with_arxiv}, with DOI: {with_doi})  repo_links={len(result.repo_links)}",
        "  primary repo candidates: "
        + ("; ".join(f"{link.url} [{link.kind}]" for link in primary) or "none"),
    ]
    baselines = [link.url for link in result.repo_links if not link.is_primary_candidate and not link.in_bibliography]
    if baselines:
        lines.append("  non-primary body links: " + "; ".join(baselines[:5]))
    for table in result.tables[:2]:
        caption = re.sub(r"\s+", " ", table.caption)[:80]
        lines.append(f"  table: {table.label or '?'} | {caption} | {len(table.rows)} rows x {max((len(r) for r in table.rows), default=0)} cols")
    hardware = _hardware_mentions(sections)
    lines.append("  hardware mentions: " + ("none" if not hardware else ""))
    lines.extend(f"    {item}" for item in hardware)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", default=str(DEFAULT_DIR), type=Path)
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.html"))
    if not files:
        print(f"No .html files in {args.directory}", file=sys.stderr)
        return 1
    for path in files:
        print(summarize(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
