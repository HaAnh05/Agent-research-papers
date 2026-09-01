from __future__ import annotations

import re
from typing import List


def generate_bibtex(
    title: str,
    authors: List[str] | None = None,
    published: str = "",
    arxiv_id: str | None = None,
) -> str:
    """Generate standard academic BibTeX citation for a research paper."""
    clean_title = " ".join((title or "Unknown Paper").split())
    author_list = " and ".join(authors) if authors else "Unknown Author"
    
    # Extract 4-digit year
    year_match = re.search(r"(\d{4})", published or "")
    year = year_match.group(1) if year_match else "2024"

    # Citation key: FirstAuthorLastName + Year + FirstKeyword
    first_author = authors[0].split()[-1].lower() if authors and len(authors) > 0 else "paper"
    first_author_clean = re.sub(r"[^a-zA-Z]", "", first_author)
    
    title_word = re.sub(r"[^a-zA-Z]", "", (title.split()[0] if title else "research")).lower()
    citation_key = f"{first_author_clean}{year}{title_word}"

    if arxiv_id:
        clean_arxiv = re.sub(r"v\d+$", "", arxiv_id.strip())
        return f"""@article{{{citation_key},
  title={{{clean_title}}},
  author={{{author_list}}},
  journal={{arXiv preprint arXiv:{clean_arxiv}}},
  year={{{year}}},
  eprint={{{clean_arxiv}}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.AI}}
}}"""
    else:
        return f"""@article{{{citation_key},
  title={{{clean_title}}},
  author={{{author_list}}},
  journal={{Research Paper Archive}},
  year={{{year}}}
}}"""
