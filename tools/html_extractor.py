"""Parse LaTeXML HTML (ar5iv / arxiv.org/html) into sections and structured data.

This module is pure: it takes an HTML string and returns an
:class:`HTMLExtraction`.  It never touches the network or the cache.

What it extracts
----------------
* ``sections`` -- ``abstract``, ``methodology``, ``setup``, ``experiments`` and
  ``limitations`` (appendix sections are included; a sub-section such as
  "A.4 Hyperparameters" is routed to ``setup`` even when its parent is not).
* ``tables`` -- ``ltx_tabular`` tables with caption, label, rows and the
  heading they appear under.
* ``references`` -- ``ltx_bibitem`` entries with arXiv id / DOI / URLs.
* ``repo_links`` -- code and project-page links with a primary-candidate flag.

Only the stdlib ``html.parser`` is used, as in the rest of the code base.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from config import config
from state import PaperReference, PaperTable, RepoLink

SECTION_KEYS: Tuple[str, ...] = ("abstract", "methodology", "setup", "experiments", "limitations")

# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------
MAX_TABLES = 40
MAX_TABLE_ROWS = 80
MAX_TABLE_COLS = 24
MAX_CELL_CHARS = 160
MAX_CAPTION_CHARS = 600
MAX_TABLE_RENDER_CHARS = 1500       # compact table text kept inside a section
MAX_REFERENCES = 500
MAX_REFERENCE_CHARS = 700
MAX_REPO_LINKS = 50
MAX_CONTEXT_CHARS = 200
MAX_MATH_CHARS = 400

# --------------------------------------------------------------------------
# Heading classification
# --------------------------------------------------------------------------
_RE_ABSTRACT = re.compile(r"\b(?:abstract|tóm tắt)\b")
_RE_REFERENCES = re.compile(r"\b(?:references?|bibliography|tài liệu tham khảo)\b")
_RE_LIMIT_EXPLICIT = re.compile(r"\b(?:limitations?|future work|giới hạn)\b")
_RE_DISCUSSION = re.compile(r"\bdiscussions?\b")
_RE_NEUTRAL = re.compile(
    r"\b(?:introduction|related works?|prior work|conclusions?|acknowledge?ments?|"
    r"acknowledgments?|broader impacts?|ethics?|ethical|reproducibility statement|"
    r"author contributions?|funding|giới thiệu|kết luận)\b"
)
# Terms that name the experimental setup itself.
_RE_SETUP_STRONG = re.compile(
    r"\b(?:set-?ups?|hyper-?parameters?|hardware|computing|compute|computational (?:resources?|cost|budget)|"
    r"configurations?|protocols?|"
    r"(?:experimental|experiments?|implementation|training|evaluation|optimi[sz]ation|dataset|"
    r"model|network|architecture|task)\s+(?:details?|settings?|procedures?|recipes?)|"
    r"details? of (?:the )?(?:experiments?|implementation|training))\b"
)
_RE_IMPLEMENTATION = re.compile(r"\bimplementations?\b")
_RE_EXPERIMENTS = re.compile(
    r"\b(?:experiments?|experimental|evaluations?|benchmarks?|results?|ablations?|empirical|"
    r"findings|thực nghiệm|kết quả)\b"
)
_RE_METHOD = re.compile(
    r"\b(?:methods?|methodology|approach(?:es)?|architectures?|models?|framework|proposed|system|"
    r"formulation|algorithms?|design|background|preliminar\w*|problem|phương pháp)\b"
)
_RE_DETAILS = re.compile(r"\bdetails?\b")
_RE_NOT_DETAILS = re.compile(r"\b(?:proofs?|derivations?|theor\w+|lemmas?)\b")


def classify_top_heading(title: str, *, appendix: bool, seen_experiments: bool) -> Optional[str]:
    """Classify a top-level heading.

    Returns one of ``abstract | methodology | setup | experiments |
    limitations | references`` or ``None`` (introduction, related work,
    conclusion, unrelated appendix material).
    """

    text = " ".join((title or "").lower().split())
    if not text:
        return None
    if _RE_ABSTRACT.search(text):
        return "abstract"
    if _RE_REFERENCES.search(text):
        return "references"
    if _RE_LIMIT_EXPLICIT.search(text):
        return "limitations"
    if _RE_SETUP_STRONG.search(text):
        return "setup"
    if _RE_EXPERIMENTS.search(text):
        return "experiments"
    if _RE_IMPLEMENTATION.search(text):
        return "setup"
    if _RE_DISCUSSION.search(text):
        return "limitations"
    if _RE_NEUTRAL.search(text):
        return None
    if _RE_METHOD.search(text):
        return "methodology"
    if appendix:
        if _RE_DETAILS.search(text) and not _RE_NOT_DETAILS.search(text):
            return "setup"
        return None
    # Descriptive numbered headings ("3 Key Design Decisions") are the paper's
    # own content: method before the experiments, experiment follow-up after.
    return "experiments" if seen_experiments else "methodology"


def classify_sub_heading(title: str) -> Optional[str]:
    """Classify a sub-heading that overrides its parent (``setup`` / ``limitations`` only)."""

    text = " ".join((title or "").lower().split())
    if not text:
        return None
    if _RE_LIMIT_EXPLICIT.search(text):
        return "limitations"
    if _RE_SETUP_STRONG.search(text) or (
        _RE_IMPLEMENTATION.search(text) and not _RE_EXPERIMENTS.search(text)
    ):
        return "setup"
    return None


# Sentences that name compute hardware are copied into ``setup`` wherever they
# appear (a paper often states "trained on 8 A100 GPUs" inside the method
# section, far beyond the method-section cap).
HARDWARE_RE = re.compile(
    r"\b(?:GPUs?|TPUs?(?:\s?v\d+)?|TPUv\d+|CPUs?|A100s?|V100s?|H100s?|H200s?|A6000|RTX\s?\d{3,4}|GTX\s?\d{3,4}|"
    r"NVIDIA|Nvidia|TFLOPs?|GPU[- ]hours?|(?:\d+\s?)?GB\s+(?:of\s+)?(?:memory|RAM|VRAM))\b"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])")
MAX_HARDWARE_SENTENCES = 8
MAX_HARDWARE_SENTENCE_CHARS = 400
MAX_HARDWARE_TOTAL_CHARS = 2000
_HARDWARE_PREFIX = "Hardware and compute mentions found elsewhere in the paper:"
_HARDWARE_SOURCES = frozenset({"methodology", "experiments", "limitations"})


# --------------------------------------------------------------------------
# Math / text helpers
# --------------------------------------------------------------------------
_TEX_SYMBOLS = {
    "pm": "±", "mp": "∓", "times": "×", "cdot": "·", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "ne": "≠", "approx": "≈", "sim": "~", "simeq": "≃", "to": "→", "rightarrow": "→",
    "leftarrow": "←", "leftrightarrow": "↔", "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔",
    "mapsto": "↦", "uparrow": "↑", "downarrow": "↓", "Uparrow": "⇑", "Downarrow": "⇓", "infty": "∞", "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "cup": "∪",
    "cap": "∩", "equiv": "≡", "propto": "∝", "ll": "≪", "gg": "≫", "nabla": "∇", "partial": "∂",
    "forall": "∀", "exists": "∃", "sum": "Σ", "prod": "Π", "int": "∫", "circ": "∘", "ast": "*",
    "star": "*", "odot": "⊙", "otimes": "⊗", "oplus": "⊕", "langle": "⟨", "rangle": "⟩", "ell": "ℓ",
    "hbar": "ℏ", "top": "⊤", "bot": "⊥", "prime": "′", "ldots": "...", "dots": "...", "cdots": "...",
    "vdots": "...", "ddots": "...", "mid": "|", "vert": "|", "Vert": "‖", "lVert": "‖", "rVert": "‖",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ",
    "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ",
    "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
    # layout-only commands
    "left": "", "right": "", "big": "", "Big": "", "bigl": "", "bigr": "", "Bigl": "", "Bigr": "",
    "displaystyle": "", "textstyle": "", "limits": "", "nolimits": "", "quad": " ", "qquad": " ",
    "hspace": "", "vspace": "", "mathstrut": "", "phantom": "", "nonumber": "", "label": "",
}
_TEX_STYLE_RE = re.compile(
    r"\\(?:bm|mathbf|mathrm|mathcal|mathbb|mathit|mathsf|mathtt|mathscr|mathfrak|text|textbf|textit|"
    r"texttt|textrm|textsf|operatorname|boldsymbol|mathop|mbox|hat|tilde|bar|vec|overline|underline|"
    r"widehat|widetilde|dot|ddot)\s*\{"
)
_TEX_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_TEX_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_TEX_SCRIPT_RE = re.compile(r"([_^])\s*\{([^{}]*)\}")
_TEX_CMD_RE = re.compile(r"\\([A-Za-z]+)")
_INVISIBLE_RE = re.compile("[\u200b-\u200f\u2060-\u2064\ufeff]")


def tex_to_text(tex: str) -> str:
    """Turn the TeX in a LaTeXML ``alttext`` attribute into short readable text."""

    text = (tex or "").strip()
    if not text:
        return ""
    if "\\" not in text and "{" not in text and "_" not in text and "^" not in text:
        return " ".join(text.split())
    text = text.replace("\\{", "\x01").replace("\\}", "\x02").replace("\\%", "%").replace("\\&", "&")
    text = text.replace("\\_", "\x03").replace("\\#", "#").replace("\\$", "$")
    text = text.replace("\\\\", " ")
    text = text.replace("\\!", "")
    text = re.sub(r"\\[,;: ]", " ", text)
    text = _TEX_STYLE_RE.sub("{", text)
    for _ in range(3):
        text = _TEX_FRAC_RE.sub(r"(\1)/(\2)", text)
    text = _TEX_SQRT_RE.sub(r"√(\1)", text)
    text = _TEX_CMD_RE.sub(lambda m: _TEX_SYMBOLS.get(m.group(1), m.group(1)), text)
    for _ in range(4):
        text = _TEX_SCRIPT_RE.sub(
            lambda m: f"{m.group(1)}{m.group(2)}" if len(m.group(2)) <= 1 else f"{m.group(1)}({m.group(2)})",
            text,
        )
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\x01", "{").replace("\x02", "}").replace("\x03", "_")
    return " ".join(text.split())


def _norm_edges(data: str) -> str:
    """Collapse whitespace but keep one leading/trailing space if present."""

    if not data:
        return ""
    core = " ".join(data.split())
    if not core:
        return " "
    return (" " if data[0].isspace() else "") + core + (" " if data[-1].isspace() else "")


def _clean_block_text(text: str) -> str:
    text = _INVISIBLE_RE.sub("", text).replace("\u00a0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _squash(text: str) -> str:
    return " ".join(_INVISIBLE_RE.sub("", text or "").replace("\u00a0", " ").split())


# --------------------------------------------------------------------------
# Reference helpers
# --------------------------------------------------------------------------
_ARXIV_NEW_RE = re.compile(
    r"(?i)(?:arxiv(?:\.org)?[\s:./]*(?:abs/|pdf/)?|\babs/|\bpdf/)(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])"
)
_ARXIV_OLD_RE = re.compile(
    r"(?i)(?:arxiv(?:\.org)?[\s:./]*(?:abs/|pdf/)?|\babs/|\bpdf/)([a-z][a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?(?![\d])"
)
_DOI_RE = re.compile(r"(?i)\b(10\.\d{4,9}/[^\s<>\"']+)")


def find_arxiv_id(text: str, urls: Optional[List[str]] = None) -> Optional[str]:
    """Return the first arXiv id (new or old style, no version) in text or URLs."""

    for source in [*(urls or []), text or ""]:
        for pattern in (_ARXIV_NEW_RE, _ARXIV_OLD_RE):
            match = pattern.search(source)
            if match:
                return match.group(1).lower() if "/" in match.group(1) else match.group(1)
    return None


def find_doi(text: str, urls: Optional[List[str]] = None) -> Optional[str]:
    """Return the first DOI found in URLs or text."""

    for source in [*(unquote(url) for url in (urls or [])), text or ""]:
        match = _DOI_RE.search(source)
        if match:
            return match.group(1).rstrip(".,;:)]}>")
    return None


_VENUE_START_RE = re.compile(r"(?i)^(?:in\b|arxiv|url\b|proceedings|journal|corr\b|technical report|preprint|available)")


_AUTHOR_YEAR_TITLE_RE = re.compile(r"\((?:19|20)\d\d[a-z]?\)\.?\s+(.{8,300}?[.?!])(?:\s|$)")
_VENUE_LOOKING_RE = re.compile(r"\d+\s*\(\d+\)|\bpp?\.\s*\d|\d+\s*[–-]\s*\d+|\bvol(?:ume)?\b", re.IGNORECASE)


def _guess_reference_title(blocks: List[str], text: str) -> str:
    quoted = re.search(r"[“\"]([^”\"]{8,300})[”\"]", text)
    author_year = _AUTHOR_YEAR_TITLE_RE.search(text)
    if quoted:
        candidate = quoted.group(1)
    elif author_year:
        candidate = author_year.group(1)
    elif len(blocks) >= 3:
        candidate = blocks[1]
    elif len(blocks) == 2 and not _VENUE_START_RE.match(blocks[1]) and not _VENUE_LOOKING_RE.search(blocks[1]):
        candidate = blocks[1]
    else:
        return ""
    candidate = re.split(r"\s+URL\s|https?://", candidate)[0]
    candidate = re.sub(r"[,.]?\s*(?:19|20)\d\d[a-z]?\.?\s*$", "", candidate.strip())
    candidate = candidate.strip(" .,;")
    return candidate[:300] if len(candidate) >= 4 else ""


# --------------------------------------------------------------------------
# Link helpers
# --------------------------------------------------------------------------
_GITHUB_RESERVED = {
    "orgs", "features", "topics", "sponsors", "marketplace", "settings", "about", "pricing", "login",
    "join", "explore", "collections", "events", "apps", "enterprise", "security", "site", "notifications",
}
_CHROME_REPOS = {"dginev/ar5iv", "arxiv/html_feedback", "brucemiller/latexml"}
_PROJECT_HOST_SUFFIXES = (".github.io", ".gitlab.io", ".netlify.app", ".pages.dev", ".vercel.app")
_CODE_HOSTS = {
    "gitlab.com", "bitbucket.org", "codeberg.org", "huggingface.co", "zenodo.org",
    "colab.research.google.com", "kaggle.com", "osf.io", "figshare.com", "sourceforge.net",
}
_DENY_HOSTS = {
    "arxiv.org", "info.arxiv.org", "ar5iv.labs.arxiv.org", "ar5iv.org", "doi.org", "dx.doi.org",
    "openreview.net", "aclanthology.org", "dlmf.nist.gov", "scholar.google.com", "semanticscholar.org",
    "creativecommons.org", "wikipedia.org", "en.wikipedia.org", "twitter.com", "x.com", "youtube.com",
    "youtu.be", "latexml.mathweb.org", "proceedings.mlr.press", "papers.nips.cc", "ieeexplore.ieee.org",
    "dl.acm.org", "link.springer.com", "nature.com", "science.org",
}

_STRONG_CUE_RE = re.compile(
    r"(?i)project (?:page|website|site|webpage)|\bour (?:code|implementation|source|codebase|project|repo|repository|"
    r"website|webpage|models?|library)\b|\bwe (?:have )?(?:publicly )?(?:release[ds]?|open[- ]?sourced?|make (?:our|the) (?:code|implementation))\b|"
    r"\b(?:code|implementations?|models?|checkpoints?|weights|data(?:sets?)?)\b[^.]{0,60}?\b(?:is|are|will be|has been|have been)\s+"
    r"(?:made\s+)?(?:publicly\s+|freely\s+|now\s+|also\s+)?(?:available|released|open[- ]?sourced?)|"
    r"\bopen[- ]?source (?:code|implementation)|\bsource code\b|\b(?:code|implementation) (?:release|repository)\b|"
    r"\b(?:released|available) (?:at|on|under)\b|\bhomepage\b|\bwebpage\b|\bwebsite\b|"
    r"\b(?:videos?|visuali[sz]ations?|demos?|more results)\b[^.]{0,40}?\b(?:at|on|see|viewable|available)\b"
)
_WEAK_CUE_RE = re.compile(r"(?i)\bcode\b|\bavailable\b|\breleas\w*|\bimplementation\b|\bgithub\b|\brepositor\w+|\bopen[- ]?source\b")
_NEGATIVE_CUE_RE = re.compile(
    r"(?i)\bbaselines?\b|\bfollow(?:ing|s|ed)?\b|\bbased on\b|\bbuilt (?:on|upon|atop)\b|\badapted from\b|\bborrow\w*\b|"
    r"\bofficial implementations? (?:of|from)\b|\bwe (?:use|used|ran|run|adopt|adopted|rely|relied)\b|\bthanks?\b|"
    r"\backnowledg\w*\b|\breference implementations?\b|\bfrom the (?:repository|repo|github)\b|\bpublic implementation\b|"
    r"\bimplementation (?:from|by|of)\b|\bunofficial\b|\bforked?\b|\bmodified version\b"
)
_PROJECT_CUE_RE = re.compile(
    r"(?i)project (?:page|website|site|webpage)|\bwebsite\b|\bwebpage\b|\bhomepage\b|\bdemos?\b|"
    r"\b(?:videos?|visuali[sz]ations?|more results)\b|\bcode\b|\bavailable (?:at|on)\b"
)
_URL_TRAIL = ".,;:)]}>\"'"


def _normalize_link(url: str) -> Optional[Tuple[str, Optional[str]]]:
    """Return ``(normalized_url, kind)``; ``kind`` is ``None`` when it needs a context cue."""

    raw = (url or "").strip().rstrip(_URL_TRAIL)
    try:
        parsed = urlparse(raw)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    parts = [segment for segment in parsed.path.split("/") if segment]
    if host == "github.com":
        if len(parts) >= 2 and parts[0].lower() not in _GITHUB_RESERVED:
            repo = re.sub(r"\.git$", "", parts[1])
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[0]) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
                return None
            if f"{parts[0]}/{repo}".lower() in _CHROME_REPOS:
                return None
            return f"https://github.com/{parts[0]}/{repo}", "github"
        if len(parts) == 1 and parts[0].lower() not in _GITHUB_RESERVED:
            return f"https://github.com/{parts[0]}", "other"
        return None
    if host.endswith(_PROJECT_HOST_SUFFIXES):
        path = parsed.path.rstrip("/")
        return f"https://{host}{path}", "project_page"
    if host in _CODE_HOSTS:
        path = parsed.path.rstrip("/")
        return f"https://{host}{path}", "other"
    if host in _DENY_HOSTS or any(host.endswith("." + denied) for denied in _DENY_HOSTS):
        return None
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{host}{path}", None


def _link_context(flow: str, start: int, end: int) -> Tuple[str, str, str]:
    """Return ``(before, link_text, after)`` limited to the link's own line."""

    line_start = flow.rfind("\n", 0, start) + 1
    line_end = flow.find("\n", end)
    if line_end < 0:
        line_end = len(flow)
    link = _squash(flow[start:end])
    if len(link) > 60:
        link = link[:57] + "..."
    before_start = max(line_start, start - 90)
    before = flow[before_start:start]
    if before_start > line_start and " " in before:
        before = before.split(" ", 1)[1]
    after_end = min(line_end, end + 90)
    after = flow[end:after_end]
    if after_end < line_end and " " in after:
        after = after.rsplit(" ", 1)[0]
    return _squash(before), link, _squash(after)


def _join_context(before: str, link: str, after: str) -> str:
    text = " ".join(part for part in (before, link, after) if part)
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS].rsplit(" ", 1)[0] or text[:MAX_CONTEXT_CHARS]
    return text


def _is_primary_candidate(
    before: str, after: str, *, in_bibliography: bool, front_matter: bool, section: Optional[str]
) -> bool:
    if in_bibliography:
        return False
    if front_matter:      # title block, abstract, first footnote
        return True
    context = f"{before} {after}"
    negative_near = bool(_NEGATIVE_CUE_RE.search(before[-80:]))
    if _STRONG_CUE_RE.search(context) and not negative_near:
        return True
    if section in {"experiments", "setup"}:
        return False
    return bool(_WEAK_CUE_RE.search(context)) and not _NEGATIVE_CUE_RE.search(context)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
_VOID_TAGS = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})
_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "svg", "head", "header", "footer", "nav", "button", "select", "option",
    "textarea", "form", "dialog", "template",
})
_SKIP_CLASSES = frozenset({
    "ltx_page_footer", "ltx_page_logo", "ltx_note_mark", "ltx_note_type", "ltx_tag_equation", "ltx_ERROR",
    "ltx_phantom", "ltx_pagination",
})
_BLOCK_BREAK_TAGS = frozenset({
    "p", "div", "li", "ul", "ol", "section", "article", "tr", "figcaption", "figure", "table", "blockquote",
    "pre", "dd", "dt", "dl", "caption", "h1", "h2", "h3", "h4", "h5", "h6",
})
_HEADING_LEVELS = {
    "part": 1, "chapter": 1, "section": 1, "appendix": 1, "bibliography": 1,
    "subsection": 2, "subsubsection": 3, "paragraph": 4, "subparagraph": 5,
}
_IGNORED_HEADINGS = frozenset({"abstract", "keywords", "classification", "acknowledgements"})
_LABEL_RE = re.compile(r"(?i)\b(table|figure|tab\.|fig\.)\s*([A-Z]?\d+(?:\.\d+)?)")


class _Frame:
    __slots__ = ("tag", "classes", "id", "skip", "kind", "data")

    def __init__(self, tag: str, classes: frozenset, ident: str) -> None:
        self.tag = tag
        self.classes = classes
        self.id = ident
        self.skip = False
        self.kind = ""
        self.data: Dict[str, Any] = {}


@dataclass
class HTMLExtraction:
    title: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    tables: List[PaperTable] = field(default_factory=list)
    references: List[PaperReference] = field(default_factory=list)
    repo_links: List[RepoLink] = field(default_factory=list)
    section_titles: List[str] = field(default_factory=list)
    heading_seen: bool = False

    @property
    def code_urls(self) -> List[str]:
        """GitHub repositories that look like the paper's own code."""

        urls: List[str] = []
        for link in self.repo_links:
            if link.kind == "github" and link.is_primary_candidate and link.url not in urls:
                urls.append(link.url)
        return urls[:8]


class _LaTeXMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: List[_Frame] = []
        self._skip = 0
        self._flow: List[str] = []
        self._flow_len = 0
        self._parts: Dict[str, List[str]] = {key: [] for key in SECTION_KEYS}

        # heading / section state
        self.title = ""
        self.section_titles: List[str] = []
        self.heading_seen = False
        self._front = True
        self._abstract_depth = 0
        self._top_kind: Optional[str] = None
        self._sub_kind: Optional[str] = None
        self._sub_level: Optional[int] = None
        self._seen_experiments = False
        self._trail: Dict[int, str] = {}

        # tables
        self._tab: Optional[Dict[str, Any]] = None
        self._tab_depth = 0
        self.raw_tables: List[Dict[str, Any]] = []

        # bibliography
        self._bib: Optional[Dict[str, Any]] = None
        self.references: List[PaperReference] = []

        # links
        self._links: List[Dict[str, Any]] = []

        # hardware / compute sentences found outside the setup section
        self._hardware: List[str] = []

    # -- routing ---------------------------------------------------------
    def _target(self) -> Optional[str]:
        if self._abstract_depth:
            return "abstract"
        if self._bib is not None or self._top_kind == "references":
            return None
        return self._sub_kind or self._top_kind

    def _flow_add(self, text: str) -> None:
        if text:
            self._flow.append(text)
            self._flow_len += len(text)

    def _section_add(self, text: str) -> None:
        target = self._target()
        if target in self._parts and text:
            self._parts[target].append(text)

    def _break(self) -> None:
        self._flow_add("\n")
        if self._tab is None and self._bib is None:
            self._section_add("\n")

    def _current_heading(self) -> str:
        for level in sorted(self._trail, reverse=True):
            return self._trail[level]
        return ""

    def _enclosing(self, kind: str) -> Optional[_Frame]:
        for frame in reversed(self._stack):
            if frame.kind == kind:
                return frame
        return None

    def _emit(self, text: str) -> None:
        if not text:
            return
        self._flow_add(text)
        heading = self._enclosing("heading")
        if heading is not None:
            heading.data["parts"].append(text)
            return
        caption = self._enclosing("caption")
        if caption is not None:
            owner = caption.data.get("owner")
            in_tag = self._enclosing("captag") is not None
            if owner is not None:
                owner.data["label_parts" if in_tag else "caption_parts"].append(text)
                if owner.data["is_table"]:
                    return
            self._section_add(text)
            return
        if self._bib is not None:
            bib = self._bib
            if self._enclosing("biblabel") is not None:
                return
            if bib["block"] is None:
                bib["block"] = []
                bib["blocks"].append(bib["block"])
            bib["block"].append(text)
            return
        if self._tab is not None:
            cell = self._tab.get("cell")
            if cell is not None:
                cell["parts"].append(text)
            return
        if self._enclosing("doctitle") is not None:
            self.title += text
            return
        self._section_add(text)

    # -- html.parser callbacks ------------------------------------------
    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        if tag in _VOID_TAGS:
            if tag == "br" and not self._skip:
                self._emit(" ")
            return
        classes = frozenset(attributes.get("class", "").split())
        frame = _Frame(tag, classes, attributes.get("id", ""))
        self._stack.append(frame)
        if (
            tag in _SKIP_TAGS
            or classes & _SKIP_CLASSES
            or any(name.startswith("ar5iv-") or "modal" in name for name in classes)
        ):
            frame.skip = True
            self._skip += 1
            return
        if self._skip:
            return
        self._classify_start(frame, attributes)

    def _classify_start(self, frame: _Frame, attributes: Dict[str, str]) -> None:
        tag, classes = frame.tag, frame.classes
        if tag == "p":
            frame.data["flow_idx"] = len(self._flow)

        if "ltx_abstract" in classes:
            frame.kind = "abstract"
            self._abstract_depth += 1
            return

        if "ltx_title" in classes:
            title_type = next((c[len("ltx_title_"):] for c in classes if c.startswith("ltx_title_")), "")
            if title_type == "document":
                frame.kind = "doctitle"
                return
            if title_type in _IGNORED_HEADINGS:
                frame.kind = "ignored_heading"
                self._skip += 1
                frame.skip = True
                return
            if title_type in _HEADING_LEVELS:
                frame.kind = "heading"
                frame.data = {"parts": [], "level": _HEADING_LEVELS[title_type], "type": title_type}
                if frame.data["level"] == 1:
                    self._front = False
                self._flow_add("\n")
                return

        if tag == "math":
            alttext = attributes.get("alttext", "")
            if alttext.strip():
                text = tex_to_text(alttext)[:MAX_MATH_CHARS]
                if attributes.get("display") == "block":
                    text = f" {text} "
                self._emit(text)
                frame.kind = "math"
                frame.skip = True
                self._skip += 1
            return

        if tag == "figcaption" or "ltx_caption" in classes:
            owner = self._enclosing("container")
            frame.kind = "caption"
            frame.data = {"owner": owner}
            return

        if "ltx_tag" in classes and self._enclosing("caption") is not None:
            frame.kind = "captag"
            return

        if tag in {"figure", "div", "span", "section"} and ("ltx_table" in classes or "ltx_figure" in classes):
            frame.kind = "container"
            frame.data = {
                "caption_parts": [], "label_parts": [], "is_table": "ltx_table" in classes,
                "rendered": False,
            }
            return

        if tag == "table" and "ltx_tabular" in classes:
            self._tab_depth += 1
            if self._tab is None:
                frame.kind = "tabular_root"
                self._tab = {
                    "rows": [], "row": None, "cell": None, "pending": {},
                    "containers": [f for f in self._stack if f.kind == "container"],
                    "section": self._current_heading(),
                    "id": frame.id,
                }
            else:
                frame.kind = "tabular_nested"
            return

        if self._tab is not None and self._tab_depth == 1:
            if tag == "tr":
                frame.kind = "row"
                self._tab["row"] = []
                return
            if tag in {"td", "th"} and self._tab["row"] is not None:
                frame.kind = "cell"
                self._tab["cell"] = {
                    "parts": [],
                    "colspan": _int_attr(attributes.get("colspan"), 50),
                    "rowspan": _int_attr(attributes.get("rowspan"), 100),
                }
                return

        if "ltx_bibitem" in classes:
            frame.kind = "bibitem"
            self._bib = {"blocks": [], "block": None, "urls": [], "label": ""}
            return
        if self._bib is not None:
            if "ltx_bibblock" in classes:
                frame.kind = "bibblock"
                self._bib["block"] = []
                self._bib["blocks"].append(self._bib["block"])
                return
            if "ltx_tag_bibitem" in classes or "ltx_role_refnum" in classes:
                frame.kind = "biblabel"
                return

        if "ltx_note" in classes:
            frame.kind = "note"
            return

        if tag == "a":
            href = _absolute_href(attributes.get("href", "").strip(), classes)
            if re.match(r"(?i)https?://", href):
                frame.kind = "anchor"
                frame.data = {
                    "href": href,
                    "start": self._flow_len,
                    "in_bib": self._bib is not None,
                    "front": self._front,
                    "section": self._target(),
                }
                if self._bib is not None and href not in self._bib["urls"]:
                    self._bib["urls"].append(href)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            return
        index = -1
        for position in range(len(self._stack) - 1, -1, -1):
            if self._stack[position].tag == tag:
                index = position
                break
        if index < 0:
            return
        while len(self._stack) > index:
            frame = self._stack.pop()
            self._close(frame)

    def handle_data(self, data: str) -> None:
        if self._skip or not data:
            return
        text = _norm_edges(data)
        if text:
            self._emit(text)

    # -- closing ---------------------------------------------------------
    def _close(self, frame: _Frame) -> None:
        if frame.skip:
            self._skip -= 1
            return
        kind = frame.kind
        if kind == "heading":
            self._finish_heading(frame)
        elif kind == "abstract":
            self._abstract_depth -= 1
        elif kind == "container":
            self._finish_container(frame)
        elif kind == "tabular_root":
            self._finish_table(frame)
            self._tab_depth -= 1
        elif kind == "tabular_nested":
            self._tab_depth -= 1
        elif kind == "cell":
            self._finish_cell()
        elif kind == "row":
            self._finish_row()
        elif kind == "bibblock":
            if self._bib is not None:
                self._bib["block"] = None
        elif kind == "bibitem":
            self._finish_bibitem()
        elif kind == "anchor":
            self._finish_anchor(frame)
        if frame.tag == "p" and "flow_idx" in frame.data:
            self._finish_paragraph(frame)
        if frame.tag in {"td", "th"}:
            self._flow_add(" ")
        elif frame.tag in _BLOCK_BREAK_TAGS:
            self._break()

    def _finish_paragraph(self, frame: _Frame) -> None:
        if (
            self._target() not in _HARDWARE_SOURCES
            or self._tab is not None
            or self._bib is not None
            or len(self._hardware) >= MAX_HARDWARE_SENTENCES
        ):
            return
        text = _squash("".join(self._flow[frame.data["flow_idx"]:]))
        if not HARDWARE_RE.search(text):
            return
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            sentence = sentence.strip()
            if (
                sentence
                and HARDWARE_RE.search(sentence)
                and sentence not in self._hardware
                and len(self._hardware) < MAX_HARDWARE_SENTENCES
            ):
                self._hardware.append(sentence[:MAX_HARDWARE_SENTENCE_CHARS])

    def _finish_heading(self, frame: _Frame) -> None:
        title = _squash("".join(frame.data["parts"]))
        level = frame.data["level"]
        if not title:
            return
        if level == 1:
            appendix = frame.data["type"] == "appendix" or bool(re.match(r"(?i)appendix\b", title))
            kind = "references" if frame.data["type"] == "bibliography" else classify_top_heading(
                title, appendix=appendix, seen_experiments=self._seen_experiments
            )
            self._top_kind = kind
            self._sub_kind = None
            self._sub_level = None
            if kind in {"experiments", "setup"}:
                self._seen_experiments = True
            self._trail = {1: title}
            self.heading_seen = True
            self.section_titles.append(title)
        else:
            if self._sub_level is not None and level <= self._sub_level:
                self._sub_kind = None
                self._sub_level = None
            self._trail = {k: v for k, v in self._trail.items() if k < level}
            self._trail[level] = title
            override = classify_sub_heading(title)
            if override and self._sub_kind is None:
                self._sub_kind = override
                self._sub_level = level
        self._section_add(f"\n{title}\n")

    def _finish_container(self, frame: _Frame) -> None:
        data = frame.data
        if data["is_table"] and not data["rendered"]:
            caption = _squash("".join(data["caption_parts"]))
            label = _squash("".join(data["label_parts"]))
            line = " ".join(part for part in (label, caption) if part)
            if line:
                self._section_add(f"\n[{line[:MAX_CAPTION_CHARS]}]\n")

    # -- tables ----------------------------------------------------------
    def _finish_cell(self) -> None:
        tab = self._tab
        if tab is None or tab["cell"] is None:
            return
        cell = tab["cell"]
        tab["cell"] = None
        text = _squash("".join(cell["parts"]))
        if len(text) > MAX_CELL_CHARS:
            text = text[: MAX_CELL_CHARS - 3] + "..."
        row: List[str] = tab["row"]
        pending: Dict[int, Tuple[int, str]] = tab["pending"]
        self._fill_pending(row, pending)
        start = len(row)
        colspan, rowspan = cell["colspan"], cell["rowspan"]
        row.extend([text] * colspan)
        if rowspan > 1:
            for offset in range(colspan):
                pending[start + offset] = (rowspan - 1, text)

    @staticmethod
    def _fill_pending(row: List[str], pending: Dict[int, Tuple[int, str]]) -> None:
        while len(row) in pending:
            column = len(row)
            remaining, text = pending[column]
            row.append(text)
            if remaining > 1:
                pending[column] = (remaining - 1, text)
            else:
                del pending[column]

    def _finish_row(self) -> None:
        tab = self._tab
        if tab is None or tab["row"] is None:
            return
        row: List[str] = tab["row"]
        pending: Dict[int, Tuple[int, str]] = tab["pending"]
        while any(column >= len(row) for column in pending):
            if len(row) in pending:
                self._fill_pending(row, pending)
            else:
                row.append("")
        tab["rows"].append(row)
        tab["row"] = None

    def _finish_table(self, frame: _Frame) -> None:
        tab = self._tab
        self._tab = None
        if tab is None:
            return
        rows = [row[:MAX_TABLE_COLS] for row in tab["rows"] if any(cell for cell in row)]
        rows = rows[:MAX_TABLE_ROWS]
        non_empty = sum(1 for row in rows for cell in row if cell)
        in_table_container = any(container.data.get("is_table") for container in tab["containers"])
        if non_empty < 2 or (len(rows) < 2 and not in_table_container):
            return
        raw = {
            "rows": rows, "containers": tab["containers"], "section": tab["section"], "id": tab["id"],
        }
        self.raw_tables.append(raw)
        for container in tab["containers"]:
            container.data["rendered"] = True
        # Keep a compact rendering in the section text so the numbers stay
        # visible to code that still reads only ``sections``.
        label, caption = _table_caption(raw)
        head = " ".join(part for part in (label, caption) if part)
        body = "\n".join(" | ".join(row) for row in rows)
        rendered = (f"[{head}]\n" if head else "") + body
        self._section_add(f"\n{rendered[:MAX_TABLE_RENDER_CHARS]}\n")

    # -- bibliography ----------------------------------------------------
    def _finish_bibitem(self) -> None:
        bib = self._bib
        self._bib = None
        if bib is None or len(self.references) >= MAX_REFERENCES:
            return
        blocks = [b for b in (_squash("".join(parts)) for parts in bib["blocks"]) if b]
        text = " ".join(blocks)
        if not text and not bib["urls"]:
            return
        urls = bib["urls"][:8]
        self.references.append(
            PaperReference(
                index=len(self.references) + 1,
                text=text[:MAX_REFERENCE_CHARS],
                title=_guess_reference_title(blocks, text),
                arxiv_id=find_arxiv_id(text, urls),
                doi=find_doi(text, urls),
                urls=urls,
            )
        )

    # -- links -----------------------------------------------------------
    def _finish_anchor(self, frame: _Frame) -> None:
        data = dict(frame.data)
        data["end"] = self._flow_len
        self._links.append(data)

    def build_links(self) -> List[RepoLink]:
        flow = "".join(self._flow)
        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for item in self._links:
            normalized = _normalize_link(item["href"])
            if normalized is None:
                continue
            url, kind = normalized
            before, link_text, after = _link_context(flow, item["start"], item["end"])
            if kind is None:
                # Unknown host: keep only when the text around it says it is a project page.
                if item["in_bib"] or not (
                    _PROJECT_CUE_RE.search(f"{before} {after}") and (item["front"] or _STRONG_CUE_RE.search(f"{before} {after}"))
                ):
                    continue
                kind = "project_page"
            primary = _is_primary_candidate(
                before, after, in_bibliography=item["in_bib"], front_matter=item["front"], section=item["section"],
            )
            key = url.lower()
            entry = merged.get(key)
            context = _join_context(before, link_text, after)
            if entry is None:
                merged[key] = {
                    "url": url, "kind": kind, "context": context,
                    "in_bibliography": item["in_bib"], "primary": primary,
                }
                order.append(key)
                continue
            entry["in_bibliography"] = entry["in_bibliography"] and item["in_bib"]
            if primary and not entry["primary"]:
                entry["primary"] = True
                entry["context"] = context
            elif entry["in_bibliography"] is False and not item["in_bib"] and not entry["primary"] and not entry["context"]:
                entry["context"] = context
        links = [
            RepoLink(
                url=merged[key]["url"],
                kind=merged[key]["kind"],
                context=merged[key]["context"],
                in_bibliography=merged[key]["in_bibliography"],
                is_primary_candidate=bool(merged[key]["primary"]) and not merged[key]["in_bibliography"],
            )
            for key in order
        ]
        return links[:MAX_REPO_LINKS]

    def build_tables(self) -> List[PaperTable]:
        tables: List[PaperTable] = []
        for raw in self.raw_tables[:MAX_TABLES]:
            label, caption = _table_caption(raw)
            tables.append(PaperTable(caption=caption, label=label, rows=raw["rows"], section=raw["section"]))
        return tables

    def section_text(self) -> Dict[str, str]:
        sections = {key: _clean_block_text("".join(parts)) for key, parts in self._parts.items()}
        if self._hardware:
            mentions: List[str] = []
            total = 0
            for sentence in self._hardware:
                if total + len(sentence) > MAX_HARDWARE_TOTAL_CHARS:
                    break
                mentions.append(sentence)
                total += len(sentence)
            if mentions:
                sections["setup"] = "\n".join([_HARDWARE_PREFIX, *mentions, sections["setup"]]).strip()
        return sections


_BARE_DOMAIN_RE = re.compile(r"(?i)^(?:[a-z0-9-]+\.)+[a-z]{2,}(?:[/:?#].*)?$")
_FILE_SUFFIX_RE = re.compile(r"(?i)\.(?:png|jpe?g|gif|svg|pdf|html?|css|js|txt|bib|tex|eps|zip)$")


def _absolute_href(href: str, classes: frozenset) -> str:
    """LaTeXML ``\\url{example.org}`` can produce a scheme-less href; add https://."""

    if "ltx_url" in classes and _BARE_DOMAIN_RE.match(href) and not _FILE_SUFFIX_RE.search(href):
        return "https://" + href
    return href


def _int_attr(value: Optional[str], maximum: int) -> int:
    try:
        number = int(str(value or "1").strip())
    except ValueError:
        return 1
    return max(1, min(number, maximum))


def _table_caption(raw: Dict[str, Any]) -> Tuple[str, str]:
    """Pick label and caption from the innermost container that has a caption."""

    for container in reversed(raw["containers"]):
        caption = _squash("".join(container.data["caption_parts"]))
        label = _squash("".join(container.data["label_parts"])).rstrip(": ").strip()
        if caption or label:
            match = _LABEL_RE.search(label or caption[:40])
            if match:
                name = match.group(1).rstrip(".").capitalize()
                name = {"Tab": "Table", "Fig": "Figure"}.get(name, name)
                label = f"{name} {match.group(2)}"
            return label, caption[:MAX_CAPTION_CHARS]
    ident = raw.get("id") or ""
    match = re.search(r"\.T(\d+)", ident) or re.fullmatch(r"T(\d+)", ident)
    return (f"Table {match.group(1)}" if match else ""), ""


_LATEXML_RE = re.compile(r"""class\s*=\s*["'][^"']*\bltx_(?:document|title_document|title_section|abstract|para|bibitem)\b""")


def looks_like_latexml(raw_html: str) -> bool:
    """True when the page carries LaTeXML paper markup (not an error/landing page)."""

    return bool(raw_html) and bool(_LATEXML_RE.search(raw_html))


def extract_html(raw_html: str) -> HTMLExtraction:
    """Parse a LaTeXML HTML page into sections, tables, references and links."""

    parser = _LaTeXMLParser()
    parser.feed(raw_html or "")
    parser.close()
    caps = {
        "abstract": config.HTML_ABSTRACT_MAX_CHARS,
        "methodology": config.HTML_METHODOLOGY_MAX_CHARS,
        "setup": config.HTML_SETUP_MAX_CHARS,
        "experiments": config.HTML_EXPERIMENTS_MAX_CHARS,
        "limitations": config.HTML_LIMITATIONS_MAX_CHARS,
    }
    sections = {key: text[: caps[key]] for key, text in parser.section_text().items()}
    return HTMLExtraction(
        title=_squash(parser.title),
        sections=sections,
        tables=parser.build_tables(),
        references=parser.references,
        repo_links=parser.build_links(),
        section_titles=parser.section_titles,
        heading_seen=parser.heading_seen,
    )
