"""FastAPI bridge for the LangGraph Research Scout workflow.

This module is the small HTTP boundary used by the React client: it owns an in-memory run
registry, streams safe operational events, and exposes persisted Markdown
reports without leaking local paths or extracted PDF text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, Literal, Mapping, Sequence
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

from config import config, provider_is_configured, resolved_provider
from graph import build_research_graph
from state import PaperItem


logger = logging.getLogger(__name__)

_PROTECTED_HTTP_URL_RE = re.compile(r"(?i)https?://[^\s,;}\])]+")


RunStatus = Literal["idle", "running", "success", "degraded", "error"]
TraceType = Literal[
    "step.started",
    "step.updated",
    "step.completed",
    "assistant.delta",
    "assistant.completed",
    "run.completed",
    "run.failed",
]


class TraceFacts(TypedDict, total=False):
    """Allowlisted, structured facts from a public state update.

    The graph state contains private PDF/cache fields and free-form model
    output.  Only these measured counters and identifiers cross the HTTP
    boundary.  ``total=False`` keeps facts sparse: an absent value means that
    the corresponding node update did not provide that measurement.
    """

    intent: Literal["direct_read", "search", "followup"]
    sources: list[str]         # answer_followup: report sections, tables and notes the answer used
    resultCount: int
    selectedCount: int
    retryCount: int
    paperCount: int
    failedPdfCount: int
    repositoryCount: int
    bibtexCount: int
    noteFallbackPaperIds: list[str]
    reportAvailable: bool
    reportId: str
    # Stage facts (chat UI).  ``paperFacts`` holds one allowlisted dict per
    # paper; the scalar keys are totals over those papers.
    paperFacts: list[dict[str, Any]]
    source: str                # read_paper: ar5iv | arxiv_html | pdf | mixed | unknown
    tableCount: int            # read_paper
    referenceCount: int        # read_paper
    hardwareFoundCount: int    # read_paper: papers whose text names GPU/TPU hardware
    insightCount: int          # write_notes
    evalSettings: dict[str, int]  # write_notes: proposed / kept / dropped
    hardwareStatedCount: int   # write_notes: papers with grounded hardware
    seeds: int                 # citation_graph
    pool: int
    nodeCount: int
    edgeCount: int
    graphStatus: str
    refsTotal: int
    refsResolved: int
    durationMs: int
    repoSource: str            # web_enrich: html | github_search | none | mixed
    sectionCount: int          # final_report: "## " headings in the report
    # Numeric measurements emitted by nodes/tools.  Keys are an allowlisted
    # timing name (for example ``llm`` or ``github.search``), never a prompt,
    # URL, cache path, or provider payload.
    timings: dict[str, int]


class APIModel(BaseModel):
    """Pydantic defaults shared by public DTOs.

    The API emits camelCase for the React client but accepts snake_case when a
    Python test or an internal caller constructs a model directly.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class TraceEvent(APIModel):
    run_id: str = Field(alias="runId")
    seq: int
    type: TraceType
    node: str
    label: str
    kind: str = "workflow"
    # Filled from ``node`` when omitted: a one-line description, the chat-UI
    # stage number (1-5, ``None`` for topic-run nodes) and the parallel group.
    description: str | None = None
    stage: int | None = None
    parallel_group: str | None = Field(default=None, alias="parallelGroup")
    status: str = "running"
    summary: str = ""
    details: str | None = None
    # ``facts`` is populated only for state-update events.  Native task
    # lifecycle events use the existing text details and intentionally carry
    # no graph-state facts.
    facts: TraceFacts | None = None
    # Present only for ``assistant.delta``.  It is the newly generated plain
    # text segment, never a task input/result or tool-call payload.
    delta: str | None = None
    links: list[dict[str, str]] = Field(default_factory=list)
    # A numeric field keeps the UI from having to parse the human-readable
    # ``details`` string.  Older events omit it and remain valid.
    duration_ms: int | None = Field(default=None, alias="durationMs", ge=0)
    timings: dict[str, int] = Field(default_factory=dict)
    timestamp: datetime

    progress: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _fill_stage_fields(self) -> "TraceEvent":
        if self.description is None:
            self.description = NODE_DESCRIPTIONS.get(self.node)
        if self.stage is None:
            self.stage = NODE_STAGES.get(self.node)
        if self.parallel_group is None:
            self.parallel_group = NODE_PARALLEL_GROUPS.get(self.node)
        return self


class RunInput(APIModel):
    query: str = ""
    paper_inputs: list[str] = Field(default_factory=list, alias="paperInputs")
    pdf_names: list[str] = Field(default_factory=list, alias="pdfNames")
    file_names: list[str] = Field(default_factory=list, alias="fileNames")


class PMRLDTO(APIModel):
    problem: str = ""
    method: str = ""
    result: str = ""
    limitation: str = ""
    status: Literal["complete", "missing", "invalid", "unknown"] = "unknown"


class InsightDTO(APIModel):
    finding: str = ""
    why_it_matters: str = Field(default="", alias="whyItMatters")


class EvalSettingDTO(APIModel):
    """One grounded result row (rows that failed grounding never reach the DTO)."""

    benchmark: str = ""
    metric: str = ""
    metric_meaning: str = Field(default="", alias="metricMeaning")
    higher_is_better: bool | None = Field(default=None, alias="higherIsBetter")
    unit: str = ""
    method_value: str = Field(default="", alias="methodValue")
    baseline_name: str = Field(default="", alias="baselineName")
    baseline_value: str = Field(default="", alias="baselineValue")
    table_label: str = Field(default="", alias="tableLabel")


class SetupDTO(APIModel):
    datasets: str = ""
    training_config: str = Field(default="", alias="trainingConfig")
    hardware: str = ""
    compute: str = ""


class RepositoryDTO(APIModel):
    url: str
    # ``html``: link found in the paper's own HTML; ``github``: GitHub search hit.
    source: Literal["html", "github"]


class SourceQualityDTO(APIModel):
    """Measured parser metadata safe to show beside one paper's source."""

    heading: Literal["heading", "fallback", "unknown"] = "unknown"
    parser_status: Literal["success", "degraded", "error", "unknown"] = Field(
        default="unknown", alias="parserStatus"
    )
    coverage_status: Literal["complete", "incomplete", "unknown"] = Field(
        default="unknown", alias="coverageStatus"
    )
    missing_sections: list[str] = Field(default_factory=list, alias="missingSections")
    source_format: Literal["pdf", "html", "mixed", "unknown"] = Field(default="unknown", alias="sourceFormat")
    code_status: Literal["verified", "unknown"] = Field(default="unknown", alias="codeStatus")
    code_urls: list[str] = Field(default_factory=list, alias="codeUrls")
    warning: str | None = None


class RepoDTO(APIModel):
    name: str
    url: str
    stars: int = 0
    framework: str = "Python"
    is_official: bool = Field(default=False, alias="isOfficial")
    description: str = ""


class PaperDTO(APIModel):
    """Safe paper metadata returned to the browser.

    Deliberately absent: ``extracted_text``, ``sections``, and
    ``local_pdf_path``.  The graph may use those fields internally, but the
    bridge never serializes them to a client.
    """

    paper_id: str = Field(alias="paperId")
    title: str
    summary: str = ""
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    subjects: list[str] = Field(default_factory=list)
    metadata_status: Literal["complete", "missing", "unknown"] = Field(default="unknown", alias="metadataStatus")
    source_type: str = Field(default="arxiv", alias="sourceType")
    arxiv_id: str | None = Field(default=None, alias="arxivId")
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    github_repos: list[RepoDTO] = Field(default_factory=list, alias="githubRepos")
    bibtex: str | None = None
    notes: PMRLDTO | None = None
    intuition: str = ""
    insights: list[InsightDTO] = Field(default_factory=list)
    eval_settings: list[EvalSettingDTO] = Field(default_factory=list, alias="evalSettings")
    setup: SetupDTO = Field(default_factory=SetupDTO)
    table_count: int = Field(default=0, alias="tableCount")
    reference_count: int = Field(default=0, alias="referenceCount")
    repository: RepositoryDTO | None = None
    pmrl_status: Literal["complete", "missing", "invalid", "unknown"] = Field(
        default="unknown", alias="pmrlStatus"
    )
    notes_quality: Literal["complete", "fallback", "invalid", "missing", "unknown"] = Field(
        default="unknown", alias="notesQuality"
    )
    source_quality: SourceQualityDTO = Field(default_factory=SourceQualityDTO, alias="sourceQuality")


class RunResult(APIModel):
    papers: list[PaperDTO] = Field(default_factory=list)
    report: str | None = None
    report_id: str | None = Field(default=None, alias="reportId")
    bibtex: str | None = None
    answer: str | None = None
    # Structured chat answer built by final_report (blocks the UI renders).
    answer_summary: dict[str, Any] | None = Field(default=None, alias="answerSummary")
    # Citation graph for a future UI: {"nodes": [...], "edges": [...], "stats": {...}}
    # (same JSON as the ``citation_graph`` state channel); ``null`` when not built.
    citation_graph: dict[str, Any] | None = Field(default=None, alias="citationGraph")


class RunSnapshot(APIModel):
    run_id: str = Field(alias="runId")
    input: RunInput
    provider: str
    model: str
    status: RunStatus
    thread_id: str | None = Field(default=None, alias="threadId")
    answer: str | None = None
    current_node: str | None = Field(default=None, alias="currentNode")
    trace_events: list[TraceEvent] = Field(default_factory=list, alias="traceEvents")
    result: RunResult = Field(default_factory=RunResult)
    error: str | None = None
    seq: int = 0
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    timings: dict[str, int] = Field(default_factory=dict)
    started_at: datetime | None = Field(default=None, alias="startedAt")
    created_at: datetime | None = Field(default=None, alias="createdAt")
    updated_at: datetime | None = Field(default=None, alias="updatedAt")


class ReportSummary(APIModel):
    report_id: str = Field(alias="reportId")
    title: str
    updated_at: datetime = Field(alias="updatedAt")


class ReportDetail(ReportSummary):
    content: str
    markdown: str


class ThreadMessage(APIModel):
    id: str
    role: Literal["user", "assistant"]
    content: str = ""
    run_id: str | None = Field(default=None, alias="runId")
    created_at: datetime = Field(alias="createdAt")
    attachments: list[str] = Field(default_factory=list)
    report_id: str | None = Field(default=None, alias="reportId")
    # These explicit fields make the message semantics easy for non-React
    # clients while ``content`` remains the canonical chat shape.
    query: str | None = None
    answer: str | None = None


class ArtifactDTO(APIModel):
    id: str
    kind: str = "report"
    title: str
    summary: str = ""
    report_id: str | None = Field(default=None, alias="reportId")


class ThreadRunDTO(APIModel):
    run_id: str = Field(alias="runId")
    status: RunStatus
    provider: str
    model: str
    current_node: str | None = Field(default=None, alias="currentNode")
    answer: str | None = None
    report_id: str | None = Field(default=None, alias="reportId")
    error: str | None = None
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")


class ThreadSummary(APIModel):
    thread_id: str = Field(alias="threadId")
    title: str
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    run_ids: list[str] = Field(default_factory=list, alias="runIds")


class ThreadSnapshot(ThreadSummary):
    messages: list[ThreadMessage] = Field(default_factory=list)
    runs: list[ThreadRunDTO] = Field(default_factory=list)
    artifacts: list[ArtifactDTO] = Field(default_factory=list)


NODE_LABELS: dict[str, str] = {
    "router": "Phân tích yêu cầu",
    "search_papers": "Tìm bài trên arXiv",
    "eval_search": "Đánh giá độ liên quan",
    "refine_query": "Tinh chỉnh truy vấn",
    "read_paper": "Đọc toàn văn",
    "write_notes": "Ghi chú có cấu trúc",
    "citation_graph": "Dựng bản đồ liên quan",
    "web_enrich": "Tìm mã nguồn và BibTeX",
    "final_report": "Viết báo cáo",
    "error_handler": "Xử lý lỗi",
    "answer_followup": "Trả lời từ báo cáo đã có",
}

# One short description line per node, shown under the label.
NODE_DESCRIPTIONS: dict[str, str] = {
    "router": "Xác định đầu vào là bài báo hay chủ đề",
    "search_papers": "Tìm ứng viên trên arXiv",
    "eval_search": "Chọn bài phù hợp nhất",
    "refine_query": "Viết lại truy vấn khi kết quả chưa đạt",
    "read_paper": "Trích xuất bảng và tài liệu tham khảo",
    "write_notes": "Rút vấn đề, phương pháp, kết quả, hạn chế",
    "citation_graph": "Tìm bài nền tảng và bài kế thừa",
    "web_enrich": "Tìm repository chính thức và tạo BibTeX",
    "final_report": "Tổng hợp báo cáo Markdown",
    "error_handler": "Dừng run và báo lỗi",
    "answer_followup": "Trả lời câu hỏi tiếp theo bằng báo cáo và ghi chú đã có",
}

# The five chat-UI progress stages.  One node = one stage ("Đọc toàn văn" and
# "Trích xuất bảng và tài liệu tham khảo" are the label and description of the
# single ``read_paper`` stage; its facts carry the table/reference counts).
# Nodes absent here (router, search, eval, refine, error_handler) are not stages.
# ``parallelGroup`` marks the three stages that run at the same time.
NODE_STAGES: dict[str, int] = {
    "read_paper": 1,
    "write_notes": 2,
    "citation_graph": 3,
    "web_enrich": 4,
    "final_report": 5,
}
NODE_PARALLEL_GROUPS: dict[str, str] = {
    "write_notes": "analysis",
    "citation_graph": "analysis",
    "web_enrich": "analysis",
}

NODE_KINDS: dict[str, str] = {
    "router": "info",
    "search_papers": "source",
    "eval_search": "evaluation",
    "refine_query": "info",
    "read_paper": "artifact",
    "web_enrich": "artifact",
    "write_notes": "artifact",
    "citation_graph": "artifact",
    "final_report": "artifact",
    "error_handler": "error",
    "answer_followup": "artifact",
}
# Nodes whose output is the assistant answer shown in the chat.
ANSWER_NODES = frozenset({"final_report", "answer_followup"})

TERMINAL_STATUSES = {"success", "degraded", "error"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_KEYS = ("pdfs", "files", "attachments", "pdf")
INPUT_KEYS = ("paper_inputs", "paperInputs", "inputs", "papers", "paper_ids", "arxivId", "arxiv_id")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _model_dump(value: Any) -> Any:
    """Convert Pydantic models and LangGraph values to ordinary containers."""

    if isinstance(value, BaseModel):
        return value.model_dump()
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return value.dict()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_model_dump(item) for item in value]
    if isinstance(value, tuple):
        return [_model_dump(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    """Return a JSON-safe value using public DTO aliases."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _mask_sensitive(text: str) -> str:
    """Mask common credential forms without changing the document length contract."""

    value = str(text or "")
    value = re.sub(
        r"(?i)(api[_ -]?key|authorization|bearer|access[_ -]?token|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        value,
    )
    value = re.sub(r"(?i)\b(sk-[A-Za-z0-9_-]{8,}|zai-[A-Za-z0-9_-]{8,})\b", "[REDACTED]", value)
    # Protect complete HTTP(S) spans while masking local paths.  This covers
    # valid URLs whose query or path happens to contain a drive-like segment,
    # such as ``https://host/?next=C:/tmp/file``.
    protected_urls: list[str] = []

    def protect_url(match: re.Match[str]) -> str:
        protected_urls.append(match.group(0))
        return f"__HTTP_URL_{len(protected_urls) - 1}__"

    value = _PROTECTED_HTTP_URL_RE.sub(protect_url, value)
    # A drive-letter path can occur inside an error message, but the old
    # pattern also matched the ``s:/`` suffix of ``https://``.  Requiring a
    # non-identifier boundary before the path keeps public URLs intact while
    # continuing to hide local paths such as ``C:/cache/paper.pdf`` and
    # ``/home/user/paper.pdf``.
    value = re.sub(
        r"(?i)(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|/(?:tmp|home|var|mnt|workspace)/)[^\s,;}\])]+",
        "[path omitted]",
        value,
    )
    # Avoid accidentally forwarding a Python traceback from a provider.
    if "traceback (most recent call last)" in value.lower():
        value = value.splitlines()[-1] if value.splitlines() else "Workflow provider error"
    for index, url in enumerate(protected_urls):
        value = value.replace(f"__HTTP_URL_{index}__", url)
    return value


def _safe_text(value: Any, limit: int = 500) -> str:
    return _mask_sensitive(str(value or "").strip())[:limit]


def _safe_answer(value: Any, limit: int = 100_000) -> str:
    """Keep the user-facing answer while masking credentials and bounding size."""

    return _mask_sensitive(str(value or "").strip())[:limit]


def _safe_stream_text(value: Any, limit: int = 100_000) -> str:
    """Mask/bound streamed text without stripping token boundary whitespace."""

    return _mask_sensitive(str(value or ""))[:limit]


def _context_text(value: Any, limit: int = 1000) -> str:
    """Sanitize compact context fields without exposing local file paths.

    Context is built from selected metadata below rather than arbitrary state.
    This final guard also removes common path-shaped values that a model might
    have copied into a title, note, or BibTeX entry.
    """

    text = _safe_text(value, limit)
    text = re.sub(r"(?i)(?:[A-Za-z]:[\\/]|/(?:tmp|home|var|mnt|workspace)/)[^\s,;}]+", "[path omitted]", text)
    text = re.sub(r"(?i)(?:file|local_pdf_path|extracted_text|sections)\s*=\s*\{[^}]*\}", "", text)
    return text.strip()[:limit]


def _details_text(details: Mapping[str, Any] | None) -> str | None:
    """Format small operational details for the browser's text-only DTO."""

    if not details:
        return None
    parts: list[str] = []
    for key, value in details.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            rendered = "yes" if value else "no"
        elif isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
        else:
            rendered = str(value)
        parts.append(f"{key}: {_safe_text(rendered, 500)}")
    return " · ".join(parts)[:1200] or None


def _safe_link(value: Any) -> str | None:
    """Return a canonical HTTPS ArXiv/GitHub link for public DTOs."""

    raw = str(value or "").strip()
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port not in {None, 80, 443}:
        return None
    if host not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org", "github.com", "www.github.com"}:
        return None
    return parsed._replace(scheme="https", netloc=host).geturl()[:2048]


_ARXIV_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5}(?:v\d+)?)(?!\d)", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(
    r"(?P<whole>(?P<image>!?)\[(?P<label>[^\]\n]{1,1000})\]\()(?P<target><[^>\n]+>|[^)\n]+)(?P<close>\))"
)
_AUTOLINK_RE = re.compile(r"<(?P<target>https?://[^>\s]+)>", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"(?<![A-Za-z0-9_`<(])https?://[^\s)>]+", re.IGNORECASE)


def _arxiv_id_from_text(value: Any) -> str | None:
    """Extract one validated modern ArXiv identifier from text or a URL."""

    match = _ARXIV_ID_RE.search(str(value or ""))
    return match.group(1) if match else None


def _canonical_arxiv_url(arxiv_id: str, *, pdf: bool = False) -> str:
    suffix = ".pdf" if pdf else ""
    route = "pdf" if pdf else "abs"
    return f"https://arxiv.org/{route}/{arxiv_id}{suffix}"


def _report_link_target(value: Any) -> str:
    """Remove Markdown angle brackets/quotes around a link destination."""

    target = str(value or "").strip()
    if len(target) >= 2 and target[0] == "<" and target[-1] == ">":
        target = target[1:-1].strip()
    if len(target) >= 2 and target[0] in {'"', "'"} and target[-1] == target[0]:
        target = target[1:-1].strip()
    return target


def _paper_link_refs(papers: Any) -> list[tuple[str, str, str]]:
    """Return safe ``(paper_id, title, arxiv_id)`` references for repair."""

    refs: list[tuple[str, str, str]] = []
    for paper in _sequence_items(papers):
        raw = _model_dump(paper)
        if not isinstance(raw, Mapping):
            continue
        arxiv_id = _arxiv_id_from_text(raw.get("arxiv_id") or raw.get("arxivId"))
        if not arxiv_id:
            continue
        paper_id = _public_identifier(raw.get("paper_id") or raw.get("paperId")) or ""
        title = _safe_text(raw.get("title"), 240)
        refs.append((paper_id, title, arxiv_id))
    return refs


def _canonical_report_placeholder(target: str, label: str, refs: Sequence[tuple[str, str, str]]) -> str | None:
    """Repair a placeholder only when its ArXiv destination is unambiguous."""

    clean_target = _report_link_target(target)
    parsed_target = urlparse(clean_target)
    target_host = (parsed_target.hostname or "").lower().rstrip(".")
    target_is_arxiv = target_host in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"} or clean_target.casefold().startswith("arxiv.org/")
    arxiv_id = _arxiv_id_from_text(clean_target) if target_is_arxiv else None
    if arxiv_id:
        return _canonical_arxiv_url(arxiv_id, pdf="/pdf/" in clean_target.casefold() or clean_target.casefold().endswith(".pdf"))

    label_id = _arxiv_id_from_text(label)
    if label_id:
        return _canonical_arxiv_url(label_id, pdf="pdf" in f"{clean_target} {label}".casefold())

    label_fold = label.strip().casefold()
    target_fold = clean_target.casefold()
    placeholder = (
        not clean_target
        or target_fold in {"#", "url", "link", "paper", "paper.pdf", "source", "source.pdf"}
        or target_fold.startswith("file:")
        or bool(re.search(r"(?:paper|source|upload|tmp|cache)[_./-].*", target_fold))
        or ":\\" in clean_target
        or clean_target.startswith(("/", "./", "../"))
    )
    if not placeholder:
        return None

    matches = [arxiv for _paper_id, title, arxiv in refs if title and (title.casefold() in label_fold or label_fold in title.casefold())]
    if len(set(matches)) == 1:
        return _canonical_arxiv_url(matches[0], pdf="pdf" in f"{clean_target} {label}".casefold())
    return None


def _sanitize_report_markdown(content: Any, papers: Any = ()) -> str:
    """Keep trusted Markdown links, repair certain placeholders, and flatten unsafe links.

    Reports are model output and can contain paths or fabricated links.  The
    browser should receive Markdown only for a destination that the bridge can
    validate.  A link with an unresolvable destination becomes its visible
    label plus a short warning, so it cannot turn into a clickable unsafe URL.
    """

    text = str(content or "")
    refs = _paper_link_refs(papers)
    # Older model outputs sometimes placed the input PDF URL in the report
    # heading and echoed orchestration instructions in the first paragraph.
    # Keep the saved artifact untouched; this same normalized view is used by
    # the run snapshot, report detail, and Markdown download.
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        heading = lines[0][2:].strip()
        raw_url_heading = bool(re.search(r"https?://|arxiv\.org/(?:abs|pdf)/", heading, re.IGNORECASE))
        if raw_url_heading:
            known_titles = [title for _, title, _ in refs if title and not re.match(r"^(?:arxiv|paper)\s+", title, re.IGNORECASE)]
            arxiv_id = _arxiv_id_from_text(heading)
            if len(known_titles) == 1:
                heading = known_titles[0]
            elif arxiv_id:
                try:
                    from tools.arxiv_search import get_cached_arxiv_metadata

                    metadata = get_cached_arxiv_metadata(arxiv_id)
                except Exception:
                    metadata = None
                heading = str(metadata.get("title") or f"arXiv {arxiv_id}") if metadata else f"arXiv {arxiv_id}"
            else:
                heading = "Research report"
            lines[0] = f"# Research Report: {heading}"
    text = "\n".join(
        line for line in lines
        if not re.match(r"^\s*(?:[*_>]+\s*)?(?:Synthesis requested\b|Analysis scope\s*:|Language Mode\s*:|PMRL quality\s*:)", line, re.IGNORECASE)
    )

    def replace_link(match: re.Match[str]) -> str:
        label = match.group("label").strip()
        target = _report_link_target(match.group("target"))
        safe_target = _safe_link(target)
        if safe_target:
            return f"[{label}]({safe_target})"
        repaired = _canonical_report_placeholder(target, label, refs)
        if repaired:
            return f"[{label}]({repaired})"
        # Preserve the visible label and a deterministic warning.  Do not
        # echo the rejected target because it may itself be a local path or a
        # credential-bearing URL.
        visible_label = "Unverified link" if re.search(r"(?:https?://|/|\\|:)" , label, re.IGNORECASE) else label
        return f"{visible_label} _(link unavailable: unverified source)_"

    text = _MARKDOWN_LINK_RE.sub(replace_link, text)

    def replace_autolink(match: re.Match[str]) -> str:
        target = match.group("target")
        safe_target = _safe_link(target)
        return f"<{safe_target}>" if safe_target else "`Unverified link` _(link unavailable: unverified source)_"

    text = _AUTOLINK_RE.sub(replace_autolink, text)

    def replace_bare_url(match: re.Match[str]) -> str:
        target = match.group(0)
        return target if _safe_link(target) else "`Unverified link` _(link unavailable: unverified source)_"

    # GFM autolinks bare HTTPS text too.  Remove rejected bare destinations so
    # a renderer cannot turn them back into a clickable link after this pass.
    text = _BARE_URL_RE.sub(replace_bare_url, text)
    return _mask_sensitive(text)


def _report_root() -> Path:
    return Path(config.REPORTS_DIR).resolve()


def _safe_report_path(report_id: str) -> Path:
    """Resolve a report ID while refusing traversal and nested paths."""

    candidate_id = str(report_id or "")
    candidate = Path(candidate_id)
    root = _report_root()
    if (
        not candidate_id
        or not re.fullmatch(r"[A-Za-z0-9_.-]+\.md", candidate_id)
        or candidate.is_absolute()
        or candidate.name != candidate_id
        or candidate.suffix.lower() != ".md"
    ):
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        resolved = (root / candidate).resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="Report not found") from None
    if resolved.parent != root:
        raise HTTPException(status_code=404, detail="Report not found")
    return resolved


def _first_heading(content: str, fallback: str) -> str:
    for line in content.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()[:240]
    return fallback


def _report_summary(path: Path) -> ReportSummary:
    try:
        stat = path.stat()
        content = _sanitize_report_markdown(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeError):
        raise HTTPException(status_code=404, detail="Report not found") from None
    return ReportSummary(
        reportId=path.name,
        title=_safe_text(_first_heading(content, path.stem), 240),
        updatedAt=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )


def list_report_files() -> list[Path]:
    """List only direct Markdown children of the configured reports folder."""

    root = _report_root()
    if not root.exists():
        return []
    try:
        return sorted(
            [item for item in root.iterdir() if item.is_file() and item.suffix.lower() == ".md"],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []


_PMRL_STATUS_VALUES = frozenset({"complete", "missing", "invalid", "unknown"})
_NOTES_QUALITY_VALUES = frozenset({"complete", "fallback", "invalid", "missing", "unknown"})
_PARSER_STATUS_VALUES = frozenset({"success", "degraded", "error", "unknown"})


def _normalise_quality_value(value: Any, *, kind: Literal["heading", "parser"] = "heading") -> str:
    if isinstance(value, bool):
        if kind == "heading":
            return "heading" if value else "fallback"
        return "success" if value else "error"
    text = str(value or "").strip().casefold()
    if kind == "heading":
        if text in {"heading", "true", "real", "detected", "section", "explicit", "parsed"}:
            return "heading"
        if text in {"fallback", "false", "inferred", "heuristic", "default", "synthetic"}:
            return "fallback"
        return "unknown"
    if text in {"success", "ok", "complete", "completed", "parsed"}:
        return "success"
    if text in {"degraded", "partial", "warning", "warn"}:
        return "degraded"
    if text in {"error", "failed", "failure"}:
        return "error"
    return "unknown"


def _source_quality(value: Any) -> SourceQualityDTO:
    """Project measured extraction metadata; absent legacy metadata is unknown."""

    raw = _model_dump(value)
    if not isinstance(raw, Mapping):
        raw = {}
    heading_raw = next(
        (
            raw.get(key)
            for key in (
                "heading",
                "heading_quality",
                "headingQuality",
                "heading_source",
                "headingSource",
                "heading_detected",
                "headingDetected",
                "used_heading_fallback",
                "usedHeadingFallback",
            )
            if key in raw
        ),
        None,
    )
    parser_raw = next(
        (
            raw.get(key)
            for key in (
                "parser_status",
                "parserStatus",
                "extraction_status",
                "extractionStatus",
                "status",
                "parser",
            )
            if key in raw
        ),
        None,
    )
    heading = _normalise_quality_value(heading_raw, kind="heading") if heading_raw is not None else "unknown"
    parser_status = _normalise_quality_value(parser_raw, kind="parser") if parser_raw is not None else "unknown"
    if parser_status == "unknown" and parser_raw is not None and str(parser_raw).strip():
        # A concrete parser identifier (for example ``pymupdf``) is measured
        # evidence that parsing completed; keep the DTO vocabulary stable.
        parser_status = "success"
    coverage_raw = str(raw.get("coverage_status") or raw.get("coverageStatus") or "").strip().casefold()
    coverage_status = coverage_raw if coverage_raw in {"complete", "incomplete"} else "unknown"
    missing_raw = raw.get("missing_sections") or raw.get("missingSections") or []
    missing_sections = [
        section for section in missing_raw
        if isinstance(section, str) and section in {"abstract", "methodology", "experiments"}
    ] if isinstance(missing_raw, list) else []
    format_raw = str(raw.get("source_format") or raw.get("sourceFormat") or "").casefold()
    source_format = format_raw if format_raw in {"pdf", "html", "mixed"} else "unknown"
    code_urls_raw = raw.get("code_urls") or raw.get("codeUrls") or []
    code_urls = [
        url for url in (_safe_link(value) for value in code_urls_raw[:8])
        if url and urlparse(url).hostname in {"github.com", "www.github.com"}
    ] if isinstance(code_urls_raw, list) else []
    return SourceQualityDTO(
        heading=heading,
        parserStatus=parser_status,
        coverageStatus=coverage_status,
        missingSections=missing_sections,
        sourceFormat=source_format,
        codeStatus="verified" if code_urls else "unknown",
        codeUrls=code_urls,
        warning=_safe_text(raw.get("warning"), 500) or None,
    )


def _notes_quality(raw: Mapping[str, Any], notes_raw: Mapping[str, Any] | None) -> tuple[str, str]:
    explicit = raw.get("notes_quality") or raw.get("notesQuality") or raw.get("pmrl_status") or raw.get("pmrlStatus")
    explicit_text = str(explicit or "").strip().casefold()
    if explicit_text in _NOTES_QUALITY_VALUES and explicit_text != "unknown":
        notes_quality = explicit_text
    elif notes_raw is None or explicit_text == "unknown":
        # A legacy PaperItem often carries the model default ``unknown``;
        # preserve that measured uncertainty.
        notes_quality = "unknown"
    else:
        notes_quality = "missing"

    if notes_quality == "complete":
        pmrl_status = "complete"
    elif notes_quality in {"fallback", "invalid"}:
        pmrl_status = "invalid"
    elif notes_quality == "missing":
        pmrl_status = "missing"
    else:
        pmrl_status = "unknown"
    return notes_quality, pmrl_status


def _as_state_mapping(state: Any) -> dict[str, Any]:
    value = _model_dump(state)
    return dict(value) if isinstance(value, Mapping) else {}


def _paper_dto(value: Any) -> PaperDTO | None:
    raw = _model_dump(value)
    if not isinstance(raw, Mapping):
        return None
    # Explicitly select fields instead of dumping PaperItem wholesale.  This
    # is the hard boundary that keeps extracted PDF text and local paths out of
    # the HTTP response.
    repos: list[RepoDTO] = []
    for repo in raw.get("github_repos", []) or []:
        repo_raw = _model_dump(repo)
        if isinstance(repo_raw, Mapping):
            try:
                repo_url = _safe_link(repo_raw.get("url"))
                if not repo_url:
                    continue
                repos.append(
                    RepoDTO(
                        name=_safe_text(repo_raw.get("name"), 240),
                        url=repo_url,
                        stars=int(repo_raw.get("stars") or 0),
                        framework=_safe_text(repo_raw.get("framework") or "Python", 120),
                        isOfficial=bool(repo_raw.get("is_official") or repo_raw.get("isOfficial")),
                        description=_safe_text(repo_raw.get("description"), 800),
                    )
                )
            except Exception:
                continue

    notes: PMRLDTO | None = None
    notes_raw_value = _model_dump(raw.get("notes"))
    notes_raw = notes_raw_value if isinstance(notes_raw_value, Mapping) else None
    notes_quality, pmrl_status = _notes_quality(raw, notes_raw)
    if notes_raw is not None:
        try:
            notes = PMRLDTO(
                problem=_safe_text(notes_raw.get("problem"), 4000),
                method=_safe_text(notes_raw.get("method"), 4000),
                result=_safe_text(notes_raw.get("result"), 4000),
                limitation=_safe_text(notes_raw.get("limitation"), 4000),
                status=pmrl_status,
            )
        except Exception:
            notes = None
            pmrl_status = "unknown"
            notes_quality = "unknown"

    quality_raw = raw.get("source_quality") or raw.get("sourceQuality") or raw.get("extraction_quality") or raw.get("extractionQuality")
    source_quality = _source_quality(quality_raw)

    notes_data = notes_raw or {}
    insights = [
        InsightDTO(
            finding=_safe_text(item.get("finding"), 600),
            whyItMatters=_safe_text(item.get("why_it_matters"), 600),
        )
        for item in _sequence_items(notes_data.get("insights"))[:8]
        if isinstance(item, Mapping) and str(item.get("finding") or "").strip()
    ]
    eval_settings = [
        EvalSettingDTO(
            benchmark=_safe_text(item.get("benchmark"), 200),
            metric=_safe_text(item.get("metric"), 200),
            metricMeaning=_safe_text(item.get("metric_meaning"), 300),
            higherIsBetter=item.get("higher_is_better") if isinstance(item.get("higher_is_better"), bool) else None,
            unit=_safe_text(item.get("unit"), 40),
            methodValue=_safe_text(item.get("method_value"), 120),
            baselineName=_safe_text(item.get("baseline_name"), 200),
            baselineValue=_safe_text(item.get("baseline_value"), 120),
            tableLabel=_safe_text(item.get("table_label"), 40),
        )
        for item in _sequence_items(notes_data.get("eval_settings"))[:32]
        if isinstance(item, Mapping)
    ]
    setup_raw = notes_data.get("setup") if isinstance(notes_data.get("setup"), Mapping) else {}
    setup = SetupDTO(
        datasets=_safe_text(setup_raw.get("datasets"), 600),
        trainingConfig=_safe_text(setup_raw.get("training_config"), 600),
        hardware=_safe_text(setup_raw.get("hardware"), 300),
        compute=_safe_text(setup_raw.get("compute"), 300),
    )
    repository: RepositoryDTO | None = None
    if source_quality.code_urls:
        repository = RepositoryDTO(url=source_quality.code_urls[0], source="html")
    elif repos:
        repository = RepositoryDTO(url=repos[0].url, source="github")

    try:
        return PaperDTO(
            paperId=_safe_text(raw.get("paper_id") or raw.get("paperId"), 260),
            title=_safe_text(raw.get("title") or "Untitled paper", 500),
            summary=_safe_text(raw.get("summary"), 1600),
            authors=[_safe_text(author, 200) for author in (raw.get("authors") or [])[:64]],
            published=_safe_text(raw.get("published"), 80),
            subjects=[_safe_text(subject, 120) for subject in (raw.get("subjects") or [])[:16] if isinstance(subject, str)],
            metadataStatus=(
                str(raw.get("metadata_status") or raw.get("metadataStatus"))
                if str(raw.get("metadata_status") or raw.get("metadataStatus")) in {"complete", "missing", "unknown"}
                else "unknown"
            ),
            sourceType=_safe_text(raw.get("source_type") or raw.get("sourceType") or "arxiv", 60),
            arxivId=_safe_text(raw.get("arxiv_id") or raw.get("arxivId"), 120) or None,
            pdfUrl=_safe_link(raw.get("pdf_url") or raw.get("pdfUrl")),
            githubRepos=repos,
            bibtex=_safe_text(raw.get("bibtex"), 6000) if raw.get("bibtex") else None,
            notes=notes,
            intuition=_safe_text(notes_data.get("intuition"), 600),
            insights=insights,
            evalSettings=eval_settings,
            setup=setup,
            tableCount=len(_sequence_items(raw.get("tables"))),
            referenceCount=len(_sequence_items(raw.get("references"))),
            repository=repository,
            pmrlStatus=pmrl_status,
            notesQuality=notes_quality,
            sourceQuality=source_quality,
        )
    except Exception:
        return None


_PUBLIC_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}")
_NOTE_FALLBACK_RE = re.compile(
    r"^\s*\[write_notes\]\s*([A-Za-z0-9][A-Za-z0-9_.-]{0,255})\s*:\s*PMRL\s+fallback\b",
    re.IGNORECASE,
)
_REPORT_ERROR_MARKERS = (
    "error generating final report",
    "research workflow execution notice",
)
_PUBLIC_INTENTS = frozenset({"direct_read", "search", "followup"})
_PUBLIC_FAILURE_MESSAGES = {
    "router": "Could not understand the request. Check the request and start again.",
    "search_papers": "Could not find papers. The search service did not respond. Start again.",
    "eval_search": "Could not evaluate search results. The language-model provider did not respond. Start again.",
    "refine_query": "Could not refine the search query. The language-model provider did not respond. Start again.",
    "read_paper": "Could not read source PDFs. Check the source files or network, then start again.",
    "web_enrich": "Could not enrich sources. Related repositories or citations may be unavailable. Start again.",
    "write_notes": "Could not write structured notes. The language-model provider did not respond. Start again.",
    "final_report": "Could not prepare the research report. The language-model provider did not respond. Start again.",
    "answer_followup": "Could not answer the follow-up question. The language-model provider did not respond. Ask again.",
    "error_handler": "Research failed. Edit the request or start again.",
    "workflow": "Research failed. Edit the request or start again.",
}
_SAFE_ERROR_PREFIXES = (
    "Không có bài báo nào trong danh sách được chọn.",
    "Tất cả các bài báo đều gặp lỗi khi đọc/tải PDF.",
)
_ARXIV_NO_RESULTS_RE = re.compile(
    r"^ArXiv không trả về bài báo nào sau (\d+) lượt tìm kiếm\.",
    re.IGNORECASE,
)


def _public_identifier(value: Any) -> str | None:
    """Return a compact identifier while rejecting paths and free-form text."""

    text = str(value or "").strip()
    return text if _PUBLIC_IDENTIFIER_RE.fullmatch(text) else None


def _public_failure_message(node: str, raw_message: Any = None) -> str:
    """Return stable recovery copy without forwarding exception details."""

    clean = _safe_text(raw_message, 1000)
    if clean:
        arxiv_match = _ARXIV_NO_RESULTS_RE.match(clean)
        if arxiv_match:
            # Rebuild the complete deterministic guard message from its count;
            # discard every suffix where a provider exception could appear.
            count = max(1, min(int(arxiv_match.group(1)), 10_000))
            return (
                f"ArXiv không trả về bài báo nào sau {count} lượt tìm kiếm. "
                "Hãy thử 2-5 từ khóa kỹ thuật bằng tiếng Anh hoặc nhập ArXiv ID trực tiếp."
            )
        if clean in _SAFE_ERROR_PREFIXES:
            return clean
    return _PUBLIC_FAILURE_MESSAGES.get(node, _PUBLIC_FAILURE_MESSAGES["workflow"])


def _sequence_items(value: Any) -> list[Any]:
    """Read list-like state channels without treating strings as sequences."""

    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


_TIMING_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")


def _duration_ms(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return min(int(number), 7 * 24 * 60 * 60 * 1000)


def _numeric_timings(value: Any, prefix: str = "", *, depth: int = 0) -> dict[str, int]:
    """Flatten bounded numeric timing metadata without exposing arbitrary state."""

    if depth > 3:
        return {}
    if isinstance(value, Mapping):
        result: dict[str, int] = {}
        for raw_key, raw_value in list(value.items())[:128]:
            key = str(raw_key).strip()
            if not key or not _TIMING_KEY_RE.fullmatch(key):
                continue
            name = f"{prefix}.{key}" if prefix else key
            duration = _duration_ms(raw_value)
            if duration is not None:
                result[name] = duration
            elif isinstance(raw_value, Mapping):
                result.update(_numeric_timings(raw_value, name, depth=depth + 1))
        return dict(list(result.items())[:128])
    duration = _duration_ms(value)
    return {prefix: duration} if prefix and duration is not None and _TIMING_KEY_RE.fullmatch(prefix) else {}


def _timings_from_mapping(value: Mapping[str, Any] | None) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("node_timings", "nodeTimings", "timings", "timing", "measurements", "performance"):
        nested = value.get(key)
        if nested is not None:
            result.update(_numeric_timings(nested))
    # A node may return its own duration directly with the update.
    for key in ("duration_ms", "durationMs"):
        if key in value:
            duration = _duration_ms(value.get(key))
            if duration is not None:
                result[key] = duration
    return dict(list(result.items())[:128])


def _state_timings(state: Mapping[str, Any] | None) -> dict[str, int]:
    """Collect measured node/tool timings from state and paper metadata."""

    if not isinstance(state, Mapping):
        return {}
    result = _timings_from_mapping(state)
    # ``processing_metadata`` is per-paper and may contain useful measured
    # durations from PDF/GitHub helpers.  Only numeric values under an
    # allowlisted key can cross the API boundary.
    for paper in _sequence_items(state.get("selected_papers")):
        raw = _model_dump(paper)
        if not isinstance(raw, Mapping):
            continue
        metadata = raw.get("processing_metadata") or raw.get("processingMetadata")
        paper_id = _public_identifier(raw.get("paper_id") or raw.get("paperId"))
        if isinstance(metadata, Mapping) and paper_id:
            result.update(_numeric_timings(metadata, f"paper.{paper_id}"))
    return dict(list(result.items())[:128])


def _valid_report(value: Any) -> bool:
    """Recognize a final Markdown report rather than an error notice."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    lowered = text.lstrip(" \t\r\n*_`#(").casefold()
    return not any(lowered.startswith(marker) for marker in _REPORT_ERROR_MARKERS)


def _failed_pdf_count(update: Mapping[str, Any]) -> int:
    """Count failed PDF inputs from the read node's bounded error channels."""

    if "error_logs" in update:
        errors = _sequence_items(update.get("error_logs"))
        # ``read_paper`` reports one entry per failed paper.  Its empty-input
        # guard is a workflow error, not a failed PDF.
        return sum(
            1
            for error in errors
            if "no selected papers" not in str(error).casefold()
            and "danh sách bài báo trống" not in str(error).casefold()
        )

    logs = _sequence_items(update.get("trace_logs"))
    failure_markers = ("lỗi", "error", "failed", "failure", "❌")
    return sum(
        1
        for log in logs
        if ("read_paper" in str(log).casefold() or "đọc" in str(log).casefold())
        and any(marker in str(log).casefold() for marker in failure_markers)
    )


def _fallback_note_ids(update: Mapping[str, Any]) -> list[str]:
    """Extract only paper IDs from write-notes fallback log lines."""

    result: list[str] = []
    seen: set[str] = set()
    for log in _sequence_items(update.get("trace_logs")):
        match = _NOTE_FALLBACK_RE.match(str(log))
        if not match:
            continue
        paper_id = _public_identifier(match.group(1))
        if paper_id and paper_id not in seen:
            result.append(paper_id)
            seen.add(paper_id)
        if len(result) >= 64:
            break
    return result


def _report_id_from_state(state: Mapping[str, Any]) -> str | None:
    # ``final_report`` stores the saved file name in ``report_id``.  The trace
    # log scan below only serves states from graphs that do not set it.
    stored = state.get("report_id")
    if isinstance(stored, str) and re.fullmatch(r"[A-Za-z0-9_.-]+\.md", stored):
        try:
            if _safe_report_path(stored).exists():
                return stored
        except HTTPException:
            pass
    logs = state.get("trace_logs") or []
    for log in reversed(logs if isinstance(logs, list) else []):
        match = re.search(r"(?:Báo cáo lưu|saved)\s*:\s*([A-Za-z0-9_.-]+\.md)", str(log), re.IGNORECASE)
        if match:
            report_id = match.group(1)
            try:
                path = _safe_report_path(report_id)
                if path.exists():
                    return report_id
            except HTTPException:
                continue
    return None


def _public_similarity_graph(value: Any) -> dict[str, Any] | None:
    """Bounded, JSON-safe copy of ``citation_graph["similarity"]`` or ``None``."""

    if not isinstance(value, Mapping) or not isinstance(value.get("nodes"), list):
        return None

    def num(item: Any, digits: int = 4) -> float | None:
        return round(float(item), digits) if isinstance(item, (int, float)) and not isinstance(item, bool) else None

    def count(item: Any) -> int | None:
        return item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None

    def text(item: Any, limit: int) -> str | None:
        return _safe_text(item, limit) if item not in (None, "") else None

    def work(item: Any, limit_abstract: int = 1500) -> dict[str, Any] | None:
        if not isinstance(item, Mapping) or not item.get("id"):
            return None
        out = {
            "id": text(item.get("id"), 120),
            "arxivId": text(item.get("arxiv_id"), 40),
            "title": text(item.get("title"), 300) or "",
            "year": count(item.get("year")),
            "citedByCount": count(item.get("cited_by_count")),
            "titleSource": text(item.get("title_source"), 12),
            "firstAuthor": text(item.get("first_author"), 120) or "",
            "abstract": text(item.get("abstract"), limit_abstract) or "",
            "similarity": num(item.get("similarity")),
            "role": text(item.get("role"), 20),
            "url": text(item.get("url"), 300) or "",
        }
        if "support" in item:
            out["support"] = count(item.get("support")) or 0
        return out

    def works(items: Any, limit: int) -> list[dict[str, Any]]:
        return [w for w in (work(i) for i in (items or [])[:limit]) if w]

    edges = [
        {"source": text(e.get("source"), 120), "target": text(e.get("target"), 120), "weight": num(e.get("weight")) or 0.0}
        for e in (value.get("edges") or [])[:600]
        if isinstance(e, Mapping) and e.get("source") and e.get("target")
    ]
    raw = value.get("stats") if isinstance(value.get("stats"), Mapping) else {}
    status = raw.get("status")
    stats = {
        "seeds": count(raw.get("seeds")) or 0,
        "pool": count(raw.get("pool")) or 0,
        "requests": count(raw.get("requests")) or 0,
        "cacheHits": count(raw.get("cache_hits")) or 0,
        "durationMs": count(raw.get("duration_ms")) or 0,
        "unverifiedDropped": count(raw.get("unverified_dropped")) or 0,
        "status": status if status in {"ok", "no_key", "error", "partial"} else "error",
    }
    return {
        "nodes": works(value["nodes"], 100), "edges": edges,
        "priorWorks": works(value.get("prior_works"), 20),
        "derivativeWorks": works(value.get("derivative_works"), 20),
        "stats": stats,
    }


def _public_citation_graph(value: Any) -> dict[str, Any] | None:
    """Return a bounded, JSON-safe copy of the citation graph, or ``None``."""

    if not isinstance(value, Mapping) or not isinstance(value.get("nodes"), list):
        return None

    def text(item: Any, limit: int) -> str | None:
        return _safe_text(item, limit) if item not in (None, "") else None

    def count(item: Any) -> int | None:
        return item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None

    nodes = []
    for node in value["nodes"][:100]:
        if not isinstance(node, Mapping) or not node.get("id"):
            continue
        nodes.append({
            "id": text(node.get("id"), 120),
            "arxiv_id": text(node.get("arxiv_id"), 40),
            "title": text(node.get("title"), 300) or "",
            "year": count(node.get("year")),
            "cited_by_count": count(node.get("cited_by_count")),
            "selected": bool(node.get("selected")),
            "role": text(node.get("role"), 20),
            "support": count(node.get("support")) or 0,
        })
    edges = [
        {"source": text(edge.get("source"), 120), "target": text(edge.get("target"), 120), "kind": "cites"}
        for edge in (value.get("edges") or [])[:400]
        if isinstance(edge, Mapping) and edge.get("source") and edge.get("target")
    ]
    raw_stats = value.get("stats") if isinstance(value.get("stats"), Mapping) else {}
    by_method = raw_stats.get("by_method") if isinstance(raw_stats.get("by_method"), Mapping) else {}
    stats = {
        "refs_total": count(raw_stats.get("refs_total")) or 0,
        "refs_resolved": count(raw_stats.get("refs_resolved")) or 0,
        "by_method": {_safe_text(k, 30): count(v) or 0 for k, v in list(by_method.items())[:10]},
        "openalex": raw_stats.get("openalex") if raw_stats.get("openalex") in {"ok", "no_key", "error"} else "error",
    }
    public: dict[str, Any] = {"nodes": nodes, "edges": edges, "stats": stats}
    similarity = _public_similarity_graph(value.get("similarity"))
    if similarity is not None:
        public["similarity"] = similarity
    return public


def _public_answer_summary(value: Any, *, depth: int = 0) -> Any:
    """Bounded, JSON-safe copy of ``state["answer_summary"]`` (``None`` when absent or not a dict)."""

    if depth == 0:
        return _public_answer_summary(dict(value), depth=1) if isinstance(value, Mapping) and value else None
    if isinstance(value, Mapping):
        return {
            _safe_text(key, 60): _public_answer_summary(item, depth=depth + 1)
            for key, item in list(value.items())[:48]
        } if depth < 7 else None
    if isinstance(value, (list, tuple)):
        return [_public_answer_summary(item, depth=depth + 1) for item in list(value)[:64]] if depth < 7 else None
    if isinstance(value, str):
        return _safe_text(value, 2000)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return None


def _result_from_state(state: Mapping[str, Any], answer: str | None = None) -> RunResult:
    papers: list[PaperDTO] = []
    for paper in state.get("selected_papers", []) or []:
        dto = _paper_dto(paper)
        if dto is not None:
            papers.append(dto)
    bibtex_entries = [paper.bibtex for paper in papers if paper.bibtex]
    report_value = state.get("final_report")
    report_valid = _valid_report(report_value) and str(state.get("status") or "").lower() not in {"error", "failed"}
    report_content = _sanitize_report_markdown(
        report_value,
        state.get("selected_papers") or [],
    ) if report_valid else None
    return RunResult(
        papers=papers,
        report=report_content,
        reportId=_report_id_from_state(state) if report_valid else None,
        answerSummary=_public_answer_summary(state.get("answer_summary")) if report_valid else None,
        bibtex="\n\n".join(bibtex_entries) if bibtex_entries else None,
        citationGraph=_public_citation_graph(state.get("citation_graph")),
        answer=_safe_answer(answer or (report_content if report_valid else None), 100_000)
        or None,
    )


def _node_from_logs(logs: Any) -> str:
    if isinstance(logs, list):
        for item in reversed(logs):
            match = re.match(r"\[([^\]]+)\]", str(item))
            if match:
                candidate = match.group(1).strip()
                return candidate if candidate in NODE_LABELS else "workflow"
    return "workflow"


def _normalise_updates(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Normalize LangGraph updates and lightweight fake graph output."""

    if isinstance(payload, tuple) and len(payload) == 2:
        node, update = payload
        return [(str(node), _as_state_mapping(update))]
    if not isinstance(payload, Mapping):
        return []

    known_nodes = set(NODE_LABELS)
    updates: list[tuple[str, dict[str, Any]]] = []
    for key, value in payload.items():
        if str(key) in known_nodes and isinstance(value, Mapping):
            updates.append((str(key), _as_state_mapping(value)))
    if updates:
        return updates

    # ``stream_mode='values'`` and simple fakes often yield a complete state.
    # Infer the node from the append-only trace prefix when available.
    state = _as_state_mapping(payload)
    if not state:
        return []
    inferred = _node_from_logs(state.get("trace_logs"))
    if inferred == "workflow":
        # Flat update fakes and some LangGraph stream modes carry the node's
        # output channels directly.  Prefer the newest measured timing key,
        # then fall back to a distinctive output channel so durationMs is
        # attached to the correct node instead of being lost on ``workflow``.
        timing_state = state.get("node_timings") or state.get("nodeTimings")
        if isinstance(timing_state, Mapping):
            for key in reversed(list(timing_state)):
                if str(key) in NODE_LABELS:
                    inferred = str(key)
                    break
        if inferred == "workflow":
            if "final_report" in state:
                inferred = "final_report"
    return [(inferred, state)]


def _merge_state(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    """Merge one graph update while respecting ResearchState append reducers."""

    for key, value in update.items():
        if key in {"trace_logs", "error_logs"} and isinstance(value, list):
            current = target.get(key)
            if not isinstance(current, list):
                target[key] = list(value)
            elif value[: len(current)] == current:
                # Complete state snapshots include the already accumulated
                # reducer values; don't duplicate them.
                target[key] = list(value)
            else:
                target[key] = current + list(value)
        elif key in {"paper_notes", "paper_enrichment"} and isinstance(value, Mapping):
            # Parallel branches write per-paper dicts under their own keys
            # (``operator.or_`` in ResearchState).
            current = target.get(key)
            target[key] = {**(current if isinstance(current, Mapping) else {}), **value}
        elif key == "node_timings" and isinstance(value, Mapping):
            current = target.get(key)
            merged_timings = dict(current) if isinstance(current, Mapping) else {}
            for node_name, timing in value.items():
                if isinstance(timing, Mapping) and isinstance(merged_timings.get(node_name), Mapping):
                    node_timing = dict(merged_timings[node_name])
                    node_timing.update(timing)
                    merged_timings[str(node_name)] = node_timing
                else:
                    merged_timings[str(node_name)] = timing
            target[key] = merged_timings
        else:
            target[key] = value


def _step_summary(node: str, state: Mapping[str, Any], update: Mapping[str, Any]) -> str:
    """Describe a node without copying arbitrary trace/log content."""

    if node == "router":
        return f"Routed request ({_safe_text(state.get('intent') or 'search', 40)})"
    if node == "search_papers":
        return f"Collected {len(state.get('search_results') or [])} paper source(s)"
    if node == "eval_search":
        selected = len(state.get("selected_papers") or [])
        return f"Evaluated relevance; selected {selected} paper(s)"
    if node == "refine_query":
        return f"Refined search query (attempt {int(state.get('retry_count') or 0)})"
    if node == "read_paper":
        return f"Parsed {len(state.get('selected_papers') or [])} paper(s)"
    if node == "web_enrich":
        return f"Looked up repositories and BibTeX for {len(state.get('selected_papers') or [])} paper(s)"
    if node == "write_notes":
        return f"Created PMRL notes for {len(state.get('selected_papers') or [])} paper(s)"
    if node == "citation_graph":
        graph = state.get("citation_graph")
        nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
        return f"Built citation graph ({len(nodes) if isinstance(nodes, list) else 0} node(s))"
    if node == "final_report":
        return "Prepared final research report"
    if node == "error_handler":
        return "Workflow entered error handling"
    return NODE_LABELS.get(node, "Workflow update")


def _step_details(node: str, state: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Build operational metadata only; never copy arbitrary graph state."""

    if node == "router":
        return {
            "intent": _safe_text(state.get("intent"), 60),
            "inputCount": len(state.get("raw_inputs") or []),
        }
    if node == "search_papers":
        return {
            "resultCount": len(state.get("search_results") or []),
            "query": _safe_text(state.get("search_query") or state.get("user_query"), 300),
        }
    if node == "eval_search":
        return {
            "passed": bool(state.get("eval_passed")),
            "selectedCount": len(state.get("selected_papers") or []),
            "feedback": _safe_text(state.get("eval_feedback"), 700),
        }
    if node == "refine_query":
        return {
            "retryCount": int(state.get("retry_count") or 0),
            "query": _safe_text(state.get("search_query"), 300),
        }
    if node == "read_paper":
        return {"paperCount": len(state.get("selected_papers") or [])}
    if node == "web_enrich":
        enrichment = update.get("paper_enrichment")
        enrichment = enrichment if isinstance(enrichment, Mapping) else {}
        repo_count = sum(
            int(entry.get("repo_count") or 0) for entry in map(_model_dump, enrichment.values()) if isinstance(entry, Mapping)
        )
        return {"paperCount": len(state.get("selected_papers") or []), "repositoryCount": repo_count}
    if node == "write_notes":
        return {"paperCount": len(state.get("selected_papers") or [])}
    if node == "citation_graph":
        graph = state.get("citation_graph")
        graph = graph if isinstance(graph, Mapping) else {}
        stats = graph.get("stats") if isinstance(graph.get("stats"), Mapping) else {}
        return {
            "nodeCount": len(graph.get("nodes") or []),
            "edgeCount": len(graph.get("edges") or []),
            "openalex": _safe_text(stats.get("openalex"), 20),
        }
    if node == "final_report":
        status = str(state.get("status") or "").lower()
        report_available = _valid_report(state.get("final_report")) and status not in {"error", "failed"}
        return {
            "reportAvailable": report_available,
            "reportId": _report_id_from_state(state) if report_available else None,
        }
    if node == "error_handler":
        return {"message": _safe_text(state.get("error_message"), 800)}
    return {}


_SOURCE_VALUES = frozenset({"ar5iv", "arxiv_html", "pdf", "mixed", "unknown"})
_SECTION_FACT_KEYS = ("abstract", "methodology", "setup", "experiments", "limitations")
_HARDWARE_RE = re.compile(
    r"(?:\b\d+\s*[x×]\s*)?(?:NVIDIA\s+)?\b(?:A100|H100|H200|V100|A6000|RTX\s?\d{4}|TPU\s?v?\d\w*|GPUs?|TPUs?)\b",
    re.IGNORECASE,
)


def _hardware_mention(sections: Any) -> str:
    """First GPU/TPU mention in the setup/experiments text (<= 60 chars), else empty."""

    if not isinstance(sections, Mapping):
        return ""
    for key in ("setup", "experiments", "methodology"):
        text = str(sections.get(key) or "")
        match = _HARDWARE_RE.search(text)
        if match:
            start = max(0, match.start() - 12)
            return " ".join(text[start:match.end() + 24].split())[:60]
    return ""


def _combined_source(values: list[str]) -> str:
    unique = {value for value in values if value in _SOURCE_VALUES}
    if not unique:
        return "unknown"
    return next(iter(unique)) if len(unique) == 1 else "mixed"


def _read_paper_facts(papers: Any) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for paper in _sequence_items(papers):
        raw = _model_dump(paper)
        if not isinstance(raw, Mapping):
            continue
        quality = raw.get("source_quality") if isinstance(raw.get("source_quality"), Mapping) else {}
        source = str(quality.get("source") or "unknown")
        sections = raw.get("sections") if isinstance(raw.get("sections"), Mapping) else {}
        hardware = _hardware_mention(sections)
        items.append({
            "paperId": _public_identifier(raw.get("paper_id")) or "",
            "source": source if source in _SOURCE_VALUES else "unknown",
            "sectionChars": {key: len(str(sections.get(key) or "")) for key in _SECTION_FACT_KEYS},
            "tableCount": len(_sequence_items(raw.get("tables"))),
            "referenceCount": len(_sequence_items(raw.get("references"))),
            "hardwareFound": bool(hardware),
            "hardware": hardware,
        })
    if not items:
        return {}
    return {
        "paperFacts": items,
        "source": _combined_source([item["source"] for item in items]),
        "tableCount": sum(item["tableCount"] for item in items),
        "referenceCount": sum(item["referenceCount"] for item in items),
        "hardwareFoundCount": sum(1 for item in items if item["hardwareFound"]),
    }


def _write_notes_facts(paper_notes: Any) -> dict[str, Any]:
    if not isinstance(paper_notes, Mapping):
        return {}
    items: list[dict[str, Any]] = []
    totals = {"proposed": 0, "kept": 0, "dropped": 0}
    insight_total = 0
    for paper_id, entry in list(paper_notes.items())[:64]:
        data = _model_dump(entry)
        if not isinstance(data, Mapping):
            continue
        notes = data.get("notes") if isinstance(data.get("notes"), Mapping) else {}
        metadata = data.get("processing_metadata") if isinstance(data.get("processing_metadata"), Mapping) else {}
        stats = metadata.get("evalSettings") if isinstance(metadata.get("evalSettings"), Mapping) else {}
        setup = notes.get("setup") if isinstance(notes.get("setup"), Mapping) else {}
        insights = len([item for item in _sequence_items(notes.get("insights")) if isinstance(item, Mapping) and str(item.get("finding") or "").strip()])
        counts = {name: _nonnegative_int(stats.get(name)) or 0 for name in totals}
        hardware = bool(str(setup.get("hardware") or "").strip())
        items.append({
            "paperId": _public_identifier(paper_id) or "",
            "insightCount": insights,
            "evalSettings": counts,
            "hardwareStated": hardware,
        })
        insight_total += insights
        for name in totals:
            totals[name] += counts[name]
    if not items:
        return {}
    return {
        "paperFacts": items,
        "insightCount": insight_total,
        "evalSettings": totals,
        "hardwareStatedCount": sum(1 for item in items if item["hardwareStated"]),
    }


def _citation_graph_facts(update: Mapping[str, Any]) -> dict[str, Any]:
    graph = _model_dump(update.get("citation_graph"))
    if not isinstance(graph, Mapping):
        return {}
    stats = graph.get("stats") if isinstance(graph.get("stats"), Mapping) else {}
    similarity = graph.get("similarity") if isinstance(graph.get("similarity"), Mapping) else None
    sim_stats = similarity.get("stats") if similarity and isinstance(similarity.get("stats"), Mapping) else {}
    # The related-papers map is the similarity graph when it was built; the
    # citation graph (references/followers) is the fallback.
    shown = similarity if similarity is not None else graph
    facts: dict[str, Any] = {
        "nodeCount": len(_sequence_items(shown.get("nodes"))),
        "edgeCount": len(_sequence_items(shown.get("edges"))),
        "graphStatus": _safe_text(sim_stats.get("status") or stats.get("openalex") or "unknown", 24),
    }
    for fact, source, key in (
        ("seeds", sim_stats, "seeds"),
        ("pool", sim_stats, "pool"),
        ("refsTotal", stats, "refs_total"),
        ("refsResolved", stats, "refs_resolved"),
    ):
        number = _nonnegative_int(source.get(key))
        if number is not None:
            facts[fact] = number
    timing = _model_dump(update.get("node_timings"))
    node_timing = timing.get("citation_graph") if isinstance(timing, Mapping) else None
    duration = _duration_ms(node_timing.get("durationMs")) if isinstance(node_timing, Mapping) else None
    if duration is not None:
        facts["durationMs"] = duration
    return facts


_REPO_SOURCES = frozenset({"html", "github_search", "none"})


def _web_enrich_facts(enrichment: Any) -> dict[str, Any]:
    if not isinstance(enrichment, Mapping):
        return {}
    items: list[dict[str, Any]] = []
    for paper_id, entry in list(enrichment.items())[:64]:
        data = _model_dump(entry)
        if not isinstance(data, Mapping):
            continue
        source = str(data.get("repo_source") or "none")
        items.append({
            "paperId": _public_identifier(paper_id) or "",
            "repoSource": source if source in _REPO_SOURCES else "none",
            "repositoryCount": _nonnegative_int(data.get("repo_count")) or 0,
            "bibtex": bool(data.get("bibtex")),
        })
    if not items:
        return {}
    sources = {item["repoSource"] for item in items}
    return {
        "paperFacts": items,
        "repoSource": next(iter(sources)) if len(sources) == 1 else "mixed",
        "repositoryCount": sum(item["repositoryCount"] for item in items),
        "bibtexCount": sum(1 for item in items if item["bibtex"]),
    }


def _step_facts(node: str, state: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Build sparse, measured facts for a public state-update event.

    The ``update`` mapping is the evidence boundary.  Reading a value from
    the merged state for an unrelated node would make a later event appear to
    have produced an earlier count, so each fact is emitted only when the
    corresponding channel is present in this node update.  Task lifecycle
    events call this helper with an empty update and therefore receive no
    facts.
    """

    if not update:
        return {}

    facts: dict[str, Any] = {}
    if node == "router" and "intent" in update:
        intent = str(update.get("intent") or "").strip()
        if intent in _PUBLIC_INTENTS:
            facts["intent"] = intent

    elif node == "search_papers" and "search_results" in update:
        facts["resultCount"] = len(_sequence_items(update.get("search_results")))

    elif node == "eval_search" and "selected_papers" in update:
        facts["selectedCount"] = len(_sequence_items(update.get("selected_papers")))

    elif node == "refine_query" and "retry_count" in update:
        retry_count = _nonnegative_int(update.get("retry_count"))
        if retry_count is not None:
            facts["retryCount"] = retry_count

    elif node == "read_paper" and any(
        key in update for key in ("selected_papers", "error_logs", "trace_logs")
    ):
        # ``selected_papers`` is the reader's valid output.  In the all-failed
        # branch the node omits that key, which correctly yields zero rather
        # than leaking the pre-read candidate count from merged state.
        facts["paperCount"] = len(_sequence_items(update.get("selected_papers")))
        facts["failedPdfCount"] = _failed_pdf_count(update)
        facts.update(_read_paper_facts(update.get("selected_papers")))

    elif node == "web_enrich" and "paper_enrichment" in update:
        facts.update(_web_enrich_facts(update.get("paper_enrichment")))

    elif node == "write_notes" and ("paper_notes" in update or "trace_logs" in update):
        fallback_ids = _fallback_note_ids(update)
        if fallback_ids:
            facts["noteFallbackPaperIds"] = fallback_ids
        facts.update(_write_notes_facts(update.get("paper_notes")))

    elif node == "citation_graph" and "citation_graph" in update:
        facts.update(_citation_graph_facts(update))

    elif node == "answer_followup" and "followup_sources" in update:
        facts["sources"] = [_safe_text(item, 80) for item in _sequence_items(update.get("followup_sources"))[:12]]

    elif node == "final_report" and "final_report" in update:
        status = str(update.get("status") or state.get("status") or "").lower()
        report_available = _valid_report(update.get("final_report")) and status not in {"error", "failed"}
        facts["reportAvailable"] = report_available
        if report_available:
            facts["sectionCount"] = len(re.findall(r"(?m)^## \S", str(update.get("final_report") or "")))
            # The report ID is sourced from the append-only save log and is
            # returned only after the file has been verified within REPORTS_DIR.
            report_id = _report_id_from_state(state)
            if report_id:
                facts["reportId"] = report_id

    measured_timings = _timings_from_mapping(update)
    if measured_timings:
        facts["timings"] = measured_timings
    return facts


_STAGE_COUNT_KEYS = (
    "tableCount", "referenceCount", "hardwareFoundCount", "insightCount", "hardwareStatedCount",
    "seeds", "pool", "nodeCount", "edgeCount", "refsTotal", "refsResolved", "durationMs", "sectionCount",
)


def _sanitize_paper_fact(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, Mapping):
        return None
    out: dict[str, Any] = {}
    paper_id = _public_identifier(item.get("paperId"))
    if paper_id:
        out["paperId"] = paper_id
    if item.get("source") in _SOURCE_VALUES:
        out["source"] = item["source"]
    if item.get("repoSource") in _REPO_SOURCES:
        out["repoSource"] = item["repoSource"]
    for key in ("tableCount", "referenceCount", "insightCount", "repositoryCount"):
        if key in item:
            number = _nonnegative_int(item.get(key))
            if number is not None:
                out[key] = number
    for key in ("hardwareFound", "hardwareStated", "bibtex"):
        if isinstance(item.get(key), bool):
            out[key] = item[key]
    if isinstance(item.get("hardware"), str):
        out["hardware"] = _safe_text(item["hardware"], 60)
    for key in ("sectionChars", "evalSettings"):
        raw = item.get(key)
        if isinstance(raw, Mapping):
            allowed = _SECTION_FACT_KEYS if key == "sectionChars" else ("proposed", "kept", "dropped")
            counts = {name: _nonnegative_int(raw.get(name)) for name in allowed if name in raw}
            out[key] = {name: number for name, number in counts.items() if number is not None}
    return out


def _sanitize_stage_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlist the stage facts added for the chat UI progress line."""

    clean: dict[str, Any] = {}
    for key in _STAGE_COUNT_KEYS:
        if key in facts:
            number = _nonnegative_int(facts.get(key))
            if number is not None:
                clean[key] = number
    if facts.get("source") in _SOURCE_VALUES:
        clean["source"] = facts["source"]
    if facts.get("repoSource") in _REPO_SOURCES | {"mixed"}:
        clean["repoSource"] = facts["repoSource"]
    if isinstance(facts.get("graphStatus"), str):
        clean["graphStatus"] = _safe_text(facts["graphStatus"], 24)
    eval_settings = facts.get("evalSettings")
    if isinstance(eval_settings, Mapping):
        counts = {name: _nonnegative_int(eval_settings.get(name)) for name in ("proposed", "kept", "dropped") if name in eval_settings}
        counts = {name: number for name, number in counts.items() if number is not None}
        if counts:
            clean["evalSettings"] = counts
    items = [_sanitize_paper_fact(item) for item in _sequence_items(facts.get("paperFacts"))[:64]]
    items = [item for item in items if item]
    if items:
        clean["paperFacts"] = items
    return clean


def _sanitize_facts(facts: Mapping[str, Any] | None) -> TraceFacts | None:
    """Enforce the facts allowlist at every event construction boundary."""

    if not isinstance(facts, Mapping):
        return None
    clean: dict[str, Any] = {}

    intent = facts.get("intent")
    if isinstance(intent, str) and intent in _PUBLIC_INTENTS:
        clean["intent"] = intent

    for key in (
        "resultCount",
        "selectedCount",
        "retryCount",
        "paperCount",
        "failedPdfCount",
        "repositoryCount",
        "bibtexCount",
    ):
        if key in facts:
            count = _nonnegative_int(facts.get(key))
            if count is not None:
                clean[key] = count

    for key in ("noteFallbackPaperIds",):
        if key not in facts:
            continue
        identifiers: list[str] = []
        seen: set[str] = set()
        for value in _sequence_items(facts.get(key)):
            identifier = _public_identifier(value)
            if identifier and identifier not in seen:
                identifiers.append(identifier)
                seen.add(identifier)
            if len(identifiers) >= 64:
                break
        if identifiers:
            clean[key] = identifiers

    if isinstance(facts.get("reportAvailable"), bool):
        clean["reportAvailable"] = facts["reportAvailable"]

    clean.update(_sanitize_stage_facts(facts))
    if isinstance(facts.get("sources"), (list, tuple)):
        sources = [_safe_text(item, 80) for item in list(facts["sources"])[:12] if isinstance(item, str) and item.strip()]
        if sources:
            clean["sources"] = sources

    timings = _numeric_timings(facts.get("timings")) if isinstance(facts.get("timings"), Mapping) else {}
    if timings:
        clean["timings"] = timings

    report_id = _public_identifier(facts.get("reportId"))
    if report_id and report_id.lower().endswith(".md"):
        clean["reportId"] = report_id

    return clean or None


def _paper_links(paper: Any, *, prefer_pdf: bool = False) -> list[dict[str, str]]:
    raw = _model_dump(paper)
    if not isinstance(raw, Mapping):
        return []
    arxiv_id = str(raw.get("arxiv_id") or raw.get("arxivId") or "").strip()
    pdf_url = _safe_link(raw.get("pdf_url") or raw.get("pdfUrl"))
    abs_url = _safe_link(f"https://arxiv.org/abs/{arxiv_id}") if arxiv_id else None
    href = pdf_url if prefer_pdf and pdf_url else abs_url or pdf_url
    if not href:
        return []
    return [{"label": _safe_text(raw.get("title") or arxiv_id or "Paper source", 120), "href": href}]


def _step_links(node: str, state: Mapping[str, Any]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    if node in {"search_papers", "eval_search", "read_paper"}:
        papers = state.get("search_results") if node == "search_papers" else state.get("selected_papers")
        for paper in (papers or [])[:8]:
            links.extend(_paper_links(paper, prefer_pdf=node == "read_paper"))
    elif node == "web_enrich":
        for paper in (state.get("selected_papers") or []):
            raw = _model_dump(paper)
            if not isinstance(raw, Mapping):
                continue
            for repo in (raw.get("github_repos") or [])[:4]:
                repo_raw = _model_dump(repo)
                if not isinstance(repo_raw, Mapping):
                    continue
                href = _safe_link(repo_raw.get("url"))
                if href:
                    links.append({"label": _safe_text(repo_raw.get("name") or "GitHub repository", 120), "href": href})
    return links[:12]


@dataclass
class _RunRecord:
    run_id: str
    input: RunInput
    provider: str
    model: str
    state: dict[str, Any]
    thread_id: str | None = None
    conversation_context: str = ""
    status: RunStatus = "running"
    current_node: str | None = None
    error: str | None = None
    trace_events: list[TraceEvent] = field(default_factory=list)
    started_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None            # set by set_terminal (measured wall time)
    # Loaded from disk: the public result and the private follow-up context of a
    # finished run whose graph state is gone.
    stored_result: "RunResult | None" = None
    stored_followup: dict[str, Any] | None = None
    started_nodes: set[str] = field(default_factory=set)
    completed_nodes: set[str] = field(default_factory=set)
    task_stream_seen: bool = False
    task_ids: dict[str, str] = field(default_factory=dict)
    task_started_at: dict[str, float] = field(default_factory=dict)
    started_task_ids: set[str] = field(default_factory=set)
    completed_task_ids: set[str] = field(default_factory=set)
    task_error: str | None = None
    assistant_answer: str = ""
    assistant_message_added: bool = False
    assistant_completed_nodes: set[str] = field(default_factory=set)
    # Message chunks can arrive once per token.  Keep the latest answer in the
    # snapshot, but throttle trace rows to bounded operational updates.
    last_answer_event_at: float = 0.0
    last_answer_event_length: int = 0


@dataclass
class _ThreadRecord:
    thread_id: str
    title: str
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    messages: list[ThreadMessage] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    artifacts: list[ArtifactDTO] = field(default_factory=list)


def _record_result(record: "_RunRecord") -> RunResult:
    """Public result of a run: from disk for a loaded run, else from its state.

    ``answerSummary.durationSec`` is overwritten with the measured wall time once
    the run has finished (the graph only estimates it).
    """

    if record.stored_result is not None:
        return record.stored_result
    result = _result_from_state(record.state, answer=record.assistant_answer)
    if result.answer_summary is not None and record.finished_at is not None:
        result.answer_summary["durationSec"] = max(0, int(round((record.finished_at - record.started_at).total_seconds())))
    return result


# Budgets of the stored follow-up context (the node trims again per question).
_FOLLOWUP_REPORT_CHARS = 150_000
_FOLLOWUP_TABLES_PER_PAPER = 40
_FOLLOWUP_TABLE_ROWS = 40
_FOLLOWUP_RELATED = 60


def _followup_context_from_state(state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Private, bounded context for the ``followup`` intent from a finished run's state.

    Holds the report id and Markdown, each paper's structured notes and tables,
    and the related-papers map.  It never leaves the server: it is not part of
    any DTO.  ``None`` when the run has no valid report.
    """

    report_value = state.get("final_report")
    status = str(state.get("status") or "").lower()
    if not _valid_report(report_value) or status in {"error", "failed"}:
        return None
    report_id = _report_id_from_state(state)
    report_text = ""
    if report_id:
        try:
            report_text = _safe_report_path(report_id).read_text(encoding="utf-8")
        except (OSError, HTTPException):
            report_text = ""
    report_text = (report_text or str(report_value))[:_FOLLOWUP_REPORT_CHARS]
    papers: list[dict[str, Any]] = []
    for paper in _sequence_items(state.get("selected_papers"))[:12]:
        raw = _model_dump(paper)
        if not isinstance(raw, Mapping):
            continue
        notes = raw.get("notes") if isinstance(raw.get("notes"), Mapping) else {}
        tables = []
        for table in _sequence_items(raw.get("tables"))[:_FOLLOWUP_TABLES_PER_PAPER]:
            if isinstance(table, Mapping):
                tables.append({
                    "label": _safe_text(table.get("label"), 40),
                    "caption": _safe_text(table.get("caption"), 300),
                    "rows": [
                        [_safe_text(cell, 80) for cell in row[:12]]
                        for row in _sequence_items(table.get("rows"))[:_FOLLOWUP_TABLE_ROWS] if isinstance(row, list)
                    ],
                })
        papers.append({
            "paperId": _safe_text(raw.get("paper_id"), 260),
            "arxivId": _safe_text(raw.get("arxiv_id"), 80),
            "title": _safe_text(raw.get("title"), 400),
            "published": _safe_text(raw.get("published"), 40),
            "methodName": _safe_text(notes.get("method_name"), 120),
            "notes": {
                key: _model_dump(notes.get(key))
                for key in ("problem", "method", "result", "limitation", "intuition", "insights", "eval_settings", "setup")
                if notes.get(key) not in (None, "", [])
            },
            "tables": tables,
        })
    graph = state.get("citation_graph")
    similarity = graph.get("similarity") if isinstance(graph, Mapping) else None
    related: list[dict[str, Any]] = []
    for node in _sequence_items(similarity.get("nodes") if isinstance(similarity, Mapping) else None)[:_FOLLOWUP_RELATED]:
        if isinstance(node, Mapping) and node.get("title"):
            related.append({
                "title": _safe_text(node.get("title"), 300),
                "abstract": _safe_text(node.get("abstract"), 1500),
                "year": node.get("year") if isinstance(node.get("year"), int) else None,
                "firstAuthor": _safe_text(node.get("first_author"), 120),
                "arxivId": _safe_text(node.get("arxiv_id"), 40),
            })
    return {"reportId": report_id, "reportText": report_text, "papers": papers, "related": related}


class ActiveRunError(RuntimeError):
    """Raised when the one-workflow-at-a-time gate is occupied."""

    def __init__(self, run_id: str):
        super().__init__("A research workflow is already running")
        self.run_id = run_id


class RunRegistry:
    """Thread-safe in-memory snapshots and append-only event history."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        # ``None`` keeps the registry purely in memory (tests, library use).
        self.persist_dir = persist_dir
        self.runs: dict[str, _RunRecord] = {}
        self.threads: dict[str, _ThreadRecord] = {}
        # Compatibility alias useful to small integration tests and callers.
        self._runs = self.runs
        self._threads = self.threads

    def create(
        self,
        run_input: RunInput,
        provider: str,
        model: str,
        state: dict[str, Any],
        *,
        thread_id: str | None = None,
        conversation_context: str = "",
    ) -> _RunRecord:
        with self.lock:
            active = next((item for item in self.runs.values() if item.status == "running"), None)
            if active is not None:
                raise ActiveRunError(active.run_id)
            if thread_id is not None and thread_id not in self.threads:
                raise KeyError(thread_id)
            run_id = str(uuid.uuid4())
            record = _RunRecord(
                run_id=run_id,
                input=run_input,
                provider=provider,
                model=model,
                state=dict(state),
                thread_id=thread_id,
                conversation_context=_safe_text(conversation_context, 12_000),
            )
            self.runs[run_id] = record
            if thread_id is not None:
                self.threads[thread_id].run_ids.append(run_id)
                self.threads[thread_id].updated_at = _now()
            self.condition.notify_all()
            return record

    def create_thread(self, title: str | None = None) -> _ThreadRecord:
        with self.lock:
            thread_id = str(uuid.uuid4())
            clean_title = _safe_text(title or "New research", 160) or "New research"
            record = _ThreadRecord(thread_id=thread_id, title=clean_title)
            self.threads[thread_id] = record
            self.condition.notify_all()
            self._persist_thread_unlocked(record)
            return record

    def get_thread(self, thread_id: str) -> _ThreadRecord | None:
        with self.lock:
            return self.threads.get(thread_id)

    def add_user_message(
        self,
        thread_id: str,
        *,
        query: str,
        run_id: str,
        attachments: Sequence[str] = (),
    ) -> ThreadMessage:
        with self.lock:
            thread = self.threads.get(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            clean_query = _safe_text(query or "Direct paper analysis", 8_000)
            message = ThreadMessage(
                id=str(uuid.uuid4()),
                role="user",
                content=clean_query,
                query=clean_query,
                runId=run_id,
                createdAt=_now(),
                attachments=[_safe_text(item, 240) for item in attachments[:32]],
            )
            thread.messages.append(message)
            if thread.title == "New research" and clean_query:
                thread.title = clean_query[:160]
            thread.updated_at = message.created_at
            self.condition.notify_all()
            return message

    def set_assistant_answer(self, run_id: str, answer: str) -> None:
        with self.lock:
            record = self.runs.get(run_id)
            if record is None:
                return
            record.assistant_answer = _safe_stream_text(answer)
            record.updated_at = _now()
            self.condition.notify_all()

    def finalize_thread(self, run_id: str) -> None:
        """Publish one assistant message and one artifact reference per run."""

        with self.lock:
            record = self.runs.get(run_id)
            if record is None or record.thread_id is None or record.assistant_message_added:
                return
            thread = self.threads.get(record.thread_id)
            if thread is None:
                return
            # Reuse the public result projection so an error-handler notice or
            # report placeholder cannot be published as a normal
            # assistant artifact in thread history.
            public_result = _record_result(record)
            answer = public_result.answer or ""
            report_id = public_result.report_id
            if not answer and not report_id:
                return
            now = _now()
            message = ThreadMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content=answer,
                answer=answer or None,
                runId=run_id,
                reportId=report_id,
                createdAt=now,
            )
            thread.messages.append(message)
            if report_id or answer:
                artifact_id = report_id or f"answer-{run_id[:12]}"
                artifact = ArtifactDTO(
                    id=artifact_id,
                    kind="report" if report_id else "answer",
                    title=_first_heading(answer, "Research answer"),
                    summary=(f"Markdown report {report_id}" if report_id else "Assistant answer from this run"),
                    reportId=report_id,
                )
                if not any(item.id == artifact.id for item in thread.artifacts):
                    thread.artifacts.append(artifact)
            thread.updated_at = now
            record.assistant_message_added = True
            self.condition.notify_all()

    def thread_context(self, thread_id: str) -> str:
        """Build bounded follow-up context from safe prior run artifacts.

        The graph's paper records contain both useful PMRL findings and
        private extraction/cache fields.  This method selects only the
        browser-safe fields needed for a follow-up (including ordinal paper
        numbers), and deliberately never forwards ``extracted_text``,
        ``sections`` or ``local_pdf_path``.
        """

        with self.lock:
            thread = self.threads.get(thread_id)
            if thread is None:
                raise KeyError(thread_id)

            lines: list[str] = []
            if thread.artifacts:
                lines.append("Prior research artifacts in this thread:")
                for artifact in thread.artifacts[-12:]:
                    report = f"; report_id={artifact.report_id}" if artifact.report_id else ""
                    lines.append(
                        f"- {_context_text(artifact.title, 240)} ({artifact.kind}){report}: "
                        f"{_context_text(artifact.summary, 500)}"
                    )

            # Include the most recent completed/degraded runs, not just report
            # metadata.  A follow-up such as "explain Method of paper 2" needs
            # the numbered paper and its PMRL method to remain addressable.
            completed = [
                self.runs[run_id]
                for run_id in thread.run_ids
                if run_id in self.runs and self.runs[run_id].status in {"success", "degraded"}
            ][-8:]
            for run_index, record in enumerate(reversed(completed), 1):
                papers = [_model_dump(paper) for paper in (record.state.get("selected_papers") or [])]
                report_id = _report_id_from_state(record.state) if record.state else None
                if not papers and record.stored_followup:
                    # A run loaded from disk keeps its notes in the follow-up context.
                    report_id = record.stored_followup.get("reportId")
                    papers = [
                        {
                            "paper_id": item.get("paperId"), "title": item.get("title"), "arxiv_id": item.get("arxivId"),
                            "notes": item.get("notes"),
                        }
                        for item in record.stored_followup.get("papers") or [] if isinstance(item, Mapping)
                    ]
                if not papers:
                    continue
                reference = "Most recent completed research run" if run_index == 1 else "Prior completed research run"
                report = f"; report_id={report_id}" if report_id else ""
                lines.append(f"{reference} {record.run_id[:12]}{report}:")
                for ordinal, raw in enumerate(papers[:12], 1):
                    if not isinstance(raw, Mapping):
                        continue
                    title = _context_text(raw.get("title") or "Untitled paper", 260)
                    arxiv_id = _context_text(raw.get("arxiv_id") or raw.get("arxivId"), 80)
                    paper_id = _context_text(raw.get("paper_id") or raw.get("paperId"), 120)
                    summary = _context_text(raw.get("summary"), 600)
                    identifier = f"; arxiv_id={arxiv_id}" if arxiv_id else ""
                    lines.append(f"{ordinal}. {title}{identifier}; paper_id={paper_id}")
                    if summary:
                        lines.append(f"   Summary: {summary}")
                    notes = _model_dump(raw.get("notes"))
                    if isinstance(notes, Mapping):
                        for label, key in (
                            ("Problem", "problem"),
                            ("Method", "method"),
                            ("Intuition", "intuition"),
                            ("Result", "result"),
                            ("Limitation", "limitation"),
                        ):
                            value = _context_text(notes.get(key), 800)
                            if value:
                                lines.append(f"   {label}: {value}")
                        for item in _sequence_items(notes.get("insights"))[:3]:
                            if isinstance(item, Mapping) and item.get("finding"):
                                lines.append(f"   Insight: {_context_text(item.get('finding'), 300)}")
                        setup = notes.get("setup") if isinstance(notes.get("setup"), Mapping) else {}
                        if setup.get("hardware"):
                            lines.append(f"   Hardware: {_context_text(setup.get('hardware'), 160)}")
                    bibtex = _context_text(raw.get("bibtex"), 900)
                    if bibtex:
                        lines.append(f"   BibTeX: {bibtex}")

            return "\n".join(lines)[:12_000]

    def followup_context(self, thread_id: str) -> dict[str, Any] | None:
        """Follow-up context of the thread's newest finished run that has a report."""

        with self.lock:
            thread = self.threads.get(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            for run_id in reversed(thread.run_ids):
                record = self.runs.get(run_id)
                if record is None or record.status not in {"success", "degraded"}:
                    continue
                if record.stored_followup is None and record.stored_result is None:
                    record.stored_followup = _followup_context_from_state(record.state)
                if record.stored_followup:
                    return record.stored_followup
            return None

    # --- persistence: one JSON file per thread -----------------------------------

    def _run_summary_unlocked(self, record: _RunRecord) -> dict[str, Any]:
        if record.stored_followup is None and record.stored_result is None:
            record.stored_followup = _followup_context_from_state(record.state)
        return {
            "runId": record.run_id,
            "status": record.status,
            "provider": record.provider,
            "model": record.model,
            "input": record.input.model_dump(mode="json", by_alias=True),
            "answer": _safe_answer(record.assistant_answer) or None,
            "error": record.error,
            "currentNode": record.current_node,
            "startedAt": record.started_at.isoformat(),
            "updatedAt": record.updated_at.isoformat(),
            "finishedAt": record.finished_at.isoformat() if record.finished_at else None,
            "result": _record_result(record).model_dump(mode="json", by_alias=True),
            # Private (never in a DTO): lets a follow-up work after a restart.
            "followupContext": record.stored_followup,
        }

    def _persist_thread_unlocked(self, thread: _ThreadRecord) -> None:
        """Write the thread and its finished runs atomically; failures only log."""

        if self.persist_dir is None:
            return
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "threadId": thread.thread_id,
                "title": thread.title,
                "createdAt": thread.created_at.isoformat(),
                "updatedAt": thread.updated_at.isoformat(),
                "runIds": [run_id for run_id in thread.run_ids if run_id in self.runs and self.runs[run_id].status in TERMINAL_STATUSES],
                "messages": [message.model_dump(mode="json", by_alias=True) for message in thread.messages],
                "artifacts": [artifact.model_dump(mode="json", by_alias=True) for artifact in thread.artifacts],
                "runs": [
                    self._run_summary_unlocked(self.runs[run_id])
                    for run_id in thread.run_ids
                    if run_id in self.runs and self.runs[run_id].status in TERMINAL_STATUSES
                ],
            }
            target = self.persist_dir / f"{thread.thread_id}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(target)
        except Exception:
            logger.exception("Could not persist thread", extra={"thread_id": thread.thread_id})

    def load_threads(self) -> int:
        """Load persisted threads (call once at startup); returns the thread count.

        A stored run is always terminal.  Its trace events are not stored, so a
        reloaded run snapshot has an empty ``traceEvents`` list.
        """

        if self.persist_dir is None or not self.persist_dir.is_dir():
            return 0
        loaded = 0
        for path in sorted(self.persist_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, Mapping) or data.get("version") != 1:
                    continue
                thread = _ThreadRecord(
                    thread_id=str(data["threadId"]),
                    title=_safe_text(data.get("title"), 160) or "New research",
                    created_at=datetime.fromisoformat(str(data["createdAt"])),
                    updated_at=datetime.fromisoformat(str(data["updatedAt"])),
                    messages=[ThreadMessage.model_validate(item) for item in data.get("messages") or []],
                    artifacts=[ArtifactDTO.model_validate(item) for item in data.get("artifacts") or []],
                )
                restored: dict[str, _RunRecord] = {}
                for item in data.get("runs") or []:
                    started = datetime.fromisoformat(str(item["startedAt"]))
                    record = _RunRecord(
                        run_id=str(item["runId"]),
                        input=RunInput.model_validate(item.get("input") or {}),
                        provider=str(item.get("provider") or ""),
                        model=str(item.get("model") or ""),
                        state={},
                        thread_id=thread.thread_id,
                        status=item.get("status") if item.get("status") in TERMINAL_STATUSES else "error",
                        current_node=item.get("currentNode"),
                        error=item.get("error"),
                        started_at=started,
                        updated_at=datetime.fromisoformat(str(item["updatedAt"])),
                        finished_at=datetime.fromisoformat(item["finishedAt"]) if item.get("finishedAt") else None,
                        assistant_answer=str(item.get("answer") or ""),
                        assistant_message_added=True,
                        stored_result=RunResult.model_validate(item.get("result") or {}),
                        stored_followup=item.get("followupContext") if isinstance(item.get("followupContext"), dict) else None,
                    )
                    restored[record.run_id] = record
                    thread.run_ids.append(record.run_id)
                with self.lock:
                    self.runs.update(restored)
                    self.threads[thread.thread_id] = thread
                loaded += 1
            except Exception:
                logger.exception("Skipping unreadable thread file", extra={"path": path.name})
        return loaded

    def get(self, run_id: str) -> _RunRecord | None:
        with self.lock:
            return self.runs.get(run_id)

    def snapshot(self, run_id: str) -> RunSnapshot:
        with self.lock:
            record = self.runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            return self._snapshot_unlocked(record)

    def _snapshot_unlocked(self, record: _RunRecord) -> RunSnapshot:
        current_progress = record.trace_events[-1].progress if record.trace_events else 0.0
        if record.status in TERMINAL_STATUSES and record.status != "error":
            current_progress = 1.0
        return RunSnapshot(
            runId=record.run_id,
            input=record.input,
            provider=record.provider,
            model=record.model,
            threadId=record.thread_id,
            answer=_safe_answer(record.assistant_answer) or None,
            status=record.status,
            currentNode=record.current_node,
            traceEvents=list(record.trace_events),
            result=_record_result(record),
            error=record.error,
            seq=record.trace_events[-1].seq if record.trace_events else 0,
            progress=current_progress,
            timings=_state_timings(record.state),
            startedAt=record.started_at,
            createdAt=record.started_at,
            updatedAt=record.updated_at,
        )

    def _thread_run_unlocked(self, record: _RunRecord) -> ThreadRunDTO:
        report_id = _record_result(record).report_id
        return ThreadRunDTO(
            runId=record.run_id,
            status=record.status,
            provider=record.provider,
            model=record.model,
            currentNode=record.current_node,
            answer=_safe_answer(record.assistant_answer) or None,
            reportId=report_id,
            error=record.error,
            createdAt=record.started_at,
            updatedAt=record.updated_at,
        )

    def _thread_summary_unlocked(self, thread: _ThreadRecord) -> ThreadSummary:
        return ThreadSummary(
            threadId=thread.thread_id,
            title=thread.title,
            createdAt=thread.created_at,
            updatedAt=thread.updated_at,
            runIds=list(thread.run_ids),
        )

    def _thread_snapshot_unlocked(self, thread: _ThreadRecord) -> ThreadSnapshot:
        return ThreadSnapshot(
            threadId=thread.thread_id,
            title=thread.title,
            createdAt=thread.created_at,
            updatedAt=thread.updated_at,
            runIds=list(thread.run_ids),
            messages=list(thread.messages),
            runs=[self._thread_run_unlocked(self.runs[run_id]) for run_id in thread.run_ids if run_id in self.runs],
            artifacts=list(thread.artifacts),
        )

    def thread_snapshot(self, thread_id: str) -> ThreadSnapshot:
        with self.lock:
            thread = self.threads.get(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            return self._thread_snapshot_unlocked(thread)

    def thread_summaries(self) -> list[ThreadSummary]:
        with self.lock:
            return [self._thread_summary_unlocked(item) for item in self.threads.values()]

    def set_state(self, run_id: str, state: Mapping[str, Any], *, current_node: str | None = None) -> None:
        with self.lock:
            record = self.runs.get(run_id)
            if record is None:
                return
            record.state = dict(state)
            if current_node:
                record.current_node = current_node
            record.updated_at = _now()
            self.condition.notify_all()

    def set_terminal(self, run_id: str, status: RunStatus, *, error: str | None = None) -> None:
        with self.lock:
            record = self.runs.get(run_id)
            if record is None:
                return
            record.status = status
            record.error = _safe_text(error, 1200) if error else None
            record.updated_at = _now()
            record.finished_at = record.updated_at
            self.condition.notify_all()
            if record.thread_id is not None and record.thread_id in self.threads:
                self._persist_thread_unlocked(self.threads[record.thread_id])

    def append_event(
        self,
        run_id: str,
        event_type: TraceType,
        *,
        node: str,
        status: str,
        summary: str = "",
        details: Mapping[str, Any] | None = None,
        facts: Mapping[str, Any] | None = None,
        delta: str | None = None,
        links: Sequence[Mapping[str, str]] | None = None,
        duration_ms: int | None = None,
        timings: Mapping[str, Any] | None = None,
        progress: float = 0.0,
    ) -> TraceEvent:
        with self.lock:
            record = self.runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            event = TraceEvent(
                runId=run_id,
                seq=len(record.trace_events) + 1,
                type=event_type,
                node=node,
                label=NODE_LABELS.get(node, node.replace("_", " ").title()),
                kind=NODE_KINDS.get(node, "workflow"),
                status=status,
                summary=_safe_text(summary, 1200),
                details=_details_text(details),
                # Facts describe a graph state update.  Lifecycle markers and
                # assistant/run events must remain factless even if a caller
                # accidentally supplies a mapping.
                facts=_sanitize_facts(facts) if event_type == "step.updated" else None,
                delta=_safe_stream_text(delta, 8_000) if delta else None,
                links=[
                    {"label": _safe_text(link.get("label"), 160), "href": safe_href}
                    for link in (links or [])
                    if (safe_href := _safe_link(link.get("href"))) and link.get("label")
                ],
                durationMs=_duration_ms(duration_ms),
                timings=_numeric_timings(timings),
                timestamp=_now(),
                progress=max(0.0, min(1.0, float(progress))),
            )
            record.trace_events.append(event)
            record.updated_at = event.timestamp
            self.condition.notify_all()
            return event

    def events_after(self, run_id: str, seq: int) -> list[TraceEvent]:
        with self.lock:
            record = self.runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            return [event for event in record.trace_events if event.seq > seq]


run_registry = RunRegistry(persist_dir=config.THREADS_DIR)
# Public alias kept short for callers that want to reset/inspect the registry.
registry = run_registry


def _initial_state(
    query: str,
    raw_inputs: Sequence[str],
    *,
    conversation_context: str = "",
    followup_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = {
        "user_query": query.strip() or "Direct Paper Analysis",
        "raw_inputs": list(raw_inputs),
        "intent": "search",
        "status": "running",
        "error_message": None,
        "search_query": query.strip(),
        "search_results": [],
        "retry_count": 0,
        "eval_passed": False,
        "eval_feedback": "",
        "selected_papers": [],
        "final_report": None,
        "citation_graph": None,
        "node_timings": {},
        "trace_logs": [],
        "error_logs": [],
        # These keys are intentionally supplied at the bridge boundary for
        # thread-aware/future graph schemas.  Current ResearchState ignores
        # unknown channels, while injected graphs can consume them directly.
        "conversation_context": _safe_text(conversation_context, 12_000),
    }
    if followup_context:
        # Structured report/notes/tables of the thread's latest finished run;
        # lets the router choose ``followup`` and ``answer_followup`` answer.
        state["followup_context"] = dict(followup_context)
    return state


def _graph_stream(graph: Any, initial_state: dict[str, Any], graph_config: dict[str, Any]) -> Iterable[Any]:
    stream = getattr(graph, "stream", None)
    if not callable(stream):
        raise TypeError("Configured research graph does not provide stream()")
    try:
        return stream(
            initial_state,
            graph_config,
            stream_mode=["tasks", "updates", "messages"],
            version="v2",
        )
    except TypeError as first_error:
        # Simple fake graphs and older LangGraph releases often omit v2 or
        # only accept the original updates stream.
        try:
            return stream(initial_state, graph_config, stream_mode="updates")
        except TypeError:
            try:
                return stream(initial_state, graph_config)
            except TypeError:
                raise first_error


def _stream_part(payload: Any) -> tuple[str, Any]:
    """Normalize LangGraph v2 parts while preserving legacy raw update maps."""

    if isinstance(payload, Mapping):
        part_type = payload.get("type") or payload.get("event")
        if part_type in {"tasks", "task", "updates", "update", "messages", "message"} and "data" in payload:
            normalized = {"task": "tasks", "update": "updates", "message": "messages"}.get(str(part_type), str(part_type))
            return normalized, payload.get("data")
    return "legacy", payload


def _task_data(data: Any) -> dict[str, Any]:
    value = _model_dump(data)
    return dict(value) if isinstance(value, Mapping) else {}


def _task_node(data: Mapping[str, Any]) -> str:
    payload = data.get("payload") or data.get("task") or {}
    payload = _model_dump(payload)
    payload_map = payload if isinstance(payload, Mapping) else {}
    value = (
        data.get("name")
        or data.get("node")
        or data.get("node_name")
        or payload_map.get("name")
        or payload_map.get("node")
        or "workflow"
    )
    return str(value)


def _task_id(data: Mapping[str, Any]) -> str:
    payload = data.get("payload") or data.get("task") or {}
    payload = _model_dump(payload)
    payload_map = payload if isinstance(payload, Mapping) else {}
    value = (
        data.get("id")
        or data.get("task_id")
        or data.get("taskId")
        or payload_map.get("id")
        or payload_map.get("task_id")
        or ""
    )
    return str(value)


def _task_is_end(data: Mapping[str, Any], known_task: bool) -> bool:
    payload = data.get("payload") or data.get("task") or {}
    payload = _model_dump(payload)
    payload_map = payload if isinstance(payload, Mapping) else {}
    values = dict(payload_map)
    values.update(data)
    if any(key in values for key in ("result", "error", "exception", "output")):
        return True
    status = str(values.get("status") or "").lower()
    if status in {"completed", "complete", "success", "failed", "error"}:
        return True
    # A v2 task end has the same id as its start but no input/triggers.
    return known_task and not any(key in values for key in ("input", "triggers"))


def _message_part(data: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(data, Mapping):
        message = data.get("message") or data.get("chunk") or data.get("data")
        metadata = data.get("metadata") or {}
        return message, metadata if isinstance(metadata, Mapping) else {}
    if isinstance(data, (tuple, list)) and len(data) >= 2:
        metadata = data[1] if isinstance(data[1], Mapping) else {}
        return data[0], metadata
    return data, {}


def _message_node(metadata: Mapping[str, Any]) -> str:
    for key in ("langgraph_node", "node", "node_name", "name"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def _message_text(message: Any) -> str:
    """Extract plain text only; tool calls and arbitrary message metadata stay out."""

    if message is None:
        return ""
    value = getattr(message, "content", message)
    if isinstance(value, str):
        return _safe_stream_text(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return _safe_stream_text("".join(parts))
    return ""


def _append_followup_chunk(previous: str, incoming: str) -> str:
    """Append a streamed chunk of the single follow-up call; a cumulative repeat replaces."""

    candidate = _safe_stream_text(incoming)
    if not candidate:
        return previous
    if len(previous) > 8 and candidate.startswith(previous):
        return candidate
    return _safe_stream_text(previous + candidate)


def _answer_lead(summary: Any) -> str:
    """The short chat answer: ``lead`` (tldr) of ``answer_summary``."""

    if isinstance(summary, Mapping):
        return _safe_stream_text(str(summary.get("lead") or summary.get("tldr") or "")).strip()
    return ""


def _task_summary(node: str, completed: bool, failed: bool = False) -> str:
    label = NODE_LABELS.get(node, node.replace("_", " ").title())
    if failed:
        return f"{label} failed"
    return f"{label} {'completed' if completed else 'started'}"


def _answer_update_details(answer: str) -> dict[str, Any]:
    return {"answerLength": len(answer), "source": "assistant_answer"}


def _public_node(value: Any) -> str:
    """Keep graph-provided node names inside the documented node allowlist."""

    node = str(value or "").strip()
    return node if node in NODE_LABELS else "workflow"


def _append_node_started(
    run_id: str,
    node: str,
    registry_instance: RunRegistry,
    record: _RunRecord,
    *,
    task_id: str | None = None,
) -> None:
    node = _public_node(node)
    if node == "workflow":
        return
    if task_id:
        task_key = str(task_id)
        if task_key in record.started_task_ids:
            return
        record.started_task_ids.add(task_key)
        record.task_started_at[task_key] = time.monotonic()
        record.started_nodes.add(node)
        # The bridge emits a synthetic router start before the graph is
        # constructed.  Avoid duplicating it for the first native router task;
        # retries still get their own task lifecycle row.
        if node == "router" and node in record.started_nodes and len(record.started_task_ids) == 1:
            return
    elif node in record.started_nodes:
        return
    else:
        record.started_nodes.add(node)
    registry_instance.append_event(
        run_id,
        "step.started",
        node=node,
        status="running",
        summary=f"Starting {NODE_LABELS.get(node, 'workflow')}" if node != "router" else "Starting workflow routing",
        progress=0.0,
    )


def _append_node_update(
    run_id: str,
    node: str,
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    registry_instance: RunRegistry,
    *,
    failed: bool = False,
    completed: bool = False,
    summary: str | None = None,
    details: Mapping[str, Any] | None = None,
    links: Sequence[Mapping[str, str]] | None = None,
    task_id: str | None = None,
    duration_ms: int | None = None,
    timings: Mapping[str, Any] | None = None,
    emit_updated: bool = True,
) -> None:
    node = _public_node(node)
    if node == "workflow":
        return
    operational_summary = summary or _step_summary(node, state, update)
    operational_details = details if details is not None else _step_details(node, state, update)
    operational_facts = _step_facts(node, state, update)
    measured_duration = _duration_ms(duration_ms)
    measured_timings = _numeric_timings(timings) if timings is not None else _timings_from_mapping(update)
    if measured_duration is None:
        measured_duration = next(
            (
                _duration_ms(update.get(key))
                for key in ("duration_ms", "durationMs")
                if key in update
            ),
            None,
        )
    if measured_duration is None:
        for key in (f"{node}.durationMs", f"{node}.duration_ms", "durationMs", "duration_ms"):
            if key in measured_timings:
                measured_duration = measured_timings[key]
                break
    # A task-end completion already carries the authoritative duration; the
    # accompanying state update (when one exists) owns the operational
    # ``step.updated`` row.  Emitting a second ``step.updated`` with the task
    # summary would render as a duplicated "completed" line in Activity.
    if emit_updated:
        registry_instance.append_event(
            run_id,
            "step.updated",
            node=node,
            status="error" if failed else "running",
            summary=operational_summary,
            details=operational_details,
            facts=operational_facts,
            links=_step_links(node, state) if links is None else links,
            duration_ms=measured_duration,
            timings=measured_timings,
            progress=0.0,
        )
    record = registry_instance.get(run_id)
    if record is None:
        return
    if task_id:
        already_completed = task_id in record.completed_task_ids
    else:
        already_completed = node in record.completed_nodes
    if completed and not already_completed:
        if task_id:
            record.completed_task_ids.add(task_id)
        else:
            record.completed_nodes.add(node)
        registry_instance.append_event(
            run_id,
            "step.completed",
            node=node,
            status="error" if failed else "completed",
            summary=operational_summary,
            details=operational_details,
            links=_step_links(node, state) if links is None else links,
            duration_ms=measured_duration,
            timings=measured_timings,
            progress=0.0,
        )


def _append_assistant_delta(
    run_id: str,
    node: str,
    answer: str,
    registry_instance: RunRegistry,
    record: _RunRecord,
) -> None:
    """Emit one bounded answer segment, leaving the complete answer in state."""

    if len(answer) <= record.last_answer_event_length:
        return
    delta = answer[record.last_answer_event_length :]
    if not delta:
        return
    registry_instance.append_event(
        run_id,
        "assistant.delta",
        node=_public_node(node),
        status="running",
        summary="Assistant answer updated",
        details=_answer_update_details(answer),
        delta=delta,
        progress=0.0,
    )
    record.last_answer_event_at = time.monotonic()
    record.last_answer_event_length = len(answer)


def _append_assistant_completed(
    run_id: str,
    node: str,
    registry_instance: RunRegistry,
    record: _RunRecord,
) -> None:
    """Flush a final delta and publish one public completion marker per node."""

    node = _public_node(node)
    if node not in ANSWER_NODES or node in record.assistant_completed_nodes:
        return
    _append_assistant_delta(run_id, node, record.assistant_answer, registry_instance, record)
    registry_instance.append_event(
        run_id,
        "assistant.completed",
        node=node,
        status="completed",
        summary="Assistant answer completed",
        details=_answer_update_details(record.assistant_answer),
        progress=0.0,
    )
    record.assistant_completed_nodes.add(node)


def _failure_progress(record: _RunRecord) -> float:
    """Never advertise a failed workflow as 100% complete."""

    if not record.trace_events:
        return 0.0
    return min(0.99, max(0.0, record.trace_events[-1].progress))


def _final_graph_state(graph: Any, graph_config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    getter = getattr(graph, "get_state", None)
    if not callable(getter):
        return state
    try:
        snapshot = getter(graph_config)
        values = getattr(snapshot, "values", snapshot)
        mapping = _as_state_mapping(values)
        if mapping:
            merged = dict(state)
            _merge_state(merged, mapping)
            return merged
    except Exception:
        # The stream already produced the best available state; a missing
        # optional checkpointer must not turn a successful run into a failure.
        pass
    return state


def _execute_run(
    run_id: str,
    graph_factory: Callable[[], Any] | Any,
    initial_state: dict[str, Any],
    registry_instance: RunRegistry,
) -> None:
    record = registry_instance.get(run_id)
    if record is None:
        return
    current_state = dict(initial_state)
    graph_config = {"configurable": {"thread_id": run_id}}
    try:
        # Router is deterministic first node in the production graph.  This
        # gives the UI immediate feedback even if the first network call is
        # slow; fake graphs can still report their own first node afterwards.
        record.started_nodes.add("router")
        registry_instance.set_state(run_id, current_state, current_node="router")
        registry_instance.append_event(
            run_id,
            "step.started",
            node="router",
            status="running",
            summary="Starting workflow routing",
            progress=0.0,
        )

        graph = graph_factory() if callable(graph_factory) else graph_factory
        for payload in _graph_stream(graph, current_state, graph_config):
            part_type, data = _stream_part(payload)

            if part_type == "messages":
                # Only ``answer_followup`` streams tokens: it makes ONE LLM call.
                # The report nodes run several LLM calls in parallel; their
                # tokens would interleave, so they are ignored (the streamed
                # answer there is the ``lead`` of ``answer_summary``).
                message, metadata = _message_part(data)
                if _message_node(metadata) != "answer_followup":
                    continue
                text = _message_text(message)
                if not text:
                    continue
                streamed = _append_followup_chunk(record.assistant_answer, text)
                if streamed == record.assistant_answer:
                    continue
                registry_instance.set_assistant_answer(run_id, streamed)
                _append_node_started(run_id, "answer_followup", registry_instance, record)
                now = time.monotonic()
                if (
                    not record.last_answer_event_length
                    or len(streamed) - record.last_answer_event_length >= 256
                    or now - record.last_answer_event_at >= 0.15
                    or streamed.endswith((".", "!", "?", "\n"))
                ):
                    _append_assistant_delta(run_id, "answer_followup", streamed, registry_instance, record)
                continue

            if part_type == "tasks":
                task = _task_data(data)
                node = _public_node(_task_node(task))
                if node == "workflow":
                    continue
                record.task_stream_seen = True
                task_id = _task_id(task)
                known_task = bool(task_id and task_id in record.task_ids)
                if task_id:
                    record.task_ids[task_id] = node
                is_end = _task_is_end(task, known_task)
                if not is_end:
                    _append_node_started(run_id, node, registry_instance, record, task_id=task_id or None)
                    record.current_node = node
                    registry_instance.set_state(run_id, current_state, current_node=node)
                    continue

                failure_value = task.get("error") or task.get("exception")
                failed = bool(failure_value) or str(task.get("status") or "").lower() in {"failed", "error"}
                record.current_node = node
                if failed:
                    # Native task exceptions may contain request paths,
                    # provider metadata, or credentials.  Public state gets a
                    # stable actionable message instead of that raw string.
                    record.task_error = f"{NODE_LABELS.get(node, node)} failed; check provider/network input"
                registry_instance.set_state(run_id, current_state, current_node=node)
                _append_node_started(run_id, node, registry_instance, record, task_id=task_id or None)
                duration_ms = None
                if task_id:
                    started_at = record.task_started_at.pop(task_id, None)
                    if started_at is not None:
                        duration_ms = max(0, int((time.monotonic() - started_at) * 1000))
                task_details: dict[str, Any] = {"source": "task"}
                if duration_ms is not None:
                    task_details["durationMs"] = duration_ms
                if failed:
                    task_details["failed"] = True
                _append_node_update(
                    run_id,
                    node,
                    current_state,
                    {},
                    registry_instance,
                    failed=failed,
                    completed=True,
                    summary=_task_summary(node, True, failed),
                    details=task_details,
                    task_id=task_id or None,
                    duration_ms=duration_ms,
                    timings=_timings_from_mapping(task),
                    emit_updated=False,
                )
                if node in ANSWER_NODES and not failed:
                    _append_assistant_completed(run_id, node, registry_instance, record)
                continue

            # ``updates`` is the v2 state-update channel.  A legacy fake graph
            # yields the same map without a type wrapper; both paths use the
            # same deterministic, public state summaries.
            updates = _normalise_updates(data if part_type == "updates" else payload)
            for node, update in updates:
                node = _public_node(node or _node_from_logs(update.get("trace_logs")))
                _merge_state(current_state, update)
                if node == "answer_followup" and isinstance(update.get("followup_answer"), str):
                    final_answer = _safe_stream_text(update["followup_answer"]).strip()
                    if final_answer:
                        registry_instance.set_assistant_answer(run_id, final_answer)
                        _append_node_started(run_id, node, registry_instance, record)
                        _append_assistant_delta(run_id, node, record.assistant_answer, registry_instance, record)
                if node == "final_report":
                    lead = _answer_lead(update.get("answer_summary"))
                    if lead:
                        registry_instance.set_assistant_answer(run_id, lead)
                        _append_node_started(run_id, node, registry_instance, record)
                        _append_assistant_delta(run_id, node, record.assistant_answer, registry_instance, record)
                node_failed = str(update.get("status") or current_state.get("status") or "").lower() == "error"
                registry_instance.set_state(run_id, current_state, current_node=node)
                if node != "workflow":
                    record.current_node = node
                    _append_node_started(run_id, node, registry_instance, record)
                    _append_node_update(
                        run_id,
                        node,
                        current_state,
                        update,
                        registry_instance,
                        failed=node_failed,
                        completed=not record.task_stream_seen or node_failed,
                    )
                    if node_failed:
                        node_error = _safe_text(update.get("error_message"), 600)
                        record.task_error = (
                            f"{NODE_LABELS.get(node, node)}: {node_error}"
                            if node_error
                            else f"{NODE_LABELS.get(node, node)} failed; check provider/network input"
                        )

        current_state = _final_graph_state(graph, graph_config, current_state)
        # Checkpointer values are authoritative for the final DTO, but never
        # expose their raw task input/result payloads.
        answer_node = "answer_followup" if current_state.get("intent") == "followup" else "final_report"
        candidate_answer = None if record.assistant_answer else (
            current_state.get("followup_answer")
            or _answer_lead(current_state.get("answer_summary"))
            or current_state.get("final_report")
        )
        if candidate_answer:
            registry_instance.set_assistant_answer(run_id, _safe_answer(candidate_answer))
        if candidate_answer or record.assistant_answer:
            _append_node_started(run_id, answer_node, registry_instance, record)
            _append_assistant_completed(run_id, answer_node, registry_instance, record)
        registry_instance.set_state(run_id, current_state, current_node=record.current_node)
        state_status = str(current_state.get("status") or "success").lower()
        errors = current_state.get("error_logs") or []
        if record.task_error:
            state_status = "error"
            current_state["status"] = "error"
            current_state["error_message"] = record.task_error
            registry_instance.set_state(run_id, current_state, current_node=record.current_node)
        if state_status == "error":
            raw_message = record.task_error or current_state.get("error_message") or (errors[-1] if errors else None)
            message = _public_failure_message(_public_node(record.current_node or "workflow"), raw_message)
            # Publish the terminal event and status atomically.  Otherwise a
            # reconnect can observe ``status=error`` and close its SSE stream
            # before ``run.failed`` has been appended.
            with registry_instance.lock:
                registry_instance.finalize_thread(run_id)
                registry_instance.append_event(
                    run_id,
                    "run.failed",
                    node=_public_node(record.current_node),
                    status="error",
                    summary=message,
                    details={"message": message},
                    progress=_failure_progress(record),
                )
                registry_instance.set_terminal(run_id, "error", error=message)
            return

        terminal_status: RunStatus = "degraded" if state_status == "degraded" or errors else "success"
        with registry_instance.lock:
            registry_instance.finalize_thread(run_id)
            registry_instance.append_event(
                run_id,
                "run.completed",
                node=_public_node(record.current_node or "final_report"),
                status=terminal_status,
                summary="Research workflow completed" if terminal_status == "success" else "Research workflow completed with warnings",
                details={"reportId": _report_id_from_state(current_state)},
                progress=1.0,
            )
            registry_instance.set_terminal(run_id, terminal_status)
    except Exception:
        node = _public_node(record.current_node or "workflow")
        logger.exception("Research workflow failed", extra={"run_id": run_id, "node": node})
        message = _public_failure_message(node)
        registry_instance.set_state(run_id, current_state, current_node=node)
        with registry_instance.lock:
            if node != "workflow" and node not in record.completed_nodes:
                _append_node_update(
                    run_id,
                    node,
                    current_state,
                    {},
                    registry_instance,
                    failed=True,
                    completed=True,
                    summary=_task_summary(node, True, True),
                    details={"message": message},
                    emit_updated=False,
                )
            registry_instance.finalize_thread(run_id)
            registry_instance.append_event(
                run_id,
                "run.failed",
                node=node,
                status="error",
                summary=message,
                details={"message": message},
                progress=_failure_progress(record),
            )
            registry_instance.set_terminal(run_id, "error", error=message)


def _parse_form_inputs(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for raw in values:
        if isinstance(raw, (list, tuple)):
            result.extend(_parse_form_inputs(raw))
            continue
        text = str(raw or "").strip()
        if not text:
            continue
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                result.extend(_parse_form_inputs(decoded))
                continue
        # Comma/newline separated IDs are accepted while repeated form keys
        # remain the canonical multipart representation.
        chunks = re.split(r"[,\n]", text) if "," in text or "\n" in text else [text]
        result.extend(item.strip() for item in chunks if item.strip())
    return result


def _validate_paper_input(value: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 2048:
        raise HTTPException(status_code=422, detail="Each paper input must be a non-empty ArXiv ID or URL")
    parsed = urlparse(text)
    if parsed.scheme:
        host = (parsed.hostname or "").lower().rstrip(".")
        try:
            port = parsed.port
        except ValueError:
            raise HTTPException(status_code=422, detail="Paper URL has an invalid port") from None
        if (
            parsed.scheme not in {"http", "https"}
            or host not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
            or parsed.username
            or parsed.password
            or port not in {None, 80, 443}
        ):
            raise HTTPException(status_code=422, detail="Paper URLs must point to a public ArXiv paper")
        match = re.fullmatch(
            r"/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?/?",
            parsed.path,
            flags=re.IGNORECASE,
        )
        if not match:
            raise HTTPException(status_code=422, detail="ArXiv URLs must identify one paper")
        # Feed the graph a canonical ID so the router always constructs the
        # known HTTPS PDF endpoint instead of downloading an arbitrary URL or
        # an ArXiv abstract HTML page.
        return match.group(1)
    if re.fullmatch(r"(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?", text, flags=re.IGNORECASE):
        return text
    # A relative/absolute path, including ``paper.pdf``, must never reach the
    # graph from client input.  PDFs must be uploaded through the UUID path.
    if "/" in text or "\\" in text or text.startswith(".") or Path(text).suffix.lower() == ".pdf":
        raise HTTPException(status_code=422, detail="Local filesystem paths are not accepted as paper inputs")
    raise HTTPException(status_code=422, detail="Paper inputs must be ArXiv IDs or HTTP(S) URLs")


async def _read_pdf_upload(upload: Any) -> tuple[str, str, bytes]:
    filename = Path(str(getattr(upload, "filename", "") or "upload.pdf")).name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF uploads are accepted")
    content_type = str(getattr(upload, "content_type", "") or "").lower()
    content = await upload.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF upload exceeds the 25 MB limit")
    # Check the file signature as well as the extension/MIME type.  Some test
    # clients omit content_type, so the magic bytes are the authoritative check.
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid PDF")
    if content_type and content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=422, detail="Only PDF uploads are accepted")
    return filename, content_type, content


async def _collect_multipart(request: Request) -> tuple[str, list[str], list[str], list[str]]:
    try:
        form = await request.form()
    except Exception:
        logger.exception("Unable to parse multipart form")
        raise HTTPException(
            status_code=400,
            detail="Unable to parse the request form. Check the upload and try again.",
        ) from None

    # Thread messages use ``message`` while the legacy run endpoint uses
    # ``query``.  Both are normalized to the same graph input contract.
    query = str(form.get("query") or form.get("message") or form.get("research_query") or "").strip()
    submitted_inputs: list[Any] = []
    for key in INPUT_KEYS:
        try:
            submitted_inputs.extend(form.getlist(key))
        except AttributeError:
            value = form.get(key)
            if value is not None:
                submitted_inputs.append(value)
    paper_inputs = [_validate_paper_input(item) for item in _parse_form_inputs(submitted_inputs)]
    # The composer promises that a user can start with an ArXiv ID. If an
    # exact ID/URL was entered in the main query field, promote it to a direct
    # paper input so the workflow does not depend on the ArXiv Search API.
    if query and not paper_inputs:
        try:
            promoted_input = _validate_paper_input(query)
        except HTTPException:
            promoted_input = None
        if promoted_input:
            paper_inputs = [promoted_input]
            query = ""
    # Only validated, normalized inputs may cross into the graph.  In
    # particular, a JSON/comma-separated form value must not be forwarded as
    # one opaque string after it has been expanded for the public DTO.
    graph_inputs = list(paper_inputs)

    uploads: list[Any] = []
    for key in UPLOAD_KEYS:
        try:
            uploads.extend(form.getlist(key))
        except AttributeError:
            value = form.get(key)
            if value is not None:
                uploads.append(value)
    # Form fields with the same name can include plain strings; only objects
    # implementing UploadFile's read contract are upload candidates.
    uploads = [item for item in uploads if hasattr(item, "read") and hasattr(item, "filename")]
    pdf_names: list[str] = []
    for upload in uploads:
        filename, _content_type, content = await _read_pdf_upload(upload)
        unique_name = f"{uuid.uuid4().hex}.pdf"
        target_dir = Path(config.PDF_CACHE_DIR) / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / unique_name
        target_path.write_bytes(content)
        graph_inputs.append(str(target_path))
        pdf_names.append(filename)

    if not query and not graph_inputs:
        raise HTTPException(status_code=422, detail="Provide a research query, ArXiv input, or PDF upload")
    # Keep the public ArXiv/URL list separate from server-generated upload
    # paths.  The latter are needed by the graph but must never appear in a
    # browser-visible RunInput DTO.
    return query, paper_inputs, graph_inputs, pdf_names


def _default_graph_factory() -> Any:
    return build_research_graph()


def _sse_line(event_type: str, payload: Any, event_id: int | None = None) -> str:
    lines = [f"event: {event_type}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    data = json.dumps(_jsonable(payload), ensure_ascii=False, separators=(",", ":"))
    lines.append(f"data: {data}")
    return "\n".join(lines) + "\n\n"


async def _event_stream(registry_instance: RunRegistry, run_id: str, after: int) -> AsyncIterator[str]:
    """Yield a current snapshot followed by events after the requested seq."""

    try:
        snapshot = registry_instance.snapshot(run_id)
    except KeyError:
        return
    # The snapshot is authoritative and lets a reconnect rebuild the UI even
    # if events were missed. New events continue after its sequence number.
    yield _sse_line("snapshot", snapshot, snapshot.seq)
    # A snapshot already contains the complete event history through
    # ``snapshot.seq``.  Continue only from the next sequence so consumers do
    # not receive duplicate event frames on first connect or reconnect.
    cursor = max(0, int(after), snapshot.seq)
    while True:
        try:
            with registry_instance.lock:
                record = registry_instance.runs.get(run_id)
                if record is None:
                    return
                events = [event for event in record.trace_events if event.seq > cursor]
                terminal = record.status in TERMINAL_STATUSES
            for event in events:
                cursor = max(cursor, event.seq)
                yield _sse_line(event.type, event, event.seq)
            if terminal:
                return
            # An async generator avoids Starlette's threadpool adapter for the
            # sync-generator variant, which can leave ASGI test transports
            # waiting for a disconnect after a finite terminal stream.  The
            # short poll still reacts promptly to background-thread updates.
            await asyncio.sleep(0.1)
        except (GeneratorExit, asyncio.CancelledError):
            return


def _public_config() -> dict[str, Any]:
    provider = resolved_provider(config.DEFAULT_PROVIDER)
    providers: list[dict[str, Any]] = []
    for name in ("gemini", "openai", "openrouter", "anthropic", "zai"):
        status: dict[str, Any] = {
            "provider": name,
            "model": config.DEFAULT_MODEL,
            "configured": provider_is_configured(name),
        }
        if name == "zai":
            status["apiBase"] = config.ZAI_API_BASE
        providers.append(status)
    return {
        "provider": provider,
        "model": config.DEFAULT_MODEL,
        "configured": provider_is_configured(provider),
        "providers": providers,
        "apiBase": config.ZAI_API_BASE if provider == "zai" else None,
    }


def create_app(
    *,
    graph_factory: Callable[[], Any] | Any | None = None,
    registry_instance: RunRegistry | None = None,
) -> FastAPI:
    """Create an API application, allowing tests to inject a fake graph."""

    chosen_registry = RunRegistry(persist_dir=config.THREADS_DIR) if registry_instance is None else registry_instance

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Reload persisted conversations when the server starts.
        chosen_registry.load_threads()
        yield

    application = FastAPI(title="Research Scout API", version="1.0", lifespan=lifespan)
    application.state.graph_factory = _default_graph_factory if graph_factory is None else graph_factory
    application.state.run_registry = chosen_registry

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return _public_config()

    @application.post("/api/threads", response_model=ThreadSnapshot, status_code=201)
    async def post_thread(request: Request) -> ThreadSnapshot:
        """Create an in-memory conversation thread without accepting secrets."""

        title: str | None = None
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if isinstance(body, Mapping):
                title = str(body.get("title") or "").strip() or None
        elif content_type:
            try:
                form = await request.form()
                title = str(form.get("title") or "").strip() or None
            except Exception:
                title = None
        registry_instance: RunRegistry = request.app.state.run_registry
        thread = registry_instance.create_thread(title)
        return registry_instance.thread_snapshot(thread.thread_id)

    @application.get("/api/threads", response_model=list[ThreadSummary])
    async def get_threads() -> list[ThreadSummary]:
        registry_instance: RunRegistry = application.state.run_registry
        return registry_instance.thread_summaries()

    @application.get("/api/threads/{thread_id}", response_model=ThreadSnapshot)
    async def get_thread(thread_id: str) -> ThreadSnapshot:
        registry_instance: RunRegistry = application.state.run_registry
        try:
            return registry_instance.thread_snapshot(thread_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Thread not found") from None

    @application.post("/api/threads/{thread_id}/messages", response_model=RunSnapshot, status_code=202)
    async def post_thread_message(request: Request, thread_id: str) -> RunSnapshot:
        registry_instance: RunRegistry = request.app.state.run_registry
        if registry_instance.get_thread(thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        query, paper_inputs, raw_inputs, pdf_names = await _collect_multipart(request)
        conversation_context = registry_instance.thread_context(thread_id)
        provider = resolved_provider(config.DEFAULT_PROVIDER)
        run_input = RunInput(query=query, paperInputs=paper_inputs, pdfNames=pdf_names, fileNames=pdf_names)
        initial_state = _initial_state(
            query,
            raw_inputs,
            conversation_context=conversation_context,
            followup_context=registry_instance.followup_context(thread_id),
        )
        try:
            record = registry_instance.create(
                run_input,
                provider,
                config.DEFAULT_MODEL,
                initial_state,
                thread_id=thread_id,
                conversation_context=conversation_context,
            )
        except ActiveRunError as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "activeRunId": exc.run_id},
            ) from None
        except KeyError:
            raise HTTPException(status_code=404, detail="Thread not found") from None
        registry_instance.add_user_message(
            thread_id,
            query=query or (paper_inputs[0] if paper_inputs else "Direct paper analysis"),
            run_id=record.run_id,
            attachments=pdf_names,
        )
        worker = threading.Thread(
            target=_execute_run,
            args=(record.run_id, request.app.state.graph_factory, initial_state, registry_instance),
            name=f"research-run-{record.run_id[:8]}",
            daemon=True,
        )
        worker.start()
        return registry_instance.snapshot(record.run_id)

    @application.post("/api/runs", response_model=RunSnapshot, status_code=202)
    async def post_run(request: Request) -> RunSnapshot:
        query, paper_inputs, raw_inputs, pdf_names = await _collect_multipart(request)
        provider = resolved_provider(config.DEFAULT_PROVIDER)
        run_input = RunInput(query=query, paperInputs=paper_inputs, pdfNames=pdf_names, fileNames=pdf_names)
        initial_state = _initial_state(query, raw_inputs)
        registry_instance: RunRegistry = request.app.state.run_registry
        # Legacy /api/runs receives the same thread-aware snapshots as the
        # chat endpoint.  Reserve the active-run gate and create the thread
        # under one lock so a concurrent request cannot leave an orphan thread.
        with registry_instance.lock:
            active = next((item for item in registry_instance.runs.values() if item.status == "running"), None)
            if active is not None:
                raise HTTPException(
                    status_code=409,
                    detail={"message": "A research workflow is already running", "activeRunId": active.run_id},
                )
            title_seed = query or (paper_inputs[0] if paper_inputs else "New research")
            thread = registry_instance.create_thread(title_seed)
            try:
                record = registry_instance.create(
                    run_input,
                    provider,
                    config.DEFAULT_MODEL,
                    initial_state,
                    thread_id=thread.thread_id,
                )
            except ActiveRunError as exc:
                registry_instance.threads.pop(thread.thread_id, None)
                raise HTTPException(
                    status_code=409,
                    detail={"message": str(exc), "activeRunId": exc.run_id},
                ) from None
            registry_instance.add_user_message(
                thread.thread_id,
                query=query or (paper_inputs[0] if paper_inputs else "Direct paper analysis"),
                run_id=record.run_id,
                attachments=pdf_names,
            )
        worker = threading.Thread(
            target=_execute_run,
            args=(record.run_id, request.app.state.graph_factory, initial_state, registry_instance),
            name=f"research-run-{record.run_id[:8]}",
            daemon=True,
        )
        worker.start()
        return registry_instance.snapshot(record.run_id)

    @application.get("/api/runs/{run_id}", response_model=RunSnapshot)
    async def get_run(run_id: str) -> RunSnapshot:
        registry_instance: RunRegistry = application.state.run_registry
        try:
            return registry_instance.snapshot(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None

    @application.get("/api/runs/{run_id}/events")
    async def get_run_events(
        request: Request,
        run_id: str,
        after: int | None = Query(default=None),
        after_seq: int | None = Query(default=None),
    ) -> StreamingResponse:
        registry_instance: RunRegistry = application.state.run_registry
        if registry_instance.get(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        header_cursor = request.headers.get("last-event-id")
        cursor = max(after or 0, after_seq or 0)
        if header_cursor and header_cursor.isdigit():
            cursor = max(cursor, int(header_cursor))
        return StreamingResponse(
            _event_stream(registry_instance, run_id, cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.get("/api/reports", response_model=list[ReportSummary])
    async def get_reports() -> list[ReportSummary]:
        return [_report_summary(path) for path in list_report_files()]

    @application.get("/api/reports/{report_id:path}/download")
    async def download_report(report_id: str) -> Response:
        path = _safe_report_path(report_id)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Report not found")
        try:
            content = _sanitize_report_markdown(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            raise HTTPException(status_code=404, detail="Report not found") from None
        return Response(
            content=content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
        )

    @application.get("/api/reports/{report_id:path}", response_model=ReportDetail)
    async def get_report(report_id: str) -> ReportDetail:
        path = _safe_report_path(report_id)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Report not found")
        try:
            content = _sanitize_report_markdown(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            raise HTTPException(status_code=404, detail="Report not found") from None
        summary = _report_summary(path)
        return ReportDetail(**summary.model_dump(), content=content, markdown=content)

    # Production serving is optional: the React build may not exist while the
    # Vite dev server is running.  API routes are registered first so mounting
    # ``/`` cannot shadow them.
    frontend_dist = config.PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.is_dir():
        resolved_dist = frontend_dist.resolve()

        @application.get("/{frontend_path:path}", include_in_schema=False)
        async def serve_frontend(frontend_path: str) -> FileResponse:
            """Serve assets directly and fall back to index.html for SPA routes."""

            candidate = (resolved_dist / frontend_path).resolve()
            try:
                candidate.relative_to(resolved_dist)
            except ValueError:
                return FileResponse(resolved_dist / "index.html")
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(resolved_dist / "index.html")

    return application


app = create_app(registry_instance=run_registry)


__all__ = [
    "ActiveRunError",
    "PaperDTO",
    "ReportDetail",
    "ReportSummary",
    "RunInput",
    "RunRegistry",
    "RunResult",
    "RunSnapshot",
    "TraceFacts",
    "TraceEvent",
    "app",
    "create_app",
    "registry",
    "run_registry",
]
