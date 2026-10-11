from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


# --- Pydantic Data Models (Structured Data & Validation) ---

class _LenientModel(BaseModel):
    """Base for LLM-produced records: ``null`` falls back to the default and
    bare numbers become strings, so one sloppy field does not discard the record."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_values(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned: Dict[str, Any] = {}
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "higher_is_better":
                value = str(value)
            cleaned[key] = value
        return cleaned


class Insight(_LenientModel):
    """One finding a reader would not get from skimming the abstract."""

    finding: str = Field(default="", description="What we learn from the paper, in one or two sentences. Not a restated number.")
    why_it_matters: str = Field(default="", description="Why a researcher should care: the consequence for design, practice or follow-up work.")


class EvalSetting(_LenientModel):
    """One key benchmark/metric row: the paper's method against its strongest baseline."""

    benchmark: str = Field(default="", description="Dataset, task or benchmark name as the paper writes it.")
    metric: str = Field(default="", description="Metric name as the paper writes it.")
    metric_meaning: str = Field(default="", description="One short sentence: what this metric measures.")
    higher_is_better: Optional[bool] = Field(default=None, description="true if a higher value is better, false if lower is better, null if unclear.")
    unit: str = Field(default="", description="Unit such as %, ms, score; empty if the paper gives none.")
    method_value: str = Field(default="", description="The paper's own method value, copied exactly from the source.")
    baseline_name: str = Field(default="", description="Name of the strongest baseline in the same row or column.")
    baseline_value: str = Field(default="", description="That baseline's value, copied exactly from the source; empty if none.")
    table_label: str = Field(default="", description='Source table label such as "Table 2"; empty if the value comes from prose.')
    source_quote: str = Field(default="", description="A short exact quote or table row from the source that holds the values.")


class Setup(_LenientModel):
    """Experimental setup; every field stays an empty string when the paper does not state it."""

    datasets: str = Field(default="", description="Datasets or benchmarks used for evaluation.")
    training_config: str = Field(default="", description="Key training configuration: optimizer, learning rate, batch size, steps, horizon, and similar.")
    hardware: str = Field(default="", description="Hardware exactly as the paper states it (GPU/TPU model and count). Empty string if not stated. Never guess.")
    compute: str = Field(default="", description="Compute budget such as training time or GPU hours. Empty string if not stated.")


class PMRLNotes(BaseModel):
    """Structured research paper notes following the PMRL framework."""
    problem: str = Field(
        description="Core problem, scientific background, challenges, and baseline limitations."
    )
    method: str = Field(
        description="Proposed methodology, key architectural designs, algorithms, formulas."
    )
    result: str = Field(
        description="Key empirical results, benchmark datasets, evaluation metrics, SOTA gains."
    )
    limitation: str = Field(
        description="Limitations, trade-offs, theoretical assumptions, and future work directions."
    )
    # Structured evidence generated in the same PMRL call.  ``eval_settings`` and
    # ``setup.hardware`` are checked against the paper source in code
    # (nodes.write_notes.ground_structured_notes) before they are used.
    method_name: str = Field(default="", description="Short name the paper gives its own method (for example a model name); empty if none.")
    intuition: str = Field(default="", description="The core idea in plain words and why it should work (English); empty if the paper gives no basis.")
    insights: List[Insight] = Field(default_factory=list)
    eval_settings: List[EvalSetting] = Field(default_factory=list)
    setup: Setup = Field(default_factory=Setup)


# --- Overview call output (final_report): one JSON object in the report language ---

class MetricInfo(_LenientModel):
    name: str = Field(default="", description="Metric name as the paper writes it.")
    meaning: str = Field(default="", description="One short sentence: what the metric measures and its unit.")
    higherIsBetter: Optional[bool] = Field(default=None, description="true if higher is better, false if lower is better, null if unclear.")


class OverviewSingle(_LenientModel):
    """Overview of ONE paper.  Camel-case names are the public JSON keys."""

    tldr: str = Field(default="", description="Two short sentences: the paper's contribution and why it matters.")
    problem: str = Field(default="", description="The problem the paper solves, in one or two short sentences.")
    intuition: str = Field(default="", description="The core idea in plain words and why it should work, in one or two short sentences.")
    keyResults: List[str] = Field(default_factory=list, description="Two to four short bullets; each holds one result with its number, dataset and baseline.")
    metrics: List[MetricInfo] = Field(default_factory=list, description="Each metric the paper uses for its key results.")
    setup: List[str] = Field(default_factory=list, description="Short bullets: datasets, key training settings, and one hardware bullet.")
    caveats: List[str] = Field(default_factory=list, description="Short bullets: limits of the evidence.")
    takeaway: str = Field(default="", description="One or two short sentences: what a researcher should do with this paper.")


class OverviewFinding(_LenientModel):
    finding: str = Field(default="", description="One finding that holds across the papers.")
    why: str = Field(default="", description="Why it matters, with the papers and numbers that support it.")


class OverviewPaper(_LenientModel):
    paperId: str = Field(default="", description="The paper_id from the context, copied exactly.")
    intuition: str = Field(default="", description="The paper's core idea in one short sentence.")
    keyResult: str = Field(default="", description="The paper's main result in one short sentence, with its number.")


class OverviewMulti(_LenientModel):
    """Overview of SEVERAL papers."""

    tldr: str = Field(default="", description="Two short sentences: what the papers share and where the field is going.")
    insights: List[OverviewFinding] = Field(default_factory=list, description="Two to four insights that hold across the papers.")
    papers: List[OverviewPaper] = Field(default_factory=list, description="One entry per paper, in the context order.")
    takeaway: str = Field(default="", description="One or two short sentences: what a researcher should do next.")


class GitHubRepoInfo(BaseModel):
    """Metadata for an open-source repository matching a paper."""
    name: str
    url: str
    stars: int = 0
    framework: str = "Python"
    is_official: bool = False
    description: str = ""


class PaperTable(BaseModel):
    """One table parsed from a paper's LaTeXML HTML (header row first)."""

    caption: str = ""
    label: str = ""                        # e.g. "Table 2"
    rows: List[List[str]] = Field(default_factory=list)
    section: str = ""                      # heading the table appears under


class PaperReference(BaseModel):
    """One bibliography entry parsed from a paper's LaTeXML HTML."""

    index: int = 0                         # 1-based position in the bibliography
    text: str = ""
    title: str = ""                        # best effort; may be empty
    arxiv_id: Optional[str] = None         # no version suffix
    doi: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class RepoLink(BaseModel):
    """A code or project-page link found in a paper's HTML."""

    url: str
    kind: Literal["github", "project_page", "other"] = "other"
    context: str = ""                      # at most 200 chars of surrounding text
    in_bibliography: bool = False
    is_primary_candidate: bool = False


class PaperItem(BaseModel):
    """Data model representing an academic paper across the lifecycle."""
    paper_id: str                          # Canonical key (e.g. arxiv_2312.00752 or local_hash)
    title: str
    summary: str = ""
    authors: List[str] = Field(default_factory=list)
    published: str = ""
    subjects: List[str] = Field(default_factory=list)
    source_type: Literal["arxiv", "local_pdf"] = "arxiv"
    arxiv_id: Optional[str] = None
    arxiv_abs_url: Optional[str] = None
    arxiv_html_url: Optional[str] = None
    pdf_url: Optional[str] = None
    local_pdf_path: Optional[str] = None
    # Metadata comes from the versioned ArXiv API record.  ``unknown`` means
    # no trustworthy record was available; ``missing`` means a record was
    # returned but one or more display fields were absent.
    metadata_status: Literal["complete", "missing", "unknown"] = "unknown"
    
    # Text Extraction
    extracted_text: Optional[str] = None    # Raw full text excerpts
    sections: Dict[str, str] = Field(default_factory=dict) # {"abstract": "...", "methodology": "...", "experiments": "...", "limitations": "..."}
    # Structured HTML extraction (empty for PDF-only papers and for cached
    # extractions that predate these fields).
    tables: List[PaperTable] = Field(default_factory=list)
    references: List[PaperReference] = Field(default_factory=list)
    repo_links: List[RepoLink] = Field(default_factory=list)
    
    # Enrichment
    github_repos: List[GitHubRepoInfo] = Field(default_factory=list)
    bibtex: Optional[str] = None
    
    # PMRL Notes
    notes: Optional[PMRLNotes] = None
    # Measured source and extraction quality.  These values are safe metadata;
    # raw extracted text and section bodies remain internal workflow state.
    source_quality: Dict[str, Any] = Field(default_factory=dict)
    notes_quality: Literal["complete", "invalid", "missing", "unknown"] = "unknown"
    # Internal timing metadata collected by tools/nodes.  API DTOs should
    # expose only allow-listed numeric measurements from this map.
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalSearchOutput(BaseModel):
    """Structured evaluation output for search quality assessment."""
    passed: bool = Field(description="True if papers are highly relevant to query; False otherwise.")
    score: float = Field(description="Relevance score from 0.0 to 1.0.")
    feedback: str = Field(description="Critique explaining why query passed or needs refinement.")
    selected_indices: List[int] = Field(
        default_factory=list,
        description="0-based indices of the best papers to select for deep reading."
    )


# --- Global LangGraph State (TypedDict with Reducers) ---

class ResearchState(TypedDict):
    """Global state flowing through the LangGraph workflow."""
    # Inputs
    user_query: str                          # Research topic or question
    raw_inputs: List[str]                    # URLs, ArXiv IDs, or local PDF paths
    
    # Router & Execution Flow
    intent: Literal["direct_read", "search", "followup"]
    status: Literal["running", "success", "degraded", "error"]
    error_message: Optional[str]

    # Structured context of the thread's latest finished report, supplied by the
    # API (see api._followup_context_from_state).  Present only when the thread
    # has a report; the ``followup`` intent needs it.  ``answer_followup`` writes
    # ``followup_answer`` and ``followup_sources``.
    followup_context: Optional[Dict[str, Any]]
    followup_answer: Optional[str]
    followup_sources: List[str]

    # Compact context supplied by the API to help optimize follow-up searches.
    # It remains separate from internal paper text/cache fields.
    conversation_context: str
    
    # Search & Evaluation
    search_query: str
    search_results: List[PaperItem]
    retry_count: int
    eval_passed: bool
    eval_feedback: str
    
    # Selected Papers for Deep Reading
    selected_papers: List[PaperItem]
    
    # Citation graph around the selected papers (nodes/citation_graph.py):
    # {"nodes": [...], "edges": [...], "stats": {...}}; None before that node ran.
    citation_graph: Optional[Dict[str, Any]]

    # Per-paper partial results of the parallel branches that follow
    # ``read_paper``.  Each branch writes only its own key (merged by paper id
    # with ``operator.or_``), never ``selected_papers``; ``final_report`` joins
    # them into the papers (nodes.final_report.merge_branch_results).
    #   paper_notes[pid]      = {"notes", "notes_quality", "processing_metadata"}   (write_notes)
    #   paper_enrichment[pid] = {"github_repos", "bibtex", "processing_metadata"}   (web_enrich)
    paper_notes: Annotated[Dict[str, Dict[str, Any]], operator.or_]
    paper_enrichment: Annotated[Dict[str, Dict[str, Any]], operator.or_]

    # Output Synthesis
    final_report: Optional[str]
    report_id: Optional[str]                 # file name of the saved report (set by final_report)
    # Structured chat answer built by code from the overview JSON and grounded
    # notes (nodes.final_report.build_answer_summary); None when not built.
    answer_summary: Optional[Dict[str, Any]]              # Comprehensive research report (Markdown)
    # Overview JSON (``OverviewSingle`` / ``OverviewMulti`` dump) made by
    # ``summarize``: ``None`` before that node ran, ``{}`` when its call failed.
    # ``summarize`` writes ``overview`` and ``answer_summary`` first;
    # ``final_report`` reads the overview and patches ``answer_summary`` later
    # (the two nodes never run in the same superstep).
    overview: Optional[Dict[str, Any]]

    # Append-only measured timings.  Each node writes its own key, preserving
    # the existing LangGraph topology while making timing visible to API
    # consumers without recording prompts, paper text, or credentials.
    node_timings: Annotated[Dict[str, Dict[str, Any]], operator.or_]
    
    # Observability & Tracing (Annotated Reducers to append logs without overwriting)
    trace_logs: Annotated[List[str], operator.add]
    error_logs: Annotated[List[str], operator.add]
