from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


# --- Pydantic Data Models (Structured Data & Validation) ---

class SummaryCards(BaseModel):
    """Compact five-card summary shown before the full PMRL disclosure."""

    tldr: str = Field(default="", description="One to three sentences stating the contribution and conclusion, in 15-80 words.")
    problem: str = Field(default="", description="The specific problem addressed, in 30-80 words.")
    method: str = Field(default="", description="The core mechanism, in 30-80 words.")
    key_results: str = Field(default="", description="One to three source-backed findings, in 30-80 words.")
    why_it_matters: str = Field(default="", description="A concise implication distinct from measured results, in 30-80 words.")


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


class PMRLGeneration(BaseModel):
    """Provider schema for the detailed PMRL call.

    Compact Results cards are intentionally absent: they are generated after
    the report is synthesized, using the report plus source evidence.
    """

    problem: str = Field(description="Core problem, scientific background, challenges, and baseline limitations.")
    method: str = Field(description="Proposed methodology, key architectural designs, algorithms, formulas.")
    result: str = Field(description="Key empirical results, benchmark datasets, evaluation metrics, SOTA gains.")
    limitation: str = Field(description="Limitations, trade-offs, theoretical assumptions, and future work directions.")
    brief_summary: str = Field(
        default="",
        description="Legacy compact conclusion plus 3 or 4 bullets, at most 150 words; empty when unavailable.",
    )
    # Structured evidence generated in the same PMRL call.  ``eval_settings`` and
    # ``setup.hardware`` are checked against the paper source in code
    # (nodes.write_notes.ground_structured_notes) before they are used.
    method_name: str = Field(default="", description="Short name the paper gives its own method (for example a model name); empty if none.")
    insights: List[Insight] = Field(default_factory=list)
    eval_settings: List[EvalSetting] = Field(default_factory=list)
    setup: Setup = Field(default_factory=Setup)


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
    # Result cards shown in the default Results summary.  They are populated
    # by the post-report summary step and remain separate from full PMRL text.
    summary_cards: SummaryCards = Field(default_factory=SummaryCards)
    # Set by local validation after the provider response.  It is metadata for
    # the API and is not a claim generated by the model.
    summary_quality: Literal["complete", "partial", "invalid", "missing", "unknown"] = "unknown"
    # Local validation metadata. The provider supplies only ``summary_cards``;
    # these statuses let API/UI consumers preserve supported cards when a
    # different card is missing or fails grounding.
    summary_card_statuses: Dict[
        str, Literal["complete", "unsupported", "invalid", "missing", "unknown"]
    ] = Field(default_factory=dict)
    # Stable, card-scoped validation reason codes for API/debug consumers.
    # These are produced locally and never contain source prose.
    summary_card_reasons: Dict[str, str] = Field(default_factory=dict)
    # Legacy compact field retained for old runs. New Results cards use
    # ``summary_cards`` and this field is not synthesized from PMRL prose.
    brief_summary: str = Field(
        default="",
        description=(
            "One-sentence conclusion followed by exactly 3 or 4 concise bullet "
            "points, at most 150 words in total."
        ),
    )
    # Structured evidence generated in the same PMRL call.  ``eval_settings`` and
    # ``setup.hardware`` are checked against the paper source in code
    # (nodes.write_notes.ground_structured_notes) before they are used.
    method_name: str = Field(default="", description="Short name the paper gives its own method (for example a model name); empty if none.")
    insights: List[Insight] = Field(default_factory=list)
    eval_settings: List[EvalSetting] = Field(default_factory=list)
    setup: Setup = Field(default_factory=Setup)


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
    intent: Literal["direct_read", "search", "direct_compare"]
    status: Literal["running", "success", "degraded", "error"]
    error_message: Optional[str]

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

    # Output Synthesis
    final_report: Optional[str]              # Comprehensive research report (Markdown)

    # Append-only measured timings.  Each node writes its own key, preserving
    # the existing LangGraph topology while making timing visible to API
    # consumers without recording prompts, paper text, or credentials.
    node_timings: Annotated[Dict[str, Dict[str, Any]], operator.or_]
    
    # Observability & Tracing (Annotated Reducers to append logs without overwriting)
    trace_logs: Annotated[List[str], operator.add]
    error_logs: Annotated[List[str], operator.add]
