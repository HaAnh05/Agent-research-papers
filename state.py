from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --- Pydantic Data Models (Structured Data & Validation) ---

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


class GitHubRepoInfo(BaseModel):
    """Metadata for an open-source repository matching a paper."""
    name: str
    url: str
    stars: int = 0
    framework: str = "Python"
    is_official: bool = False
    description: str = ""


class PaperItem(BaseModel):
    """Data model representing an academic paper across the lifecycle."""
    paper_id: str                          # Canonical key (e.g. arxiv_2312.00752 or local_hash)
    title: str
    summary: str = ""
    authors: List[str] = Field(default_factory=list)
    published: str = ""
    source_type: Literal["arxiv", "local_pdf"] = "arxiv"
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    local_pdf_path: Optional[str] = None
    
    # Text Extraction
    extracted_text: Optional[str] = None    # Raw full text excerpts
    sections: Dict[str, str] = Field(default_factory=dict) # {"abstract": "...", "methodology": "...", "experiments": "...", "limitations": "..."}
    
    # Enrichment
    github_repos: List[GitHubRepoInfo] = Field(default_factory=list)
    bibtex: Optional[str] = None
    
    # PMRL Notes
    notes: Optional[PMRLNotes] = None


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
    
    # Search & Evaluation
    search_query: str
    search_results: List[PaperItem]
    retry_count: int
    eval_passed: bool
    eval_feedback: str
    
    # Selected Papers for Deep Reading
    selected_papers: List[PaperItem]
    
    # Output Synthesis
    benchmark_matrix: Optional[str]          # Markdown benchmark comparison table
    final_report: Optional[str]              # Comprehensive research report (Markdown)
    
    # Observability & Tracing (Annotated Reducers to append logs without overwriting)
    trace_logs: Annotated[List[str], operator.add]
    error_logs: Annotated[List[str], operator.add]
