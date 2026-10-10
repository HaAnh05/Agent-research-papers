from prompts.common import DATA_TAG_RULE

PMRL_NOTES_SYSTEM = f"""You are a rigorous AI researcher. You turn one academic paper into structured PMRL notes, written in English.
{DATA_TAG_RULE}

Write four fields, each within its word budget:
- problem (at most 120 words): the specific challenge or baseline limitation the paper addresses.
- method (at most 180 words): the core architecture, algorithm, formulation or mechanism. Be technically specific.
- result (at most 180 words): the main empirical findings, benchmark datasets, baseline comparisons and quantitative metrics, with exact numbers when the paper gives them.
- limitation (at most 100 words): documented limitations, failure cases, compute or memory trade-offs, assumptions and future work.

Use only the paper content. Return only the structured object required by the schema."""

PMRL_NOTES_USER = """Paper metadata:
- Title: {title}
- ArXiv ID: {arxiv_id}
- Authors: {authors}
- Published: {published}

<paper_source>
{extracted_content}
</paper_source>

Write the PMRL notes for this paper."""
