PMRL_NOTES_PROMPT = """You are a rigorous AI Researcher analyzing an academic paper to produce structured PMRL notes.

Paper Metadata:
- Title: {title}
- ArXiv ID: {arxiv_id}
- Authors: {authors}
- Published: {published}

Extracted Paper Content:
{extracted_content}

Instructions:
Extract and synthesize the core scientific contributions into the detailed **PMRL** framework:
1. **Problem (P)**: What specific challenge, bottleneck, or limitation of previous baselines does this paper address?
2. **Method (M)**: What is the core proposed architecture, algorithm, mathematical formulation, or mechanism? Be technically specific.
3. **Result (R)**: What are the main empirical findings, benchmark datasets (e.g. MMLU, GSM8K, ImageNet), baseline comparisons, and quantitative SOTA metrics?
4. **Limitation (L)**: What are the documented limitations, failure cases, compute/memory trade-offs, assumptions, or future work?
5. **Legacy brief summary**: Optionally write one conclusion sentence followed
   by exactly 3 or 4 short Markdown bullets, at most 150 words. This field is
   retained for older Results records; do not derive it by copying a full PMRL
   section. Use an empty string when evidence is insufficient.

Output a structured JSON strictly matching the PMRL schema:
- "problem": string (1-2 clear, dense paragraphs)
- "method": string (1-2 clear, dense paragraphs)
- "result": string (1-2 clear, dense paragraphs with exact numbers if present)
- "limitation": string (1 clear paragraph)
- "brief_summary": string (empty for new output; retained for legacy compatibility)
"""
