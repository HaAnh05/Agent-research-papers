"""Prompt for the post-report five-card Results summary."""

RESULTS_SUMMARY_PROMPT = """You are preparing the default Results view for one research paper.

Paper metadata:
- Title: {title}
- ArXiv ID: {arxiv_id}
- Authors: {authors}
- Published: {published}
- Output language: {language_mode}

Detailed PMRL notes:
{pmrl_notes}

Final report excerpt:
{final_report}

Source evidence (the only authority for factual claims and numbers):
{source_text}

Return five concise Markdown-safe prose fields. Respect every word budget
exactly — count the words of each field before returning it, and shorten any
field that exceeds its limit. A field over its limit is discarded entirely,
so staying inside the budget matters more than adding one more detail.
- tldr: 15–80 words in one to three sentences stating the contribution and main conclusion. Aim for 40–60 words.
- problem: 30–80 words describing the concrete problem the paper addresses. Aim for 40–60 words.
- method: 30–80 words describing the core mechanism only (key idea and essential components; short inline symbols are fine, long derivations and hyperparameter lists are out). Aim for 40–60 words.
- key_results: 30–80 words with one to three measured findings. Include dataset, unit, and comparison target when reported. If experiments or numbers are not present in the source, say that the result was not extracted; never invent a metric. Aim for 40–60 words.
- why_it_matters: 30–80 words explaining practical or research meaning, clearly separate from measured results. Aim for 40–60 words.

Write every field in the requested output language.

The five fields together must contain 180–400 words. Give each field a
distinct angle and wording: do not repeat a sentence, do not copy a paragraph,
and do not restate the same finding in two fields with synonyms. Every factual
statement must be supported by the source evidence. Use an empty field if the
source is too incomplete to support it; do not fill a gap with a generic
claim. Return only the structured object required by the schema.
"""
