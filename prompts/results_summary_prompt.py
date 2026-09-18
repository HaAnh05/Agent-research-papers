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

Return five concise Markdown-safe prose fields:
- tldr: 15–50 words in one or two sentences stating the contribution and main conclusion.
- problem: 30–60 words describing the concrete problem the paper addresses.
- method: 30–60 words describing the core mechanism. Keep formulas, hyperparameters, and long derivations in the detailed view.
- key_results: 30–60 words with one to three measured findings. Include dataset, unit, and comparison target when reported. If experiments or numbers are not present in the source, say that the result was not extracted; never invent a metric.
- why_it_matters: 30–60 words explaining practical or research meaning, clearly separate from measured results.

Write every field in the requested output language.

The five fields together must contain 180–300 words. Do not repeat a sentence
or copy a paragraph between fields. Every factual statement must be supported
by the source evidence. Use an empty field if the source is too incomplete to
support it; do not fill a gap with a generic claim. Return only the structured
object required by the schema.
"""
