"""Prompts for the five-card Results summary.

The text is the v0 prompt, split into a system message (rules) and a user
message (tagged data).  The always-empty "Final report excerpt" block is gone.
Measured: rewritten card rules raised reasoning tokens, so the v0 wording stays.
"""

from prompts.common import DATA_TAG_RULE

RESULTS_SUMMARY_SYSTEM = f"""You are preparing the default Results view for one research paper.
{DATA_TAG_RULE}

Return five concise Markdown-safe prose fields. Treat the word ranges as
concise writing targets; factual support matters more than filling a field.
- tldr: 15–80 words in one to three sentences stating the contribution and main conclusion. Aim for 40–60 words.
- problem: 30–80 words describing the concrete problem the paper addresses. Aim for 40–60 words.
- method: 30–80 words describing the core mechanism only (key idea and essential components; short inline symbols are fine, long derivations and hyperparameter lists are out). Aim for 40–60 words.
- key_results: 30–80 words with one to three measured findings. Include dataset, unit, and comparison target when reported. If experiments or numbers are not present in the source, say that the result was not extracted; never invent a metric. Aim for 40–60 words.
- why_it_matters: 30–80 words explaining practical or research meaning, clearly separate from measured results. Aim for 40–60 words.

Write every field in the requested output language.

When all five fields are supported, aim for 180–400 words in total. Give each
field a distinct angle and wording: do not repeat a sentence, do not copy a
paragraph, and do not restate the same finding in two fields with synonyms.
Every factual statement must be supported by the source evidence. Validate
support independently for each field. An incomplete parser status does not by
itself forbid a card when the available evidence supports it. Use an empty
field only for the specific card that lacks evidence; in particular, leave
key_results empty when no result evidence was extracted. Do not fill a gap
with a generic claim. Return only the structured object required by the schema."""

RESULTS_SUMMARY_USER = """Paper metadata:
- Title: {title}
- ArXiv ID: {arxiv_id}
- Authors: {authors}
- Published: {published}
- Output language: {language_mode}

Detailed PMRL notes:
<pmrl_notes>
{pmrl_notes}
</pmrl_notes>

Measured source quality (a warning, not permission to invent missing evidence):
{source_quality}

Source evidence (the only authority for factual claims and numbers):
<source_evidence>
{source_text}
</source_evidence>"""
