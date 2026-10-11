from prompts.common import DATA_TAG_RULE, STE_RULES

# One streamed call.  Rules live in the system message; the report excerpts,
# notes, tables and the question are tagged data in the user message.
FOLLOWUP_SYSTEM = f"""You are a research assistant. You answer ONE follow-up question about a research report that already exists in this conversation.
{DATA_TAG_RULE}

Rules:
- Answer only from the material in the user message: report sections, structured notes, tables and related papers. Add no outside fact, number or link.
- Write in the language mode given in the user message.
- Cite the source of each claim in square brackets. Use the exact source names listed in <sources>, for example [Báo cáo §2], [Table 17] or [Notes: arxiv_2605.05737].
- If the material does not contain the answer, say so plainly in one sentence. Then suggest running a full analysis of that paper or topic. Do not guess.
- Answer the question that was asked. Put the answer first. Use at most 8 short bullets, or at most 150 words.
- Take every number from the material. Copy it exactly.
- Do not print internal labels, file names or pipeline text.

{STE_RULES}"""

FOLLOWUP_USER = """Language Mode:
{language_mode}

Question:
<question>{question}</question>

Sources you may cite:
<sources>
{sources}
</sources>

Report sections:
<report_sections>
{report_sections}
</report_sections>

Structured notes:
<paper_notes>
{paper_notes}
</paper_notes>

Tables:
<tables>
{tables}
</tables>

Related papers from the research map:
<related_papers>
{related_papers}
</related_papers>

Answer the question now."""
