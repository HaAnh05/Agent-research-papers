BENCHMARK_MATRIX_PROMPT = """You are a careful research benchmark analyst.
Return one structured comparison artifact for the papers below. Keep findings
short and preserve source evidence. Do not infer a metric, dataset, unit, or
number that is absent from a paper's PDF source excerpt.

Paper evidence:
{papers_pmrl_summary}

For each paper, return at most 4 concise findings and metric claims. A metric
claim must include the exact metric name, dataset, unit, value, and a short
verbatim source_quote copied from that paper's PDF source excerpt. When any part
is missing, leave the metric claim out.

For rows, use one shared metric, dataset, and unit only when every value has
matching evidence. Put the paper IDs in values and source_quotes. Set
comparable=false with reason beginning "Not directly comparable" whenever the
papers use different datasets, metrics, units, protocols, or unsupported
evidence. Keep synthesis to 3-5 short trade-off statements.

Return JSON matching this shape:
{{
  "papers": [{{
    "paper_id": "...",
    "title": "...",
    "findings": ["..."],
    "metrics": [{{"metric":"...","dataset":"...","unit":"...","value":"...","source_quote":"..."}}]
  }}],
  "rows": [{{
    "metric":"...", "dataset":"...", "unit":"...",
    "values": {{"paper_id":"value"}},
    "source_quotes": {{"paper_id":"quote"}},
    "comparable": false,
    "reason":"Not directly comparable: ..."
  }}],
  "synthesis": ["..."]
}}
"""
