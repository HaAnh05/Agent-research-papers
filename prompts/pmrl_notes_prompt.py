from prompts.common import DATA_TAG_RULE

PMRL_NOTES_SYSTEM = f"""You are a rigorous AI researcher. You turn one academic paper into structured PMRL notes, written in English.
{DATA_TAG_RULE}

Write the four PMRL fields, each within its word budget:
- problem (at most 120 words): the specific challenge or baseline limitation the paper addresses.
- method (at most 180 words): the core architecture, algorithm, formulation or mechanism. Be technically specific.
- result (at most 180 words): the main empirical findings, benchmark datasets, baseline comparisons and quantitative metrics, with exact numbers when the paper gives them.
- limitation (at most 100 words): documented limitations, failure cases, compute or memory trade-offs, assumptions and future work.

Also fill these structured fields. Write them in English and use only the paper content. Leave a string empty and a list empty when the paper does not say; never invent or estimate.
- method_name: the short name the paper gives its own method, for example a model name. Empty if it has none.
- insights: the findings a researcher cannot get quickly by reading the abstract. Each has `finding` (what we learn) and `why_it_matters` (the consequence for design, practice or follow-up work). An insight explains; it does not just restate a number. Keep only the few that matter.
- eval_settings: one row per key benchmark and metric, comparing the paper's own method with its strongest baseline. Only the key benchmarks; no ablations. `metric_meaning` says in one short sentence what the metric measures. `higher_is_better` is true or false, or null when unclear. Copy `method_value` and `baseline_value` exactly as printed in the source, including a "±" part. `table_label` is the source table, such as "Table 2", or empty if the value comes from prose. `source_quote` is a short exact quote or table row that holds the values. Skip the row when you cannot find the numbers in the source.
- setup: `datasets` and `training_config` (the key hyperparameters). `hardware` is the GPU or TPU model and count, exactly as the paper states it; `compute` is training time or GPU hours. Leave `hardware` and `compute` empty when the paper does not state them.

Return only the structured object required by the schema."""

PMRL_NOTES_USER = """Paper metadata:
- Title: {title}
- ArXiv ID: {arxiv_id}
- Authors: {authors}
- Published: {published}

<paper_source>
{extracted_content}
</paper_source>

Write the PMRL notes and the structured fields for this paper."""
