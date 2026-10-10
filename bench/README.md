# bench: latency and quality harness

The harness treats the graph as a black box. It works with any version of `nodes/`, `tools/` and `prompts/`.

## Run one scenario

```bash
.venv/bin/python bench/run_scenario.py \
  --scenario bench/baselines/direct_compare_3papers_20261010/run.json \
  --label after_parallel
# or: --query "..." --input https://arxiv.org/abs/2303.04137 --input ...
```

- Output dir: `bench/runs/<timestamp>_<label>/` (override with `--out`; `bench/runs/` is git-ignored).
- `--fresh-llm-cache` (default on, `--no-fresh-llm-cache` to disable) points `config.SUMMARY_CACHE_DIR` at a temp dir, so PMRL and report caches miss. PDF, text and GitHub caches stay warm.
- `--dry-run` uses a stub graph (no network, no LLM) to test the plumbing.
- This calls the real provider. A cold run took about 15 minutes in the baseline.

Files written: `run.json` (same schema as the baseline: `total_seconds`, `status`, `node_wall_seconds`, `node_timings`, `llm_calls` with `start_s/seconds/prompt_chars/thread/node`, `http_calls`), `state.json` (paper `extracted_text`/`sections` replaced by `extracted_text_chars`/`sections_chars`), `report.md`, `source_text/<paper_id>.json`, `cache/` (LLM cache files of the run).

## Compare against a baseline

```bash
.venv/bin/python bench/verify.py \
  --baseline bench/baselines/direct_compare_3papers_20261010 \
  --candidate bench/runs/<run_dir>
```

Prints a table and writes `verify.json` (`<candidate>/verify.json`; `bench/runs/<name>_verify.json` when the candidate is a baseline layout). Exit code 1 on any HARD failure.

Flags: `--strict` (QUALITY warnings fail), `--strict-latency` (a missed target fails), `--target-seconds 60`, `--word-ratio LO HI` (0.6 1.6), `--report-ratio LO HI` (0.6 1.6), `--grounding-drop 0.05`, `--max-equations 3`, `--out PATH`.

## Checks

HARD (always fail the run):

| Check | Meaning |
|---|---|
| `status_not_error` | final state `status` is not `error` |
| `same_paper_ids` | candidate has the same paper id set as the baseline |
| `pmrl_notes_nonempty` | every paper has non-empty problem/method/result/limitation |
| `report_structure` | one H1, headings `## 1.` to `## 5.` in order, at least one `###` per paper in section 2 (code fences ignored) |
| `arxiv_links` | `https://arxiv.org/abs/<id>` present for each arXiv paper |
| `no_internal_lines` | no internal labels (PMRL status, language mode, `report_*.md`, `summary_quality`, `pipeline:` labels) |

QUALITY (WARN by default, FAIL under `--strict`):

| Check | Meaning |
|---|---|
| `summary_quality_rank` | per paper, complete > partial > other; not worse than baseline |
| `complete_cards` | total card statuses `complete` not lower than baseline |
| `pmrl_word_counts` | each P/M/R/L field word count within 0.6x-1.6x baseline |
| `report_length` | report characters within 0.6x-1.6x baseline |
| `number_grounding` | share of report numbers found in the union of source texts (`,`/`.` decimals and `%` tolerated; arXiv ids, URLs, code, LaTeX and bare integers below 10 skipped) not below baseline minus 5 points |
| `display_equations` | at most 3 `$$` blocks per paper section |

LATENCY (exit code 1 only with `--strict-latency`): total seconds, per-node wall seconds, longest LLM call, speedup vs baseline, and PASS/FAIL against `--target-seconds`.

## Search recall bench (`bench/recall/`)

Measures how many of a survey's cited papers a search system finds from the survey's topic. Gold set = survey bibliographies. No LLM calls.

Step 1. Build the gold set (needs `data/arxiv/metadata.parquet`; fetches ar5iv HTML once, then reads the cache):

```bash
.venv/bin/python -I bench/recall/build_goldset.py            # 200 cs.AI surveys, seed 42
# offline, local HTML only (survey ids bypass selection):
.venv/bin/python -I bench/recall/build_goldset.py --html-dir data/arxiv/html_probe \
    --ids 2205.09991 2303.04137 --min-refs 1 --out-dir /tmp/recall_dry
```

Flags: `--category`, `--year-from/--year-to` (id prefix), `--title-regex`, `--sample N`, `--seed`, `--min-refs 15`, `--out-dir`, `--metadata`.
Output in `data/arxiv/recall/` (git-ignored): `goldset.jsonl`, `goldset_stats.json`, `html/` (raw HTML cache). The committed `bench/recall/goldset_manifest.json` has survey ids, queries and gold id lists only.
Resolution order per reference: explicit arXiv id, DOI `10.48550/arXiv.<id>`, DOI match in metadata, exact normalized title (4 or more words, unambiguous). Only references older than the survey count; the survey itself is dropped.

Step 2. Run the systems:

```bash
.venv/bin/python -I bench/recall/run_recall.py --systems arxiv_api            # baseline
.venv/bin/python -I bench/recall/run_recall.py --systems arxiv_api,local_fts  # after feat/local-index
.venv/bin/python -I bench/recall/run_recall.py --systems fake --goldset /tmp/recall_dry/goldset.jsonl   # offline plumbing check
```

Each (system, query) result is cached in `data/arxiv/recall/results/`; a rerun costs nothing. Output: `bench/results/recall/<YYYY-MM-DD>/recall.{json,md}` with Recall@5/20/100, Precision@5, HitRate@5, MRR and latency p50/p90.

Runtime estimates:

| Step | Estimate |
|---|---|
| Gold-set build, 200 surveys | about 3 s per ar5iv page, so about 10 min, plus about 1 min for the metadata scans |
| `arxiv_api`, 200 queries | at least 3.5 s per query, so about 12 min |
| `local_fts`, 200 queries | seconds (local index) |

`arxiv_search` returns at most 10 results, so `arxiv_api` Recall@20 and Recall@100 equal Recall@10. The table column `max results` shows this.
