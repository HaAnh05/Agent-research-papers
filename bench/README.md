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
