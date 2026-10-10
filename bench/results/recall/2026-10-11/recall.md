# Search recall - 2026-10-11

Gold set: 139 surveys, 14658 gold ids (mean 105.5 per survey). k searched = 100.

| System | Recall@5 | Recall@10 | Recall@20 | Recall@100 | Precision@5 | HitRate@5 | MRR | p50 s | p90 s | queries | errors | max results |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arxiv_api | 0.006 | 0.008 | 0.008 | 0.008 | 0.060 | 0.201 | 0.128 | 0.70 | 0.95 | 139 | 0 | 10 |
| local_fts | 0.016 | 0.025 | 0.039 | 0.105 | 0.160 | 0.482 | 0.326 | 0.55 | 1.03 | 139 | 0 | 99 |

Recall@k = gold ids found in the top k / gold ids. `max results` is the longest result list a system returned: Recall@k for k above it equals Recall at that length. Latency comes from the first (uncached) run.
