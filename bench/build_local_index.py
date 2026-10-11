"""Build the local arXiv FTS5 index and time queries against it.

    .venv/bin/python bench/build_local_index.py \
        --parquet data/arxiv/metadata.parquet --out data/arxiv/index.sqlite

Prints build wall time, DB size, and p50/p90 query latency for max_results 5 and 100.
No network, no LLM.  ``--bench-only`` skips the build and times an existing index.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.local_index import build_index, search_local  # noqa: E402

BENCH_QUERIES = [
    "diffusion policy robot manipulation",
    "retrieval augmented generation survey",
    "graph neural networks molecules",
    "large language model reasoning chain of thought",
    "vision transformer image classification",
    "reinforcement learning from human feedback",
    "state space models long sequence Mamba",
    "mixture of experts language model scaling",
    "federated learning privacy differential",
    "self-supervised contrastive representation learning",
    "neural machine translation low resource languages",
    "speech recognition end to end",
    "object detection real time",
    "point cloud 3D semantic segmentation",
    "knowledge graph embedding link prediction",
    "multi-agent reinforcement learning cooperation",
    "text to image generation latent diffusion",
    "code generation large language models benchmark",
    "hallucination detection large language models",
    "parameter efficient fine-tuning LoRA",
    "quantization of neural networks",
    "adversarial robustness deep networks",
    "recommendation systems collaborative filtering",
    "time series forecasting transformer",
    "medical image segmentation U-Net",
    "autonomous driving trajectory prediction",
    "survey of agents based on large language models",
    "instruction tuning data quality",
    "in-context learning theory",
    "graph transformer",
    "protein structure prediction",
    "causal inference observational data",
    "Bayesian optimization hyperparameter",
    "quantum error correction surface code",
    "dark matter direct detection",
    "convex optimization stochastic gradient descent convergence",
    "sparse attention efficient transformer long context",
    "evaluation of retrieval for question answering",
    "video generation world model",
    "tool use language model agents",
    "C++ template metaprogramming",
    "BERT",
    "attention is all you need",
    "DeepSeek-V2 mixture of experts",
    "survey on hallucination in multimodal models",
    "neural radiance fields novel view synthesis",
    "symbolic regression",
    "explainable AI feature attribution",
    "curriculum learning",
    "benchmark for long-context language models",
]


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def _time_queries(index_path: Path, k: int, queries: list[str]) -> list[float]:
    timings = []
    for query in queries:
        started = time.perf_counter()
        search_local(query, k, index_path=index_path)
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def _report(label: str, timings: list[float]) -> None:
    print(
        f"{label:<28} n={len(timings):<3} p50={statistics.median(timings):8.1f} ms  "
        f"p90={_percentile(timings, 90):8.1f} ms  max={max(timings):8.1f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", default=str(ROOT / "data/arxiv/metadata.parquet"))
    parser.add_argument("--out", default=str(ROOT / "data/arxiv/index.sqlite"))
    parser.add_argument("--categories-prefix", default="cs.", help='keep papers with any category starting with this prefix (default "cs." keeps ~1M papers, ~2.4 GB); pass "" for all 3.1M (~6.2 GB)')
    parser.add_argument("--batch-rows", type=int, default=50_000)
    parser.add_argument("--optimize", action="store_true", help="run FTS5 'optimize' after the build (slower build)")
    parser.add_argument("--bench-only", action="store_true", help="skip the build; time the existing index")
    parser.add_argument("--no-bench", action="store_true", help="build only")
    args = parser.parse_args()

    out = Path(args.out)
    if not args.bench_only:
        wall = time.perf_counter()
        stats = build_index(
            args.parquet, out,
            categories_prefix=args.categories_prefix,
            batch_rows=args.batch_rows,
            optimize=args.optimize,
        )
        print(f"\nBuild wall time: {time.perf_counter() - wall:,.1f} s ({stats['papers']:,} papers)")
    print(f"DB size: {out.stat().st_size / 1e9:.2f} GB ({out})")
    if args.no_bench:
        return 0

    print(f"\nQuery latency over {len(BENCH_QUERIES)} queries")
    _report("k=5   first pass (cold)", _time_queries(out, 5, BENCH_QUERIES))
    _report("k=5   second pass (warm)", _time_queries(out, 5, BENCH_QUERIES))
    _report("k=100 first pass", _time_queries(out, 100, BENCH_QUERIES))
    _report("k=100 second pass (warm)", _time_queries(out, 100, BENCH_QUERIES))

    print("\nTop-5 titles")
    for query in BENCH_QUERIES[:3]:
        print(f"- {query}")
        for paper in search_local(query, 5, index_path=out):
            print(f"    {paper.arxiv_id}  {paper.title[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
