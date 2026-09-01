from __future__ import annotations

import argparse
import asyncio
import sys
from pprint import pprint

from config import config
from graph import build_research_graph


def main():
    parser = argparse.ArgumentParser(description="AI Research Paper Scout & Benchmark Assistant CLI")
    parser.add_argument("--query", type=str, default="", help="Research topic or scientific question")
    parser.add_argument("--inputs", nargs="*", default=[], help="ArXiv URLs, ArXiv IDs (e.g. 1706.03762), or local PDF file paths")
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider override (gemini, openai, openrouter, anthropic)")
    parser.add_argument("--model", type=str, default=None, help="LLM Model name override")
    args = parser.parse_args()

    if not args.query and not args.inputs:
        print("Usage: python main.py --query 'Diffusion Models in Medical Imaging'")
        print("   or: python main.py --inputs '1706.03762' '2005.14165'")
        sys.exit(1)

    if args.provider:
        config.DEFAULT_PROVIDER = args.provider
    if args.model:
        config.DEFAULT_MODEL = args.model

    print("\n" + "="*70)
    print("🚀 AI Research Paper Assistant - LangGraph Engine")
    print(f"Provider: {config.DEFAULT_PROVIDER} | Model: {config.DEFAULT_MODEL}")
    print(f"Query: '{args.query}' | Raw Inputs: {args.inputs}")
    print("="*70 + "\n")

    initial_state = {
        "user_query": args.query,
        "raw_inputs": args.inputs,
        "intent": "search",
        "status": "running",
        "error_message": None,
        "search_query": args.query,
        "search_results": [],
        "retry_count": 0,
        "eval_passed": False,
        "eval_feedback": "",
        "selected_papers": [],
        "benchmark_matrix": None,
        "final_report": None,
        "trace_logs": [],
        "error_logs": [],
    }

    graph = build_research_graph()
    thread_config = {"configurable": {"thread_id": "cli_run_01"}}

    # Stream graph execution
    for event in graph.stream(initial_state, thread_config, stream_mode="values"):
        logs = event.get("trace_logs", [])
        if logs:
            print(f"⏱️  {logs[-1]}")

    final_state = graph.get_state(thread_config).values

    print("\n" + "="*70)
    print("📑 FINAL RESEARCH REPORT")
    print("="*70 + "\n")
    print(final_state.get("final_report", "No report generated."))
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
