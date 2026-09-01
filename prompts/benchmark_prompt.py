BENCHMARK_MATRIX_PROMPT = """You are a Principal AI Benchmarking Specialist.
Construct a comprehensive, rigorous **Benchmark Comparison Matrix** comparing the following research papers.

Papers Summary & PMRL Notes:
{papers_pmrl_summary}

Instructions:
1. Build a detailed, beautifully formatted **Markdown Comparison Matrix Table** with columns:
   - | Dimension / Feature | Paper 1 ({paper_1_name}) | Paper 2 ({paper_2_name}) | ... |
   Include rows for:
   - **Core Architecture / Novel Mechanism**
   - **Parameter Scale / Compute Efficiency**
   - **Primary Benchmark Datasets**
   - **Key Quantitative SOTA Metrics (Accuracy / Speed / Loss)**
   - **Key Strengths (Pros)**
   - **Key Limitations / Bottlenecks (Cons)**
   - **Open Source Code / GitHub Repo**

2. Write a concise **Cross-Paper Trade-off & Synthesis Analysis**:
   - Compare computational complexity vs performance gains.
   - Practical recommendations: Under which specific engineering/research scenario should a practitioner choose Paper A over Paper B?

Output clean Markdown content ready to be embedded directly into the research report.
"""
