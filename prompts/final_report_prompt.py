FINAL_REPORT_PROMPT = """You are an Academic Director and Senior AI Researcher.
Synthesize the comprehensive scientific research report based on all gathered data, PMRL paper notes, GitHub repositories, and benchmark matrix.

User Goal / Topic:
"{user_query}"

Language Mode:
{language_mode}

Individual Paper Analyses (PMRL + GitHub + BibTeX):
{detailed_papers_breakdown}

{optional_benchmark_matrix_section}

Instructions:
Generate a publication-grade, thorough **Comprehensive Research Report** in Markdown format following this structure.

Write in {language_mode}. Keep the structure and academic quality consistent with the requested language.

# Comprehensive Research Report: {user_query}

## 1. Executive Summary
- Synthesis of the research landscape and state-of-the-art developments regarding the topic.
- High-level takeaways from the analyzed paper(s).

## 2. In-Depth Paper Breakdowns (PMRL Analysis)
For each paper analyzed:
### [Paper Title] (ArXiv: [ID])
- **Problem**: Core motivation and existing baseline gaps.
- **Method & Architecture**: Detailed breakdown of the technical mechanism.
- **Key Empirical Results**: Datasets, quantitative metrics, and improvements.
- **Limitations & Future Work**: Known constraints and theoretical assumptions.
- **Code & Reproducibility**: GitHub links, stars, and framework info.
- **Citation**: Formatted BibTeX block.

## 3. Comparative Benchmark Analysis
(Include the comparison matrix table and trade-off synthesis if multiple papers were evaluated).

## 4. Key Takeaways & Strategic Recommendations
- Actionable architectural recommendations for researchers & ML engineers.
- What open challenges remain in this field.

## 5. Bibliography & References
- Complete list of citations with ArXiv and PDF links.
"""
