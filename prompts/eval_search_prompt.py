EVAL_SEARCH_PROMPT = """You are a Principal AI Scientist evaluating the relevance and quality of academic search results from ArXiv.

User Research Goal / Query:
"{user_query}"

ArXiv Search Results ({total_results} candidates):
{candidate_papers_text}

Instructions:
1. Assess how well these papers address the user's research goal.
2. Determine if the search quality passes (`passed = True`) or is poor/off-topic (`passed = False`).
3. If passed, select up to {top_k} best paper indices (0-indexed) that provide maximum insight.
4. If failed, provide actionable feedback on why the results are inadequate and what terminology is missing.

Your output must follow the structured JSON schema:
- "passed": boolean (True/False)
- "score": float (0.0 to 1.0)
- "feedback": string (Concise evaluation)
- "selected_indices": list of integers (e.g. [0, 1, 2])
"""
