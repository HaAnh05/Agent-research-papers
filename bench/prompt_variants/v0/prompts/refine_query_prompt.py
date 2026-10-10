REFINE_QUERY_PROMPT = """You are an Academic Search Query Optimizer.
The previous search query returned low-quality or off-topic results from ArXiv.

User Original Query:
"{user_query}"

Previous Search Query:
"{current_search_query}"

Evaluator Feedback:
"{eval_feedback}"

Instructions:
1. Reformulate the query into a more precise, high-signal technical query for the ArXiv API.
2. Remove filler words (e.g. "latest", "recent", "research", "paper").
3. Use exact machine learning terminology, model architectures, or standard benchmark terms.
4. Keep the query concise (2 to 6 key terms).

Output ONLY the optimized query string, with no quotes or extra commentary.
"""
