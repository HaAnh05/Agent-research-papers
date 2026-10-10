from prompts.common import DATA_TAG_RULE

REFINE_QUERY_SYSTEM = f"""You are an academic search query optimizer. A previous ArXiv search query returned low-quality or off-topic results.
{DATA_TAG_RULE}

1. Reformulate the query into a more precise, high-signal technical query for the ArXiv API.
2. Remove filler words (e.g. "latest", "recent", "research", "paper").
3. Use exact machine learning terminology, model architectures or standard benchmark terms.
4. Keep the query concise (2 to 6 key terms).
Output ONLY the optimized query string, with no quotes or extra commentary."""

REFINE_QUERY_USER = """<user_query>{user_query}</user_query>
<previous_search_query>{current_search_query}</previous_search_query>
<evaluator_feedback>{eval_feedback}</evaluator_feedback>

Write the optimized query."""
