from __future__ import annotations

import time
from typing import Any, Dict
from state import ResearchState


def error_handler_node(state: ResearchState) -> Dict[str, Any]:
    """Gracefully handles exceptions and degraded workflow states without crashing the application."""
    started = time.perf_counter()
    err_msg = state.get("error_message") or "An unexpected error occurred during execution."
    logs = [f"[error_handler] Kích hoạt Fallback Error Handler: {err_msg}"]

    report_fallback = f"""# Research Workflow Execution Notice

**Status:** Degraded / Terminated with Notice
**Reason:** {err_msg}

### Logs & Diagnostics:
- Query: `{state.get('user_query', 'N/A')}`
- Intent: `{state.get('intent', 'N/A')}`
- Retries attempted: `{state.get('retry_count', 0)}`

Vui lòng kiểm tra lại kết nối mạng, API Keys trong file `.env`, hoặc thử lại với từ khóa học thuật chính xác hơn.
"""

    return {
        "status": "error",
        "final_report": report_fallback,
        "node_timings": {"error_handler": {"durationMs": max(0, int((time.perf_counter() - started) * 1000))}},
        "trace_logs": logs
    }
