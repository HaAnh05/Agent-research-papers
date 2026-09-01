from __future__ import annotations

import os
import uuid
import streamlit as st

from config import config
from graph import build_research_graph
from state import ResearchState

from text_utils import compact_message

st.set_page_config(
    page_title="AI Research Paper Scout",
    page_icon="book",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Giao diện gọn, tập trung vào nội dung
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    h1, h2, h3 { font-family: 'Segoe UI', Roboto, sans-serif; font-weight: 600; }
    .stMetric { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }
    .paper-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Research Paper Scout")
st.caption("LangGraph • GitHub enrichment • PMRL notes")

# --- Cấu hình bên trái ---
with st.sidebar:
    st.header("Cấu hình")
    
    provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "openai", "openrouter", "anthropic"],
        index=0,
    )
    
    default_models = {
        "gemini": "gemini-3.1-flash-lite",
        "openai": "gpt-4o-mini",
        "openrouter": "google/gemini-2.5-flash",
        "anthropic": "claude-3-5-sonnet-20241022"
    }
    
    model_name = st.text_input("Tên Model", value=default_models.get(provider, "gemini-3.1-flash-lite"))
    
    with st.expander("API keys"):
        if provider == "gemini":
            k = st.text_input("GEMINI_API_KEY", type="password", value=config.GEMINI_API_KEY)
            if k: config.GEMINI_API_KEY = k
        elif provider == "openai":
            k = st.text_input("OPENAI_API_KEY", type="password", value=config.OPENAI_API_KEY)
            if k: config.OPENAI_API_KEY = k
        elif provider == "openrouter":
            k = st.text_input("OPENROUTER_API_KEY", type="password", value=config.OPENROUTER_API_KEY)
            if k: config.OPENROUTER_API_KEY = k
        elif provider == "anthropic":
            k = st.text_input("ANTHROPIC_API_KEY", type="password", value=config.ANTHROPIC_API_KEY)
            if k: config.ANTHROPIC_API_KEY = k

        gh_token = st.text_input("GITHUB_TOKEN", type="password", value=config.GITHUB_TOKEN)
        if gh_token: config.GITHUB_TOKEN = gh_token

    config.DEFAULT_PROVIDER = provider
    config.DEFAULT_MODEL = model_name

    st.divider()
    st.markdown("### Luồng xử lý")
    st.markdown("""
    1. Router: phân loại ý định và chuẩn hóa query.
    2. Search: chọn bài báo phù hợp từ ArXiv.
    3. Read: trích xuất nội dung từ PDF.
    4. Enrich: tìm GitHub và BibTeX.
    5. PMRL: tóm tắt vấn đề, cách làm, kết quả.
    6. Benchmark: so sánh giữa các bài báo.
    7. Report: xuất báo cáo tổng hợp.
    """)

# --- SESSION STATE INITIALIZATION ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = None

if "last_logs" not in st.session_state:
    st.session_state.last_logs = []

# --- INPUT SECTION ---
col_in1, col_in2 = st.columns([2, 1])

with col_in1:
    query_input = st.text_input(
        "Nhập chủ đề nghiên cứu hoặc câu hỏi học thuật (Tiếng Việt hoặc English):",
        value=st.session_state.get("current_query", ""),
        placeholder="Ví dụ: Tôi muốn tìm 1 vài bài báo unlearning trong AI",
    )

with col_in2:
    direct_links = st.text_input(
        "Hoặc nhập ArXiv IDs / URLs (cách nhau bởi dấu phẩy):",
        placeholder="Ví dụ: 1706.03762, 2005.14165",
    )

uploaded_files = st.file_uploader(
    "Hoặc tải lên file PDF từ máy tính:",
    type=["pdf"],
    accept_multiple_files=True
)

# Start Analysis Button
if st.button("Bắt đầu phân tích", type="primary", use_container_width=True):
    raw_inputs = []
    if direct_links.strip():
        raw_inputs.extend([x.strip() for x in direct_links.split(",") if x.strip()])

    if uploaded_files:
        for uf in uploaded_files:
            save_path = config.PDF_CACHE_DIR / f"upload_{uf.name}"
            save_path.write_bytes(uf.getbuffer())
            raw_inputs.append(str(save_path))

    if not query_input.strip() and not raw_inputs:
        st.warning("Nhập chủ đề hoặc thêm link/file PDF.")
    else:
        st.session_state.current_query = query_input.strip()
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.last_logs = []
        
        initial_state = {
            "user_query": query_input.strip() or "Direct Paper Analysis",
            "raw_inputs": raw_inputs,
            "intent": "search",
            "status": "running",
            "error_message": None,
            "search_query": query_input.strip(),
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
        thread_cfg = {"configurable": {"thread_id": st.session_state.thread_id}}

        log_placeholder = st.empty()
        with st.spinner("Đang chạy quy trình..."):
            try:
                for event in graph.stream(initial_state, thread_cfg, stream_mode="values"):
                    st.session_state.workflow_state = event
                    logs = event.get("trace_logs", [])
                    st.session_state.last_logs = logs
                    if logs:
                        log_placeholder.info(compact_message(logs[-1]))
                st.success("Hoàn tất.")
            except Exception as exc:
                st.error(f"Lỗi: {str(exc)}")

# --- DISPLAY TABS ---
tab_report, tab_benchmark, tab_pmrl, tab_trace = st.tabs([
    "Báo cáo",
    "Benchmark",
    "PMRL & GitHub",
    "Trace logs"
])

state = st.session_state.workflow_state

with tab_report:
    if state and state.get("final_report"):
        st.markdown(state["final_report"])
        st.download_button(
            "Tải báo cáo Markdown",
            data=state["final_report"],
            file_name=f"research_report_{st.session_state.thread_id}.md",
            mime="text/markdown"
        )
    else:
        st.info("Chưa có báo cáo. Nhập chủ đề rồi bấm 'Bắt đầu phân tích'.")

with tab_benchmark:
    if state and state.get("benchmark_matrix"):
        st.markdown(state["benchmark_matrix"])
    else:
        st.info("Ma trận so sánh sẽ tự động được tạo khi có từ 2 bài báo trở lên.")

with tab_pmrl:
    if state and state.get("selected_papers"):
        papers = state["selected_papers"]
        st.subheader(f"Đã phân tích {len(papers)} bài báo:")
        for idx, p in enumerate(papers, 1):
            with st.expander(f"{idx}. {p.title} ({p.arxiv_id or 'Local PDF'})", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Tác giả:** {', '.join(p.authors) if p.authors else 'N/A'}")
                    st.write(f"**Ngày xuất bản:** {p.published or 'N/A'}")
                with col2:
                    if p.pdf_url:
                        st.link_button("Mở PDF", p.pdf_url)

                if p.github_repos:
                    st.markdown("**GitHub liên quan:**")
                    for r in p.github_repos:
                        official_badge = "Official" if r.is_official else "Community"
                        st.markdown(f"- [{r.name}]({r.url}) — ⭐ {r.stars} | `{r.framework}` | *{official_badge}*")

                if p.notes:
                    st.markdown("#### PMRL:")
                    st.info(f"**Problem:**\n\n{p.notes.problem}")
                    st.success(f"**Method:**\n\n{p.notes.method}")
                    st.warning(f"**Result:**\n\n{p.notes.result}")
                    st.error(f"**Limitation:**\n\n{p.notes.limitation}")

                if p.bibtex:
                    with st.expander("Trích dẫn BibTeX"):
                        st.code(p.bibtex, language="bibtex")
    else:
        st.info("Chưa có dữ liệu phân tích PMRL.")

with tab_trace:
    trace_items = state.get("trace_logs", []) if state else st.session_state.last_logs
    if trace_items:
        for log in trace_items:
            msg = compact_message(log)
            if "Lỗi" in msg or "error" in msg.lower():
                st.error(msg)
            elif "hoàn tất" in msg.lower() or "thành công" in msg.lower() or "passed" in msg.lower():
                st.success(msg)
            elif "benchmark" in msg.lower() or "github" in msg.lower() or "pmrl" in msg.lower() or "report" in msg.lower():
                st.info(msg)
            else:
                st.write(msg)
    else:
        st.info("Trace log sẽ hiển thị ở đây khi chạy quy trình.")
