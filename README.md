# 📚 AI Research Paper Scout & Benchmark Assistant

Backend phân tích, tìm kiếm và đối chiếu bài báo nghiên cứu khoa học xây dựng trên **LangGraph StateGraph**, hỗ trợ bóc tách PDF 2 cột, tìm kiếm mã nguồn GitHub đa tầng, trích xuất cấu trúc PMRL (Problem-Method-Result-Limitation), và tự động lập ma trận so sánh SOTA Benchmark.

---

## 🏗️ Kiến Trúc Hệ Thống (Workflow)

```
[Start] (User Query / ArXiv Links / Local PDFs)
   │
   ▼
[router] ──────────────────────────(CE1: direct_read)──────────────────────────┐
   │ (CE1: search)                                                             │
   ▼                                                                           │
[search_papers] (ArXiv API + Lock & Jitter)                                    │
   │                                                                           │
   ▼                                                                           │
[eval_search] ──(CE2: refine)──► [refine_query] ──► (Loop retry_count++)       │
   │                                                                           │
   ├──(CE2: pass / max_retries)                                                │
   ▼                                                                           │
[human_feedback] (Tùy chọn HITL Breakpoint)                                    │
   │                                                                           │
   ▼                                                                           │
[read_paper] ◄─────────────────────────────────────────────────────────────────┘
   │ (Two-column layout parsing + Section extractor via asyncio.to_thread)
   ▼
[web_enrich] (Multi-tier GitHub Search + Standard BibTeX Generator)
   │
   ▼
[write_notes] (PMRL Generator với Pydantic Validation)
   │
   ├──(CE3: >= 2 papers)──► [compare_benchmark] (Lập bảng ma trận so sánh) ──┐
   │                                                                         │
   └──(CE3: 1 paper)─────────────────────────────────────────────────────────┼──► [final_report] ──► [End]
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Kích hoạt môi trường Conda `langraph`

Môi trường Conda đã được khởi tạo sẵn tại `D:\Lib\miniconda3\envs\langraph`.

```bash
conda activate langraph
```

### 2. Cấu hình API Keys

Tạo file `.env` từ `.env.example` và điền ít nhất 1 LLM API Key (vd: Google Gemini, OpenAI, hoặc OpenRouter):

```env
# Điền ít nhất 1 API key
GEMINI_API_KEY=your_gemini_api_key_here
# hoặc OPENAI_API_KEY=your_openai_key_here

DEFAULT_PROVIDER=gemini
DEFAULT_MODEL=gemini-2.5-flash

# Tùy chọn: Token GitHub để tăng quota tìm kiếm repo (60 -> 5000 req/h)
GITHUB_TOKEN=
```

---

## 💻 Cách Sử Dụng

### 1. Khởi chạy Giao diện Trực quan (Streamlit UI)

```bash
streamlit run app.py
```
👉 Mở trình duyệt tại: `http://localhost:8501`

### 2. Chạy từ Dòng Lệnh (CLI)

- **Tìm kiếm theo chủ đề nghiên cứu (Search Flow):**
  ```bash
  python main.py --query "Mixture of Experts in Large Language Models"
  ```

- **Đọc và phân tích 1 bài báo trực tiếp (Direct Read):**
  ```bash
  python main.py --inputs "https://arxiv.org/abs/1706.03762"
  ```

- **Đối chiếu và lập ma trận so sánh nhiều bài báo (Direct Compare):**
  ```bash
  python main.py --inputs "1706.03762" "2005.14165"
  ```

---

## 🧪 Chạy Bộ Kiểm Thử Tự Động (PyTest)

```bash
pytest tests/ -v
```

Bộ kiểm thử bao gồm:
- `test_tools.py`: Kiểm tra chuẩn hóa Cache Key, sinh BibTeX, bóc tách Regex Section.
- `test_nodes.py`: Kiểm thử độc lập từng Node với mock state.
- `test_graph_flow.py`: Kiểm tra toàn bộ cấu trúc StateGraph, các nhánh điều kiện CE1, CE2, CE3.

---

## 📂 Cấu Trúc Thư Mục

```
d:\LABAI\agentresearch\
├── config.py                 # Cấu hình tập trung (paths, models, limits)
├── state.py                  # Pydantic Models & TypedDict ResearchState
├── graph.py                  # Lắp ráp StateGraph hoàn chỉnh
├── app.py                    # Giao diện Streamlit tương tác
├── main.py                   # Script chạy CLI
│
├── prompts/                  # Quản lý Prompt templates tách biệt
│   ├── router_prompt.py
│   ├── eval_search_prompt.py
│   ├── refine_query_prompt.py
│   ├── pmrl_notes_prompt.py
│   ├── benchmark_prompt.py
│   └── final_report_prompt.py
│
├── tools/                    # Công cụ chuyên biệt
│   ├── arxiv_search.py       # Thread-safe ArXiv API (Lock + Exponential Backoff)
│   ├── pdf_parser.py         # Bóc tách PDF 2 cột + Section extractor
│   ├── github_enricher.py    # Tìm kiếm GitHub đa tầng (Title -> ArXiv ID -> Author)
│   ├── bibtex_generator.py   # Format BibTeX citation
│   ├── cache_manager.py      # Chuẩn hóa Cache key & lưu đệm
│   └── llm_provider.py       # Adapter đa LLM (Gemini, OpenAI, OpenRouter)
│
├── nodes/                    # 10 Nodes độc lập của LangGraph
│   ├── router.py
│   ├── search.py
│   ├── eval_search.py
│   ├── refine_query.py
│   ├── human_feedback.py
│   ├── read_paper.py
│   ├── web_enrich.py
│   ├── write_notes.py
│   ├── compare_benchmark.py
│   ├── final_report.py
│   └── error_handler.py
│
└── tests/                    # Bộ kiểm thử tự động với PyTest
    ├── test_tools.py
    ├── test_nodes.py
    └── test_graph_flow.py
```
