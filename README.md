# Research Scout

Research Scout là research assistant dạng chat chạy trên một LangGraph. React là giao diện chính; FastAPI quản lý thread trong RAM, SSE trace/answer và thư viện report. CLI (`main.py`) vẫn được giữ. [FEPlan.md](FEPlan.md) ghi các quyết định, tiến độ đã kiểm chứng và acceptance gate còn mở.

## Workflow

```text
query / ArXiv / PDF
        │
        ▼
      Router ───────────── direct paper ────────────────┐
        │ search                                        │
        ▼                                                │
  ArXiv search → relevance evaluation → refine query  ─┐ │
        │                         ▲                    │ │
        └─────────────────────────┴────────────────────┘ │
                                                         ▼
PDF parsing → GitHub + BibTeX → PMRL notes → final report
```

Graph chỉ chạy một workflow tại một thời điểm. Run đang chạy được giữ trong memory của FastAPI; Markdown report hoàn tất được lưu dưới `reports/`.

## Chạy React + FastAPI

Yêu cầu khuyến nghị: Python 3.11/3.12 và Node.js 20+.
Các lệnh sau giả định bạn đang ở thư mục gốc của repository, không phải trong `frontend/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm ci
```

Cấu hình `.env` từ `.env.example`. Ví dụ Z.AI:

```env
ZAI_API_KEY=your_key
ZAI_API_BASE=https://api.z.ai/api/paas/v4/
DEFAULT_PROVIDER=zai
DEFAULT_MODEL=glm-5.3-flash
```

Chạy backend ở terminal thứ nhất:

```bash
source .venv/bin/activate
uvicorn api:app --reload --port 8000
```

Chạy frontend ở terminal thứ hai:

```bash
cd frontend
npm run dev
```

Mở <http://localhost:5173>. Vite proxy `/api` sang `localhost:8000`.

Sau khi build, FastAPI có thể serve cả bundle tại `localhost:8000`:

```bash
cd frontend
npm run build
cd ..
uvicorn api:app --port 8000
```

## UI

Một màn hình chat duy nhất (spec ngắn trong [`DESIGN.md`](DESIGN.md)):

- Landing "Bạn muốn hiểu điều gì?" với một composer nhận topic, ArXiv ID/URL hoặc PDF; khi gửi, composer chuyển xuống thanh chat.
- Mỗi run có một dòng tiến độ mở rộng được: 5 stage với mô tả, facts đo thật và thời gian; ba stage phân tích chạy "song song".
- Câu trả lời dựng từ `result.answerSummary` (các block); có nút Tải `report.md`, Sao chép BibTeX, Bản đồ liên quan.
- Panel bên phải hiện đồ thị bài liên quan (`citationGraph.similarity`) với tab Tương tự / Nền tảng / Kế thừa; từ một bài có thể "Hỏi về bài này" hoặc "Phân tích bài này".
- Xem UI không cần backend: `cd frontend && npm run dev`, mở <http://localhost:5173/?mock=1> (dữ liệu mẫu, chỉ có ở dev server).

Trace chỉ chứa operational metadata và summary đã có trong state; API không trả raw prompt, credential, local PDF path, extracted PDF text hoặc chain-of-thought.

## API

| Endpoint | Mục đích |
| --- | --- |
| `GET /api/config` | Provider/model và configured status công khai |
| `POST /api/threads` | Tạo thread trong RAM |
| `GET /api/threads` và `GET /api/threads/{id}` | Danh sách và nội dung thread trong phiên |
| `POST /api/threads/{id}/messages` | Gửi tin nhắn và bắt đầu LangGraph run |
| `POST /api/runs` | Tạo run multipart; trả `409` nếu đã có run active |
| `GET /api/runs/{id}` | Snapshot hiện tại |
| `GET /api/runs/{id}/events` | SSE snapshot authoritative, rồi tiếp tục từ sequence mới |
| `GET /api/reports` | Danh sách Markdown report |
| `GET /api/reports/{id}` | Nội dung report |
| `GET /api/reports/{id}/download` | Tải report |

Upload chỉ nhận PDF có magic bytes hợp lệ, giới hạn 25 MB, được đổi sang UUID phía server. Client không thể gửi filesystem path làm paper input.

## Provider

Các provider hiện có: `gemini`, `zai`, `openai`, `openrouter`, `anthropic`.

Z.AI dùng `ChatOpenAI` với `ZAI_API_BASE` tường minh. Các call có Pydantic output dùng JSON mode rồi validate locally. Cấu hình cũ `DEFAULT_PROVIDER=openai` + `OPENAI_API_BASE` trỏ tới Z.AI vẫn được nhận diện như một migration alias; cấu hình mới nên dùng `ZAI_*`.

Frontend không nhận hoặc lưu API key.

## CLI

```bash
python main.py --query "Mixture of Experts in Large Language Models"
python main.py --inputs "1706.03762"
python main.py --inputs "1706.03762" "2005.14165"
```

## Kiểm thử

```bash
.venv/bin/python -m pytest -q

cd frontend
npm run typecheck
npm run build
```

Backend tests dùng fake graph để kiểm event order/reconnect, single-run gate, upload validation, DTO masking, report traversal và Z.AI JSON validation.

## Cấu trúc chính

```text
api.py                 FastAPI bridge, registry và SSE
graph.py               LangGraph StateGraph
state.py               ResearchState và Pydantic models
nodes/                 10 workflow nodes
tools/                 ArXiv, PDF, GitHub, BibTeX, LLM adapters
frontend/              React/Vite + TypeScript
main.py                CLI
reports/               Markdown reports đã tạo
tests/                 Backend tests
```
