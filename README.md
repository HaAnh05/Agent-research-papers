# Research Scout

Research Scout là research assistant dạng chat chạy trên một LangGraph. React là giao diện chính; FastAPI quản lý thread trong RAM, SSE trace/answer và thư viện report. Streamlit và CLI vẫn được giữ làm fallback. [FEPlan.md](FEPlan.md) ghi các quyết định, tiến độ đã kiểm chứng và acceptance gate còn mở.

## Workflow

```text
query / ArXiv / PDF
        │
        ▼
      Router ─── direct answer → END
        ├────────────────── direct paper ────────────────┐
        │ search                                         │
        ▼                                                │
  ArXiv search → relevance evaluation → refine query  ─┐ │
        │                         ▲                    │ │
        └─────────────────────────┴────────────────────┘ │
                                                         ▼
PDF parsing → GitHub + BibTeX → PMRL notes → benchmark → final report
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

- `/` — Landing tối giản với một composer nhận topic, ArXiv ID/URL và PDF hợp lệ.
- `/run/:id` — Pipeline semantic khi run đang chạy; chỉ sau snapshot hoàn tất mới thay bằng PMRL Results.
- `/library` và `/library/:reportId` — route tương thích cũ, redirect về Landing vì UI hiện tại không có dashboard/library shell.

Trace chỉ chứa operational metadata và summary đã có trong state; API không trả raw prompt, credential, local PDF path, extracted PDF text hoặc chain-of-thought.

Pipeline dùng Task Rows, Activity disclosure và factual count chips lấy từ trace thật; không hiển thị Thinking/agent shell hay progress giả. Bản đồ các primitive được dùng nằm trong [`docs/beautiful-ui-component-map.md`](docs/beautiful-ui-component-map.md); copyright notice và license nằm trong `frontend/NOTICE`.

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

## Streamlit fallback và CLI

```bash
streamlit run app.py

python main.py --query "Mixture of Experts in Large Language Models"
python main.py --inputs "1706.03762"
python main.py --inputs "1706.03762" "2005.14165"
```

## Kiểm thử

```bash
.venv/bin/python -m pytest -q

cd frontend
npm run typecheck
npm test
npm run build
npm run test:e2e
```

Backend tests dùng fake graph để kiểm event order/reconnect, single-run gate, upload validation, DTO masking, report traversal và Z.AI JSON validation. Frontend tests kiểm navigation, composer validation, SSE reconciliation, trace, Markdown/GFM/KaTeX, result states và mocked desktop/mobile workflow.

## Cấu trúc chính

```text
api.py                 FastAPI bridge, registry và SSE
graph.py               LangGraph StateGraph
state.py               ResearchState và Pydantic models
nodes/                 10 workflow nodes
tools/                 ArXiv, PDF, GitHub, BibTeX, LLM adapters
frontend/              React/Vite + TypeScript
app.py                 Streamlit fallback
main.py                CLI
reports/               Markdown reports đã tạo
tests/                 Backend tests
```
