# Research Scout — UI spec (chat UI, v2)

Supersedes the DESIGN.md v1 (Landing → Pipeline → Results workspace). Reference mockup: `docs/ui-mockup/research-scout-v3.html` (desktop only). Plan: `docs/plans/chat-ui-backend.md`.

## Screen

One chat screen. Landing and chat are two states of the same layout; no router.

- **Landing**: headline "Bạn muốn hiểu điều gì?", big composer (textarea, attach-PDF button, send), chips "Phân tích một bài" / "Tìm theo chủ đề" that fill the composer.
- **Send**: the composer animates (FLIP) into a bottom bar; the headline fades out. User messages are right-aligned bubbles.
- **Turn** = user bubble + progress line + answer.
- **Follow-up chips** sit above the composer after a result. The composer stays for follow-ups.
- **Right panel**: related-papers graph. Opens by itself when a run delivers a graph; closable; reopened by "Bản đồ liên quan" (top bar or answer).

## Tokens

Background `#FAFAF8`, surface `#FFFFFF`, ink `#1B1B19`, accent `#3D63B5`, ok `#4F7F62`, "paper không nêu" `#9A5531`. Newsreader for the headline and the answer prose; Inter for all UI text. All tokens are CSS variables in `frontend/src/styles.css`.

## Progress line

Collapsed: pulse dot + current stage names + elapsed time + chevron ("Ghi chú có cấu trúc… đang chạy 1:23" while running; "Đã phân tích bài trong 3 phút 33 giây" when done). Expanded: one row per step with label, description, measured facts, an LLM line (see below) and time (`m:ss`). Stages 2–5 carry `parallelGroup` and show the tag "song song". The six stages: Đọc toàn văn, Ghi chú có cấu trúc, Tóm tắt nhanh, Dựng bản đồ liên quan, Tìm mã nguồn và BibTeX, Viết báo cáo đầy đủ.

**Live timers.** A running step counts up every second from the client time of its `step.started` event; the header counts from the first event of the run. When `step.completed` arrives the row shows the server `durationMs`. One interval per progress line; it stops when the run ends.

**LLM line.** When a step reports `facts.llmCalls`, the row has a line with commas, for example "LLM 1:12, 9.4k in, 3.1k out, 9.4k reasoning". Several parallel calls show "2 lệnh LLM, dài nhất 1:12" and the token sums. A total is shown only when every call reported that number; a missing value is left out. Steps without `stage` (search, eval, refine, and router only for topic runs) are listed first; the router row is hidden for direct paper runs. Waiting stages are shown until their events arrive. A run with an `answer_followup` step (router intent `followup`) is a quick answer: no progress line, only "Trả lời từ báo cáo đã có, N giây", the streamed text, and "Nguồn: …" from that step's `facts.sources`.

## Answer (from `result.answerSummary`)

Header (title, arXiv id, date, repository), `lead`, then `blocks` in order. Unknown block types are ignored; missing fields are hidden.

**Early summary.** The answer renders as soon as the `answer.summary` SSE event arrives (stage "Tóm tắt nhanh" done), while the progress line keeps running for "Viết báo cáo đầy đủ" (background). The early summary has no `foundations` block, `actions.graphNodes` = 0 and `actions.reportId` = null. When the graph or BibTeX arrives later, the API sends `answer.summary` again with the patched summary and `citationGraph`: the "Nền tảng" block appears, the graph panel opens and "Bản đồ liên quan" shows up. `run.completed` carries the final summary and `reportReady: true`. After a reload, `GET /api/runs/{id}` returns the same partial result while the run goes on.

**Report readiness.** `reportReady` is false until `final_report` ends. Until then "Tải `report.md`" is grey, not clickable (`aria-disabled`), tooltip "Báo cáo đầy đủ đang được viết…". After `reportReady` it is the normal black button. "Sao chép BibTeX" is enabled as soon as `result.bibtex` exists. Follow-up chips are disabled and the composer shows "Đang viết báo cáo đầy đủ, hỏi tiếp được khi xong…" until `reportReady` (no queue: the API allows one run at a time).

| Block | Heading / rendering |
| --- | --- |
| `problem`, `intuition` | "Vấn đề", "Ý tưởng cốt lõi" — paragraph |
| `keyResults`, `results` | "Kết quả chính" — bullets, then table; "Mọi số khớp nguồn" pill + "Nguồn: Table N" |
| `metrics` | "Đại lượng đo" — definition list |
| `setup` | "Thiết lập" — text + "<missing>: paper không nêu" chips |
| `caveats`, `foundations` | "Cần lưu ý", "Nền tảng" |
| `takeaway` | "Nên dùng khi nào" |
| `insights` (multi) | "Insight chung" |
| `paperCards` (multi) | expandable cards: title, year, intuition; expanded: key result, links |

Actions: Tải `report.md` (`GET /api/reports/{id}/download`), Sao chép BibTeX (`result.bibtex`), Bản đồ liên quan; duration on the right.

## Graph panel

Force layout (seeded, 500 iterations) with a label-collision layer; node radius = citations, colour = year, origin = black. Tabs Tương tự / Nền tảng / Kế thừa. A node opens a detail card with the abstract and two buttons: "Hỏi về bài này" (fills the composer) and "Phân tích bài này" (sends a message with form field `arxivId` in the same thread; hidden when the node has no arXiv id).

## Code map

`frontend/src/` — `App.tsx` (layout, chat state), `lib/useChat.ts` (thread, send, SSE), `lib/run.ts` (events → view model, normalizers), `lib/useNow.ts` (1 s clock for live timers), `lib/format.ts` (facts and LLM lines), `lib/api.ts` (HTTP/SSE transport), `components/` (`Composer`, `ProgressLine`, `Answer`, `GraphPanel`), `types.ts` (all API types), `dev/` (mock mode, `?mock=1` on the dev server only).

## Non-goals

Mobile layout, thread list/library, dark mode, any UI for data the API does not return.
