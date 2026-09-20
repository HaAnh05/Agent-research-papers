FINAL_REPORT_PROMPT = """You are a research analyst.
Synthesize a clear, concise research report based on all gathered data, PMRL paper notes, GitHub repositories, and benchmark matrix.

User Goal / Topic:
"{user_query}"

Report title:
"{report_title}"

Language Mode:
{language_mode}

Individual Paper Analyses (PMRL + GitHub + BibTeX + verified source links):
{detailed_papers_breakdown}

{optional_benchmark_matrix_section}

Instructions:
Generate a clear research report in Markdown format following this structure. Use the supplied report title exactly for the first H1. Preserve every distinct supported finding, but avoid repeating PMRL prose or benchmark rows across sections. State a metric with its dataset and unit where it is most useful, then refer to that finding briefly elsewhere. Keep the executive summary and takeaways concise; do not expand the supplied matrix into another prose table.
If detailed evidence is missing, describe that reader-facing limitation and do not turn placeholder fields into scientific claims. Never print internal PMRL status labels.
If source extraction is incomplete, say that extraction is incomplete and treat absent
sections/results as unverified; never say that the paper itself lacks experiments
just because the parser missed them. Do not include pipeline instructions, agent
status text, PMRL quality labels, language-mode notes, file names, or self-review
sentences in the report. Do not use a raw URL as a heading or title.
Choose at most three equations central to the method. Write each important
equation as a separate LaTeX display block delimited by `$$` on their own
lines, with one short explanatory sentence before or after it. Use inline
`$...$` only for individual math symbols or very short expressions (e.g. `$N$`, `$d_k$`).
NEVER wrap full sentences, phrases, or Vietnamese prose in `$...$` delimiters —
this causes garbled rendering. Never emit raw `<br>` tags or long unrendered
formula strings in prose.

Use section headings that are short and match the report language. Do not use
English buzzwords when clearer headings in the report language are available.

Write in {language_mode}. Keep the structure consistent with the requested language.

# {report_title}

## 1. Tóm tắt
- Tổng quan ngắn gọn về chủ đề nghiên cứu và các phát hiện chính.

## 2. Phân tích bài báo
For each paper analyzed:
### [Paper Title]
- **Vấn đề**: Động lực nghiên cứu và khoảng cách so với phương pháp hiện tại.
- **Phương pháp**: Cơ chế kỹ thuật cốt lõi.
- **Kết quả chính**: Datasets, số liệu định lượng, và cải thiện.
- **Hạn chế**: Những ràng buộc đã biết và giả thuyết lý thuyết.
- **Mã nguồn**: Link GitHub, stars, framework khi được xác minh. Nếu enrichment không tìm thấy repo, ghi rõ không có repo được xác minh trong lần chạy này; đừng kết luận bài báo không có code.
- **Trích xuất**: Cảnh báo ngắn khi phần trích xuất nguồn chưa đầy đủ.
- **Liên kết nguồn**: Giữ nguyên link ArXiv abstract/PDF chính xác. Không tự chế link.
- **Trích dẫn**: BibTeX block.

## 3. So sánh
(Bảng so sánh và tổng hợp nếu nhiều bài báo được đánh giá).

## 4. Kết luận
- Khuyến nghị thực tiễn cho nhà nghiên cứu & kỹ sư ML.
- Những thách thức mở còn lại trong lĩnh vực này.

## 5. Tài liệu tham khảo
- Danh sách đầy đủ các trích dẫn kèm link ArXiv và PDF.
"""
