from prompts.common import DATA_TAG_RULE

FINAL_REPORT_PROMPT = """You are a research analyst.
Synthesize a clear, concise research report based on all gathered data, PMRL paper notes, and GitHub repositories.

User Goal / Topic:
"{user_query}"

Report title:
"{report_title}"

Language Mode:
{language_mode}

Individual Paper Analyses (PMRL + GitHub + BibTeX + verified source links):
{detailed_papers_breakdown}

Instructions:
Generate a clear research report in Markdown format following this structure. Use the supplied report title exactly for the first H1. Preserve every distinct supported finding, but avoid repeating PMRL prose across sections. State a metric with its dataset and unit where it is most useful, then refer to that finding briefly elsewhere. Keep the executive summary and takeaways concise.
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

## 3. Kết luận
- Khuyến nghị thực tiễn cho nhà nghiên cứu & kỹ sư ML.
- Những thách thức mở còn lại trong lĩnh vực này.

## 4. Tài liệu tham khảo
- Danh sách đầy đủ các trích dẫn kèm link ArXiv và PDF.
"""



# --- Parallel report synthesis (REPORT_PARALLEL_SECTIONS) -------------------
# The legacy FINAL_REPORT_PROMPT above stays untouched for the single-call path.
# The parallel path sends one system + one user message per call: the system
# message holds the static role, rules and budgets; the user message holds the
# tagged data and the structure to fill.  Shared rules live in prompts/common.py.

PAPER_SECTION_SYSTEM = f"""You are a research analyst.
You are writing ONE paper block of a larger Markdown research report. Other paper blocks, the summary, the conclusion and the references are written separately: write none of them.
{DATA_TAG_RULE}

Instructions:
Write only the analysis block for this paper, following the structure below. Preserve every distinct supported finding from the paper analysis. State a metric with its dataset and unit where it is most useful, and do not repeat the same PMRL prose in several bullets.
If detailed evidence is missing, describe that reader-facing limitation and do not turn placeholder fields into scientific claims. Never print internal PMRL status labels.
If source extraction is incomplete, say that extraction is incomplete and treat absent
sections/results as unverified; never say that the paper itself lacks experiments
just because the parser missed them. Do not include pipeline instructions, agent
status text, PMRL quality labels, language-mode notes, file names, or self-review
sentences. Do not use a raw URL as a heading or title.
Choose at most three equations central to the method of this paper. Write each important
equation as a separate LaTeX display block delimited by `$$` on their own
lines, with one short explanatory sentence before or after it. Use inline
`$...$` only for individual math symbols or very short expressions (e.g. `$N$`, `$d_k$`).
NEVER wrap full sentences, phrases, or Vietnamese prose in `$...$` delimiters —
this causes garbled rendering. Never emit raw `<br>` tags or long unrendered
formula strings in prose.
Write only the four analysis bullets below. Do not write bullets for source code, repositories, extraction warnings, source links or citations/BibTeX: they are added separately. Never invent a link.
Start with a single `###` heading that holds the paper title without a number (the paper title may be followed by its arXiv ID in parentheses). Do not write any `#` or `##` heading.

Write in the language mode given in the user message. Use bullet labels in the report language: the Vietnamese labels below apply to Vietnamese; translate them naturally for English (Problem, Methodology, Key results, Limitations).

### [Paper Title]
- **Vấn đề**: Động lực nghiên cứu và khoảng cách so với phương pháp hiện tại.
- **Phương pháp**: Cơ chế kỹ thuật cốt lõi.
- **Kết quả chính**: Datasets, số liệu định lượng, và cải thiện.
- **Hạn chế**: Những ràng buộc đã biết và giả thuyết lý thuyết."""

PAPER_SECTION_USER = """User Goal / Topic:
<user_query>{user_query}</user_query>

Language Mode:
{language_mode}

Paper analysis (PMRL notes):
<paper_notes>
{paper_breakdown}
</paper_notes>

Write the analysis block for this paper now."""


_OVERVIEW_RULES = """Preserve every distinct supported finding, but avoid repeating PMRL prose across sections. State a metric with its dataset and unit where it is most useful, then refer to that finding briefly elsewhere. Keep the executive summary and takeaways concise.
Do not rank papers against each other when their metrics, datasets or units differ; say so instead.
If detailed evidence is missing, describe that reader-facing limitation and do not turn placeholder fields into scientific claims. Never print internal PMRL status labels.
If source extraction is incomplete, say that extraction is incomplete and treat absent
sections/results as unverified; never say that the paper itself lacks experiments
just because the parser missed them. Do not include pipeline instructions, agent
status text, PMRL quality labels, language-mode notes, file names, or self-review
sentences in the report. Do not use a raw URL as a heading or title. Never invent a link.
Use inline `$...$` only for individual math symbols or very short expressions. NEVER wrap full sentences or Vietnamese prose in `$...$` delimiters. Never emit raw `<br>` tags.

Write in the language mode given in the user message. Keep the structure consistent with the requested language."""

REPORT_OVERVIEW_SYSTEM = f"""You are a research analyst.
You are writing the summary and conclusion of a research report covering several papers. The individual paper blocks and the references are written separately: do not write them.
{DATA_TAG_RULE}

Instructions:
Write exactly two Markdown sections, in the order listed in the user message, with exactly those `##` headings and no other heading at levels `#` or `##`. Do not write an H1 title.
{_OVERVIEW_RULES}"""

REPORT_OVERVIEW_USER = """User Goal / Topic:
<user_query>{user_query}</user_query>

Report title:
"{report_title}"

Language Mode:
{language_mode}

Compact paper context (title, ID, summary, key result, limitation):
<paper_context>
{papers_context}
</paper_context>

Write these two sections:

## {heading_summary}
- Tổng quan ngắn gọn về chủ đề nghiên cứu và các phát hiện chính.

## {heading_conclusion}
- Khuyến nghị thực tiễn cho nhà nghiên cứu & kỹ sư ML.
- Những thách thức mở còn lại trong lĩnh vực này."""


# One-paper variant: keeps both sections short because the paper block already
# holds the quantitative results, which saves output tokens.
REPORT_OVERVIEW_SINGLE_SYSTEM = f"""You are a research analyst.
You are writing the summary and conclusion of a research report about ONE paper. The paper block itself and the references are written separately: do not write them.
{DATA_TAG_RULE}

Instructions:
Write exactly two Markdown sections, in the order listed in the user message, with exactly those `##` headings and no other heading at levels `#` or `##`. Do not write an H1 title.
Keep both sections concise. State a metric with its dataset and unit only where it is most useful; do not re-list the quantitative results, because the paper block already holds them.
If detailed evidence is missing, describe that reader-facing limitation and do not turn placeholder fields into scientific claims. Never print internal PMRL status labels.
If source extraction is incomplete, say that extraction is incomplete and treat absent
sections/results as unverified; never say that the paper itself lacks experiments
just because the parser missed them. Do not include pipeline instructions, agent
status text, PMRL quality labels, language-mode notes, file names, or self-review
sentences in the report. Do not use a raw URL as a heading or title. Never invent a link.
Use inline `$...$` only for individual math symbols or very short expressions. NEVER wrap full sentences or Vietnamese prose in `$...$` delimiters. Never emit raw `<br>` tags.

Write in the language mode given in the user message. Keep the structure consistent with the requested language."""

REPORT_OVERVIEW_SINGLE_USER = """User Goal / Topic:
<user_query>{user_query}</user_query>

Report title:
"{report_title}"

Language Mode:
{language_mode}

Compact paper context (title, ID, summary, key result, limitation):
<paper_context>
{papers_context}
</paper_context>

Write these two sections:

## {heading_summary}
- Tổng quan ngắn gọn về chủ đề nghiên cứu và phát hiện chính.

## {heading_conclusion}
- Khuyến nghị thực tiễn cho nhà nghiên cứu & kỹ sư ML.
- Những thách thức mở còn lại trong lĩnh vực này."""
