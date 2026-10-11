// Dev-only fixtures built from the approved mockup (docs/ui-mockup/research-scout-v3.html):
// the `SUMMARY` object, the graph `D`, and the answers of the follow-up examples.
import type { AnswerSummary, PaperCard } from '../types'
import graphJson from './mockGraph.json'

type RawGraph = typeof graphJson

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v)) as T

/** The mockup graph with the given arXiv ids marked as origin (the rest `similar`). */
export function mockGraph(originArxivIds: string[]): RawGraph {
  const g = clone(graphJson)
  g.nodes.forEach((n) => {
    n.role = n.arxivId && originArxivIds.includes(n.arxivId) ? 'origin' : 'similar'
  })
  return g
}

export const REFLECT_ID = '2605.05737'
export const CHATCOT_ID = '2305.14323'
export const MULTI_IDS = [REFLECT_ID, '2310.12426', CHATCOT_ID, '2304.13007', '2305.03268']

export const REFLECT_SUMMARY: AnswerSummary = {
  kind: 'single',
  paper: { paperId: 'arxiv:2605.05737', arxivId: REFLECT_ID, title: 'ReFlect: An Effective Harness System for Complex Long-Horizon LLM Reasoning', published: '05/2026' },
  lead: 'ReFlect chuyển việc phát hiện và sửa lỗi ra khỏi prompt, sang một lớp code bao quanh model. Nhờ vậy, model nhỏ có harness vượt model lớn không có harness, mà dùng ít token hơn.',
  tldr: 'ReFlect chuyển việc phát hiện và sửa lỗi ra khỏi prompt, sang một lớp code bao quanh model.',
  blocks: [
    { type: 'problem', text: 'Trên tác vụ suy luận dài nhiều bước, lỗi tích lũy âm thầm. Chain-of-thought, ReAct và tự phê bình đều giả định model tự nhận ra lỗi của mình, nhưng giả định này không đúng.' },
    { type: 'intuition', text: 'Model đánh giá lập luận của mình bằng cùng một cách nghĩ, nên thường không thấy lỗi. ReFlect đặt bước kiểm tra vào code bên ngoài: CHECK độc lập chạy sau mỗi bước, và chỉ khi CHECK thất bại thì REFLECT mới được gọi.' },
    {
      type: 'keyResults',
      items: [
        'Tự phê bình bằng prompt gần như không bắt được lỗi: 90/100 khối reflection không nêu vấn đề nào.',
        'Claude Haiku 4.5 có ReFlect đạt trung bình 48,0%; Claude Sonnet 4.5 không có harness chỉ đạt 27,1%.',
        'Trên tập 70B, Full ReFlect dùng 1.993 token mỗi bài và đạt 48,8%; Reflexion dùng 32.062 token và đạt 27,8%.',
      ],
    },
    {
      type: 'metrics',
      items: [
        { name: 'Success rate', meaning: 'Tỉ lệ bài giải đúng trên 50 bài mỗi domain, tính bằng %.', higherIsBetter: true },
        { name: 'Acc / 1K tokens', meaning: 'Độ chính xác chia cho số nghìn token đã dùng; đo hiệu quả chi phí.', higherIsBetter: true },
      ],
    },
    {
      type: 'setup',
      items: ['6 domain (SWE-bench, QASPER, Proof, AIME, ALFWorld, FinQA), 50 bài mỗi domain, 6 model ở 3 tầng năng lực. Model 70B chạy trên vLLM, bf16, tensor-parallel 4 GPU.'],
      hardware: '',
      missing: ['Loại GPU'],
    },
    {
      type: 'results',
      grounded: true,
      source: ['Table 3', 'Table 17'],
      methodName: 'ReFlect',
      rows: [
        { setting: 'Claude Haiku 4.5, TB 6 domain', metric: 'Success %', method: '48.0', baseline: '25.5 Direct CoT', delta: '+22.5', higherIsBetter: true },
        { setting: 'Claude Sonnet 4.5, TB 6 domain', metric: 'Success %', method: '56.4', baseline: '27.1 Direct CoT', delta: '+29.3', higherIsBetter: true },
        { setting: 'Qwen2.5-72B, TB 6 domain', metric: 'Success %', method: '41.8', baseline: '35.2 Direct CoT', delta: '+6.6', higherIsBetter: true },
        { setting: 'Tập 70B, cùng ngân sách', metric: 'Acc / 1K tok', method: '24.5', baseline: '0.9 Reflexion', delta: '+23.6', higherIsBetter: true },
      ],
    },
    {
      type: 'caveats',
      items: [
        'SWE-bench dùng bộ chấm cấu trúc riêng (Appendix L), nên điểm SWE không so trực tiếp được với bài khác.',
        'QASPER gần như không cải thiện (6,3–13,0).',
        'Bản harness không phụ thuộc domain chỉ đạt 17,0%, thấp hơn cả Direct CoT.',
      ],
    },
    { type: 'foundations', items: ['Chain-of-Thought', 'Self-Consistency', 'Tree of Thoughts', 'Reflexion', 'Self-Refine', 'ReAct'] },
    { type: 'takeaway', text: 'Dùng ReFlect khi tác vụ dài, có dạng tính toán rõ và bạn viết được công cụ kiểm tra cho từng domain. Không nên dùng khi chưa có công cụ kiểm tra: bản harness chung chung thua cả Direct CoT.' },
  ],
  actions: { reportId: 'report_reflect_20261011', bibtexCount: 1, graphNodes: 40 },
  durationSec: 213,
}

export const CHATCOT_SUMMARY: AnswerSummary = {
  kind: 'single',
  paper: { paperId: 'arxiv:2305.14323', arxivId: CHATCOT_ID, title: 'ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models', published: '05/2023' },
  lead: 'ChatCoT biến chain-of-thought thành một cuộc hội thoại nhiều lượt; ở mỗi lượt, model chọn giữa suy luận tiếp hoặc gọi công cụ.',
  blocks: [
    { type: 'intuition', text: 'Chat model quen với hội thoại nhiều lượt. ChatCoT dùng đúng thế mạnh này: mỗi bước suy luận và mỗi lần gọi công cụ là một lượt trao đổi.' },
    { type: 'takeaway', text: 'So với ReFlect: cả hai đều đưa công cụ vào quá trình suy luận, nhưng ChatCoT để model tự quyết khi nào kiểm tra, còn ReFlect đặt việc kiểm tra vào code bên ngoài model.' },
  ],
  actions: { reportId: 'report_chatcot_20261011', bibtexCount: 1, graphNodes: 40 },
  durationSec: 192,
}

const CARDS: PaperCard[] = [
  { paperId: 'arxiv:2605.05737', title: 'ReFlect: An Effective Harness System for Complex Long-Horizon LLM Reasoning', year: '2026', intuition: 'Đặt việc phát hiện và sửa lỗi vào code quanh model; Haiku 4.5 có harness đạt 48,0% so với 25,5% khi không có.', keyResult: 'Trung bình 6 domain: 48,0% với Haiku 4.5 + ReFlect so với 25,5% Direct CoT (Table 3).' },
  { paperId: 'arxiv:2310.12426', title: 'MAF: Multi-Aspect Feedback for Improving Reasoning in Large Language Models', year: '2023', intuition: 'Gộp nhiều nguồn phản hồi, mỗi nguồn chuyên một loại lỗi, thay cho một bộ phản hồi chung.', keyResult: 'Phản hồi theo từng khía cạnh cho kết quả tốt hơn một bộ phản hồi chung trên các tập suy luận toán và logic.' },
  { paperId: 'arxiv:2305.14323', title: 'ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models', year: '2023', intuition: 'Chuỗi suy luận thành hội thoại nhiều lượt; model chọn suy luận tiếp hoặc gọi công cụ.', keyResult: 'Vượt các baseline dùng công cụ trên bài toán toán học nhiều bước.' },
  { paperId: 'arxiv:2304.13007', title: 'Answering Questions by Meta-Reasoning over Multiple Chains of Thought', year: '2023', intuition: 'Suy luận trên nhiều chuỗi CoT cùng lúc thay vì chỉ bỏ phiếu đáp án cuối.' },
  { paperId: 'arxiv:2305.03268', title: 'Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework', year: '2023', intuition: 'Sửa các bước suy luận theo tri thức bên ngoài để tăng độ đúng sự thật.', keyResult: 'Tăng độ đúng sự thật trên các bộ câu hỏi nhiều bước.' },
]

export const MULTI_SUMMARY: AnswerSummary = {
  kind: 'multi',
  papers: CARDS.map((c) => ({ paperId: c.paperId, arxivId: c.paperId?.replace('arxiv:', ''), title: c.title, published: String(c.year) })),
  lead: 'Năm bài chọn ra cùng trả lời một câu hỏi: làm sao để model không tin vào lỗi của chính nó. Hướng mới nhất chuyển việc kiểm tra từ prompt sang công cụ và code bên ngoài.',
  blocks: [
    {
      type: 'insights',
      items: [
        { finding: 'Phản hồi từ chính model là điểm yếu chung.', why: 'MAF và Verify-and-Edit lấy phản hồi từ nguồn ngoài; ReFlect cho thấy tự phê bình bằng prompt bỏ sót phần lớn lỗi.' },
        { finding: 'Công cụ dần trở thành một phần của chuỗi suy luận.', why: 'ChatCoT gọi công cụ trong từng lượt; ReFlect định tuyến bài toán theo dạng tính toán.' },
      ],
    },
    { type: 'paperCards', items: CARDS },
    { type: 'foundations', items: ['Chain-of-Thought', 'Self-Consistency', 'Reflexion', 'ReAct'] },
    { type: 'takeaway', text: 'Bắt đầu từ ReFlect nếu bạn cần kiểm tra bằng code; đọc MAF và Verify-and-Edit nếu phản hồi phải đến từ nguồn tri thức ngoài.' },
  ],
  actions: { reportId: 'report_multi_20261011', bibtexCount: 5, graphNodes: 40 },
  durationSec: 425,
}

export const REFLECT_BIBTEX = `@article{huang2026reflect,
  title   = {ReFlect: An Effective Harness System for Complex Long-Horizon LLM Reasoning},
  author  = {Huang, Fan},
  journal = {arXiv preprint arXiv:2605.05737},
  year    = {2026}
}`

export const CHATCOT_BIBTEX = `@article{chen2023chatcot,
  title   = {ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models},
  author  = {Chen, Zhipeng},
  journal = {arXiv preprint arXiv:2305.14323},
  year    = {2023}
}`

export const MULTI_BIBTEX = [REFLECT_BIBTEX, CHATCOT_BIBTEX, '@article{nathani2023maf,\n  title = {MAF},\n  year  = {2023}\n}', '@article{yoran2023meta,\n  title = {Meta-Reasoning over Multiple Chains of Thought},\n  year  = {2023}\n}', '@article{zhao2023ve,\n  title = {Verify-and-Edit},\n  year  = {2023}\n}'].join('\n\n')

export const QUICK_ANSWER = `Vì model đánh giá lại chính lập luận của mình bằng cùng một cách nghĩ, nên nó thường không thấy lỗi.

Trong 100 khối reflection mà tác giả kiểm tra, 90 khối không nêu vấn đề nào; chúng là các mẫu câu lặp lại. Các model được thử chấp nhận đáp án sai ít nhất 76% số lần. ReFlect thay bước này bằng cơ chế CHECK độc lập, và chỉ gọi REFLECT khi CHECK thất bại.

*Từ báo cáo ReFlect, mục Insight và Table 15–16.*`

export const TABLE_ANSWER = `Table 17 so các phương pháp trên tập 70B (Llama-3.3-70B và Qwen2.5-72B, 300 bài mỗi model) theo 3 đại lượng: số token mỗi bài, độ chính xác, và độ chính xác trên mỗi 1.000 token.

| Phương pháp | Token / bài | Accuracy % | Acc / 1K tok |
| --- | ---: | ---: | ---: |
| Direct (1 run) | 2,001 | 26.9 | 13.4 |
| ReAct (1 run) | 4,087 | 26.5 | 6.5 |
| Self-Refine (3 vòng) | 8,361 | 27.2 | 3.2 |
| Reflexion (3 episode) | 32,062 | 27.8 | 0.9 |
| **Full ReFlect** | **1,993** | **48.8** | **24.5** |

*Mọi số khớp nguồn. Nguồn: Table 17.*`

export const MOCK_REPORT = `# Research Report: ReFlect

Bản report mẫu cho chế độ mock.
`
