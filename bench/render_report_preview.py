"""Render a sample report fully OFFLINE so a human can eyeball the new format.

Usage::

    .venv/bin/python -I bench/render_report_preview.py [--out data/arxiv/preview_report.md]

Inputs are the saved LaTeXML pages in ``data/arxiv/html_probe/`` (git-ignored
local samples).  Every LLM call is answered by ``FakeLLM`` below: the PMRL
structured output and the section / overview text are HAND-WRITTEN stand-ins,
not model output.  Everything else is the real pipeline code: HTML extraction,
PMRL grounding (``write_notes.analyze_paper_notes``), the results table, the
hardware line and the report assembly (``final_report``).  The fake Diffuser
output deliberately carries one invented row and one invented hardware claim to
show that grounding drops them (see the printed counts).

No network, no real LLM, no cache writes outside a temporary directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # works under ``python -I``, which drops the script dir
    sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402
from nodes import final_report as fr  # noqa: E402
from nodes.write_notes import PMRLCoreGeneration, analyze_paper_notes  # noqa: E402
from state import PaperItem  # noqa: E402
from text_utils import detect_response_language  # noqa: E402
from tools.bibtex_generator import generate_bibtex  # noqa: E402
from tools.html_extractor import extract_html  # noqa: E402

PROBE_DIR = ROOT / "data" / "arxiv" / "html_probe"
DEFAULT_OUT = ROOT / "data" / "arxiv" / "preview_report.md"
QUERY = "Diffusion model cho planning và điều khiển robot"

META: Dict[str, Dict[str, Any]] = {
    "2205.09991": {"authors": ["Michael Janner", "Yilun Du", "Joshua B. Tenenbaum", "Sergey Levine"], "published": "2022-05"},
    "2209.14988": {"authors": ["Ben Poole", "Ajay Jain", "Jonathan T. Barron", "Ben Mildenhall"], "published": "2022-09"},
    "2303.04137": {"authors": ["Cheng Chi", "Siyuan Feng", "Yilun Du", "Zhenjia Xu", "Eric Cousineau", "Benjamin Burchfiel", "Shuran Song"], "published": "2023-03"},
}

# --- FAKE structured PMRL output (hand-written; numbers copied from the parsed tables) ---

FAKE_PMRL: Dict[str, Dict[str, Any]] = {
    "2205.09991": {
        "problem": "Learned planners usually pair a dynamics model with a separate trajectory optimizer, which exploits model errors and fails on long horizons with sparse reward.",
        "method": "Diffuser is a diffusion model over whole trajectories (states and actions). Planning is iterative denoising of the full trajectory, with a learned return predictor for guidance and inpainting to fix a start or goal state.",
        "result": "Diffuser scores above 100 on all Maze2D sizes, beats prior baselines in multi-task Multi2D and block stacking, and is competitive with the best offline RL methods on D4RL locomotion.",
        "limitation": "Planning needs many denoising steps per plan, so it is slower than a feed-forward policy. Plans are executed open-loop in the maze experiments.",
        "method_name": "Diffuser",
        "insights": [
            {"finding": "Planning is treated as denoising a whole trajectory at once instead of rolling a one-step model forward, so one-step prediction errors do not compound over the horizon.",
             "why_it_matters": "This is what lets it keep working where model-free credit assignment breaks down, as in sparse-reward mazes."},
            {"finding": "The model is trained on trajectories only; goals and rewards enter at test time as inpainting constraints or guidance.",
             "why_it_matters": "A new goal or reward does not require retraining, so one trained model serves many tasks."},
            {"finding": "On standard offline RL locomotion the method is competitive but does not beat the strongest baseline on average.",
             "why_it_matters": "Choose it for long-horizon planning and flexibility, not as a drop-in winner on D4RL locomotion."},
        ],
        "eval_settings": [
            {"benchmark": "Maze2D (single-task average)", "metric": "Normalized score", "metric_meaning": "Return normalized so that 100 equals a reference expert policy.",
             "higher_is_better": True, "unit": "", "method_value": "119.5", "baseline_name": "IQL", "baseline_value": "47.0",
             "table_label": "Table 1", "source_quote": "Single-task Average | 16.2 | 7.7 | 47.0 | 119.5"},
            {"benchmark": "Multi2D (multi-task average)", "metric": "Normalized score", "metric_meaning": "Return normalized so that 100 equals a reference expert policy.",
             "higher_is_better": True, "unit": "", "method_value": "129.4", "baseline_name": "MPPI", "baseline_value": "21.5",
             "table_label": "Table 1", "source_quote": "Multi-task Average | 21.5 | - | 16.9 | 129.4"},
            {"benchmark": "D4RL locomotion (average)", "metric": "Normalized score", "metric_meaning": "Return normalized so that 100 equals an expert policy.",
             "higher_is_better": True, "unit": "", "method_value": "77.5", "baseline_name": "TT", "baseline_value": "78.9",
             "table_label": "Table 2", "source_quote": "Average | Average | 51.9 | 77.6 | 77.0 | 74.7 | 78.9 | 42.1 | 72.9 | 47.8 | 77.5"},
            {"benchmark": "Block stacking (average)", "metric": "Stacking score", "metric_meaning": "100 is a perfectly executed stack, 0 is a failed one.",
             "higher_is_better": True, "unit": "", "method_value": "54.4", "baseline_name": "CQL", "baseline_value": "8.1",
             "table_label": "Table 3", "source_quote": "Average | 0.0 | 8.1 | 54.4"},
            # FAKE ungrounded row: the numbers are not in the paper, so write_notes must drop it.
            {"benchmark": "Hopper-medium (invented)", "metric": "Normalized score", "metric_meaning": "Return normalized to an expert.",
             "higher_is_better": True, "unit": "", "method_value": "91.37", "baseline_name": "TT", "baseline_value": "66.21",
             "table_label": "Table 2", "source_quote": ""},
        ],
        "setup": {
            "datasets": "D4RL Maze2D and Multi2D, D4RL locomotion (Medium-Expert, Medium, Medium-Replay), block stacking.",
            "training_config": "U-Net with 6 residual blocks, Adam, learning rate 4e-05, batch size 32, 500k steps, 20 diffusion steps for locomotion and 100 for block stacking.",
            # FAKE invented hardware: the paper does not state it, so grounding must drop it.
            "hardware": "8x NVIDIA A100 80GB",
            "compute": "",
        },
    },
    "2209.14988": {
        "problem": "Text-to-3D synthesis needs large labelled 3D datasets that do not exist, while text-to-image diffusion models are plentiful.",
        "method": "DreamFusion optimizes a NeRF so that its renders from random views look like samples of a frozen 2D text-to-image diffusion model. A score-distillation loss turns the 2D model into a prior without back-propagating through it.",
        "result": "Evaluated with CLIP R-Precision on 153 COCO prompts; the geometry (textureless) variant shows large gaps to reimplemented baselines.",
        "limitation": "Uses a 64x64 diffusion model, so details are coarse. Score distillation tends to give over-saturated, mode-seeking results.",
        "method_name": "DreamFusion",
        "insights": [
            {"finding": "A pretrained 2D diffusion model can act as a differentiable prior for a 3D representation, so no 3D training data is needed.",
             "why_it_matters": "It moves the data bottleneck from 3D assets to the much larger 2D image-text corpora."},
            {"finding": "Color R-Precision alone can look good while the geometry is flat, so the paper also scores textureless renders.",
             "why_it_matters": "Without a geometry-aware metric, a text-to-3D method can be rewarded for painting texture on flat shapes."},
        ],
        "eval_settings": [
            {"benchmark": "COCO prompts (Color renders), CLIP B/32", "metric": "R-Precision", "metric_meaning": "How often CLIP retrieves the correct caption for a render among distractors.",
             "higher_is_better": True, "unit": "%", "method_value": "75.1", "baseline_name": "Dream Fields (reimpl.)", "baseline_value": "78.6",
             "table_label": "Table 1", "source_quote": "DreamFusion | 75.1 | 42.5 | 77.5 | 46.6 | 79.7 | 58.5"},
            {"benchmark": "COCO prompts (textureless Geo), CLIP B/32", "metric": "R-Precision", "metric_meaning": "Same retrieval test on textureless renders, which probes geometry.",
             "higher_is_better": True, "unit": "%", "method_value": "42.5", "baseline_name": "Dream Fields (reimpl.)", "baseline_value": "1.3",
             "table_label": "Table 1", "source_quote": ""},
        ],
        "setup": {
            "datasets": "153 prompts from the object-centric COCO validation subset used by Dream Fields.",
            "training_config": "mip-NeRF 360 scene model optimized per prompt against the 64x64 Imagen model.",
            "hardware": "TPUv4 machine with 4 chips",
            "compute": "",
        },
    },
    "2303.04137": {
        "problem": "Behavior cloning policies struggle with multimodal action distributions and high-dimensional action sequences.",
        "method": "Diffusion Policy represents a visuomotor policy as a conditional denoising process over a short action sequence, conditioned on observations. CNN and Transformer denoisers are compared, with receding-horizon execution.",
        "result": "Across 15 tasks from 4 benchmarks it outperforms prior behavior cloning methods, with an average success-rate improvement of 46.9%.",
        "limitation": "Inherits behavior cloning limits: it needs demonstrations and is less effective with little data. Inference is slower than a single forward pass.",
        "method_name": "Diffusion Policy",
        "insights": [
            {"finding": "Predicting a short sequence of actions and executing part of it (receding horizon) gives smooth, temporally consistent behavior that single-step policies lack.",
             "why_it_matters": "Action-sequence prediction, not only the diffusion formulation, is part of why it works on real robots."},
            {"finding": "The CNN backbone is the safer default, while the Transformer variant helps on tasks with sharp action changes.",
             "why_it_matters": "It tells practitioners which denoiser to try first and when to switch."},
        ],
        "eval_settings": [
            {"benchmark": "Robomimic ToolHang (ph), state observation", "metric": "Success rate (max / average over checkpoints)", "metric_meaning": "Fraction of evaluation episodes that complete the task, at the best checkpoint and averaged over the last checkpoints.",
             "higher_is_better": True, "unit": "", "method_value": "1.00/0.87", "baseline_name": "LSTM-GMM", "baseline_value": "0.67/0.31",
             "table_label": "Table 1", "source_quote": "LSTM-GMM | 1.00/0.96 | 1.00/0.93 | 1.00/0.91 | 1.00/0.81 | 0.95/0.73 | 0.86/0.59 | 0.76/0.47 | 0.62/0.20 | 0.67/0.31 | 0.67/0.61"},
            {"benchmark": "Real-world Push-T", "metric": "Success rate", "metric_meaning": "Fraction of real-robot trials that end with the T block in the target region.",
             "higher_is_better": True, "unit": "", "method_value": "0.95", "baseline_name": "LSTM-GMM", "baseline_value": "0.20",
             "table_label": "Table 6", "source_quote": ""},
            {"benchmark": "Real-world Push-T", "metric": "IoU", "metric_meaning": "Overlap between the final T block and the target pose.",
             "higher_is_better": True, "unit": "", "method_value": "0.80", "baseline_name": "LSTM-GMM", "baseline_value": "0.25",
             "table_label": "Table 6", "source_quote": ""},
        ],
        "setup": {
            "datasets": "Robomimic, Push-T, Block Push and Franka Kitchen in simulation; real-world Push-T, sauce pouring and spreading, mug flipping.",
            "training_config": "Batch size 256 for state-based and 64 for image-based experiments, cosine learning-rate schedule with warmup, DDIM with 16 inference steps on real-world tasks.",
            "hardware": "Nvidia 3080 GPU",
            "compute": "",
        },
    },
}

# --- FAKE report text (what a section call would plausibly return) ---

FAKE_SECTIONS: Dict[str, str] = {
    "2205.09991": """### Planning with Diffusion for Flexible Behavior Synthesis (arXiv 2205.09991)
- **Vấn đề**: Các planner học được thường ghép một mô hình động lực học với một bộ tối ưu quỹ đạo riêng. Bộ tối ưu khai thác lỗi của mô hình, và cách này hỏng khi tầm nhìn dài, phần thưởng thưa.
- **Phương pháp**: Diffuser là mô hình khuếch tán trên toàn bộ quỹ đạo (trạng thái và hành động). Lập kế hoạch là khử nhiễu lặp cả quỹ đạo; một bộ dự đoán return dẫn hướng quá trình, còn inpainting cố định trạng thái đầu hoặc đích.
- **Kết quả chính**:
  - **Insight**:
    1. Diffuser khử nhiễu cả quỹ đạo một lần thay vì cuốn chiếu mô hình một bước, nên sai số một bước không cộng dồn theo tầm nhìn. Vì vậy nó vẫn chạy được ở nơi gán công trạng của model-free thất bại, như mê cung phần thưởng thưa.
    2. Mô hình chỉ học từ quỹ đạo; đích và phần thưởng được đưa vào lúc test qua inpainting hoặc guidance. Một đích hay reward mới không cần huấn luyện lại, nên một mô hình phục vụ được nhiều tác vụ.
    3. Trên locomotion offline RL chuẩn, Diffuser cạnh tranh nhưng không vượt baseline mạnh nhất về trung bình. Hãy chọn nó cho planning tầm dài và tính linh hoạt, không phải để thắng D4RL locomotion.
  - **Đại lượng đo**:
    - Normalized score: return chuẩn hoá sao cho 100 bằng một chính sách chuyên gia tham chiếu; cao hơn là tốt hơn.
    - Stacking score: 100 là xếp khối hoàn hảo, 0 là thất bại; cao hơn là tốt hơn.
  - **Thiết lập**: Datasets: D4RL Maze2D/Multi2D, D4RL locomotion, xếp khối. Huấn luyện: U-Net 6 khối residual, Adam, learning rate 4e-05, batch 32, 500k bước; 20 bước khuếch tán cho locomotion, 100 cho xếp khối.
  - **Nhận xét**: Kế hoạch Maze2D được thực thi open-loop. Nhiều baseline có điểm lấy từ bài khác thay vì chạy lại. Điểm từng tác vụ locomotion không đồng đều: Diffuser thấp hơn baseline tốt nhất ở một số tác vụ.
- **Hạn chế**: Mỗi kế hoạch cần nhiều bước khử nhiễu nên chậm hơn policy feed-forward.
""",
    "2209.14988": """### DreamFusion: Text-to-3D using 2D Diffusion (arXiv 2209.14988)
- **Vấn đề**: Tổng hợp text-to-3D cần tập dữ liệu 3D gán nhãn lớn, vốn chưa tồn tại, trong khi mô hình khuếch tán text-to-image rất sẵn.
- **Phương pháp**: DreamFusion tối ưu một NeRF sao cho ảnh render từ các góc nhìn ngẫu nhiên giống mẫu của mô hình khuếch tán 2D đóng băng. Hàm mất mát score distillation biến mô hình 2D thành prior mà không lan truyền ngược qua nó.
- **Kết quả chính**:
  - **Insight**:
    1. Mô hình khuếch tán 2D đã huấn luyện có thể làm prior khả vi cho biểu diễn 3D, nên không cần dữ liệu 3D. Nút thắt dữ liệu chuyển từ tài sản 3D sang kho ảnh-văn bản 2D lớn hơn nhiều.
    2. R-Precision theo màu có thể cao dù hình học phẳng, nên bài báo chấm thêm ảnh render không texture. Thiếu thước đo nhạy hình học, phương pháp text-to-3D có thể được thưởng vì "vẽ" texture lên hình phẳng.
  - **Đại lượng đo**:
    - R-Precision: tỷ lệ CLIP truy xuất đúng caption của ảnh render giữa các caption gây nhiễu; tính bằng %, cao hơn là tốt hơn.
  - **Thiết lập**: Datasets: 153 prompt thuộc tập COCO validation hướng vật thể của Dream Fields. Huấn luyện: scene mip-NeRF 360 tối ưu riêng cho từng prompt với mô hình Imagen 64x64. Hardware: TPUv4, 4 chip.
  - **Nhận xét**: Baseline Dream Fields tái cài đặt đạt R-Precision màu cao hơn DreamFusion ở nhiều cấu hình CLIP nhưng hình học rất kém, nên chỉ số màu đơn lẻ dễ gây hiểu lầm.
- **Hạn chế**: Dùng mô hình 64x64 nên chi tiết thô; score distillation có xu hướng bão hoà màu và tìm mode.
""",
    "2303.04137": """### Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (arXiv 2303.04137)
- **Vấn đề**: Behavior cloning khó biểu diễn phân phối hành động đa mode và chuỗi hành động nhiều chiều.
- **Phương pháp**: Diffusion Policy mô hình hoá policy thị giác-vận động như quá trình khử nhiễu có điều kiện trên một chuỗi hành động ngắn, điều kiện theo quan sát. Bài so sánh denoiser CNN và Transformer, thực thi theo receding horizon.
- **Kết quả chính**:
  - **Insight**:
    1. Dự đoán cả chuỗi hành động rồi chỉ thực thi một phần (receding horizon) cho hành vi mượt, nhất quán theo thời gian mà policy một bước không có. Vì vậy phần thành công trên robot thật một phần đến từ dự đoán chuỗi hành động, không chỉ từ khuếch tán.
    2. Backbone CNN là mặc định an toàn hơn; biến thể Transformer giúp ở tác vụ có hành động đổi đột ngột. Điều này cho người dùng biết nên thử denoiser nào trước.
  - **Đại lượng đo**:
    - Success rate (max / trung bình các checkpoint): tỷ lệ episode hoàn thành tác vụ; cao hơn là tốt hơn.
    - IoU: mức chồng lấp giữa khối T cuối cùng và vị trí đích; cao hơn là tốt hơn.
  - **Thiết lập**: Datasets: Robomimic, Push-T, Block Push, Franka Kitchen (mô phỏng) và Push-T, đổ/phết nước sốt, lật cốc (thực tế). Huấn luyện: batch 256 (state) và 64 (ảnh), lịch cosine có warmup, DDIM 16 bước suy luận. Hardware: Nvidia 3080 GPU (suy luận).
  - **Nhận xét**: Bảng 1 lấy số baseline từ nhiều nguồn (tái chạy hoặc từ bài gốc), và một số đánh giá robomimic chỉ dùng 22 lần khởi tạo do lỗi mã đánh giá.
- **Hạn chế**: Thừa hưởng giới hạn của behavior cloning: cần demonstration và kém hiệu quả khi ít dữ liệu; suy luận chậm hơn một lần forward.
""",
}

FAKE_OVERVIEW = """## 1. Tóm tắt
- Cả ba bài đều dùng khuếch tán như một cách tinh chỉnh lặp: Diffuser khử nhiễu cả quỹ đạo, Diffusion Policy khử nhiễu một chuỗi hành động, DreamFusion khử nhiễu qua gradient của một prior 2D. Cái chung là sai số một bước không cộng dồn và đầu ra không bị ép về một mode.
- Chỗ khác nhau là điều kiện nằm ở đâu: Diffuser đưa đích vào lúc test bằng inpainting và guidance, Diffusion Policy điều kiện theo quan sát lúc huấn luyện, DreamFusion không có dữ liệu 3D mà mượn prior 2D. Vì chỉ số, bộ dữ liệu và đơn vị khác nhau, không xếp hạng ba bài với nhau.

## 3. Kết luận
- Chọn Diffuser khi cần planning tầm dài hoặc đổi đích lúc test, Diffusion Policy khi có demonstration và cần hành vi mượt trên robot thật.
- Còn mở: chi phí suy luận của nhiều bước khử nhiễu và đánh giá closed-loop trên robot thật.
"""


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Structured:
    def __init__(self, llm: "FakeLLM", schema: Any) -> None:
        self._llm, self._schema = llm, schema

    def invoke(self, payload: Any) -> Any:
        assert self._schema is PMRLCoreGeneration, "the preview only fakes the PMRL structured call"
        return PMRLCoreGeneration.model_validate(self._llm.pmrl_for(_user_text(payload)))


def _user_text(payload: Any) -> str:
    return str(getattr(payload[-1], "content", payload[-1])) if isinstance(payload, list) else str(payload)


class FakeLLM:
    """FAKE model: returns the hand-written texts above, chosen by the arXiv ID in the prompt."""

    def pmrl_for(self, user: str) -> Dict[str, Any]:
        return FAKE_PMRL[self._arxiv_id(user)]

    @staticmethod
    def _arxiv_id(user: str) -> str:
        return next(i for i in FAKE_PMRL if f"ArXiv ID: {i}" in user)

    def with_structured_output(self, schema: Any) -> _Structured:
        return _Structured(self, schema)

    def invoke(self, payload: Any) -> _Message:
        user = _user_text(payload)
        if "Write the analysis block for this paper now." in user:
            return _Message(FAKE_SECTIONS[self._arxiv_id(user)])
        if "Write these two sections:" in user:
            return _Message(FAKE_OVERVIEW)
        raise AssertionError("unexpected fake LLM call")


def build_paper(arxiv_id: str) -> PaperItem:
    extraction = extract_html((PROBE_DIR / f"ar5iv_{arxiv_id}.html").read_text(encoding="utf-8", errors="replace"))
    meta = META[arxiv_id]
    return PaperItem(
        paper_id=f"arxiv_{arxiv_id}",
        title=extraction.title,
        summary=extraction.sections.get("abstract", ""),
        authors=meta["authors"],
        published=meta["published"],
        arxiv_id=arxiv_id,
        sections=dict(extraction.sections),
        tables=list(extraction.tables),
        references=list(extraction.references),
        repo_links=list(extraction.repo_links),
        source_quality={"coverageStatus": "complete", "source": "ar5iv", "codeUrls": extraction.code_urls},
        bibtex=generate_bibtex(extraction.title, meta["authors"], meta["published"], arxiv_id),
    )


def render(out: Path) -> str:
    config.DEFAULT_PROVIDER = "gemini"                      # native structured-output path (not the Z.AI JSON path)
    config.SUMMARY_CACHE_DIR = Path(tempfile.mkdtemp(prefix="preview_cache_"))
    llm = FakeLLM()
    language = detect_response_language(QUERY)

    papers = []
    for arxiv_id in META:
        paper, status, _ = analyze_paper_notes(build_paper(arxiv_id), llm=llm)
        print(f"{arxiv_id}: notes {status}; grounding {json.dumps(paper.processing_metadata.get('evalSettings'))}")
        papers.append(paper)

    title = fr.report_title(QUERY, papers)
    blocks = [fr._write_paper_section(llm, paper, i, QUERY, language)[0] for i, paper in enumerate(papers, 1)]
    overview, _ = fr._write_overview(llm, QUERY, title, language, papers)
    report = fr._assemble_parallel_report(title, language, papers, blocks, overview)
    banner = "<!-- PREVIEW: PMRL output and section text are FAKE (hand-written); tables, hardware line and layout come from the real code. -->\n\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(banner + report, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    render(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
