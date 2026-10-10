# Research Report: các phương pháp diffusion cho robot

## 1. Tóm tắt

Báo cáo tổng hợp ba nghiên cứu đại diện cho ba hướng ứng dụng mô hình khuếch tán (diffusion) liên quan đến robot:

- **Diffusion Policy** (arXiv 2303.04137): dùng DDPM có điều kiện làm biểu diễn chính sách cho học bắt chước thị giác–vận động; đạt **cải thiện tương đối trung bình 46,9%** trên 15 tác vụ manipulation thuộc 4 benchmark.
- **Diffuser** (arXiv 2205.09991): đặt mô hình khuếch tán trực tiếp lên quỹ đạo, biến lập kế hoạch thành lấy mẫu có điều kiện (guidance + inpainting); đạt **điểm trên 100 trên cả ba kích thước mê cung Maze2d**, vượt chính sách tham chiếu, thích nghi đa mục tiêu không cần huấn luyện lại.
- **DreamFusion** (arXiv 2209.14988): dùng prior khuếch tán 2D đóng băng để tối ưu NeRF sinh hình 3D từ văn bản (Score Distillation Sampling) — nhánh bổ trợ tạo tài sản 3D cho robot/mô phỏng; đạt **58,5% nhất quán chú thích trên kết xuất không texture**.

Điểm chung: cả ba khai thác năng lực của diffusion trong việc biểu diễn đầu ra **đa mode, chiều cao**. Ba bài báo dùng metric, bộ dữ liệu và đơn vị khác nhau (tỷ lệ thành công manipulation, điểm lập kế hoạch mê cung, CLIP R-Precision) nên **không thể xếp hạng định lượng chéo**. Điểm nghẽn chung là chi phí lấy mẫu lặp tại thời điểm suy luận.

## 2. Phân tích bài báo

### Diffusion Policy (arXiv 2303.04137)

- **Vấn đề**: Học bắt chước thị giác–vận động yêu cầu biểu diễn phân phối có điều kiện $p(a \mid o)$ trên hành động liên tục, nhưng mọi biểu diễn trước đó đều thất bại theo một cách cụ thể: chính sách hồi quy tường minh trung bình hóa các mode của phân phối đa mode; hỗn hợp Gauss/phân loại (BC-RNN/LSTM-GMM, BET) phải chỉ định trước số mode, và khi dự đoán từng bước độc lập, các hành động liên tiếp có thể rơi vào mode khác nhau tạo rollout rung giật (dao động qua trái/phải quanh khối T); IBC học hàm năng lượng nhưng khó lấy mẫu trong không gian hành động chiều cao, mặt năng lượng không trơn và thiên lệch về một mode. Ngoài ra, chính sách một bước dễ overfit hành động idle (khi tạm dừng teleop) và bị kẹt trong rollout thực nếu không lọc dữ liệu idle; điều khiển vận tốc chịu lỗi cộng dồn, còn điều khiển vị trí giảm lỗi cộng dồn nhưng đa mode hơn — BC-RNN và BET thậm chí suy giảm khi chuyển sang điều khiển vị trí.
- **Phương pháp**: Biểu diễn chính sách là quá trình khử nhiễu khuếch tán có điều kiện (DDPM) trên **chuỗi hành động**. Mạng dự đoán nhiễu $\epsilon_\theta$, được điều kiện bởi quan sát, huấn luyện với mục tiêu chuẩn:

$$
\mathcal{L}(\theta)=\mathbb{E}_{k,\epsilon}\left[\left\lVert \epsilon_{\theta}\!\left(a^{k},\,o,\,k\right)-\epsilon\right\rVert^{2}\right],\qquad a^{k}=\alpha_{k}\,a^{0}+\sigma_{k}\,\epsilon,
$$

trong đó $a^0$ là chuỗi hành động sạch, $a^k$ là phiên bản bị nhiễu ở bước khuếch tán thứ $k$, và $o$ là quan sát. Suy luận bắt đầu từ nhiễu Gauss, tinh chỉnh qua nhiều bước khử nhiễu kiểu Langevin, tạo ra chuỗi hành động chiều cao với nhất quán thời gian tự nhiên và miễn nhiễm dữ liệu idle. Ba đóng góp kỹ thuật: (1) *receding horizon control* — dự đoán chuỗi $T_p$ bước, chỉ thực hiện $T_a$ bước đầu rồi lập kế hoạch lại; (2) *điều kiện hóa thị giác* — đặc trưng quan sát tiêm trực tiếp vào backbone khử nhiễu (qua FiLM) thay vì chỉ vào loss; (3) hai biến thể kiến trúc: Diffusion Policy-C (CNN tích chập thời gian 1D) và Diffusion Policy-T (transformer kiểu GPT khử nhiễu cả chuỗi). Chính sách dùng điều khiển delta vị trí end-effector thay cho điều khiển vận tốc phổ biến trước đó.
- **Kết quả chính**: Trên 15 tác vụ thuộc 4 benchmark manipulation (PushT, Robomimic, ManiSkill, RoboTurk), vượt các baseline SOTA (LSTM-GMM/BC-RNN, IBC, BET) với **cải thiện tương đối trung bình 46,9%**. Trong tác vụ đẩy khối T đa mode, chính sách học cả hai mode và cam kết đúng một mode mỗi rollout, trong khi LSTM-GMM/IBC thiên lệch một mode và BET không cam kết do thiếu nhất quán thời gian. Khi chuyển từ điều khiển vận tốc sang vị trí trên Square, Kitchen, p4, Diffusion Policy-C/-T tăng tỷ lệ thành công tương đối trong khi BC-RNN và BET giảm. Ablation trên PushT và Square: thay đổi tỷ lệ thành công tương đối trong khoảng 0 đến −0,6 theo action horizon từ 1 tới 128 bước (đạt đỉnh ở horizon trung bình); dự đoán chuỗi hành động mang lại độ bền với độ trễ replanning 0–7 bước.
- **Hạn chế**: Nội dung trích xuất không có phần hạn chế riêng; các đánh đổi sau được ghi nhận hoặc suy trực tiếp: mỗi kế hoạch đòi hỏi nhiều bước khử nhiễu nên suy luận chậm hơn chính sách hồi quy một bước; receding horizon giảm nhẹ chi phí và ablation độ trễ cho thấy suy giảm êm, nhưng horizon tối ưu là sự đánh đổi với độ trễ — chọn horizon sai có thể mất tới ~0,6 tỷ lệ thành công tương đối. Lợi thế trong điều khiển vị trí giả định đa mode là khó khăn chủ đạo; tác giả định hướng dòng chính sách dựa trên generative diffusion, gợi ý công việc tương lai về sampler nhanh.
- **Mã nguồn**: [44David/diffusion-policy](https://github.com/44David/diffusion-policy) (⭐ 0, Python).
- **Trích xuất**: Trích xuất nguồn chưa đầy đủ — tên tác giả, ngày xuất bản không được ghi nhận; con số 46,9% chỉ có trong phần tóm tắt kết quả, các trích đoạn PDF chủ yếu định tính. Các chi tiết thiếu được coi là chưa xác minh, không có nghĩa là bài báo không có thông tin đó.
- **Liên kết nguồn**: [ArXiv abstract](https://arxiv.org/abs/2303.04137) | [PDF](https://arxiv.org/pdf/2303.04137.pdf)
- **Trích dẫn**:

```bibtex
@article{paper2024arxiv,
  title={ArXiv 2303.04137},
  author={Unknown Author},
  journal={arXiv preprint arXiv:2303.04137},
  year={2024},
  eprint={2303.04137},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

### Diffuser (arXiv 2205.09991)

- **Vấn đề**: RL dựa mô hình tách rời học mô hình khỏi điều khiển: mạng học động môi trường bằng hồi quy có giám sát rồi gắn vào bộ tối ưu quỹ đạo cổ điển (iLQG, MPPI, planner lấy mẫu). Cách ghép này giòn: bộ tối ưu mạnh khai thác triệt để lỗi, discontinuity và vùng lạc quan của mô hình học, trả về kế hoạch giống adversarial example chống lại chính mô hình. Về phía model-free offline RL (BCQ, CQL, IQL), các phương pháp này yếu với gán công quả dài hạn, phần thưởng thưa (Maze2d cần hàng trăm bước để nhận một lần +1) và suy giảm mạnh khi chuyển từ đơn nhiệm sang đa nhiệm; MPPI kém ngay cả khi được cho động lực học đúng trên quãng đường dài. Câu hỏi trung tâm: gom càng nhiều pipeline tối ưu quỹ đạo vào bài toán mô hình hóa càng tốt, sao cho lấy mẫu từ mô hình và lập kế hoạch gần như là một, đồng thời giữ linh hoạt lúc test (mục tiêu chưa thấy, điều kiện phần thưởng, ràng buộc) mà không cần huấn luyện lại.
- **Phương pháp**: Diffuser là mô hình khuếch tán xác định trực tiếp trên **quỹ đạo** (chuỗi trạng thái–hành động). Quá trình xuôi làm nhiễu Gauss dần quỹ đạo; quá trình ngược, cài đặt bằng temporal U-Net, tinh chỉnh mẫu từ nhiễu thuần thành quỹ đạo hợp lệ. Lập kế hoạch trở thành lấy mẫu sinh có điều kiện qua hai cơ chế: (1) *classifier guidance* — một guide khả vi (ví dụ mô hình phần thưởng học được) hiệu chỉnh dự đoán nhiễu ở mỗi bước khử nhiễu:

$$
\hat{\epsilon}\left(\tau^{i}\right)=\epsilon_{\theta}\left(\tau^{i},\,i\right)+\lambda\,\nabla_{\tau^{i}} r\left(\tau^{i}\right),
$$

trong đó $r$ là hàm phần thưởng và $\lambda$ là hệ số định hướng mẫu về các quỹ đạo có phần thưởng cao — đảm nhiệm vai trò hàm chi phí trong tối ưu quỹ đạo; (2) *inpainting* — chốt một tập phần tử quỹ đạo (trạng thái bắt đầu/đích) ở giá trị cố định tại mọi bước khử nhiễu, phần còn lại được "vá" như inpainting ảnh. Vì định nghĩa mục tiêu (guidance), điều kiện hóa (inpainting) và tổng hợp quỹ đạo thống nhất trong một thủ tục lấy mẫu, không cần bộ tối ưu ngoài; tác vụ mới lúc test chỉ cần đổi guide hoặc mặt nạ.
- **Kết quả chính**: Trên Maze2d (U-Maze, Medium, Large; chỉ thưởng 1 tại đích, không shaping), điều kiện hóa start/goal bằng inpainting và thực thi open-loop đạt **điểm trên 100 ở cả ba kích thước** — vượt chính sách tham chiếu (đơn vị điểm không được ghi rõ trong trích đoạn). Ở biến thể đa nhiệm Multi2d (đích ngẫu nhiên mỗi episode), Diffuser đạt như đơn nhiệm chỉ bằng đổi điều kiện đầu vào, không huấn luyện lại, trong khi IQL + hindsight relabeling giảm mạnh; MPPI với động lực học đúng cũng kém, cho thấy độ khó dài hạn ngay cả khi không có lỗi dự đoán. Trên bài toán xếp khối từ 10.000 quỹ đạo demo PDDLStream: BCQ 0,0/0,0/0,0 (TB 0,0); CQL 24,4/0,0/0,0 (TB 8,1); Diffuser 58,7 ± 2,5 / 45,6 ± 3,1 / 58,9 ± 3,4 (TB 54,4) theo ba thiết lập Unconditional Stacking / Conditional Stacking / Rearrangement — tổng quát hóa tới cấu hình đích chưa thấy mà baseline gần như thất bại. Bài báo dành một nghiên cứu riêng cho thời gian chạy, chỉ ra cách tăng tốc hiệu quả nhất với thiệt hại hiệu năng tối thiểu.
- **Hạn chế**: Linh hoạt đổi bằng chi phí tính toán lúc test — nhiều bước khử nhiễu thay vì một forward pass (tác giả dành riêng một nghiên cứu về tăng tốc). Trên Maze2d, quỹ đạo được thực thi hoàn toàn open-loop nên độ bền closed-loop với lỗi cộng dồn chưa được minh chứng trong kết quả báo cáo. Định nghĩa tác vụ đòi hỏi guide khả vi (cần mô hình phần thưởng phù hợp) hoặc ràng buộc inpainting biết trước (start/goal). Là planner học từ dữ liệu, Diffuser chỉ tổng hợp hành vi nhất quán với phân phối quỹ đạo huấn luyện, nên thành công phụ thuộc độ phủ dữ liệu, kể cả khả năng khâu (stitch) dữ liệu dị chất lượng.
- **Mã nguồn**: Không có repository được xác minh trong lần chạy này (không vì vậy kết luận bài báo không công bố code).
- **Trích xuất**: Trích xuất chưa ghi nhận tác giả và ngày xuất bản; các số liệu trên đến từ trích đoạn kết quả của nguồn.
- **Liên kết nguồn**: [ArXiv abstract](https://arxiv.org/abs/2205.09991) | [PDF](https://arxiv.org/pdf/2205.09991.pdf)
- **Trích dẫn**:

```bibtex
@article{paper2024arxiv,
  title={ArXiv 2205.09991},
  author={Unknown Author},
  journal={arXiv preprint arXiv:2205.09991},
  year={2024},
  eprint={2205.09991},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

### DreamFusion (arXiv 2209.14988)

- **Vấn đề**: Tổng hợp text-to-3D bị chặn bởi hai điều kiện tiên quyết còn thiếu: bộ dữ liệu 3D gán nhãn quy mô lớn và kiến trúc khử nhiễu dữ liệu 3D hiệu quả. Các baseline zero-shot trước đó tránh diffusion, tối ưu biểu diễn 3D (NeRF trong Dream Fields, mesh trong CLIP-Mesh) theo embedding CLIP; lối CLIP-guidance này giòn: kết xuất có texture khớp chú thích trong khi hình học vẫn phẳng hoặc vô đạo, và bản cài đặt lại Dream Fields của chính bài báo gần như ngẫu nhiên khi đánh giá hình học bằng kết xuất không texture. Thách thức: tận dụng prior khuếch tán 2D pretrained làm giám sát sinh cho tối ưu 3D — hình học nhất quán góc nhìn, relight được, từ văn bản — mà không cần dữ liệu 3D, không sửa mô hình ảnh, và không có giám sát tham chiếu (không tồn tại "cảnh 3D đúng" nên Chamfer Distance hay PSNR không áp dụng được).
- **Phương pháp**: Tối ưu NeRF khởi tạo ngẫu nhiên bằng gradient descent theo kiểu DeepDream, dùng mô hình text-to-image khuếch tán 2D đóng băng (Imagen) làm prior học được: các kết xuất 2D từ góc camera ngẫu nhiên phải đạt loss thấp. Cơ chế lõi là **Score Distillation Sampling (SDS)** dựa trên chưng cất mật độ xác suất: kết xuất $x$ bị nhiễu ở bước $t$ ngẫu nhiên theo $x_t = \alpha_t x + \sigma_t \epsilon$, U-Net đóng băng dự đoán nhiễu $\epsilon_\phi(x_t; y, t)$ theo prompt $y$, và gradient truyền về tham số 3D $\theta$:

$$
\nabla_{\theta}\,\mathcal{L}_{\mathrm{SDS}}
=\mathbb{E}_{t,\epsilon}\left[\,w(t)\left(\epsilon_{\phi}(x_t;\,y,\,t)-\epsilon\right)\frac{\partial x_t}{\partial \theta}\right],
$$

tức hàm điểm (score) của mô hình khuếch tán kéo các kết xuất về vùng có khả năng cao dưới prompt. Vì giám sát đến từ dự đoán nhiễu chứ không phải chuỗi lấy mẫu ngược đầy đủ, mô hình ảnh không cần fine-tune hay sửa kiến trúc. NeRF kết xuất albedo, pháp tuyến và độ sâu với chiếu sáng kiểu Phong, cho phép prompt phụ thuộc góc nhìn và relighting. Bốn lựa chọn tối ưu được ablation xác nhận là cần thiết cho hình học đúng: tăng cường khoảng góc nhìn rộng (ViewAug), prompt phụ thuộc góc nhìn (ViewDep), tối ưu cả kết xuất có chiếu sáng (Lighting), và tối ưu kết xuất hình học không texture (Textureless).
- **Kết quả chính**: Đánh giá bằng **CLIP R-Precision** (độ chính xác truy xuất chú thích đúng trong các caption gây nhiễu, đơn vị %) trên **153 prompt đối tượng từ tập validation COCO** của Dream Fields, so với Dream Fields, bản cài đặt lại dùng biểu diễn 3D của DreamFusion, CLIP-Mesh và oracle ảnh COCO gốc. Mặc dù baseline có lợi thế dùng CLIP khi huấn luyện (DreamFusion không dùng CLIP), DreamFusion vượt Dream Fields và CLIP-Mesh trên ảnh màu và tiệm cận ảnh ground-truth. Trên kết xuất không texture — thước đo hình học thực sự, vì metric hiện tại dễ đạt cao khi sơn texture lên hình học phẳng — kết xuất của DreamFusion nhất quán với chú thích **58,5%** thời gian, trong khi bản cài lại Dream Fields gần ngẫu nhiên. Ablation cộng dồn ViewAug/ViewDep/Lighting/Textureless cải thiện hình học đáng kể và +12,5% trên kết xuất đầy đủ; đồng thời cho thấy kết xuất albedo đánh lừa — cấu hình nền đạt điểm albedo cao nhất nhưng hình học hỏng (ví dụ chó nhiều đầu). Bộ tài sản 3D, video và mesh được phát hành tại dreamfusion3d.github.io.
- **Hạn chế**: Tối ưu theo từng prompt thay vì generator feed-forward: mỗi cảnh cần gradient descent lặp trên NeRF, khoảng **1,5 giờ mỗi prompt trên TPU v4** — chậm và không amortize. Đánh giá bị giới hạn: không có cảnh 3D đúng nên chỉ còn metric tự động dựa trên CLIP, vốn không phản ánh chất lượng hình học; so sánh R-Precision cũng vốn bất công vì baseline huấn luyện với CLIP. Hình học không được đảm bảo bởi loss chưng cất một mình — cần đủ bốn lựa chọn trên, nếu không xuất hiện artifact như động vật nhiều đầu hay hình học phẳng được sơn texture. Chất lượng bị chặn bởi prior 2D đóng băng và khả năng hiểu văn bản của nó; phần thảo luận chỉ thêm lấy mẫu chậm và cấu trúc 3D đôi khi vô đạo, thấp đa dạng là hướng tương lai.
- **Mã nguồn**: Không có repository được xác minh trong lần chạy này (không vì vậy kết luận bài báo không công bố code).
- **Trích xuất**: Trích xuất chưa ghi nhận tác giả và ngày xuất bản; mọi kết quả định lượng đến từ trích đoạn của nguồn.
- **Liên kết nguồn**: [ArXiv abstract](https://arxiv.org/abs/2209.14988) | [PDF](https://arxiv.org/pdf/2209.14988.pdf)
- **Trích dẫn**:

```bibtex
@article{paper2024arxiv,
  title={ArXiv 2209.14988},
  author={Unknown Author},
  journal={arXiv preprint arXiv:2209.14988},
  year={2024},
  eprint={2209.14988},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## 3. So sánh

Ba phương pháp dùng chung nền tảng toán học (DDPM) nhưng đóng vai trò khác nhau trong hệ robot:

| Tiêu chí | Diffusion Policy (2303.04137) | Diffuser (2205.09991) | DreamFusion (2209.14988) |
|---|---|---|---|
| Vai trò của diffusion | Biểu diễn chính sách (DDPM có điều kiện trên chuỗi hành động) | Mô hình quỹ đạo kiêm planner (DDPM trên chuỗi trạng thái–hành động) | Prior 2D đóng băng, chưng cất vào NeRF (không phải chính sách) |
| Đầu ra | Chuỗi hành động robot ($T_p$ bước mỗi lần lập kế hoạch) | Quỹ đạo trạng thái–hành động hoàn chỉnh | Mô hình 3D (màu, pháp tuyến, độ sâu) |
| Điều kiện hóa | Quan sát tiêm vào backbone khử nhiễu (FiLM) | Classifier guidance (guide phần thưởng) + inpainting (start/goal) | Prompt văn bản qua SDS |
| Thực thi | Receding horizon, điều khiển delta vị trí end-effector | Open-loop hoặc replanning | Không thuộc vòng điều khiển — tạo nội dung 3D |
| Kết quả tiêu biểu | +46,9% cải thiện tương đối trung bình trên 15 tác vụ / 4 benchmark manipulation | Điểm Maze2d > 100 mọi kích thước; xếp khối trung bình 54,4 so với CQL 8,1 và BCQ 0,0 | CLIP R-Precision 58,5% trên kết xuất không texture (153 prompt COCO) |
| Điểm nghẽn tính toán | Nhiều bước khử nhiễu mỗi lần lập kế hoạch | Nhiều bước lấy mẫu mỗi quỹ đạo | ~1,5 giờ tối ưu mỗi prompt (TPU v4) |
| Mã nguồn | Repo diffusion-policy (0 sao, Python) | Chưa xác minh repo | Chưa xác minh repo |

**Tổng hợp**:

- Không tồn tại metric, bộ dữ liệu hay đơn vị chung giữa ba bài báo — thay đổi tỷ lệ thành công manipulation, điểm lập kế hoạch mê cung, và CLIP R-Precision — nên không thể lập bảng xếp hạng định lượng chéo; các giao thức đánh giá quá khác nhau để gộp thành một leaderboard.
- Chủ đề chung: cả ba đều khai thác năng lực của diffusion với đầu ra chiều cao, đa mode — thứ mà các biểu diễn chính sách/planner trước đây xử lý kém.
- Diffuser đánh đổi phản hồi từng bước bằng lấy mẫu quỹ đạo open-loop: vượt chuyên gia trên Maze2d và thích nghi mục tiêu ngẫu nhiên chỉ bằng đổi điều kiện đầu vào, không huấn luyện lại.
- DreamFusion bỏ hướng dẫn CLIP, thay bằng prior 2D đóng băng qua SDS, và vẫn vượt các baseline được huấn luyện bằng CLIP về CLIP R-Precision, đồng thời đạt 58,5% nhất quán chú thích trên hình học không texture.
- Diffusion Policy là phương pháp duy nhất trong ba bài vận hành ở mức chính sách phản hồi lặp (receding horizon); Diffuser hoạt động ở mức lập kế hoạch quỹ đạo; DreamFusion nằm ngoài vòng điều khiển, bổ trợ bằng sinh tài sản 3D.

## 4. Kết luận

**Khuyến nghị thực tiễn:**

- **Chính sách học bắt chước cho manipulator với demo teleop đa mode**: chọn Diffusion Policy; ưu tiên điều khiển delta vị trí end-effector (lợi thế chỉ diffusion khai thác được, trong khi BC-RNN/BET suy giảm), chọn action horizon trung bình (đỉnh trong ablation 1–128 bước), và tính toán ngân sách suy luận vì mỗi kế hoạch cần nhiều bước khử nhiễu; receding horizon cùng dự đoán chuỗi giúp kháng độ trễ replanning 0–7 bước và dữ liệu idle.
- **Offline planning/RL dài hạn, mục tiêu thay đổi lúc test**: chọn Diffuser; cần guide phần thưởng khả vi hoặc biết trước start/goal để inpaint; lưu ý kết quả được báo cáo ở chế độ open-loop — cần kiểm chứng closed-loop trước khi triển khai thực tế.
- **Tài sản 3D cho mô phỏng/embodied AI từ văn bản**: cách tiếp cận DreamFusion (SDS với prior 2D đóng băng) hoạt động không cần dữ liệu 3D, nhưng chi phí ~1,5 giờ/prompt và hình học chỉ đạt khi áp đủ các kỹ thuật ViewAug, ViewDep, Lighting, Textureless.
- Khi so sánh các phương pháp diffusion cho robot, lưu ý các benchmark không giao thoa: số liệu manipulation success (Diffusion Policy), điểm maze planning (Diffuser) và CLIP R-Precision (DreamFusion) không quy đổi cho nhau.

**Thách thức mở:**

- **Chi phí lấy mẫu lặp**: cả ba đều chậm hơn phương pháp một bước; sampler nhanh, distillation và chính sách diffusion ít bước là hướng được chính các tác giả chỉ ra.
- **Đánh giá phân mảnh**: thiếu benchmark dùng chung cho các phương pháp diffusion–robot, nên chưa có leaderboard hợp nhất; metric hiện hành (CLIP cho 3D) không phản ánh chất lượng hình học.
- **Tính biểu đạt của điều kiện hóa**: Diffuser yêu cầu guide khả vi hoặc ràng buộc biết trước; mở rộng phạm vi tác vụ có thể chỉ định lúc test là vấn đề chưa giải quyết.
- **Phụ thuộc dữ liệu và giả định**: Diffuser phụ thuộc độ phủ phân phối quỹ đạo huấn luyện; Diffusion Policy giả định đa mode là khó khăn chủ đạo; DreamFusion bị chặn bởi prior 2D đóng băng (cấu trúc 3D đôi khi vô đạo, thấp đa dạng).

## 5. Tài liệu tham khảo

1. **Diffusion Policy (arXiv 2303.04137)** — [ArXiv abstract](https://arxiv.org/abs/2303.04137) | [PDF](https://arxiv.org/pdf/2303.04137.pdf) | [Mã nguồn](https://github.com/44David/diffusion-policy)
2. **Diffuser (arXiv 2205.09991)** — [ArXiv abstract](https://arxiv.org/abs/2205.09991) | [PDF](https://arxiv.org/pdf/2205.09991.pdf)
3. **DreamFusion (arXiv 2209.14988)** — [ArXiv abstract](https://arxiv.org/abs/2209.14988) | [PDF](https://arxiv.org/pdf/2209.14988.pdf)

Ghi chú: thông tin tác giả và năm xuất bản không được ghi nhận trong trích xuất nguồn; khối BibTeX đầy đủ cho từng bài báo nằm ở mục 2.
