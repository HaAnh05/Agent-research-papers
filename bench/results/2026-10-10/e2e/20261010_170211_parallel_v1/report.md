# Research Report: các phương pháp diffusion cho robot

## 1. Tóm tắt

- Chủ đề của báo cáo là các phương pháp dựa trên mô hình diffusion cho robot. Ba công trình được phân tích đại diện cho ba mức trừu tượng khác nhau: **Diffusion Policy** (arXiv:2303.04137) — chính sách thị giác–vận động ở mức chuỗi hành động ngắn hạn; **Diffuser** (arXiv:2205.09991) — diffusion trên toàn bộ quỹ đạo trạng thái–hành động cho hoạch định; **DreamFusion** (arXiv:2209.14988) — dùng diffusion 2D đóng băng làm prior (Score Distillation Sampling) để tối ưu NeRF tạo tài sản 3D, tức ứng dụng ngoài điều khiển trực tiếp.
- Diffusion Policy đạt cải thiện trung bình **46.9%** (so với các phương pháp trước đó: IBC, BET, BC-RNN/LSTM-GMM) trên **15 nhiệm vụ thuộc 4 benchmark** (Push-T, Robomimic, RLBench, robot thật); xử lý tốt phân phối hành động đa mode — mỗi rollout cam kết một mode duy nhất.
- Diffuser, với kỹ thuật inpainting điểm đầu/cuối, đạt **score trên 100 ở cả ba kích thước Maze2D** (U-Maze, Medium, Large; đơn vị không được nêu trong trích xuất), vượt cả expert tham chiếu; giữ nguyên hiệu năng khi mục tiêu thay đổi ngẫu nhiên theo episode (multi-task không cần huấn luyện lại); đạt trung bình **54.4** trên các nhiệm vụ xếp khối so với 8.1 (CQL) và 0.0 (BCQ).
- DreamFusion vượt Dream Fields và CLIP-Mesh về **CLIP R-Precision trên 153 prompt COOC** (đơn vị %), đạt **58.5%** nhất quán caption trên render không texture; các lựa chọn ablation (ViewAug, ViewDep, Lighting, Textureless) cải thiện hình học và tăng **+12.5%** trên full render.
- Phát hiện xuyên suốt: độ biểu đạt đa mode và cơ chế điều kiện hóa linh hoạt là lợi thế chung; cái giá là suy luận nhiều bước khử nhiễu và, với DreamFusion, một đợt tối ưu gradient riêng cho từng prompt.
- Lưu ý về dữ liệu nguồn: trích xuất được cung cấp không đầy đủ — phần tóm tắt ngắn của cả ba bài bỏ trống, và mục hạn chế tường minh chỉ xuất hiện ở Diffusion Policy; hạn chế của Diffuser và DreamFusion được suy luận từ nội dung phương pháp/kết quả và cần đối chiếu với bản gốc để xác nhận.
- Vì metric, dataset và đơn vị của ba bài không giao nhau, báo cáo không xếp hạng các bài với nhau theo số liệu.

## 2. Phân tích bài báo

### Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (arXiv 2303.04137)

- **Vấn đề**: Trong học bắt chước (imitation learning) cho chính sách thị giác–vận động, phân phối hành động tại cùng một trạng thái thường đa phương (multimodal) — ví dụ end-effector có thể đẩy khối quanh phía trái hoặc phía phải — và mỗi dạng biểu diễn chính sách trước đó thất bại theo kiểu riêng: chính sách tường minh với hồi quy vô hướng trung bình hòa các mode; biểu diễn GMM/phân loại (LSTM-GMM) phải định trước số mode và dễ sụp về một mode; chính sách ngầm (IBC) học hàm năng lượng trên cặp quan sát–hành động nhưng việc lấy mẫu từ bề mặt năng lượng không trơn không mở rộng tốt lên không gian hành động chiều cao. Vì khó lấy mẫu này, việc dự đoán chuỗi hành động phần lớn bị tránh: BC-RNN/LSTM-GMM và BET dự đoán từng hành động từ các phân phối đa phương độc lập, nên các hành động liên tiếp rơi vào mode khác nhau tạo ra quỹ đạo rung giật, không nhất quán về mode; BET thiếu tính nhất quán hành động theo thời gian nên không cam kết một mode duy nhất; chính sách một bước dễ overfit các hành động nhàn rỗi/tạm dừng của teleoperation và bị kẹt trong rollout trên robot thật. Ngoài ra, phần lớn behavior cloning dùng điều khiển vận tốc, vốn chịu lỗi tích lũy (compounding errors), trong khi điều khiển vị trí bộc lộ tính đa phương mạnh hơn mà các phương pháp hiện có không biểu diễn được.

- **Phương pháp**: Diffusion Policy biểu diễn chính sách như một quá trình khử nhiễu diffusion có điều kiện (DDPM): thay vì xuất trực tiếp hành động, mạng học gradient của hàm score của phân phối hành động, và ở inference hành động được sinh bằng cách tinh chỉnh lặp một nhiễu Gaussian qua các bước động học Langevin ngẫu nhiên có điều kiện theo quan sát. Với mẫu thứ $k$ trong batch (chuỗi hành động mục tiêu độ dài $T_p$ là $a_0^k$), quá trình forward thêm nhiễu theo bước khuếch tán $\tau$:

$$a_\tau^k = \sqrt{\bar{\alpha}_\tau}\, a_0^k + \sqrt{1-\bar{\alpha}_\tau}\, \epsilon_\tau^k, \quad \epsilon_\tau^k \sim \mathcal{N}(0, I)$$

Mạng khử nhiễu $\epsilon_\theta$ được huấn luyện dự đoán nhiễu đã thêm vào, có điều kiện trên quan sát trực giác $O_t$:

$$L = \mathrm{MSE}\left(\epsilon_\tau^k,\; \epsilon_\theta(a_\tau^k, \tau, O_t)\right)$$

Ở inference, chuỗi hành động được khử nhiễu lặp từ nhiễu Gaussian qua $K$ bước theo dạng cập nhật ngược chuẩn của DDPM:

$$a_{\tau-1}^k = \frac{1}{\sqrt{\alpha_\tau}}\left(a_\tau^k - \frac{1-\alpha_\tau}{\sqrt{1-\bar{\alpha}_\tau}}\, \epsilon_\theta(a_\tau^k, \tau, O_t)\right) + \sigma_\tau z, \quad z \sim \mathcal{N}(0, I)$$

Ba đóng góp kỹ thuật then chốt đưa cách tiếp cận này lên robot: (1) điều khiển receding-horizon — mô hình dự đoán chuỗi dài với prediction horizon $T_p = 16$ nhưng chỉ thực thi $T_a = 8$ bước đầu (action horizon) dựa trên $T_o = 2$ bước quan sát rồi lập kế hoạch lại, vừa đảm bảo tính nhất quán hành động theo thời gian vừa giữ tính phản ứng; (2) điều kiện hóa thị giác — đặc trưng từ vision backbone đóng băng pretrained trên ImageNet được tiêm qua FiLM (biến thể CNN, DiffusionPolicy-C, dùng temporal convolution 1D) hoặc qua cross-attention (DiffusionPolicy-T), thay vì khử nhiễu ảnh đồng thời; (3) time-series diffusion transformer thay backbone temporal convolution cho dữ liệu hành động dạng chuỗi thời gian. Chính sách được huấn luyện ở chế độ điều khiển vị trí, và DDIM với 10 bước khử nhiễu được dùng để giảm độ trễ inference.

- **Kết quả chính**: Trên 15 tác vụ từ 4 benchmark thao tác robot (Push-T mô phỏng, Robomimic, RLBench và các tác vụ robot thật), Diffusion Policy liên tục vượt các phương pháp học robot SOTA trước đó (IBC, BET, BC-RNN/LSTM-GMM) với cải thiện trung bình 46,9% (đơn vị: % so với baseline tốt nhất, tính trung bình trên toàn bộ 15 tác vụ). Định tính, trên các tác vụ đẩy đa phương (Fig. 3), phương pháp học cả hai mode (đẩy trái hoặc phải) và chỉ cam kết một mode trong mỗi rollout, trong khi LSTM-GMM và IBC thiên lệch về một mode và BET không cam kết được mode nào do thiếu tính nhất quán hành động theo thời gian. Khi chuyển từ điều khiển vận tốc sang điều khiển vị trí (Fig. 4, tác vụ Square và Kitchen), BC-RNN và BET giảm hiệu năng trong khi Diffusion Policy-C/-T cải thiện, cho thấy diffusion khai thác được lợi thế của điều khiển vị trí trong dự đoán chuỗi hành động. Phép ablation về action horizon (Fig. 5, PushT và Square) quét 1–128 bước cho thấy horizon lớn hơn tăng độ bền trước độ trễ được tiêm thêm 0–7 bước nhưng đánh đổi một phần độ chính xác, củng cố thiết kế receding-horizon.

- **Hạn chế**: Công thức diffusion tạo ra đánh đổi tính toán/độ trễ: inference đòi hỏi nhiều bước khử nhiễu Langevin ngẫu nhiên lặp lại, khiến nó chậm và đắt hơn các chính sách hồi quy một lần, chỉ được giảm bớt một phần nhờ DDIM. Điều khiển receding-horizon đòi hỏi dự đoán các hành động tương lai dư thừa để có tính nhất quán thời gian, và lợi ích này dựa trên giả định của điều khiển vị trí rằng hành động tương lai có thể dự đoán được; bài báo lưu ý nếu cây thực thi (plant) hoặc chính sách phi tuyến thì việc dự đoán hành động tương lai có thể trở nên khó khăn hơn đáng kể và đa phương trở lại. Hiệu năng cũng nhạy với siêu tham số action horizon, đánh đổi độ bền độ trễ với độ chính xác (Fig. 5), và việc thực thi các khối hành động theo kiểu bán mở vòng (semi-open-loop) trên robot thật đặt ra lo ngại về độ nhạy độ trễ — đây chính là động cơ cho phân tích độ bền độ trễ.

- **Mã nguồn**: [44David/diffusion-policy](https://github.com/44David/diffusion-policy) (⭐ 0; Python). Số sao và ngôn ngữ phản ánh thời điểm trích xuất; đây là repo được ghi nhận trong lần chạy này và chưa xác minh được thêm thông tin framework chi tiết.

- **Trích xuất**: Các phần chính (vấn đề, phương pháp, kết quả, hạn chế, repo, liên kết ArXiv) có trong bản trích xuất; tuy nhiên các bảng kết quả chi tiết theo từng benchmark/tác vụ (ví dụ tỉ lệ thành công tuyệt đối từng bảng) không có mặt trong phần trích xuất, nên số liệu định lượng xác minh được ở cấp khối này chủ yếu là mức cải thiện trung bình 46,9% cùng các thiết lập horizon. Đây là giới hạn của việc trích xuất nguồn, không phải dấu hiệu bài báo thiếu thí nghiệm.

- **Liên kết nguồn**: [ArXiv abstract](https://arxiv.org/abs/2303.04137) | [PDF](https://arxiv.org/pdf/2303.04137.pdf)

- **Trích dẫn**:

```bibtex
@article{chi2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
  journal={arXiv preprint arXiv:2303.04137},
  year={2023},
  eprint={2303.04137},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

### Planning with Diffusion for Flexible Behavior Synthesis (arXiv:2205.09991)

- **Vấn đề**: Model-based RL truyền thống tách việc ra quyết định thành hai giai đoạn rời nhau: học có giám sát một mô hình động lực học xấp xỉ, rồi ghép mô hình đó vào một bộ tối ưu hóa quỹ đạo cổ điển (iLQG, MPPI, direct transcription). Cách ghép này giòn vỡ trong thực nghiệm: khi bộ tối ưu hóa mạnh gặp mô hình học chưa hoàn hảo, nó khai thác có hệ thống lỗi của mô hình, khiến kế hoạch sinh ra giống các "adversarial examples" chống lại chính mô hình (Talvitie, 2014; Ke et al., 2018) hơn là quỹ đạo khả thi. Song song, điều khiển dài hạn với phần thưởng thưa (Maze2D, hàng trăm bước giữa xuất phát và đích) làm thất bại credit assignment ngay với các thuật toán model-free mạnh nhất, còn các phương pháp dựa trên policy thiếu linh hoạt tại test-time: đổi cấu hình đích đòi hỏi huấn luyện lại hoặc relabel dữ liệu. Bài báo hỏi: nếu gói càng nhiều càng tốt quy trình tối ưu hóa quỹ đạo vào chính bài toán mô hình hóa sinh dữ liệu, sao cho lấy mẫu từ mô hình và lập kế hoạch với nó gần như trùng nhau?

- **Phương pháp**: Diffuser là một mô hình diffusion xác suất định nghĩa trực tiếp trên toàn bộ quỹ đạo trạng thái–hành động, thay vì trên từng bước hoặc trên mô hình động lực học. Tiến trình thuận cố định làm nhiễu quỹ đạo, tiến trình ngược học được lặp khử nhiễu từ nhiễu thuần; một chuỗi diffusion ngược cho ra một kế hoạch hoàn chỉnh được thực thi open-loop. Hai kỹ thuật diffusion chuẩn được diễn giải lại thành toán tử lập kế hoạch: (1) classifier-guided sampling — gradient của một bộ phân loại reward đã huấn luyện lệch mỗi bước khử nhiễu về phía quỹ đạo lợi nhuận cao, cho phép khai báo mục tiêu tại test-time theo cách khả vi; (2) image inpainting — kẹp (clamp) các bước thời gian chỉ định (trạng thái đầu và đích) ở giá trị cố định trong khi mô hình điền phần còn lại, cung cấp conditioning theo mục tiêu. Vì toàn quỹ đạo được sinh đồng thời thay vì từng bước, bài toán credit assignment dài hạn được hấp thụ vào mô hình sinh; hành vi đa nhiệm thu được chỉ bằng cách đổi điều kiện (ví dụ vị trí đích mới) mà không huấn luyện lại.

  Cặp tiến trình thuận – ngược trên quỹ đạo $\tau$ với chỉ số bước diffusion $i$:

  $$q(\tau^{i} \mid \tau^{i-1}) = \mathcal{N}\!\left(\tau^{i};\ \sqrt{1-\beta^{i}}\,\tau^{i-1},\ \beta^{i}\mathbf{I}\right)$$

  Tiến trình thuận phá hủy quỹ đạo thực dần theo lịch hệ số nhiễu $\beta^{i}$ cho tới nhiễu thuần.

  $$p_\theta(\tau^{i-1} \mid \tau^{i}) = \mathcal{N}\!\left(\tau^{i-1};\ \mu_\theta(\tau^{i}, i),\ \Sigma^{i}\right)$$

  Tiến trình ngược do mạng $\theta$ dự đoán, khử nhiễu từng bước; sau một chuỗi đầy đủ, đầu ra là một kế hoạch trọn vẹn.

  $$p(\tau^{i-1} \mid \tau^{i}) = \mathcal{N}\!\left(\tau^{i-1};\ \mu_\theta(\tau^{i}, i) + \Sigma\,\gamma\,\nabla_{\tau^{i}} J(\tau^{i}),\ \Sigma\right)$$

  Ở dạng có hướng dẫn, gradient của hàm lợi nhuận $J(\tau^i)$ do mô hình reward học được cung cấp được cộng vào trung bình dự đoán, kéo mỗi bước khử nhiễu về phía quỹ đạo điểm cao; các ràng buộc kiểu đích được áp đặt bằng cách kẹp một số bước thời gian trong quỹ đạo.

- **Kết quả chính**:
  - Maze2D (U-Maze, Medium, Large): phần thưởng chỉ bằng 1 tại ô đích, không shaping, horizon hàng trăm bước. Diffuser với inpainting start/goal đạt trên 100 điểm chuẩn hóa ở cả ba kích thước mê cung, vượt expert policy tham chiếu; các baseline model-free gặp khó khăn, và MPPI chạy với động lực học ground-truth cũng kém — cho thấy nút thắt dài hạn nằm ở khâu tối ưu hóa chứ không phải lỗi dự đoán.
  - Multi2D (mục tiêu ngẫu nhiên theo từng episode): Diffuser đạt như trong thiết lập đơn nhiệm mà không huấn luyện lại, nhờ conditioning tự nhiên; phương pháp model-free mạnh nhất là IQL với hindsight experience relabeling (Kostrikov et al., 2022; Andrychowicz et al., 2017) giảm mạnh điểm số.
  - Block stacking (10.000 quỹ đạo trình diễn PDDLStream; reward 1 cho mỗi lần xếp khối thành công, 0 nếu thất bại): Unconditional Stacking đạt 58.7±2.5 so với 24.4 của CQL và 0.0 của BCQ; Conditional Stacking 45.6±3.1 và Rearrangement 58.9±3.4, trong khi BCQ và CQL đều 0.0; trung bình 54.4 so với 8.1 (CQL) và 0.0 (BCQ) — thể hiện khả năng tổng quát hóa tới các cấu hình đích không xuất hiện trong huấn luyện.
  - Bài báo còn phân tích thời gian chạy thực tế của lập kế hoạch bằng diffusion và những cách tăng tốc lấy mẫu hiệu quả nhất với tổn thất hiệu năng tối thiểu.

- **Hạn chế**: Nội dung bài được trích xuất không có mục hạn chế/future work tường minh, nên các hạn chế dưới đây được suy ra từ chính tài liệu trình bày. Việc tác giả dành riêng một phần phân tích cho thời gian chạy và tăng tốc cho thấy khử nhiễu lặp trên toàn quỹ đạo tốn kém về tính toán tại thời điểm ra quyết định, và tăng tốc thô sơ làm suy giảm chất lượng kế hoạch. Về cấu trúc phương pháp: kế hoạch được thực thi open-loop như một quỹ đạo lấy mẫu duy nhất (closed-loop replanning và độ nhạy với lỗi mô hình trong lúc thực thi không được đề cập trong phần trích xuất); điều khiển theo reward phụ thuộc vào việc huấn luyện riêng một bộ phân loại reward; conditioning dạng inpainting giả định ràng buộc có thể diễn đạt thành trạng thái cố định tại các bước thời gian đã biết. Không có bảo đảm hình thức nào về tính tối ưu của kế hoạch hay khả năng thỏa mãn ràng buộc.

- **Mã nguồn**: Không có repository nào được xác minh trong lần chạy enrichment này; điều này không có nghĩa là bài báo không có mã nguồn công khai.

- **Trích xuất**: Phần trích xuất nguồn cho bài báo này thiếu mục hạn chế tường minh, nên các hạn chế được suy luận từ nội dung phương pháp và phân tích runtime; các kết quả ngoài những số liệu nêu trên chưa được kiểm chứng trong lần chạy này.

- **Liên kết nguồn**: [ArXiv abstract](https://arxiv.org/abs/2205.09991) | [PDF](https://arxiv.org/pdf/2205.09991.pdf)

- **Trích dẫn**:

```bibtex
@article{janner2022planning,
  title={Planning with Diffusion for Flexible Behavior Synthesis},
  author={Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  journal={arXiv preprint arXiv:2205.09991},
  year={2022},
  eprint={2205.09991},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

### DreamFusion: Text-to-3D using 2D Diffusion (arXiv 2209.14988)

- **Vấn đề**: Các mô hình diffusion text-to-image vừa đạt chất lượng đột phá nhờ huấn luyện trên hàng tỷ cặp ảnh–văn bản, nhưng việc kế thừa thành công này sang 3D bị chặn bởi hai nút thắt nền tảng: không tồn tại tập dữ liệu 3D có chú thích (caption) quy mô lớn, và không có kiến trúc hiệu quả để khử nhiễu dữ liệu 3D thể tích ở quy mô tương đương ảnh 2D. Các baseline zero-shot text-to-3D trước đó như Dream Fields và CLIP-Mesh né vấn đề dữ liệu bằng cách tối ưu biểu diễn 3D theo embedding CLIP, nhưng CLIP là mô hình discriminative nên guidance cho ra render thiếu thực và hình học kém, đồng thời không thể ép buộc tính nhất quán đa góc nhìn mà một image prior có khả năng sinh mới cung cấp. Góc độ đánh giá cũng là bài toán mở: vì không có scene 3D "đúng" ứng với một prompt, các metric tham chiếu chuẩn như Chamfer Distance (tái tạo) hay PSNR (view synthesis) không áp dụng được, còn các metric dựa trên CLIP hiện có không phản ánh chất lượng hình học — thường cho điểm cao khi texture chỉ được sơn lên hình học phẳng.

- **Phương pháp**: DreamFusion né việc thiếu dữ liệu 3D bằng cách dùng một mô hình diffusion text-to-image 2D đã huấn luyện trước và đóng băng như một prior trên các ảnh render, rồi tối ưu bằng gradient descent — theo lối DeepDream — một bộ sinh 3D tham số khởi tạo ngẫu nhiên dưới dạng Neural Radiance Field (NeRF) với thành phần density, màu albedo và chiếu sáng phụ thuộc góc nhìn. Đóng góp cốt lõi là hàm mất mát dựa trên probability density distillation, hiện thực hóa thành Score Distillation Sampling (SDS): một góc camera được lấy mẫu, render $x$ của NeRF được cộng nhiễu Gaussian theo một bước thời gian diffusion $t$ ngẫu nhiên,

$$x_t = \sqrt{\bar{\alpha}_t}\, x + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

  sau đó hàm mất mát SDS đo sai lệch giữa nhiễu mà U-Net diffusion đóng băng $\hat{\epsilon}_\phi$ dự đoán cho prompt $y$ và nhiễu thực tế đã cộng vào:

$$\mathcal{L}_{\mathrm{SDS}}(\theta, x) = \mathbb{E}_{t,\epsilon}\left[\, w(t)\,\big\| \hat{\epsilon}_\phi(x_t;\, t, y) - \epsilon \big\|_2^2 \right]$$

  Gradient truyền vào tham số NeRF $\theta$ chỉ phụ thuộc hiệu giữa nhiễu dự đoán và nhiễu mẫu, lan truyền ngược qua differentiable renderer:

$$\nabla_\theta \mathcal{L}_{\mathrm{SDS}} \;\propto\; \mathbb{E}_{t,\epsilon}\left[\, w(t)\left(\hat{\epsilon}_\phi(x_t;\, t, y) - \epsilon\right)\frac{\partial x}{\partial \theta} \right]$$

  Cách làm này không cần dữ liệu huấn luyện 3D và không đòi hỏi chỉnh sửa mô hình diffusion ảnh, cho thấy mô hình diffusion ảnh huấn luyện trước có thể đóng vai trò prior hiệu quả. Để đạt hình học chính xác, bài báo kết hợp các lựa chọn tối ưu được cô lập trong ablation: vùng góc nhìn lớn (ViewAug), prompt phụ thuộc góc nhìn (ViewDep), tối ưu trên render có chiếu sáng/đổ bóng bên cạnh render albedo không ánh sáng (Lighting), và tối ưu trên ảnh hình học đổ bóng không texture (Textureless). NeRF thu được có thể xem từ mọi góc, chiếu sáng lại dưới điều kiện ánh sáng tùy ý và ghép vào môi trường 3D bất kỳ.

- **Kết quả chính**:
  - Đánh giá bằng CLIP R-Precision — độ chính xác (%) mà CLIP truy xuất đúng caption trong một tập caption gây nhiễu khi cho xem một render — trên 153 prompt đối tượng (object-centric) từ tập validation COCO của Dream Fields, đo trên ba loại render: albedo màu, đổ bóng đầy đủ và không texture. Mặc dù các baseline được lợi thế không công bằng khi dùng CLIP ngay trong huấn luyện, DreamFusion vẫn vượt Dream Fields, một bản reimplement Dream Fields nâng cấp dùng chính biểu diễn 3D của DreamFusion, và CLIP-Mesh trên render màu, đồng thời tiệm cận mức của oracle là các cặp ảnh–caption MS-COCO ground-truth.
  - Trên render không texture — thước đo hướng tới chất lượng hình học — DreamFusion nhất quán với caption 58,5% số lần, trong khi bản reimplement Dream Fields gần như ở mức ngẫu nhiên (chance).
  - Ablation cộng dần ViewAug, ViewDep, Lighting, Textureless cho thấy hình học cải thiện rõ rệt với từng lựa chọn, và render đầy đủ tăng thêm 12,5% CLIP R-Precision. Ablation này cũng phơi bày bẫy đo lường: mô hình cơ sở đạt điểm render albedo cao nhất nhưng hình học lại kém (một chú chó nhiều đầu), chứng tỏ render albedo có thể gây hiểu lầm và hình học chính xác đòi hỏi prompt theo góc nhìn, chiếu sáng và render không texture.
  - Về định tính, mô hình sinh được các cảnh có tính kết hợp (compositional) và được so sánh với baseline trên các prompt như "a hamburger", "a vase with pink flowers", hay "a matte painting of a castle made of cheesecake surrounded by a moat made of ice cream"; thư viện asset 3D, video mở rộng và mesh được công bố tại trang dự án dreamfusion3d.github.io.

- **Hạn chế**:
  - Đánh giá định lượng buộc phải dựa vào CLIP R-Precision vì không có scene 3D ground-truth để áp dụng Chamfer Distance hay PSNR; bản thân các metric CLIP không hoàn hảo — chúng có thể cho điểm cao khi texture được vẽ lên hình học phẳng, và điểm render albedo không đảm bảo hình học đúng như ablation đã chỉ ra — nên chất lượng hình học không được phản ánh đầy đủ.
  - Chất lượng hình học phụ thuộc vào một bộ lựa chọn tối ưu cụ thể (vùng góc nhìn rộng, prompt theo góc nhìn, render có chiếu sáng, render không texture); thiếu các lựa chọn này, quá trình tối ưu tạo ra hình học suy biến.
  - DreamFusion là quy trình gradient descent kiểu DeepDream thực hiện cho từng prompt trên một NeRF khởi tạo ngẫu nhiên, chứ không phải mô hình sinh feed-forward, kéo theo chi phí tính toán đáng kể cho mỗi asset được sinh ra và sự phụ thuộc vào thiên kiến cũng như chất lượng của prior diffusion 2D đóng băng.

- **Mã nguồn**: Không có repository nào được xác minh trong lần chạy enrichment này, nên không thể cung cấp link GitHub, số star hay framework; điều này không có nghĩa là bài báo không công bố mã. Trang dự án dreamfusion3d.github.io (thư viện asset, video, mesh) được nhắc trong bài nhưng không được xác minh là repo mã nguồn.

- **Trích xuất**: Nội dung nguồn được trích xuất chủ yếu ở mức tóm tắt/trang ArXiv; phần trích xuất không hiện mục hạn chế tường minh của bài báo — các hạn chế nêu trên được suy ra từ thiết kế và kết quả đánh giá được báo cáo, nên có thể chưa phủ toàn bộ giới hạn mà tác giả trình bày trong bản đầy đủ. Các phần nằm ngoài phạm vi trích xuất được coi là chưa kiểm chứng.

- **Liên kết nguồn**: [ArXiv abstract](https://arxiv.org/abs/2209.14988) | [PDF](https://arxiv.org/pdf/2209.14988.pdf)

- **Trích dẫn**:

```bibtex
@article{poole2022dreamfusion,
  title={DreamFusion: Text-to-3D using 2D Diffusion},
  author={Ben Poole and Ajay Jain and Jonathan T. Barron and Ben Mildenhall},
  journal={arXiv preprint arXiv:2209.14988},
  year={2022},
  eprint={2209.14988},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## 3. So sánh

Ba bài dùng chung nền tảng toán học (quá trình khử nhiễu) nhưng ở cấp độ ứng dụng khác nhau, nên bảng dưới so sánh định tính, không phải xếp hạng.

| Tiêu chí | Diffusion Policy | Diffuser | DreamFusion |
|---|---|---|---|
| Vai trò của diffusion | Phân phối chuỗi hành động ngắn hạn, khử nhiễu có điều kiện theo quan sát (DDPM) | Toàn bộ quỹ đạo trạng thái–hành động; một chuỗi khử nhiễu cho cả kế hoạch | Prior 2D đóng băng (SDS) cho tối ưu NeRF theo prompt |
| Điều kiện hóa | Quan sát (thị giác + cảm biến) | Inpainting start/goal; hướng dẫn bằng reward classifier | Văn bản (text-to-image) |
| Đầu ra / thực thi | Chuỗi hành động, receding horizon | Một kế hoạch quỹ đạo, thực thi open-loop | Tài sản 3D (NeRF), không phải hành vi robot |
| Kết quả tiêu biểu | +46.9% trung bình (15 task, 4 benchmark) | Score >100 trên Maze2D; 54.4 so với 8.1 (CQL) trên xếp khối | CLIP R-Precision vượt baselines trên 153 prompt; 58.5% (textureless) |
| Hạn chế chính | Trễ suy luận do nhiều bước khử nhiễu (DDIM chỉ giảm một phần); nhạy với hyperparameter action-horizon; dựa trên giả định position control | Chi phí denoise toàn quỹ đạo tại thời điểm quyết định; tăng tốc thô làm giảm chất lượng; thực thi open-loop, closed-loop replanning không được đề cập trong trích xuất | Tối ưu riêng từng prompt tốn kém; CLIP metric không phản ánh đầy đủ chất lượng hình học |
| Giao thức đánh giá | Success/score trên benchmark robot | Score trên Maze2D và xếp khối (đơn vị không nêu) | CLIP R-Precision không tham chiếu (%) trên COCO prompts |

**Tổng hợp**

- Không thể xếp hạng số: không có metric, dataset hay đơn vị nào dùng chung giữa ba bài; kết quả chỉ có ý nghĩa so sánh trong phạm vi từng bài và baseline của bài đó.
- Đa mode là lợi thế lặp lại: Diffusion Policy học cả hai mode trên nhiệm vụ đẩy đa mode và cam kết một mode mỗi rollout (trong khi LSTM-GMM/IBC thiên lệch và BET không cam kết); Diffuser giải được Maze2D thưởng thưa dài hạn, nơi ngay cả MPPI với động lực học ground-truth cũng kém — cho thấy nút thắt nằm ở tối ưu hóa chứ không phải dự đoán.
- Điều kiện hóa mang lại tính linh hoạt không cần huấn luyện lại: Diffuser giữ hiệu năng khi mục tiêu ngẫu nhiên hóa, DreamFusion tổng hợp 3D với không dữ liệu 3D huấn luyện — đổi lại là chi phí tại thời điểm chạy.
- Về kiểu điều khiển: khi chuyển từ velocity sang position control (Square, Kitchen), Diffusion Policy cải thiện trong khi BC-RNN và BET suy giảm, cho thấy diffusion khai thác tốt lợi thế của position control trong dự đoán chuỗi hành động.
- Giao thức đánh giá rời rạc (success/score robot so với CLIP R-Precision reference-free) là trở ngại lớn cho mọi so sánh trực tiếp; DreamFusion thêm vào không được đánh giá trên nhiệm vụ robot nào.

## 4. Kết luận

**Khuyến nghị thực tiễn cho nhà nghiên cứu & kỹ sư ML**

- Chọn mức trừu tượng theo bài toán: điều khiển thị giác–vận động thời gian thực → chính sách diffusion kiểu Diffusion Policy; hoạch định dài hạn, thưởng thưa, nhiều mục tiêu → quỹ đạo kiểu Diffuser với conditioning thay vì huấn luyện lại; tạo tài sản 3D/môi trường số → SDS kiểu DreamFusion. Không phương pháp nào tốt nhất trên mọi mặt.
- Lập kế hoạch cho chi phí suy luận: nhiều bước khử nhiễu là điểm nghẽn chung; cần đo latency thực tế, dùng các kỹ thuật giảm bước (như DDIM, chỉ giảm một phần), và lưu ý rằng tăng tốc thô làm suy giảm chất lượng hoạch định.
- Tinh chỉnh action-horizon cẩn thận: horizon lớn hơn tăng độ bền với độ trễ (0–7 bước được kiểm tra trong ablation) nhưng giảm một chút độ chính xác; đây là hyperparameter nhạy của Diffusion Policy.
- Kiểm tra giả định điều khiển: receding horizon dựa trên giả định position control và hành động tương lai dự đoán được; nếu plant hoặc policy phi tuyến mạnh, tính đa mode quay trở lại và bài toán khó hơn đáng kể.
- Với kế hoạch quỹ đạo open-loop, cân nhắc bổ sung cơ chế replanning bên ngoài mô hình, vì độ nhạy với sai lệch mô hình trong lúc thực thi chưa được xử lý trong nội dung trích xuất.
- Với tài sản 3D, không dùng CLIP R-Precision làm thước đo hình học duy nhất: texture phủ trên hình học phẳng có thể đạt điểm cao, và điểm albedo cao không đảm bảo hình học đúng (trường hợp hình chó nhiều đầu đạt điểm albedo cao nhất).
- Khi thiết kế thí nghiệm, báo cáo cả chất lượng lẫn thời gian suy luận để các nghiên cứu sau có cơ sở so sánh.

**Thách thức mở**

- Giảm số bước khử nhiễu tại suy luận mà không mất tính đa mode và chất lượng mẫu.
- Thực thi closed-loop cho hoạch định diffusion toàn quỹ đạo, cùng với việc xử lý độ nhạy đối với lỗi mô hình khi thực thi.
- Dự đoán hành động tương lai trong hệ phi tuyến, nơi tính đa mode tái xuất hiện và làm khó receding horizon.
- Thiếu protocol và metric chuẩn hóa cho phép so sánh các phương pháp diffusion ở các mức trừu tượng khác nhau; đánh giá 3D thiếu ground truth (Chamfer/PSNR không áp dụng được khi không có scene tham chiếu).
- Về xác thực nguồn: các hạn chế của Diffuser và DreamFusion nêu trên là suy luận từ nội dung trình bày do trích xuất không chứa mục hạn chế tường minh; cần đọc nguyên văn hai bài này để kiểm chứng trước khi dùng làm căn cứ thiết kế.

## 5. Tài liệu tham khảo

1. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** (Cheng Chi, Zhenjia Xu, Siyuan Feng et al., 2023-03-07) — [ArXiv abstract](https://arxiv.org/abs/2303.04137) | [PDF](https://arxiv.org/pdf/2303.04137.pdf) | [Mã nguồn](https://github.com/44David/diffusion-policy)
2. **Planning with Diffusion for Flexible Behavior Synthesis** (Michael Janner, Yilun Du, Joshua B. Tenenbaum et al., 2022-05-20) — [ArXiv abstract](https://arxiv.org/abs/2205.09991) | [PDF](https://arxiv.org/pdf/2205.09991.pdf)
3. **DreamFusion: Text-to-3D using 2D Diffusion** (Ben Poole, Ajay Jain, Jonathan T. Barron et al., 2022-09-29) — [ArXiv abstract](https://arxiv.org/abs/2209.14988) | [PDF](https://arxiv.org/pdf/2209.14988.pdf)
