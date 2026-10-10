# Research Report: các phương pháp diffusion cho robot

## 1. Tóm tắt

Báo cáo tổng hợp và so sánh ba công trình đại diện cho ba tầng ứng dụng khác nhau của mô hình diffusion trong hệ robot:

- **Diffusion Policy** (arXiv:2303.04137) biểu diễn chính sách visuomotor như một quá trình khử nhiễu có điều kiện (DDPM) trên không gian hành động: thay vì hồi quy trực tiếp một hành động, mạng $\epsilon_\theta$ học gradient score của phân phối hành động và tinh chỉnh dần một mẫu nhiễu Gaussian thành hành động, giúp xử lý tốt tính đa phương thức.
- **Diffuser** (arXiv:2205.09991) đặt diffusion trực tiếp trên toàn bộ trajectory (trạng thái – hành động) bằng một temporal U-Net, biến việc hoạch định thành sampling ngược: kế hoạch được "inpaint" từ nhiễu với trạng thái đầu và đích cố định.
- **DreamFusion** (arXiv:2209.14988) hoạt động ở tầng nội dung chứ không phải điều khiển: sinh tài sản 3D từ văn bản bằng cách chưng cất gradient (Score Distillation Sampling) từ một mô hình diffusion 2D đóng băng, không cần dữ liệu huấn luyện 3D.

Các phát hiện chính: Diffusion Policy vượt các baseline imitation learning trung bình 46,9% trên 15 nhiệm vụ thao tác; Diffuser đạt điểm trên 100 ở cả ba kích thước mê cung Maze2D và cho phép thay đổi mục tiêu lúc kiểm thử không cần huấn luyện lại; DreamFusion đạt 58,5% CLIP R-Precision trên render không texture, vượt các baseline được huấn luyện trực tiếp với CLIP về hình học.

Lưu ý về dữ liệu nguồn: phần tóm tắt ngắn của cả ba bài không có sẵn trong dữ liệu trích xuất, nên báo cáo này dựa trên các trích đoạn về phương pháp, kết quả chính và hạn chế; những nội dung không nằm trong trích xuất được coi là chưa kiểm chứng.

## 2. Phân tích bài báo

### Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (arXiv 2303.04137)

- **Vấn đề**: Trong học bắt chước cho điều khiển khớp kéo robot, từ một quan sát duy nhất có thể tồn tại nhiều hành vi hợp lệ cùng nhất quán với người thể nghiệm (ví dụ đẩy khối T quanh bên trái hoặc bên phải), tức là phân bố hành động đa chế độ — và các baseline hành vi-cloning hiện tại xử lý kém tình huống này. Hồi quy scalar tường minh trung bình hóa các chế độ; mô hình hỗn hợp (LSTM-GMM/BC-RNN) và chính sách ẩn dạng năng lượng (IBC) thiên lệch về một chế độ; Behavior Transformer (BET) thiếu tính nhất quán hành động theo thời gian nên dao động giật cục giữa các quỹ đạo hợp lệ trong cùng một rollout. Các chính sách ẩn biểu cảm hơn nhưng khó huấn luyện ổn định và khó lấy mẫu không gian hành động chiều cao với mặt năng lượng không trơn. Nút thắt thứ hai là chiều kích đầu ra và chế độ điều khiển: các phương pháp trước tránh dự đoán chuỗi hành động dài vì lấy mẫu đa chế độ chiều cao kiểu GMM/k-means (BC-RNN, BET) phải chỉ định trước số chế độ, nên chúng dựa vào điều khiển vận tốc một bước hoặc horizon ngắn — dễ mắc lỗi compounding; điều khiển vị trí ít mắc lỗi compounding và phù hợp dự đoán chuỗi hơn, nhưng đa chế độ của nó rõ nét hơn — đúng vùng mà các chính sách trước thất bại — nên ưu thế của nó vẫn chưa được khai thác.

- **Phương pháp**: Diffusion Policy biểu diễn chính sách thị giác-vận động như một quá trình khử nhiễu diffusion có điều kiện (DDPM) trên hành động: thay vì hồi quy trực tiếp, mạng dự đoán nhiễu $\epsilon_\theta$ học gradient của hàm điểm (score) của phân bố hành động. Tiến trình nhiễu hóa thuận làm nhiễu Gaussian chuỗi hành động $x_t$ qua $k$ bước:

  $$q(x_t^k \mid x_t^{k-1}) = \mathcal{N}\!\left(x_t^k;\ \sqrt{\alpha_k}\, x_t^{k-1},\ (1-\alpha_k) I\right)$$

  Mạng nhiễu được huấn luyện bằng mục tiêu khử nhiễu chuẩn, dự đoán nhiễu $\epsilon$ đã thêm vào mẫu tại bước $k$ dựa trên quan sát $o_t$:

  $$\mathcal{L} = \mathbb{E}_{k,\epsilon}\left[\left\lVert \epsilon - \epsilon_\theta(o_t, x_t^k, k) \right\rVert^2\right]$$

  Tại suy luận, một mẫu Gaussian ban đầu được tinh chỉnh lặp qua các bước khử nhiễu Langevin ngẫu nhiên có điều kiện trên quan sát:

  $$x_t^{k-1} = \frac{1}{\sqrt{\alpha_k}}\left(x_t^k - \frac{1-\alpha_k}{\sqrt{1-\bar{\alpha}_k}}\,\epsilon_\theta(o_t, x_t^k, k)\right) + \sigma_k z$$

  Hướng diffusion này xử lý mượt phân bố đa chế độ, mở rộng tốt sang đầu ra chiều cao mà vẫn giữ tính biểu cảm, và huấn luyện ổn định. Ba đóng góp để áp dụng vào điều khiển robot thật: (1) receding horizon control — dự đoán chuỗi hành động tương lai chiều cao, chỉ thực thi một phần rồi lập kế hoạch lại với quan sát mới, tạo tính nhất quán hành động theo thời gian và độ bền với các hành động nhàn rỗi/tạm dừng phổ biến trong teleoperation (loại bẫy các chính sách một bước như BC-RNN và IBC); (2) conditioning thị giác — ảnh camera và trạng thái robot được đưa vào qua bộ mã hóa quan sát bằng FiLM trong backbone temporal-convolution 1D (DiffusionPolicy-C) hoặc như token cross-attention trong diffusion transformer chuỗi thời gian (DiffusionPolicy-T), thay vì nối thô, giúp tách rời mã hóa quan sát khỏi khử nhiễu hành động; (3) kiến trúc time-series diffusion transformer. Chính sách vận hành ở chế độ điều khiển vị trí, nơi tính biểu cảm đa chế độ của diffusion khai thác được ưu thế của điều khiển vị trí mà tránh được nhược điểm thông thường của nó.

- **Kết quả chính**: Trên 15 nhiệm vụ thuộc 4 benchmark điều khiển khớp kéo (bao gồm mô phỏng 2D PushT, RoboMimic mô phỏng, và các nhiệm vụ teleoperation robot thật), Diffusion Policy liên tục vượt các phương pháp imitation learning tiên tiến trước đó, với cải thiện trung bình 46,9% so với các baseline như BC-RNN (LSTM-GMM), IBC và BET. Trên bài toán đẩy đa chế độ, rollout 40 bước cho thấy mô hình học được cả hai chế độ hành động và cam kết đúng một chế độ duy nhất mỗi rollout, trong khi LSTM-GMM và IBC thiên lệch về một chế độ và BET không cam kết được do thiếu nhất quán hành động theo thời gian. Các thí nghiệm ablation ủng hộ các giả thuyết thiết kế: chuyển từ điều khiển vận tốc sang điều khiển vị trí làm suy giảm BC-RNN và BET nhưng cải thiện Diffusion Policy (Hình 4, các nhiệm vụ Square/Kitchen/p4), chứng tỏ phương pháp khai thác lợi thế của điều khiển vị trí mà né được nhược điểm; quét action horizon từ 1 đến 128 bước đối chiếu với độ trễ quan sát 0–7 bước (Hình 5) cho thấy lợi ích của dự đoán chuỗi — nhất quán thời gian và độ bền với hành động nhàn rỗi — cùng một sự đánh đổi đo được giữa chiều dài horizon và độ bền trễ, được báo cáo bằng mức thay đổi tương đối của tỷ lệ thành công so với giá trị tối đa của từng nhiệm vụ.

- **Hạn chế**: Các đánh đổi được ghi nhận xoay quanh suy luận lặp và thiết kế horizon. Thứ nhất, sinh mỗi chunk hành động đòi hỏi nhiều bước khử nhiễu Langevin ngẫu nhiên với số lần lặp chỉ định trước và không có tiêu chí hội tụ tường minh, khiến suy luận chậm hơn các chính sách hồi quy một bước — mối lo thực tiễn cho điều khiển robot thật ở tần số cao. Thứ hai, lược đồ receding-horizon tạo ra sự đánh đổi giữa chiều dài action horizon và độ bền với độ trễ quan sát: phép quét ở Hình 5 cho thấy mức thay đổi tương đối của tỷ lệ thành công suy giảm khi độ trễ tăng, nên horizon phải được tinh chỉnh theo từng nhiệm vụ. Thứ ba, lợi thế so với điều khiển vận tốc gắn chặt với khả năng biểu diễn đa chế độ của diffusion trong chế độ điều khiển vị trí, nên mức cải thiện có thể thu hẹp ở các nhiệm vụ hoặc chế độ điều khiển ít đa chế độ hơn. Cần lưu ý rằng bản trích xuất nguồn không ghi nhận mục Hạn chế/Công việc tương lai riêng biệt của bài báo, do đó các điểm trên được suy ra từ bằng chứng ablation và phần thảo luận thiết kế; những phần không xuất hiện trong bản trích xuất được coi là chưa kiểm chứng.
- **Mã nguồn**: [44David/diffusion-policy](https://github.com/44David/diffusion-policy) (⭐ 0, Python)
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

### Planning with Diffusion for Flexible Behavior Synthesis (arXiv: 2205.09991)

- **Vấn đề**: Trong RL dựa trên mô hình truyền thống, việc ra quyết định bị tách thành hai khâu: học chỉ được dùng để xấp xỉ mô hình động lực học chưa biết (dưới dạng bài toán học có giám sát), phần còn lại được giao cho một bộ tối ưu hóa quỹ đạo cổ điển kiểu iLQG/MPC. Tác giả chỉ ra rằng tổ hợp này dễ gãy vỡ: bộ tối ưu hóa mạnh thường khai thác triệt để các sai lệch của mô hình đã học, khiến kế hoạch thu được giống adversarial examples hơn là quỹ đạo khả thi, gần tối ưu. Điểm nghẽn cũng không thuần túy nằm ở khâu mô hình hóa: MPPI chạy với động lực học ground-truth vẫn kém trên điều khiển phần thưởng thưa dài hạn, còn các thuật toán offline RL không dùng mô hình gặp khó khăn về credit assignment khi việc đạt đích đòi hỏi hàng trăm bước. Về tính linh hoạt, các agent offline model-free như IQL chỉ hoạt động tốt trong kịch bản một mục tiêu đã huấn luyện và suy giảm mạnh khi vị trí đích được ngẫu nhiên hóa; các phương pháp trước đó như BCQ và CQL đạt gần bằng 0 trên các biến thể có điều kiện (xếp khối theo thứ tự định trước, tái bố trí khớp cấu hình mới) — tức là không thích nghi được với cấu hình mục tiêu mới lúc kiểm thử nếu không huấn luyện lại. Câu hỏi nghiên cứu của bài báo là: có thể gói càng nhiều quy trình tối ưu hóa quỹ đạo vào bản thân bài toán mô hình hóa, sao cho "lấy mẫu từ mô hình" và "lập kế hoạch với mô hình" trở thành gần như cùng một thao tác.

- **Phương pháp**: Diffuser là một mô hình khuếch tán xác suất định nghĩa trực tiếp trên quỹ đạo (trạng thái, hành động). Tiến trình khuếch tán thuận làm nhiễu dần một quỹ đạo hoàn chỉnh:

  $$q(\tau^{i} \mid \tau^{i-1}) = \mathcal{N}\!\left(\tau^{i};\ \sqrt{1-\beta_i}\,\tau^{i-1},\ \beta_i\,\mathbf{I}\right)$$

  trong đó $\beta_i$ là lịch trình phương sai nhiễu tại bước $i$. Mô hình ngược học được — một U-Net theo thời gian xử lý chuỗi (trạng thái, hành động) như một ảnh trải theo trục thời gian — được huấn luyện để hoàn tác từng bước làm nhiễu:

  $$p_\theta(\tau^{i-1} \mid \tau^{i}) = \mathcal{N}\!\left(\tau^{i-1};\ \mu_\theta(\tau^{i}, i),\ \Sigma\right)$$

  Lập kế hoạch khi đó trở thành lấy mẫu kế tiếp (ancestral sampling): bắt đầu từ nhiễu thuần túy, mô hình lặp lại tinh chỉnh thành một quỹ đạo mạch lạc, qua đó xóa bỏ ranh giới giữa việc ước lượng mô hình và bộ tối ưu hóa bên ngoài. Hai công cụ sinh mẫu chuẩn được diễn giải lại thành chiến lược lập kế hoạch. Thứ nhất, lấy mẫu có hướng dẫn theo classifier: gradient của tín hiệu giá trị/thưởng học được dẫn các mẫu khử nhiễu về vùng có phần thưởng cao, thay thế việc cực tiểu hóa chi phí tường minh của bộ tối ưu quỹ đạo:

  $$\hat{p}(\tau^{i-1} \mid \tau^{i}, \mathcal{O}) \propto p_\theta(\tau^{i-1} \mid \tau^{i})\, p(\mathcal{O} \mid \tau^{i-1})$$

  với $\mathcal{O}$ là tập điều kiện (ví dụ: mục tiêu). Thứ hai, inpainting: một tập con phần tử quỹ đạo (trạng thái bắt đầu, trạng thái đích) bị kẹp cứng tại giá trị cho trước ở mọi bước khử nhiễu trong khi mô hình "điền" phần còn lại. Vì điều kiện hóa diễn ra ngay lúc lấy mẫu, một mô hình duy nhất đóng vai trò bộ lập kế hoạch đa nhiệm: đổi mục tiêu chỉ cần đổi điều kiện, và cùng cơ chế đó hỗ trợ các đặc tả linh hoạt lúc kiểm thử như thứ tự xếp khối hay cấu hình sắp xếp lại mong muốn.

- **Kết quả chính**: Trên Maze2D (D4RL; chỉ có phần thưởng 1 tại vị trí đích, các episode kéo dài hàng trăm bước), Diffuser lập kế hoạch bằng cách inpaint trạng thái bắt đầu và đích cố định rồi thực thi quỹ đạo được lấy mẫu ở chế độ vòng mở: đạt trên 100 điểm ở cả ba kích thước mê cung (U-Maze, Medium, Large), vượt chính sách expert tham chiếu và các baseline offline RL model-free mạnh nhất (IQL là mạnh nhất), trong khi MPPI với động lực học ground-truth cho kết quả kém — cho thấy lập kế hoạch dài hạn khó ngay cả khi không có sai số dự đoán. Ở biến thể đa nhiệm Multi2D, khi đích được ngẫu nhiên hóa mỗi episode, Diffuser hoạt động ngang bằng thiết lập đơn nhiệm mà không cần huấn luyện lại (chỉ đổi điều kiện đích), trong khi IQL với hindsight relabeling suy giảm đáng kể. Về tính linh hoạt lúc kiểm thử, trên bài toán xếp khối huấn luyện từ 10.000 quỹ đạo minh họa sinh bởi PDDLStream (phần thưởng thưa 1 khi xếp đặt thành công), Diffuser đạt 58.7 ± 2.5 điểm ở Unconditional Stacking, 45.6 ± 3.1 điểm ở Conditional Stacking (thứ tự khối định trước) và 58.9 ± 3.4 điểm ở Rearrangement (khớp khối theo cấu hình mới), trung bình 54.4; so sánh: CQL chỉ đạt 24.4 điểm ở nhiệm vụ không điều kiện và 0.0 ở cả hai nhiệm vụ có điều kiện (trung bình 8.1), còn BCQ bằng 0.0 trên mọi nhiệm vụ (trung bình 0.0) — Diffuser tổng quát hóa tới các cấu hình đích chưa xuất hiện trong huấn luyện, nơi các baseline model-free thất bại hoàn toàn. Bài báo còn phân tích khía cạnh thời gian chạy thực tế, chỉ ra các cách hiệu quả để tăng tốc lấy mẫu khuếch tán với mức suy giảm hiệu năng tối thiểu.

- **Hạn chế**: Ràng buộc được tài liệu hóa rõ nhất là chi phí tính toán: khử nhiễu lặp khiến lập kế hoạch tốn kém tại thời điểm ra quyết định, tạo ra sự đánh đổi vốn có giữa chi phí tính toán và chất lượng kế hoạch; chính vì vậy tác giả dành phần phân tích cho các cách tăng tốc (ví dụ: giảm số bước khử nhiễu) với thiệt hại hiệu năng tối thiểu. Trong đánh giá Maze2D, quỹ đạo được lấy mẫu chỉ dùng thuần túy như một kế hoạch vòng mở, nên độ bền vững trước sai số cộng dồn khi thực thi vòng kín kiểu receding-horizon chưa được kiểm chứng trong các kết quả trình bày. Chất lượng lập kế hoạch còn phụ thuộc vào tín hiệu hướng dẫn (mô hình giá trị/thưởng) đã học, và việc đặc tả nhiệm vụ linh hoạt đòi hỏi biến mục tiêu/ràng buộc thành các phần tử quỹ đạo có thể inpaint — cơ chế này có thể không bao phủ được mọi loại ràng buộc tùy ý. Cuối cùng, khả năng khai thác dữ liệu không đồng nhất chỉ được minh họa trên các bộ điều khiển được nghiên cứu; phần tổng quát hóa liên miền rộng hơn chưa được kiểm chứng trong kết quả được trình bày.
- **Mã nguồn**: Không tìm thấy repository được xác minh trong lần chạy này; đừng kết luận bài báo không có code.
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

- **Vấn đề**: Việc sinh 3D từ văn bản bị chặn bởi cả dữ liệu lẫn kiến trúc: các đột phá về sinh ảnh từ văn bản đến từ mô hình diffusion huấn luyện trên hàng tỷ cặp ảnh–văn bản, trong khi một hướng tiếp cận 3D tương đương đòi hỏi bộ dữ liệu 3D gán nhãn quy mô lớn và kiến trúc khử nhiễu dữ liệu 3D hiệu quả — cả hai đều chưa tồn tại. Các baseline zero-shot trước đó (Dream Fields, CLIP-Mesh) dẫn dắt biểu diễn 3D bằng mô hình phân biệt ảnh–văn bản CLIP thay vì một prior sinh; vì CLIP chỉ được huấn luyện để so khớp tương phản caption–ảnh, nó có thể bị khai thác bằng các rendering đối kháng, phi vật lý, dẫn tới tài sản có hình học phẳng, texture chỉ vẽ trên bề mặt và artefact như vật thể nhiều đầu. Thách thức thứ hai là đánh giá: với sinh zero-shot từ prompt tùy ý, không có "cảnh 3D đúng" để đối chiếu, nên các metric chuẩn như Chamfer Distance và PSNR không áp dụng được; các metric dựa trên CLIP lại thường cho điểm cao ngay cả khi texture được vẽ lên hình học phẳng, không phản ánh chất lượng hình học.

- **Phương pháp**: DreamFusion dùng một mô hình diffusion 2D văn bản→ảnh đã huấn luyện sẵn và đóng băng làm prior để tối ưu một bộ sinh ảnh tham số, không cần dữ liệu huấn luyện 3D và không chỉnh sửa mô hình diffusion. Hàm mất mát dựa trên chưng cất mật độ xác suất, gọi là Score Distillation Sampling (SDS): rendering $x$ của một Neural Radiance Field (NeRF) khởi tạo ngẫu nhiên được làm nhiễu thành

  $$x_t = \alpha_t\, x + \sigma_t\, \varepsilon,$$

  với $\varepsilon$ là nhiễu ngẫu nhiên; điểm khử nhiễu $\hat{\varepsilon}_\phi$ của mô hình diffusion được đánh giá tại $x_t$ theo điều kiện prompt $y$, cho hàm mất mát

  $$\mathcal{L}_{\mathrm{SDS}}(\theta) = \mathbb{E}_{t,\varepsilon}\!\left[\, w(t)\,\big\| \hat{\varepsilon}_\phi(x_t;\, t,\, y) - \varepsilon \big\|^2 \right],$$

  và gradient thu được được lan truyền ngược qua một volume renderer khả vi vào tham số $\theta$ của NeRF:

  $$\nabla_\theta \mathcal{L}_{\mathrm{SDS}}(\theta) \approx \mathbb{E}_{t,\varepsilon}\!\left[\, w(t)\,\big(\hat{\varepsilon}_\phi(x_t;\, t,\, y) - \varepsilon\big)\,\frac{\partial x}{\partial \theta} \right].$$

  Quá trình tối ưu diễn ra theo kiểu DeepDream: NeRF được cập nhật bằng gradient descent sao cho các rendering 2D từ góc camera ngẫu nhiên đạt loss chưng cất thấp, tạo ra mô hình 3D xem được từ mọi góc, chiếu sáng lại được bằng nguồn sáng tùy ý và ghép được vào môi trường 3D bất kỳ. Để hình học chính xác, bài_report xếp chồng bốn cải tiến tối ưu: (i) ViewAug — tăng cường trên phạm vi góc nhìn rộng; (ii) ViewDep — condition trên prompt phụ thuộc góc nhìn ("viewed from the front/back/side"); (iii) Lighting — tối ưu thêm rendering có đèn điểm và shading Lambertian bên cạnh rendering màu albedo không chiếu sáng; (iv) Textureless — tối ưu thêm ảnh hình học không texture (bản đồ pháp tuyến) bên cạnh rendering có texture.

- **Kết quả chính**: Đánh giá bằng CLIP R-Precision (theo Dream Fields) trên 153 prompt hướng vật thể của tập con COCO validation — độ chính xác truy hồi (%) mà CLIP tìm đúng caption trong một tập caption gây nhiễu khi được xem rendering của cảnh. Baseline gồm Dream Fields, CLIP-Mesh, một bản tái triển khai Dream Fields nâng cấp với biểu diễn 3D của DreamFusion, và một oracle đánh giá trực tiếp các cặp ảnh–caption MS-COCO gốc. Trên ảnh màu, DreamFusion vượt cả hai baseline — dù chúng có lợi thế không công bằng là dùng CLIP ngay trong huấn luyện — và tiệm cận hiệu năng của ảnh ground truth. Trên rendering không texture dùng để thăm dò hình học, bản tái triển khai Dream Fields gần như ngẫu nhiên, trong khi hình học của DreamFusion khớp caption 58,5% số lần. Ablation cộng dồn ViewAug → ViewDep → Lighting → Textureless cho thấy hình học cải thiện ở từng bước và đạt mức +12,5% trên full renderings; đáng chú ý, mô hình albedo-only đạt R-Precision cao nhất trên albedo renders nhưng hình học kém (con chó nhiều đầu), chứng minh metric dựa trên ảnh có texture/albedo gây hiểu nhầm. Định tính, phương pháp sinh các cảnh 3D mạch lạc, có tính tổ hợp ("a hamburger", "a vase with pink flowers", lâu đài bánh phô mai với hào bằng kem) vượt Dream Fields và CLIP-Mesh; thư viện tài sản 3D, video mở rộng và mesh được công bố trên trang dự án dreamfusion3d.github.io.

- **Hạn chế**: Về đánh giá, do không có ground truth cho sinh zero-shot, Chamfer Distance và PSNR không dùng được, và bằng chứng ablation ở trên cho thấy ngay cả CLIP R-Precision cũng thiếu tin cậy, nên chất lượng hình học phải được thăm dò riêng qua rendering không texture. Về chi phí, phương pháp tối ưu một NeRF hoàn chỉnh cho từng prompt bằng gradient descent thay vì sinh feed-forward amortized, tốn cỡ hàng giờ thời gian bộ tăng tốc cho mỗi tài sản. Hình học chính xác còn phụ thuộc vào việc xếp chồng nhiều heuristic (tăng cường góc nhìn, prompt theo góc nhìn, chiếu sáng, shading không texture); thiếu một thành phần sẽ tạo artefact như bộ phận vật thể bị nhân đôi. Về lý thuyết, mục tiêu chưng cất mật độ xác suất mang tính mode-seeking: bộ sinh chỉ cần khớp các mode có likelihood cao của prior 2D thay vì toàn bộ tính đa dạng của nó, nên chất lượng lẫn đa dạng bị chặn trên bởi prior 2D đóng băng; tác giả chỉ hướng tương lai gồm sinh nhanh hơn/amortized và metric đánh giá nhận biết hình học.
- **Mã nguồn**: Không tìm thấy repository được xác minh trong lần chạy này; đừng kết luận bài báo không có code.
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

| Khía cạnh | Diffusion Policy | Diffuser | DreamFusion |
|---|---|---|---|
| Mức áp dụng | Sinh hành động cho chính sách điều khiển | Hoạch định trên trajectory | Sinh tài sản 3D (không phải điều khiển) |
| Đại lượng khuếch tán | Hành động, có điều kiện theo quan sát | Chuỗi trạng thái–hành động (như ảnh theo trục thời gian) | Gradient SDS chưng cất từ prior 2D đóng băng |
| Điều kiện hóa | Quan sát hiện tại, thực thi receding-horizon | Inpaint trạng thái đầu–đích; goal đổi được lúc kiểm thử | Prompt văn bản, thông qua mô hình 2D |
| Suy luận | Nhiều bước Langevin khử nhiễu | Nhiều bước khử nhiễu tại thời điểm hoạch định | Tối ưu gradient một NeRF cho mỗi prompt |
| Điểm mạnh chính | Xử lý đa phương thức; hưởng lợi từ điều khiển vị trí | Hoạch định dài hạn vượt expert; đa nhiệm vụ không cần huấn luyện lại | Không cần dữ liệu 3D; hình học nhất quán với prompt |
| Hạn chế chính | Chậm hơn hồi quy một bước; horizon phải tinh chỉnh | Chi phí tính toán; kế hoạch open-loop; phụ thuộc guidance | Metric thiếu tin cậy; chi phí tối ưu mỗi prompt |

**Kết quả định lượng (nêu một lần với dataset và đơn vị):**

- Diffusion Policy: cải thiện tương đối trung bình **46,9%** so với BC-RNN, IBC và BET trên 15 nhiệm vụ thuộc 4 benchmark thao tác (2D PushT, RoboMimic, teleoperation robot thật). Trên bài toán đẩy đa phương thức (rollout 40 bước), chính sách học được nhiều mode hành động nhưng cam kết đúng một mode mỗi rollout, trong khi LSTM-GMM và IBC thiên lệch về một mode còn BET không cam kết do thiếu tính nhất quán thời gian của hành động.
- Diffuser: **điểm số (score) trên 100** ở cả ba kích thước mê cung (U-Maze, Medium, Large) của Maze2D (D4RL), vượt chính sách expert tham chiếu và các baseline offline RL model-free (IQL mạnh nhất), trong khi MPPI với động lực học ground-truth cũng yếu. Trên bài toán xếp khối với 10.000 demonstra PDDLStream: **58,7 ± 2,5** (Unconditional Stacking) và **45,6 ± 3,1** (Conditional Stacking). Trong biến thể đa nhiệm vụ Multi2D, hiệu năng giữ nguyên khi goal đổi mỗi episode mà không huấn luyện lại, trong khi IQL với hindsight relabeling giảm mạnh.
- DreamFusion: **CLIP R-Precision 58,5%** (đơn vị %) trên render không texture đo hình học, qua 153 prompt thuộc tập COCO validation object-centric của Dream Fields; baseline Dream Fields được cài lại gần mức ngẫu nhiên. Trên ảnh màu, DreamFusion vượt cả Dream Fields lẫn CLIP-Mesh dù hai baseline này dùng CLIP trong huấn luyện.

**Về khả năng so sánh chéo:** các số liệu trên **không thể so sánh trực tiếp** giữa ba bài — dataset khác nhau (benchmark thao tác so với Maze2D/D4RL so với prompt COCO), metric khác nhau (tỉ lệ thành công/cải thiện tương đối so với điểm hoạch định so với CLIP R-Precision) và giao thức đánh giá khác nhau. Ma trận benchmark đi kèm ghi nhận mọi dòng metric là "không so sánh trực tiếp được", do đó báo cáo này **không xếp hạng** các phương pháp theo số liệu.

**Đánh đổi chung và riêng:**

- Chi phí suy luận lặp là điểm chung: cả ba đều cần nhiều bước khử nhiễu hoặc tối ưu lặp. Diffusion Policy chậm hơn chính sách hồi quy một bước; Diffuser phân tích riêng việc giảm số bước khử nhiễu với tổn thất chất lượng tối thiểu (đánh đổi tính toán – chất lượng nội tại); DreamFusion tốn chi phí cấp hàng giờ cho mỗi prompt thay vì sinh feed-forward.
- Diffusion Policy: số bước khử nhiễu được đặt trước, không có tiêu chí hội tụ tường minh; scheme receding-horizon tạo đánh đổi giữa độ dài horizon hành động (quét 1–128 bước) và độ bền với độ trễ quan sát (0–7 bước), nên horizon phải tinh chỉnh theo từng nhiệm vụ. Đáng chú ý, chuyển từ điều khiển vận tốc sang vị trí làm giảm BC-RNN và BET nhưng lại cải thiện Diffusion Policy.
- Diffuser: trong Maze2D, trajectory lấy mẫu được dùng thuần túy làm kế hoạch open-loop nên độ bền trước lỗi tích lũy khi thực thi closed-loop chưa được kiểm chứng; chất lượng hoạch định phụ thuộc tín hiệu guidance (value/reward) đã học.
- DreamFusion: zero-shot text-to-3D không có ground truth, khiến các metric tham chiếu không áp dụng được; CLIP R-Precision có thể cao ngay cả khi hình học phẳng chỉ được sơn texture, nên chất lượng hình học phải kiểm tra riêng bằng render không texture.

## 4. Kết luận

**Khuyến nghị thực tiễn:**

- Chọn phương pháp theo tầng của hệ thống, không theo "điểm số chung": diffusion trên hành động (Diffusion Policy) phù hợp cho chính sách imitation đa phương thức trong thao tác; diffusion trên trajectory (Diffuser) phù hợp cho hoạch định dài hạn, đa nhiệm vụ và thay đổi mục tiêu lúc kiểm thử; SDS (DreamFusion) phù hợp làm công cụ sinh tài sản 3D khi thiếu dữ liệu 3D, với vai trò gián tiếp đối với pipeline robot.
- Dự trù ngân sách tính toán cho suy luận lặp: với điều khiển tần số cao trên robot thật, cần đo tốc độ thực tế và cân nhắc giảm số bước khử nhiễu (theo phân tích của Diffuser) chấp nhận tổn thất chất lượng nhỏ.
- Tinh chỉnh horizon hành động theo từng nhiệm vụ, vì độ dài horizon đánh đổi trực tiếp với độ bền trước độ trễ quan sát (Diffusion Policy).
- Kiểm chứng cách thực thi: kế hoạch diffusion open-loop (Diffuser trong Maze2D) cần được đánh giá thêm trong chế độ closed-loop trước khi suy rộng sang robot thật.
- Cẩn trọng với metric: với sinh 3D, không dựa chỉ vào CLIP R-Precision mà phải kiểm tra hình học bằng render không texture; với hoạch định, báo cáo cả khả năng đa nhiệm vụ khi đổi goal.
- Không so sánh hay xếp hạng các phương pháp này qua số liệu chéo bài, vì dataset, metric và đơn vị không tương thích.

**Thách thức mở:**

- Tiêu chí dừng và tốc độ cho quá trình khử nhiễu lặp trong điều khiển thời gian thực, khi số bước hiện được đặt trước mà không có điều kiện hội tụ tường minh.
- Độ bền của kế hoạch diffusion khi thực thi closed-loop: lỗi tích lũy, độ trễ cảm biến và tương tác động chưa được kiểm chứng đầy đủ trong các kết quả đã trích xuất.
- Thiết kế tín hiệu guidance (value/reward) cho hoạch định diffusion — chất lượng kế hoạch phụ thuộc trực tiếp vào nó.
- Metric đánh giá đáng tin cậy cho sinh nội dung 3D khi không có ground truth.
- Thiếu benchmark chuẩn hóa dùng chung cho các tầng ứng dụng khác nhau của diffusion trong robot; cho đến khi có benchmark như vậy, so sánh định lượng chéo phương pháp chỉ có thể dừng ở mức định tính như trong báo cáo này.

## 5. Tài liệu tham khảo

1. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** (Cheng Chi, Zhenjia Xu, Siyuan Feng et al., 2023-03-07) — [ArXiv abstract](https://arxiv.org/abs/2303.04137) | [PDF](https://arxiv.org/pdf/2303.04137.pdf) | [Mã nguồn](https://github.com/44David/diffusion-policy)
2. **Planning with Diffusion for Flexible Behavior Synthesis** (Michael Janner, Yilun Du, Joshua B. Tenenbaum et al., 2022-05-20) — [ArXiv abstract](https://arxiv.org/abs/2205.09991) | [PDF](https://arxiv.org/pdf/2205.09991.pdf)
3. **DreamFusion: Text-to-3D using 2D Diffusion** (Ben Poole, Ajay Jain, Jonathan T. Barron et al., 2022-09-29) — [ArXiv abstract](https://arxiv.org/abs/2209.14988) | [PDF](https://arxiv.org/pdf/2209.14988.pdf)
