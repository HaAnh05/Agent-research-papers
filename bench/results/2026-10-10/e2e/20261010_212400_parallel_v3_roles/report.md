# Research Report: các phương pháp diffusion cho robot

## 1. Tóm tắt

Báo cáo so sánh ba công trình áp dụng mô hình khuếch tán (diffusion) vào robot và các bài toán lân cận, tất cả cùng dựa trên một cơ chế chung: khử nhiễu lặp (iterative denoising), tức tinh chỉnh dần nhiễu ngẫu nhiên thành đầu ra có cấu trúc. Điểm khác biệt cốt lõi nằm ở không gian đầu ra được khử nhiễu.

- *Diffusion Policy* (arXiv 2303.04137) biểu diễn chính sách thị giác–vận động như quá trình khử nhiễu có điều kiện trên chuỗi hành động cao chiều, kèm receding horizon control. Theo phần kết quả được trích, phương pháp đạt mức cải thiện trung bình 46,9% trên 15 tác vụ thuộc 4 benchmark thao tác tay máy, vượt LSTM-GMM, IBC và BET nhờ nắm được tính đa phương thức của phân phối hành động và cam kết một mode trong mỗi lần thực thi.
- *Diffuser* (arXiv 2205.09991) đặt mô hình khuếch tán trên toàn bộ quỹ đạo trạng thái, biến lập kế hoạch thành lấy mẫu có điều kiện bằng classifier guidance và inpainting. Trên Maze2D, mô hình đạt điểm chuẩn hóa trên 100 ở cả ba kích thước mê cung và dùng lại một mô hình duy nhất cho nhiều goal ngẫu nhiên mà không cần huấn luyện lại.
- *DreamFusion* (arXiv 2209.14988) không điều khiển robot mà sinh nội dung 3D từ văn bản bằng cách chưng cất prior khuếch tán 2D vào một NeRF, đạt 58,5% CLIP R-Precision trên ảnh không kết cấu (153 prompt COCO). Có thể hữu ích như nguồn tài sản 3D cho mô phỏng, nhưng chưa được đánh giá trên tác vụ robot trong các trích đoạn.

Trade-off xuyên suốt là chi phí suy luận của khử nhiễu lặp, độ nhạy với độ trễ quan sát, và sự phụ thuộc vào điều kiện hóa lúc kiểm thử. Do ba bài dùng dataset, metric và đơn vị khác nhau, không có so sánh số trực tiếp giữa chúng.

## 2. Phân tích bài báo

### Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (arXiv: 2303.04137)

- **Vấn đề**: Chính sách thị giác–động cơ của robot phải ánh xạ quan sát hình ảnh thành hành động, nhưng các biểu diễn behavior-cloning tiêu chuẩn bộc lộ nhiều hạn chế: chính sách tường minh (hồi quy vô hướng, hỗn hợp Gaussian, hành động phân loại) khó biểu diễn phân bố hành động đa modal ở chiều cao; chính sách ngầm kiểu IBC học một hàm năng lượng có landscape không trơn, khiến việc sampling ở chiều cao trở nên khó khăn; còn các phương pháp dự đoán trình tự buộc phải khai báo trước số mode phân bố (GMM trong BC-RNN, k-means trong BET). Chính sách một bước còn dễ overfit vào các hành động "đứng yên" trong dữ liệu demo và cho ra hành vi đa modal giật cục, thiếu nhất quán theo thời gian. Đây là khoảng trống so với các baseline BC hiện hành mà việc đại diện chính sách bằng diffusion nhằm giải quyết.

- **Phương pháp**: Diffusion Policy biểu diễn chính sách như một quá trình khử nhiễu diffusion có điều kiện: mạng học gradient của hàm score của phân bố hành động, và tại thời gian suy luận tinh chỉnh dần từ nhiễu thành hành động qua chuỗi các bước Langevin dynamics ngẫu nhiên; hành động được dự đoán dưới dạng chuỗi chiều cao thay vì từng bước đơn, giúp DDPM giữ trọn tính biểu đạt và tạo ra hành động nhất quán theo thời gian, bền với các hành động idle. Ba đóng góp kỹ thuật chính cho robot vật lý gồm: (1) receding horizon control — chỉ thực thi một phần horizon hành động dự đoán, cho phép dùng điều khiển vị trí với ít sai số tích lũy hơn; (2) điều kiện hoá quá trình khử nhiễu bằng quan sát hình ảnh; (3) hai biến thể kiến trúc: diffusion transformer theo trình tự thời gian (DiffusionPolicy-T) và biến thể CNN (DiffusionPolicy-C). Cách tiếp cận này tránh được yêu cầu chỉ định số mode như GMM/k-means. Cơ chế nhiễu thuận làm nhiễu chuỗi hành động gốc $A_t^0$ theo mức nhiễu $k$:

$$q(A_t^k \mid A_t^0)=\mathcal{N}\!\left(\sqrt{\bar{\alpha}_k}\,A_t^0,\;(1-\bar{\alpha}_k)\,\mathbf{I}\right)$$

trong đó $A_t$ là chuỗi hành động dự đoán và $k$ chạy từ $K$ về $0$. Mạng được huấn luyện để dự đoán nhiễu $\epsilon^k$ từ chuỗi hành động đã nhiễu, mức nhiễu và quan sát $O_t$:

$$\mathcal{L}=\mathrm{MSE}\!\left[\epsilon^k-\epsilon_\theta\!\left(A_t^k,\,k,\,O_t\right)\right]$$

Tại thời gian suy luận, hành động được sinh bằng các bước khử ngược có điều kiện quan sát:

$$A_t^{k-1}=\frac{1}{\sqrt{\alpha_k}}\left(A_t^k-\frac{1-\alpha_k}{\sqrt{1-\bar{\alpha}_k}}\,\epsilon_\theta(A_t^k,\,k,\,O_t)\right)+\sigma_k z$$

- **Kết quả chính**: Trên tổng cộng 15 tác vụ thuộc 4 benchmark điều khiển tay máy, Diffusion Policy liên tục vượt các phương pháp học robot tốt nhất hiện nay với cải thiện trung bình 46.9% về tỷ lệ thành công. Trên các tác vụ đẩy đa modal (đẩy khối T sang trái hoặc phải), Diffusion Policy học được cả hai mode và cam kết một mode trong mỗi rollout, trong khi LSTM-GMM và IBC thiên lệch về một mode còn BET không cam kết được do thiếu nhất quán hành động theo thời gian. Khi chuyển từ điều khiển vận tốc sang điều khiển vị trí trên các tác vụ Square và Kitchen p4, BC-RNN và BET giảm hiệu năng trong khi cả DiffusionPolicy-C lẫn DiffusionPolicy-T đều cải thiện, tận dụng được lợi thế của điều khiển vị trí. Các ablation trên PushT và Square đặc trưng hoá đánh đổi theo action-horizon (biến đổi tương đối của tỷ lệ thành công trên các horizon từ 1 đến 128 bước) cùng độ bền của mô hình trước trễ quan sát.

- **Hạn chế**: Đánh đổi lớn nhất là độ trễ suy luận do khử nhiễu lặp, tương tác với trễ quan sát: ablation cho thấy biến đổi tương đối của tỷ lệ thành công giảm tới khoảng -0.6 khi trễ quan sát tăng lên 7 bước — vấn đề mà receding horizon control được thiết kế để giảm nhẹ; việc chọn action-horizon cũng là sự cân bằng giữa tính phản ứng nhanh và nhất quán thời gian. Tuyên bố rằng điều khiển vị trí phù hợp với Diffusion Policy hơn điều khiển vận tốc mang tính suy đoán một phần (quy về tính đa modal hành động nổi bật hơn và sai số tích lũy giảm). Lợi ích của phương pháp phụ thuộc vào khả năng scale của DDPM theo chiều dữ liệu đầu ra. Phần tài liệu nguồn không có mục hạn chế tường minh, nên các điểm trên là các trade-off được ghi nhận trong bài chứ không phải danh sách hạn chế chính thức của tác giả.
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

### Planning with Diffusion for Flexible Behavior Synthesis (arXiv 2205.09991)

- **Vấn đề**: Trong model-based RL, mô hình động lực học thường được học như một bài toán supervised rồi giao việc ra quyết định cho bộ tối ưu hóa quỹ đạo cổ điển; kết hợp này thất bại trong thực nghiệm vì optimizer mạnh khai thác mô hình đã học, sinh ra kế hoạch giống adversarial examples hơn là quỹ đạo tối ưu. Ngược lại, các thuật toán model-free yếu ở task dài hạn, reward thưa: trong Maze2D, agent cần hàng trăm bước để tới goal nhưng chỉ nhận reward tại goal, khiến credit assignment không đáng tin. Các planner hiện có cũng thiếu linh hoạt test-time: điều chỉnh controller đơn-nhiệm vụ (IQL) sang goal ngẫu nhiên đòi hỏi huấn luyện lại với hindsight relabeling và làm giảm hiệu năng đáng kể, trong khi MPPI với dynamics ground-truth cũng lập kế hoạch kém ở horizon dài.

- **Phương pháp**: Diffuser là một mô hình diffusion xác suất trên toàn bộ quỹ đạo $\tau$, gộp việc tối ưu hóa quỹ đạo vào bản thân bài toán mô hình hóa, nên sampling từ mô hình và planning gần như trùng nhau. Tiến trình khuếch tán xuôi dần phá hủy quỹ đạo bằng nhiễu:

  $$q(\tau^i \mid \tau^{i-1}) = \mathcal{N}\!\left(\tau^i;\ \sqrt{1-\beta_i}\,\tau^{i-1},\ \beta_i I\right)$$

  còn tiến trình ngược học được tinh chỉnh lặp lại từ nhiễu thành quỹ đạo mạch lạc:

  $$p_\theta(\tau^{i-1} \mid \tau^i) = \mathcal{N}\!\left(\tau^{i-1};\ \mu_\theta(\tau^i, i),\ \Sigma_\theta(\tau^i, i)\right)$$

  Điều khiển tại test-time tái diễn giải hai kỹ thuật từ miền ảnh: (1) classifier guidance, thiên hướng quá trình khử nhiễu về mục tiêu mong muốn (ví dụ reward) bằng cách lấy mẫu từ phân bố tỉ lệ với tiến trình ngược nhân với hàm mũ của reward,

  $$\hat{p}_\theta(\tau^{i-1} \mid \tau^i) \propto p_\theta(\tau^{i-1} \mid \tau^i)\, e^{r(\tau^{i-1})}$$

  và (2) inpainting, kẹp cứng các trạng thái chỉ định (điểm bắt đầu và vị trí goal) trong khi khử nhiễu lấp đầy phần còn lại của quỹ đạo. Vì goal được đưa vào như điều kiện trên cùng một mô hình, Diffuser đa nhiệm vụ một cách tự nhiên: đổi task chỉ là đổi conditioning mà không cần huấn luyện lại; quỹ đạo mẫu được thực thi open-loop trong các thí nghiệm maze.

- **Kết quả chính**: Trên Maze2D (U-Maze, Medium, Large), Diffuser dùng inpainting để điều kiện hóa start/goal đạt điểm trên 100 (thang điểm chuẩn hóa) ở mọi kích thước maze, vượt cả chính sách expert tham chiếu, trong khi IQL — baseline model-free tốt nhất — gặp khó khăn với credit assignment qua hàng trăm bước. Ở biến thể Multi2D với goal ngẫu nhiên, Diffuser hoạt động ngang bằng thiết lập đơn nhiệm mà không cần huấn luyện lại (chỉ thay đổi goal điều kiện), trong khi IQL thích nghi bằng hindsight relabeling giảm hiệu năng đáng kể; MPPI với dynamics ground-truth cũng hoạt động kém, cho thấy khó khăn của planning dài hạn ngay cả khi không có sai số dự đoán. Trên bài toán xếp khối (10.000 quỹ đạo demonstration PDDLStream; reward bằng 1 khi đặt khối thành công), Diffuser đạt 58.7±2.5 (xếp khối vô điều kiện), 45.6±3.1 (xếp khối có điều kiện) và 58.9±3.4 (sắp xếp lại), trung bình 54.4, so với CQL ở 24.4/0.0/0.0 (trung bình 8.1) và BCQ ở 0.0 trên mọi task.

- **Hạn chế**: Chi phí thời gian chạy của khử nhiễu lặp là ràng buộc được ghi nhận, thúc đẩy phần nghiên cứu về các yếu tố runtime thực tế và tăng tốc planning với mức mất mát hiệu năng tối thiểu. Hiệu quả phụ thuộc vào conditioning test-time (guidance và inpainting) và vào việc goal nhận diện được trong dữ liệu (trong Maze2D, goal là trạng thái duy nhất có reward khác 0); các kế hoạch maze được thực thi open-loop. Vẫn còn khoảng cách hiệu năng (ví dụ 45.6 trên xếp khối có điều kiện) và đánh giá giới hạn ở navigation Maze2D cùng các bài toán chẩn đoán xếp khối; khả năng phục hồi từ dữ liệu không đồng nhất được liệt kê trong bài nhưng không được trình bày chi tiết trong trích đoạn. Lưu ý cho người đọc: trích đoạn nguồn không chứa mục limitations riêng, nên các hạn chế trên được tổng hợp từ các lưu ý rải rác trong bài báo thay vì một đánh giá hạn chế chính thức của tác giả.
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

### DreamFusion: Text-to-3D using 2D Diffusion (arXiv:2209.14988)

- **Vấn đề**: Chuyển các đột phá của text-to-image diffusion sang tổng hợp 3D đòi hỏi bộ dữ liệu 3D gán nhãn quy mô lớn và kiến trúc denoising hiệu quả cho dữ liệu 3D — cả hai đều chưa tồn tại. Các bộ sinh zero-shot text-to-3D trước đó (Dream Fields, CLIP-Mesh) tối ưu bằng CLIP nên có thể đạt điểm cao trong khi hình học kém, ví dụ vẽ kết cấu lên các hình phẳng. Bên cạnh đó, các chỉ số dựa trên tham chiếu như Chamfer Distance hay PSNR khó áp dụng vì không tồn tại "cảnh 3D đúng" tương ứng với một prompt văn bản, còn các chỉ số tự động hiện có không đo được chất lượng hình học.

- **Phương pháp**: DreamFusion tối ưu một NeRF khởi tạo ngẫu nhiên bằng gradient descent theo kiểu DeepDream, dùng một mô hình diffusion text-to-image 2D đã huấn luyện trước làm prior — không cần dữ liệu 3D và không sửa đổi mô hình diffusion. Cơ chế cốt lõi là chưng cất mật độ xác suất: NeRF được render từ các góc camera ngẫu nhiên, và gradient để các ảnh 2D này "khớp" prior có điều kiện văn bản được xấp xỉ bởi

$$\nabla_\theta \mathcal{L}(\theta, x(\theta)) \triangleq \mathbb{E}_{t,\epsilon}\left[\, w(t)\left(\hat{\epsilon}_\phi(x_t;\, t, y) - \epsilon\right)\frac{\partial x}{\partial \theta}\right]$$

trong đó $\theta$ là tham số NeRF, $y$ là prompt, và thay mẫu nhiễu $\epsilon$ bằng dự đoán nhiễu $\hat{\epsilon}_\phi$ của mô hình diffusion giúp tránh phải tính Jacobian của U-Net. Ảnh nhiễu được tạo bằng quá trình forward

$$x_t = \alpha_t\, x + \sigma_t\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

và dự đoán nhiễu liên hệ với score của phân phối có điều kiện qua

$$\nabla_{x_t}\log p_\phi(x_t \mid y) \approx -\hat{\epsilon}_\phi(x_t;\, t, y)/\sigma_t$$

Bài báo chỉ ra các lựa chọn tối ưu hóa then chốt cho hình học chính xác: đa dạng hóa góc nhìn (ViewAug), prompt phụ thuộc góc nhìn (ViewDep), tối ưu cả ảnh có chiếu sáng bên cạnh ảnh albedo (Lighting), và tối ưu ảnh hình học không kết cấu (Textureless). Mô hình 3D thu được có thể quan sát từ mọi góc, thắp sáng lại và ghép vào môi trường 3D tùy ý.

- **Kết quả chính**: Đánh giá bằng CLIP R-Precision (%) trên 153 prompt của tập con object-centric trong COCO validation của Dream Fields; baseline gồm Dream Fields, CLIP-Mesh, một bản tái hiện Dream Fields nâng cấp dùng biểu diễn 3D của DreamFusion, và một oracle trên ảnh MS-COCO có chú thích gốc. Mặc dù các baseline CLIP có lợi thế dùng CLIP ngay trong huấn luyện, DreamFusion vẫn vượt cả hai trên ảnh màu và tiệm cận hiệu năng oracle. Trên ảnh không kết cấu (thăm dò hình học), DreamFusion nhất quán với caption 58.5% số lần trong khi bản tái hiện Dream Fields gần bằng mức ngẫu nhiên. Ablation cộng dồn ViewAug → ViewDep → Lighting → Textureless cho thấy hình học cải thiện với từng lựa chọn, và ảnh render đầy đủ tăng +12.5% R-Precision.

- **Hạn chế**: Bài báo không có mục hạn chế riêng, nhưng tài liệu hóa rằng zero-shot text-to-3D thiếu ground truth nên Chamfer Distance/PSNR không khả thi, khiến việc đánh giá phụ thuộc vào CLIP R-Precision — chỉ số không phản ánh chất lượng hình học: mô hình cơ sở đạt điểm cao nhất trên ảnh albedo nhưng hình học kém (ví dụ chú chó nhiều đầu). Hình học chính xác đòi hỏi kết hợp prompt phụ thuộc góc nhìn, chiếu sáng và render không kết cấu; đánh giá dựa trên CLIP có thể bị nhiễu vì một số baseline huấn luyện cùng CLIP; và đánh giá "biết hình học" vẫn là bài toán mở. Trong bối cảnh so sánh các phương pháp diffusion cho robot, DreamFusion thuộc nhánh dùng diffusion 2D làm prior để sinh tài sản 3D chứ không phải phương pháp điều khiển hay planning robot; paper không đánh giá trên tác vụ robot, nên giá trị tiềm năng cho robot (ví dụ sinh mô hình 3D cho mô phỏng) chưa được kiểm chứng trong bài báo.
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

Bảng dưới đây đối chiếu định tính ba cách khai thác cơ chế khuếch tán; vì metric, dataset và đơn vị khác nhau hoàn toàn, bảng không xếp hạng các bài theo nhau.

| Tiêu chí | Diffusion Policy | Diffuser | DreamFusion |
|---|---|---|---|
| Vai trò trong hệ robot | Chính sách điều khiển trực tiếp (visuomotor) | Trình lập kế hoạch quỹ đạo kiểu model-based | Sinh tài sản 3D, ngoài vòng điều khiển |
| Đối tượng khử nhiễu | Chuỗi hành động cao chiều | Quỹ đạo trạng thái đầy đủ | Tham số NeRF, thông qua render 2D |
| Điều kiện hóa lúc suy luận | Quan sát thị giác; receding horizon | Classifier guidance và inpainting theo goal/reward | Prompt văn bản qua prior 2D |
| Kết quả tiêu biểu | Cải thiện trung bình 46,9% trên 15 tác vụ, 4 benchmark (không có tỉ lệ thành công tuyệt đối theo tác vụ trong trích đoạn) | Maze2D: trên 100 điểm chuẩn hóa mọi kích thước; xếp khối 58,7 ± 2,5 (vô điều kiện) và 45,6 ± 3,1 (có điều kiện) so với CQL trung bình 8,1 và BCQ 0,0 | 58,5% CLIP R-Precision trên textureless renders; vượt Dream Fields và CLIP-Mesh trên render màu, tiệm cận oracle |
| Điểm yếu chính | Độ trễ suy luận; thay đổi tương đối của tỉ lệ thành công giảm tới khoảng −0,6 khi độ trễ quan sát tăng đến 7 bước | Thử nghiệm mê cung chạy open-loop; phụ thuộc goal nhận diện được trong dữ liệu; còn khoảng cách ở chế độ có điều kiện | Không có ground truth 3D; CLIP R-Precision không phản ánh chất lượng hình học |

**Tổng hợp**

- Cùng một cơ chế khử nhiễu lặp, ba không gian đầu ra khác nhau: chuỗi hành động (Diffusion Policy), quỹ đạo trạng thái (Diffuser), tham số NeRF qua render 2D (DreamFusion).
- Diffusion Policy giải quyết tính đa phương thức ở cấp chuỗi: trên tác vụ đẩy T-block, mô hình học cả hai mode và cam kết một mode mỗi rollout, trong khi LSTM-GMM và IBC thiên lệch về một mode còn BET thiếu temporal consistency. Khi chuyển từ điều khiển vận tốc sang điều khiển vị trí, cả hai biến thể DiffusionPolicy-C và DiffusionPolicy-T cải thiện trong khi BC-RNN và BET giảm hiệu năng.
- Diffuser xử lý credit assignment dài hạn: IQL (baseline model-free mạnh nhất) tụt đáng kể trên Multi2D dù được thích ứng bằng hindsight relabeling, và MPPI với động lực học ground truth cũng kém — cho thấy khó khăn nằm ở lập kế hoạch dài hạn chứ không chỉ ở sai số dự đoán mô hình. Khả năng tái dùng một mô hình cho nhiều goal chỉ bằng cách đổi điều kiện là lợi thế riêng so với các baseline model-free.
- DreamFusion tránh hoàn toàn dữ liệu huấn luyện 3D; ablation lần lượt thêm ViewAug, ViewDep, Lighting và Textureless, cải thiện hình học từng bước và tăng 12,5% R-Precision trên full renderings. Lưu ý một số baseline được huấn luyện với CLIP (lợi thế so sánh), song DreamFusion vẫn vượt các baseline này.
- Về hạn chế trích xuất: một số giá trị và phần nội dung không có mặt trong các đoạn PDF đã trích, nên việc đối chiếu định lượng liên bài bị giới hạn; những mục thiếu được xem là chưa kiểm chứng thay vì được diễn giải thành kết luận khoa học.

## 4. Kết luận

**Khuyến nghị thực tiễn cho nhà nghiên cứu và kỹ sư ML**

- Chọn dạng khuếch tán theo bài toán: thao tác đa phương thức nên dùng diffusion trên chuỗi hành động với receding horizon control; lập kế hoạch dài hạn, thưởng thưa nên dùng diffusion trên quỹ đạo với guidance/inpainting; nhu cầu tài sản 3D cho mô phỏng có thể dùng chưng cất prior 2D nhưng bắt buộc kiểm tra hình học bằng tay.
- Thiết kế vòng điều khiển phải tính trước chi phí khử nhiễu lặp: đo độ trễ quan sát thực tế của hệ, và chọn độ dài horizon cân bằng giữa khả năng phản ứng và tính nhất quán thời gian (dải khảo sát trong bài là 1–128 bước).
- Với Diffusion Policy, cân nhắc điều khiển vị trí thay vì điều khiển vận tốc; tuy nhiên đây là giả thuyết của tác giả (gắn với đa phương thức mạnh hơn và sai số tích lũy nhỏ hơn), cần kiểm chứng trên hệ thống cụ thể trước khi áp dụng.
- Với Diffuser, đảm bảo goal nhận diện được trong dữ liệu huấn luyện và kiểm chứng hành vi closed-loop, vì các thử nghiệm mê cung được thực thi open-loop.
- Chỉ so sánh số liệu trong nội bộ từng bài với baseline tương ứng; không đối chiếu chéo giữa các bài do khác benchmark, metric và đơn vị.

**Thách thức mở**

- Độ trễ suy luận của khử nhiễu lặp trong điều khiển vòng kín thời gian thực, đặc biệt dưới độ trễ quan sát biến động.
- Sự phụ thuộc vào điều kiện hóa lúc kiểm thử (guidance, inpainting, prompt); khả năng phục hồi từ dữ liệu dị chủng được Diffuser liệt kê nhưng chưa được mô tả chi tiết trong trích đoạn.
- Khoảng cách hiệu năng còn lại của Diffuser ở chế độ lập kế hoạch có điều kiện so với vô điều kiện trong thử nghiệm xếp khối.
- Đánh giá hình học cho text-to-3D vẫn là bài toán ngỏ: thiếu ground truth 3D, CLIP R-Precision là proxy dễ gây nhiễu và không phản ánh lỗi hình học (ví dụ mô hình chó có nhiều đầu).
- Cầu nối giữa sinh nội dung 3D kiểu DreamFusion và huấn luyện robot (dữ liệu mô phỏng, tài sản tương tác) chưa được đánh giá trong các tài liệu đã thu thập.

## 5. Tài liệu tham khảo

1. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** (Cheng Chi, Zhenjia Xu, Siyuan Feng et al., 2023-03-07) — [ArXiv abstract](https://arxiv.org/abs/2303.04137) | [PDF](https://arxiv.org/pdf/2303.04137.pdf) | [Mã nguồn](https://github.com/44David/diffusion-policy)
2. **Planning with Diffusion for Flexible Behavior Synthesis** (Michael Janner, Yilun Du, Joshua B. Tenenbaum et al., 2022-05-20) — [ArXiv abstract](https://arxiv.org/abs/2205.09991) | [PDF](https://arxiv.org/pdf/2205.09991.pdf)
3. **DreamFusion: Text-to-3D using 2D Diffusion** (Ben Poole, Ajay Jain, Jonathan T. Barron et al., 2022-09-29) — [ArXiv abstract](https://arxiv.org/abs/2209.14988) | [PDF](https://arxiv.org/pdf/2209.14988.pdf)
