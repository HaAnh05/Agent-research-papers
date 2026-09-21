/** Exact representative excerpt from the current GauS-SLAM generated report. */
export const GAUS_SLAM_REPORT_EXCERPT = String.raw`
### GauS-SLAM: Dense RGB-D SLAM with Gaussian Surfels

Kết xuất độ sâu có bước hiệu chỉnh độ sâu với tham số $B = 4$.

$$N_{\text{gs}} > \tau_l \cdot H \cdot W, \qquad \tau_l = 1.5$$

Khung hình chính được chèn khi tỷ lệ cảnh mới quan sát vượt ngưỡng:

$$p_{\text{new}} > \tau_k, \qquad \tau_k = 0.01$$

Pose camera được tối ưu qua hàm mất mát kết hợp quang học và độ sâu:

$$\mathcal{L} = \lambda_1 \, \mathcal{L}_{\text{rgb}} + \lambda_2 \, \mathcal{L}_{\text{depth}}$$

| Hệ thống | ATE-RMSE trên Replica (cm) |
|---|---:|
| **GauS-SLAM** | **0.06** |
| GS-ICP | 0.16 |
| SplaTAM | 0.36 |
`
