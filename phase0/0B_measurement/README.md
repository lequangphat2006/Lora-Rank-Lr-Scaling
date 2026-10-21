# Pha 0B: kiểm định công cụ đo (chưa làm)

Đo độ lớn cập nhật hiệu dụng `E_W = median_m ‖ΔW_m‖_F / ‖W0_m‖_F` (không gian trọng số), `E_f` (thay đổi tương đối RMS của biểu diễn cuối trên lô thăm dò cố định 256 mẫu) và chỉ số kết hợp `ρ`. Mã nguồn đo chưa có trong repo; cần gắn vào `code/run_pilot.py` (log tại các bước {10, 25, 50, 100, 200}, E_f tại {50, 100}).

Năm phép kiểm (ngưỡng là đề xuất, chốt trong R0):

| Phép kiểm | Cách làm | Ngưỡng đạt (đề xuất) |
|---|---|---|
| T1 Phục hồi tổng hợp | Dựng B, A với độ kết hợp và độ lớn thành phần biết trước; tính E_W, ρ̂, κ̂_ρ; lặp theo r, d, bf16/fp32 | Lệch của ρ̂ và κ̂_ρ ≤ 5%, không phụ thuộc d |
| T2 Test–retest | Rank {4, 16, 64}, 5 seed, 1 LR, 100 bước, D1; ICC theo module | ICC(E_W) ≥ 0,9; ICC(ρ) ≥ 0,7 |
| T3 Hội tụ giữa hàm đo | η̂ từ E_W và từ E_f trên D1, D2 | \|log₁₀ η̂_W − log₁₀ η̂_f\| ≤ ε_x/2 trên ≥ 80% cặp (cell, rank) |
| T4 Bất biến phân tích | Trung vị/trung bình/theo loại module; t_p ∈ {25, 50, 100, 200}; Adam ε; bf16/fp32 | Thứ tự rank và dấu độ dốc không đổi; σ̂ đổi ≤ 0,1 |
| T5 Thao tác | Trộn hướng chung vào khởi tạo A với λ ∈ {0, 0,5, 0,9}; đo ρ | ρ tăng đơn điệu theo λ ở mọi rank thử |

T1 chạy được hoàn toàn trên CPU bằng numpy; nên làm trước.
Cổng: G0B-a (T1, T2 với E_W, T3, T4) cho phép dùng R-probe; G0B-b (T2 với ρ, T5) cho phép dùng ρ để giải thích k.
