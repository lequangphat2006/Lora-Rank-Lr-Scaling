# Dữ liệu khởi điểm từ dự án trước (chỉ để tham chiếu)

Nguồn: kho `Peft-Rank-Audit`, commit `2ebb1c53e46af2ad7366e1449d2c8da15a355dee` (2026-10-20). Bài kiểm toán đó (XLM-R-base, UIT-VSFC, α = 2r, fp16) đã hoàn thành và được quản lý riêng; thư mục này chỉ giữ những phần mà đề xuất v3 dùng làm điểm xuất phát.

| File | Nội dung | Dùng cho |
|---|---|---|
| `audit_tuning/tuning_full.csv` | 69 lần tune LR (seed 400–402, rank {1,4,8,16,32}, 6 epoch, LR 5e-5 đến 2e-3) | Pha 0 bước 0.1 (hồi quy σ̂₀ trên 5 rank); hiệu chỉnh σ, c₂ của mô phỏng 0A |
| `audit_tuning/tuning_decision.json` | Bảng LR chọn theo rank của dự án trước | Tham chiếu so sánh với σ̂₀ |
| `audit_R1_global_merged.csv` | 160 lượt R1 (16 seed, 3 và 6 epoch) | Tham chiếu |
| `audit_smoke_timing.csv` | 2 lượt smoke (rank 1 và 32, 3 epoch) | Tham chiếu thời gian chạy |

Lưu ý: mọi số liệu ở đây dùng **Macro-F1 trên tập calibration** làm tiêu chí tune, trong khi đề tài mới dùng **NLL**. Số σ = 0,0136 và c₂ = 0,047 trong mô phỏng 0A lấy từ Macro-F1, nên phải đo lại bằng NLL ở pilot 0C.
Thời gian chạy đo được: ≈ 132 s/lượt (3 epoch) và ≈ 249 s/lượt (6 epoch) cho XLM-R-base trên T4 (Kaggle, fp16), gần như không đổi theo rank (244–252 s ở 6 epoch).
