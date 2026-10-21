# Pha 0C: pilot bề mặt NLL (chưa làm)

Thiết kế gốc (v3): D1 (XLM-R-base, UIT-VSFC), rank {2, 8, 64}, 6 LR, 3 seed = 54 lượt, tham số hóa P1.

Đầu ra cần có (mở rộng so với v3, xem `docs/PROPOSAL_v3.1.md` mục A5):
- NLL validation theo (rank, LR, seed); σ của NLL giữa seed; c₂ của NLL từng rank (hệ số bậc hai theo x = log₁₀ η).
- Ước lượng độ cong có đổi theo rank không: c₂(r) ∝ (r/8)^γ, báo cáo γ̂ và khoảng tin cậy. Giao thức G0 v3.2 gần như chặn mọi kết luận khi |γ| ≥ 0,15 ở c₂ ≥ 0,2, nên đây là số quyết định.
- Kiểm mô hình bề mặt chia sẻ hình dạng và vách collapse theo rank.
- GPU-giờ thực đo mỗi lượt (tham chiếu XLM-R-base: ≈ 132 s ở 3 epoch, ≈ 249 s ở 6 epoch trên T4).

Lệnh gợi ý (Kaggle, mỗi GPU một tiến trình): xem `docs/SETUP_KAGGLE.md`. LR dùng bước √2 quanh 5e-4 và mở rộng về phía thấp cho rank cao.
