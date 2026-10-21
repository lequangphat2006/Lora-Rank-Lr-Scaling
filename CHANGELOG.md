# Changelog

## [Chưa phát hành]

### 2026-10-21: khởi tạo repo
- Tách thành dự án riêng, không thừa kế lịch sử git hay đăng ký trước của `Peft-Rank-Audit` (xem `docs/MIGRATION_from_peft_rank_audit.md`).
- Thêm `docs/PROPOSAL_v3.1.md`: chỉnh đề xuất v3 theo kết quả mô phỏng nhận dạng Pha 0A (A1 đến A9).
- Thêm `phase0/`: bộ kiểm toán 0A v3.2 và log (từ `_inbox` của dự án trước, đã bỏ bản trùng và bản thử cũ), đặc tả 0B và 0C.
- `code/rankcap.py`: thêm `gamma_for`, `alpha_for` và `--alpha_mode p:<p>` (γ_r = 2·(r/8)^(−p), α = γ_r·r không làm tròn); `code/run_pilot.py`: cập nhật mô tả tham số.
- Thêm `test_gamma_alpha_parametrisations` (đạt) và `test_alpha_mode_scaling_p_modes` (đã chạy PASS trên torch 2.13.0+cpu / peft 0.20.0, 2026-10-21).
- `data_prior/`: 69 lần tune, R1 và smoke của dự án trước, ghi nguồn commit.
- Viết lại `README.md`, `CITATION.cff`, `.gitignore`, `docs/SETUP_KAGGLE.md`, `scripts/`; bỏ toàn bộ tham chiếu đăng ký trước cũ.

### 2026-10-21: ma trận quyết định 0C + 0A-bis
- 0A-bis: thêm `g0_bis.py` tính regret R-known/R-μA/R-est (R-probe hoãn 0B); log `bis_01_regret.txt`.
- 0C: thêm ma trận quyết định (chốt trong R0): c₂, số seed, γ, bề mặt chia sẻ, GPU-giờ.
- Tách hai tiêu chí: regret đạt ở 3 seed (bis_01) vs phân loại G0 vẫn theo README 0A (c₂ ≥ 0,2 hoặc 0,1 với bước √2 + 8 seed).
