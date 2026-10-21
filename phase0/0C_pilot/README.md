# Pha 0C: pilot bề mặt NLL (chưa làm)

Thiết kế gốc (v3): D1 (XLM-R-base, UIT-VSFC), rank {2, 8, 64}, 6 LR, 3 seed = 54 lượt, tham số hóa P1.

Đầu ra cần có (mở rộng so với v3, xem `docs/PROPOSAL_v3.1.md` mục A5):
- NLL validation theo (rank, LR, seed); σ của NLL giữa seed; c₂ của NLL từng rank (hệ số bậc hai theo x = log₁₀ η).
- Ước lượng độ cong có đổi theo rank không: c₂(r) ∝ (r/8)^γ, báo cáo γ̂ và khoảng tin cậy. Giao thức G0 v3.2 gần như chặn mọi kết luận khi |γ| ≥ 0,15 ở c₂ ≥ 0,2, nên đây là số quyết định.
- Kiểm mô hình bề mặt chia sẻ hình dạng và vách collapse theo rank.
- GPU-giờ thực đo mỗi lượt (tham chiếu XLM-R-base: ≈ 132 s ở 3 epoch, ≈ 249 s ở 6 epoch trên T4).

Lệnh gợi ý (Kaggle, mỗi GPU một tiến trình): xem `docs/SETUP_KAGGLE.md`. LR dùng bước √2 quanh 5e-4 và mở rộng về phía thấp cho rank cao.

---

## Ma trận quyết định (chốt trong R0)

### Quy tắc dừng / mở rộng theo c₂(NLL)

| c₂(NLL) ước lượng | CI 95% của c₂ | Hành động 0C | Preview cho R1 (không ràng buộc) |
|---|---|---|---|
| ≥ 0,2 | CI dưới ≥ 0,15 | Dừng ở 54 lượt (3 seed × 6 LR × 3 rank) | Giữ 3 seed ở pilot; Phase 1 theo A5: bước 2, 5 seed |
| 0,1–0,2 | CI giao [0,1; 0,2] | Mở rộng lên 8 seed × 14 LR (bước √2) = +112 lượt | Chuyển sang bước √2, 8 seed |
| < 0,1 | CI trên < 0,1 | Dừng sớm; không tăng seed; kích hoạt nhánh "Go hẹp" (A5) hoặc chuyển sang cell D2/D3 | G0 không đạt |
| Bất kỳ | CI rộng > 0,1 | Mở rộng seed để thu hẹp CI | Chờ kết quả |

Cột "Preview cho R1" không ràng buộc — R1 sẽ khóa thiết kế Phase 1 dựa trên kết quả 0C. Cột "Hành động 0C" là quy tắc của chính pilot, có ràng buộc.

### Quy tắc về số seed (bổ sung sau 0A-bis)

- **Regret (bis_01):** R-est đạt δ_R ở 3 seed, mọi c₂ và mọi bước lưới. Regret không đòi hỏi tăng seed.
- **Phân loại G0 (README 0A):** vẫn đòi c₂ ≥ 0,2 với 3 seed, hoặc c₂ ≥ 0,1 với bước √2 và 8 seed. Hai tiêu chí này độc lập, không thay thế nhau.
- **Trạng thái:** regret đạt ở 3 seed (bis_01); phân loại G0 vẫn theo README 0A.
- **Giới hạn:** R-est gần 0 vì mô phỏng lý tưởng (parabol đúng, nhiễu Gauss, γ = 0, không có vách collapse). Không thay thế được kiểm tra γ và collapse ở 0C; regret trên dữ liệu thật chủ yếu đến từ sai mô hình, không phải nhiễu.

### Quy tắc về độ cong đổi theo rank (γ)

- Nếu γ̂ (c₂(r) ∝ (r/8)^γ) có CI vượt quá |γ| = 0,15, mọi kết luận G0 bị chặn bất kể c₂ — theo giao thức v3.2.
- Nếu |γ̂| < 0,15 với CI hẹp, giao thức v3.2 được giữ nguyên.

### Quy tắc về bề mặt chia sẻ

- Nếu mô hình bề mặt chia sẻ bị từ chối, ghi cờ "hai ước lượng bất đồng" — kết luận bị chặn theo giao thức v3.2.
- Nếu vách collapse xuất hiện ở rank cao, ghi vị trí; không ảnh hưởng G0 nhưng là dữ liệu cho R1.

### Quy tắc về GPU-giờ

- Ghi GPU-giờ thực đo mỗi lượt (132 s ở 3 epoch, 249 s ở 6 epoch trên T4).
- Nếu tổng GPU-giờ 0C vượt 2× dự toán (54 × 132 s ≈ 2 giờ), dừng và báo cáo — không tự động mở rộng.
