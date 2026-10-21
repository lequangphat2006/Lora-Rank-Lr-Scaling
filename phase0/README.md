# Pha 0: nhận dạng, kiểm định công cụ đo, pilot

Pha 0 trả lời hai câu trước khi tốn GPU lớn (xem `docs/PROPOSAL_v3.1.md`, mục 6 của bản v3 gốc):
(A) thiết kế có phân biệt được ba chữ ký số mũ Hκ=1, HμA, Hκ=1/2 không; (B) công cụ đo cơ chế có đủ bền để gánh claim không.

| Phần | Việc | Trạng thái (2026-10-21) | Thư mục |
|---|---|---|---|
| 0A | Mô phỏng nhận dạng (CPU) | **Đã chạy**, giao thức v3.2. Kết luận: **chưa phán quyết được G0** vì thiếu c₂/σ của NLL; xem bảng bên dưới | `0A_identification/` |
| 0A-bis | Mô phỏng lại đúng thiết kế đã lập ngân sách (7 rank P1 + 4 rank P2/P3, regret của các quy tắc) | **Đã chạy** 2026-10-21: R-est đạt δ_R = 0,02 ở 3 seed; R-μA không phải fallback an toàn cho k1; xem `logs_bis/bis_01_regret.txt` | `0A_identification/` |
| 0B | Năm phép kiểm công cụ đo T1–T5 | Chưa làm (chưa có code log E_W, E_f, ρ) | `0B_measurement/` |
| 0C | Pilot bề mặt NLL: D1, rank {2, 8, 64}, 6 LR, 3 seed (54 lượt) | Chưa làm | `0C_pilot/` |
| 0.1 | Hồi quy σ̂₀ trên 5 rank, bootstrap theo seed (69 lần tune của audit) | Chưa thấy kết quả trong repo | `data_prior/audit_tuning/` |

## Kết quả 0A (giao thức v3.2)

Mô phỏng: bề mặt ℓ = ℓ_min + c₂(x − x\*)² + nhiễu, x = log₁₀ η, 7 rank {1,…,64}, ba mức p {0, 1/2, 1}, lưới LR trải 2 decade (2e-5 đến 2e-3), σ = 0,0136, c₂ ∈ {0,047; 0,1; 0,2; 0,5}.
σ và c₂ lấy từ **Macro-F1** của audit cũ, không phải NLL.

1. Script power gốc cho ~100% vì phép kiểm quá dễ (chỉ kiểm a tại p = 0); đối chứng âm (nhiễu ×10 hoặc c₂ ≈ 0) tụt về 4–33%. Power không phải là G0 và không được dùng làm bằng chứng nhận dạng được.
2. Bootstrap theo seed với 3 seed cho độ phủ CI chỉ 0,82–0,92 (chỉ có 10 cách lấy lại mẫu); bootstrap tham số đạt 0,93–0,95 nhưng chỉ trên bề mặt parabol lý tưởng nên lạc quan.
3. Khi bề mặt thật khác mô hình (plateau, vách collapse), ước lượng đỉnh parabol từng rank lệch hệ thống (â +0,1 đến +0,4; b̂ −0,3 đến −0,47).
4. Ước lượng bề mặt chia sẻ hình dạng sửa được độ lệch đó và qua G0 ở mọi c₂ kể cả 0,047, nhưng sụp về độ chính xác 0,33–0,45 khi độ cong đổi theo rank (γ = ±0,3).
5. Giao thức cuối (bước 11): phân loại bằng ước lượng chia sẻ, chặn kết luận khi nó bất đồng với ước lượng từng rank (cờ). Thước đo là `unsafe%` = P(không bị chặn và phân loại sai), mục tiêu ≤ 5% ở mọi bề mặt và mọi γ.

| c₂ (σ = 0,0136) | Bước 2 (7 điểm LR), 5 seed | Bước √2 (14 điểm), 5 seed | Bước √2 (14 điểm), 8 seed |
|---|---|---|---|
| ≥ 0,2 | an toàn | an toàn | an toàn |
| 0,1 | **không an toàn** (unsafe tới 18–26%) | gần an toàn (unsafe trung bình 4,9%, tối đa 33% ở γ = +0,3) | an toàn (unsafe tối đa 0,4%) |
| 0,047 | không an toàn | không an toàn | không an toàn (unsafe 14–41% khi γ ≠ 0) |

Chưa kiểm: 3 seed (mức của v3) không có trong bảng bước 11 (chỉ n = 5 và 8); thiết kế 4 rank cho P2/P3; regret của R-est, R-known, R-μA (không có trong bất kỳ log nào); rank 128/256; GP; bước 1,25 ở bước 08; nhiễu phụ thuộc vị trí LR; chỉ σ = 0,0136 được chạy ở giao thức cuối.

**Việc tiếp theo quyết định G0:** pilot 0C để đo c₂/σ của NLL và xem độ cong có đổi theo rank không (γ̂). Bảng ánh xạ sang thiết kế ở `docs/PROPOSAL_v3.1.md`, mục A5.

## Chạy lại 0A

```bash
cd phase0/0A_identification
pip install numpy scipy
python run_all.py quick     # vài phút, chỉ kiểm tra môi trường, KHÔNG dùng để kết luận
python run_all.py           # đầy đủ, 60–90 phút trên 2 nhân; log ghi đè vào logs/
G0_SEED_OFFSET=1000 G0_PROCS=4 python run_all.py   # bản lặp độc lập (ví dụ Kaggle)
```

Log đã lưu trong `0A_identification/logs/` là bản chạy đầy đủ v3.2 trên máy local; `logs/kaggle_seed_offset_1000_v3.1.txt` là bản lặp v3.1 với hạt giống khác. README gốc của bộ kiểm toán (đặc tả từng bước) nằm ở `0A_identification/README.md`.
