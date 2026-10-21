# R0 — Giao thức G0 v3.2 và các cổng Phase 0

**Ngày nộp:** [điền khi nộp]
**DOI:** [điền khi nộp]
**Tag repo:** [điền khi nộp]
**Nền tảng:** OSF Registries
**Loại:** Đăng ký trước (preregistration), chỉ cho Phase 0 (0B và 0C)
**Nguồn đề xuất:** `docs/proposal_v3_original_2026-10-20.pdf` (bản gốc) và `docs/PROPOSAL_v3.1.md` (đính chính A1–A8)

## 0. Bối cảnh và phạm vi

- **0A (đã chạy trước R0, exploratory).** Mô phỏng nhận dạng được chạy và chỉnh qua nhiều phiên bản (v1 → v3.2); ngưỡng, bộ phân loại và giao thức đổi sau khi xem kết quả. R0 **không** khóa 0A. Nó chỉ khóa giao thức cuối (v3.2) được chép nguyên văn ở mục 1.
- **0A-bis (đã chạy trước R0, exploratory).** Mô phỏng regret của R-known, R-μA, R-est trên đúng thiết kế đã lập ngân sách (P1: 7 rank; P2/P3: {2, 8, 32, 64}), chạy ngày 2026-10-21, commit `6160cbd`, log `phase0/0A_identification/logs_bis/bis_01_regret.txt`. Kết quả là dữ liệu phát triển, **không** phải cam kết của R0. R-probe không được tính (cần E_W từ 0B). Mô phỏng lý tưởng (parabol đúng, nhiễu Gauss, γ = 0, không có vách collapse), nên regret gần 0 của R-est không thay thế kiểm tra γ và collapse ở 0C.
- **R0 khóa:** (1) giao thức G0 v3.2; (2) năm phép kiểm 0B và ngưỡng; (3) cổng G0, G0B-a, G0B-b; (4) ma trận quyết định 0C; (5) quy tắc regret dùng trong các cổng (mục 3).
- **Thứ tự thời gian:** R0 được nộp **trước mọi lượt GPU** của 0B và 0C, và trước mọi phép chạy T1–T5 (kể cả T1 trên CPU).
- **Ngoài phạm vi:** RQ1–RQ6, giả thuyết HμA/H-bão hòa/H-trôi và toàn bộ thiết kế Phase 1 thuộc R1. Dữ liệu `data_prior/` (kiểm toán Peft-Rank-Audit, đo bằng Macro-F1) chỉ để tham chiếu và hiệu chỉnh mô phỏng; nó là dữ liệu có trước và được phân tích post hoc.

## 1. Giao thức G0 v3.2 (nguyên văn)

Chép nguyên văn từ `phase0/0A_identification/README.md` (mục «Giao thức G0 cuối (bước 11) và kết quả» đến hết «Giới hạn đã biết của bộ kiểm toán») và phần đầu `logs/11_giao_thuc_dong_thuan.txt`.

### 1.1. Đặc tả và kết quả mô phỏng (README 0A)

## Giao thức G0 cuối (bước 11) và kết quả
Bộ phân loại ở mọi bước 08–11 là QDA **bền với ngoại lệ** (ước lượng cắt tỉa 90% có hiệu chỉnh nhất quán), vì QDA thường làm kết quả nhảy giữa các lần chạy khi estimator có đuôi nặng (đặc biệt `mask`).
Quy trình: (1) hiệu chuẩn lớp và phân phối nhất quán 4 chiều (â,b̂ của `shared` và `mask`) trên 4 bề mặt, γ=0; (2) ngưỡng cờ = phân vị 95% trên 5 bề mặt khác, γ=0 (tỉ lệ báo động giả ≈ 5%); (3) phán quyết: kết luận khi không bị cờ; bị cờ ⇒ không kết luận.
Kết quả (600 mẫu hiệu chuẩn, 300 mẫu kiểm mỗi ô, 9 bề mặt, γ ∈ {−0.3, −0.15, 0, 0.15, 0.3}):
| c₂ (σ=0.0136) | Bước 2, 5 seed | Bước √2, 5 seed | Bước √2, 8 seed |
|---|---|---|---|
| ≥ 0.2 | An toàn (unsafe 0%) | An toàn | An toàn |
| 0.1 | KHÔNG an toàn (unsafe tới 18–26%) | Gần an toàn (unsafe max 33% ở γ=+0.3 trên một bề mặt) | An toàn (unsafe ≤ 0.4%) |
| 0.047 | KHÔNG an toàn | KHÔNG an toàn | KHÔNG an toàn (unsafe 14–41%): phân loại đúng khi giả định chia sẻ đúng, nhưng chẩn đoán không đủ công suất để phát hiện khi sai |
Ở c₂ ≥ 0.2, mọi trường hợp γ ≠ 0 đều bị chặn gần 100% còn γ=0 chỉ bị chặn ~5% (đúng thiết kế); độ chính xác khi được phép kết luận ≥ 0.99. Ở c₂ ≈ 0.047 chẩn đoán mất công suất (chỉ chặn 5–50% các trường hợp vi phạm) nên giao thức không an toàn.
Hệ quả: với c₂ ≥ 0.2 (hoặc 0.1 với bước √2 và 8 seed), G0 có thể được phán quyết trên dữ liệu thật bằng giao thức này. Dưới 0.1, đi theo nhánh "Go hẹp" của v3 mục 6.3.
Giới hạn: giao thức nhạy với γ=±0.15 (bị chặn gần hết ở c₂ ≥ 0.2). Nếu dữ liệu thật có độ cong đổi nhẹ theo rank thì sẽ luôn bị chặn; khi đó cần mở rộng `shared` thêm tham số γ.

## Giới hạn đã biết của bộ kiểm toán
- Chưa mô phỏng: tập rank 5/7+128/256, ít mức p hơn, estimator GP, bước lưới 1.25 cùng bước 08, độ lệch một phía kiểu khác, nhiễu phụ thuộc vị trí LR.
- Bootstrap tham số (bước 05) dựa vào giả định parabol nên lạc quan khi bề mặt thật khác.
- Các con số 0.9 và [0.93, 0.97] là ngưỡng của v3; sai số Monte Carlo của độ phủ ~±0.005 ở 2,000 lần lặp mỗi giả thuyết.

### 1.2. Định nghĩa thước đo (phần đầu log bước 11, nguyên văn, giữ nguyên không dấu)

```
g0_protocol.py - GIAO THUC G0 DAY DU: phan loai bang estimator chia se + chan doan DONG THUAN voi estimator mask.
Y tuong: khi hinh dang be mat dung (chia se giua cac rank), shared va mask cho ket qua khop theo mot cach co the hieu chuan (mask co do lech co dinh);
khi sai (vd do cong doi theo rank, gamma != 0) hai estimator tach nhau => CO CHAN ('flag'). Do:
  acc_all      : do chinh xac phan loai 3 lop (khong dung co)
  flag%        : ty le bi co chan (khong ket luan)
  unsafe%      : P(khong bi chan VA phan loai sai)  <- cai thuc su gay hai; muc tieu <= 5% o MOI be mat, MOI gamma
  acc|unflag   : chinh xac khi duoc phep ket luan; muc tieu >= 0.9 tai gamma=0

```

### 1.3. Điều kiện kết luận trên dữ liệu thật

Theo `docs/PROPOSAL_v3.1.md` mục A1: nếu cờ bật thì **không được kết luận về số mũ** (trạng thái CHƯA KẾT LUẬN).

## 2. Năm phép kiểm 0B và ngưỡng đã chốt

Đại lượng đo: `E_W = median_m ‖ΔW_m‖_F / ‖W0_m‖_F`; `E_f` = thay đổi tương đối RMS của biểu diễn cuối trên lô thăm dò cố định 256 mẫu; chỉ số kết hợp `ρ`. Log tại các bước {10, 25, 50, 100, 200}, E_f tại {50, 100}.

**Dải seed (cố định):** seed huấn luyện = **[0, 1, 2, 3, 4]** cho T2; các phép kiểm khác dùng ít nhất 5 seed lấy từ cùng dải này.

**ε_x cho T3:** ε_x = nửa bước lưới LR của phép đo 0B. 0B dùng lưới bước √2 (log₁₀√2 ≈ 0,1505 decade), nên **ε_x = 0,075 decade** và ngưỡng T3 = ε_x/2 ≈ **0,0376 decade**. Phương án thay thế đã khai báo trước: nếu 0B dùng bước 2 (0,301 decade) thì ε_x = 0,15 và ngưỡng T3 = 0,075. ε_x ở đây chỉ áp dụng cho T3 của 0B; ε_x của Phase 2 do R2 chốt.

| Phép kiểm | Cách làm | Ngưỡng đạt (chốt) |
|---|---|---|
| T1 Phục hồi tổng hợp | Dựng B, A với độ kết hợp và độ lớn thành phần biết trước; tính E_W, ρ̂, κ̂_ρ; lặp theo r, d, bf16/fp32 | Lệch của ρ̂ và κ̂_ρ ≤ 5%, không phụ thuộc d |
| T2 Test–retest | Rank {4, 16, 64}, seed [0, 1, 2, 3, 4], 1 LR, 100 bước, D1; ICC theo module | ICC(E_W) ≥ 0,9; ICC(ρ) ≥ 0,7 |
| T3 Hội tụ giữa hàm đo | η̂ từ E_W và từ E_f trên D1, D2 | \|log₁₀ η̂_W − log₁₀ η̂_f\| ≤ ε_x/2 (≈ 0,0376 decade) trên ≥ 80% cặp (cell, rank) |
| T4 Bất biến phân tích | Trung vị/trung bình/theo loại module; t_p ∈ {25, 50, 100, 200}; Adam ε; bf16/fp32 | Thứ tự rank và dấu độ dốc không đổi; σ̂ đổi ≤ 0,1 |
| T5 Thao tác | Trộn hướng chung vào khởi tạo A với λ ∈ {0, 0,5, 0,9}; đo ρ | ρ tăng đơn điệu theo λ ở mọi rank thử |

**Quy tắc ngoại lệ cho T2 (khai báo trước), áp dụng riêng cho từng đại lượng:**
- ICC(E_W) (thuộc G0B-a): nếu không đạt ở rank 64 nhưng đạt ở cả rank 4 và rank 16, ghi **cờ rank 64** và **không** đánh trượt G0B-a. Không đạt ở rank 4 hoặc 16 thì T2 (E_W) trượt.
- ICC(ρ) (thuộc G0B-b): quy tắc tương tự; cờ rank 64 không đánh trượt G0B-b.
- Cờ rank 64 (nếu có) được ghi lại và chuyển sang R1; không dùng để đánh trượt cổng nhưng là thông tin cho thiết kế Phase 1 ở rank cao.

## 3. Cổng G0, G0B-a, G0B-b

| Cổng | Điều kiện đạt | Cho phép | Trạng thái lúc nộp |
|---|---|---|---|
| **G0** | Trên mọi bề mặt kiểm khác bề mặt hiệu chuẩn và γ ∈ {−0,3; −0,15; 0; +0,15; +0,3}: (i) `unsafe%` ≤ 5%; (ii) độ chính xác khi được phép kết luận ≥ 0,9 tại γ = 0; (iii) tỉ lệ cờ tại γ = 0 xấp xỉ 5%. Độ phủ CI chỉ báo cáo (bootstrap tham số), không quyết định. Trên dữ liệu thật: cờ bật ⇒ không kết luận số mũ | Kết luận về số mũ (RQ1) | Chưa phán quyết: cần c₂ và γ̂ của NLL từ 0C |
| **G0B-a** | T1, T2 (với E_W), T3, T4 đạt (theo quy tắc ngoại lệ T2 ở mục 2) | Dùng R-probe | Chưa chạy |
| **G0B-b** | T2 (với ρ), T5 đạt | Dùng ρ để giải thích k | Chưa chạy |

**Quy tắc regret dùng trong cổng (Q1):** regret(r, p) = c₂·(x_quy tắc − x_thật)², x = log₁₀ η; δ_R = **0,02**.
- **Tiêu chí chính:** nếu có **một cặp (r, p) xấu nhất** của thiết kế có regret > δ_R thì quy tắc đó **trượt**.
- **Tiêu chí phụ:** regret trung bình trên các cặp; chỉ báo cáo, không quyết định.
- Hệ quả đã biết từ 0A-bis (exploratory): R-μA trượt tiêu chí chính ở `k1` khi c₂ ≥ 0,1 và ở `khalf` khi c₂ ≥ 0,2; R-est không trượt ở mọi c₂ đã mô phỏng. Nhánh «Go hẹp» (c₂ < 0,1) chưa được kiểm bằng regret.

## 4. Ma trận quyết định 0C

Pilot 0C: D1 (XLM-R-base, UIT-VSFC), rank {2, 8, 64}, 6 LR, 3 seed = 54 lượt, tham số hóa P1. Đầu ra bắt buộc: NLL validation theo (rank, LR, seed); σ của NLL giữa seed; c₂ của NLL từng rank; γ̂ (c₂(r) ∝ (r/8)^γ) cùng khoảng tin cậy; kiểm mô hình bề mặt chia sẻ và vách collapse; GPU-giờ thực đo mỗi lượt.

### 4.1. Quy tắc dừng / mở rộng theo c₂(NLL)

Số seed ở cột «Hành động 0C» là seed của **pilot** để đo c₂ và **ràng buộc** theo R0. Cột «Preview cho R1» chỉ là dự kiến từ `PROPOSAL_v3.1.md` A5, **không ràng buộc** R1; R1 sẽ khóa thiết kế Phase 1 dựa trên kết quả 0C.

| c₂(NLL) ước lượng | CI 95% của c₂ | Hành động 0C | Preview cho R1 (không ràng buộc) |
|---|---|---|---|
| ≥ 0,2 (và \|γ̂\| nhỏ) | CI dưới ≥ 0,15 | Dừng ở 54 lượt (3 seed × 6 LR × 3 rank); không tăng seed | Bước 2 (7 điểm LR), 5 seed |
| 0,1 đến dưới 0,2 | CI giao [0,1; 0,2] | Mở rộng lên 8 seed × 14 LR (bước √2) = +112 lượt | Bước √2 (14 điểm LR), 8 seed |
| < 0,1 | CI trên < 0,1 | Dừng sớm; không tăng seed | G0 không đạt: nhánh «Go hẹp» (bỏ claim số mũ, giữ RQ3, RQ4) hoặc chuyển sang cell D2/D3 |
| Bất kỳ | CI rộng > 0,1 | Mở rộng seed để thu hẹp CI | Chờ kết quả |

**Thứ tự áp dụng:** quy tắc «CI rộng > 0,1» được xét trước; được mở rộng seed **một lần** (đến 8 seed); sau đó phân loại lại theo ba dòng đầu với ước lượng mới. Trường hợp không khớp dòng nào được xử lý như dòng «CI rộng». **Điểm dừng:** nếu sau khi mở rộng đến 8 seed CI của c₂ vẫn > 0,1 (hoặc vẫn không khớp dòng nào), ghi cờ «không đủ công suất đo c₂», dừng 0C, G0 ở trạng thái CHƯA KẾT LUẬN; **không** mở rộng seed lần hai. Bước tiếp theo do R1 hoặc một amendment công khai quyết định.

### 4.2. Quy tắc về số seed

- **Regret (bis_01, exploratory):** R-est đạt δ_R ở 3 seed ở mọi c₂ và mọi bước lưới đã mô phỏng; regret không đòi tăng seed.
- **Phân loại G0 (mục 1):** đòi c₂ ≥ 0,2 (bước 2 hoặc √2, 5 seed), hoặc c₂ = 0,1 với bước √2 và 8 seed. 3 seed không được giao thức bước 11 kiểm. Hai tiêu chí độc lập, không thay thế nhau.

### Quy tắc về độ cong đổi theo rank (γ)

- Nếu γ̂ (c₂(r) ∝ (r/8)^γ) có CI vượt quá |γ| = 0,15, mọi kết luận G0 bị chặn bất kể c₂ — theo giao thức v3.2.
- **Bước tiếp theo khi bị chặn (theo `PROPOSAL_v3.1.md` A5):** mở rộng ước lượng chia sẻ thêm tham số γ, mô phỏng lại (CPU), rồi mới chọn thiết kế Phase 1. Đây không phải nhánh «Go hẹp» (nhánh này dành cho c₂ < 0,1). Việc có chuyển sang cell D2/D3 hay không do R1 quyết.
- Nếu |γ̂| < 0,15 với CI hẹp, giao thức v3.2 được giữ nguyên.

### Quy tắc về bề mặt chia sẻ

- Nếu mô hình bề mặt chia sẻ bị từ chối, ghi cờ "hai ước lượng bất đồng" — kết luận bị chặn theo giao thức v3.2.
- Nếu vách collapse xuất hiện ở rank cao, ghi vị trí; không ảnh hưởng G0 nhưng là dữ liệu cho R1.

### Quy tắc về GPU-giờ

- Ghi GPU-giờ thực đo mỗi lượt (132 s ở 3 epoch, 249 s ở 6 epoch trên T4).
- Nếu tổng GPU-giờ 0C vượt 2× dự toán (54 × 132 s ≈ 2 giờ), dừng và báo cáo — không tự động mở rộng.

### 4.3. Ghi nhận bắt buộc

Báo cáo đầy đủ mọi kết quả 0C (kể cả khi G0 không đạt); không loại lượt nào sau khi thấy kết quả trừ lượt hỏng kỹ thuật được ghi rõ nguyên nhân.

## 5. Cam kết

- Mọi phân tích của 0B và 0C chạy sau R0 được gắn nhãn **preregistered**.
- 0A và 0A-bis được gắn nhãn **exploratory**; `data_prior/` là **post hoc**.
- Không thay đổi giao thức, ngưỡng hay ma trận sau khi nộp mà không có **amendment công khai** ghi ngày và lý do.
- Mọi sai khác giữa dữ liệu thật và giả định mô phỏng (γ ≠ 0, plateau, collapse) được báo cáo, không điều chỉnh ngưỡng để «cho qua».
- Bài báo nêu rõ rằng 0A và 0A-bis không được đăng ký trước, như một hạn chế.

## 6. Tài liệu tham khảo nội bộ (tại tag repo ghi ở đầu file)

- `docs/proposal_v3_original_2026-10-20.pdf`, `docs/proposal_v3_original_2026-10-20.docx`, `docs/PROPOSAL_v3.1.md`
- `phase0/README.md`, `phase0/0A_identification/README.md`, `phase0/0A_identification/logs/` (đặc biệt `11_giao_thuc_dong_thuan.txt`)
- `phase0/0A_identification/g0_final.py`, `g0_protocol.py`, `g0_bis.py`, `logs_bis/bis_01_regret.txt`
- `phase0/0B_measurement/README.md`, `phase0/0C_pilot/README.md`
- `data_prior/README.md`, `registrations/README.md`
