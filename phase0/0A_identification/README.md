> Bộ kiểm toán Pha 0A, phiên bản v3.2, chuyển nguyên từ dự án trước (thư mục `_inbox`, chưa từng được commit). Tổng hợp kết quả và trạng thái: `../README.md`. Tên "Hκ=1, HμA, Hκ=1/2" là ba giả thuyết của `docs/PROPOSAL_v3.1.md`.

# Bộ kiểm toán Pha 0A (cổng G0) — hướng dẫn

## Chạy trên máy khác (Kaggle) để kiểm tra tính đồng nhất
- `G0_SEED_OFFSET=0` (mặc định): kết quả giống hệt bản gốc cùng phiên bản numpy ⇒ kiểm tra tái lập môi trường. Khác biệt lớn ⇒ môi trường (numpy/BLAS) có vấn đề.
- `G0_SEED_OFFSET=1000` (hoặc số khác): hạt giống khác ⇒ bản lặp độc lập. Kết luận (PASS/FAIL, vùng an toàn theo c₂) phải giống nhau; con số chênh nhẹ trong sai số Monte Carlo (~±0.01–0.03).
- `G0_PROCS=4`: số tiến trình song song (Kaggle thường có 4 nhân; mặc định 2).
Ví dụ trên Kaggle: `!G0_SEED_OFFSET=1000 G0_PROCS=4 python run_all.py`

## Chạy
```
pip install numpy scipy
python run_all.py quick     # ~2-3 phút, chỉ để kiểm tra môi trường; KHÔNG dùng để kết luận
python run_all.py           # đầy đủ (2,000 lần lặp/giả thuyết; ước tính 40–60 phút, bước 09–10 lâu nhất), log lưu trong logs/
```
Mọi file .py để chung một thư mục. Chỉ cần numpy và scipy (không cần sklearn).

## Các script và mục đích
| Bước | Script | Trả lời câu hỏi | Cách đọc |
|---|---|---|---|
| 01 | diag_power.py | Script power gốc có lỗi không? | Gốc: power ~100% vì phép kiểm dễ (chỉ a tại p=0, nhiễu chia √seed). Đối chứng âm (nhiễu ×10, c₂≈0): tụt về 6–31% ⇒ không lỗi. Power ≠ G0. |
| 02 | g0_final.py neg | Đối chứng âm cho G0 | Nhiễu ×10 hoặc c₂≈0: phân loại phải ~0.33–0.47. |
| 03–04 | g0_final.py sweep | G0 với bootstrap theo seed | Phân loại cao nhưng độ phủ CI 0.83–0.92 (thấp): bootstrap 3 seed chỉ có 10 cách lấy mẫu lại. |
| 05 | g0_pboot.py | G0 với bootstrap tham số | Độ phủ 0.93–0.95, PASS — nhưng chỉ trên bề mặt parabol lý tưởng (cận trên lạc quan). |
| 06 | g0_robust2.py | Thêm plateau + vách collapse; 3 lớp vs 5 lớp; loại điểm collapse | G0 là 3 lớp; 2 giả thuyết trung gian chỉ để đo độ nhạy. Điểm gãy ở c₂/σ. |
| 07 | g0_robust3.py | Bước lưới (2, √2, 1.25) × seed (5, 8) | Lưới mịn hơn bù độ cong thấp hiệu quả hơn thêm seed. |
| 08 | g0_robustness.py | **Hiệu chỉnh trên bề mặt A, kiểm trên bề mặt B** | Kiểm tra quan trọng nhất; xem phần dưới. |
| 09 | g0_shared.py | Estimator bề mặt chia sẻ (f chung, dịch theo x*(r,p)=b0+b1·L+b2·p·L) có sửa được độ lệch ở bước 08 không? | So `mask` với `shared` trên 9 bề mặt. Độ lệch giảm còn ±0.05; G0 pass từ c₂≥0.1. Cột `oracle` của `shared` không đáng tin (ngoại lệ làm QDA lệch); dùng cột `mixture`. |
| 11 | g0_protocol.py | **Giao thức G0 cuối**: phân loại bằng `shared`, kèm cờ chặn khi `shared` và `mask` bất đồng (đã hiệu chuẩn ngưỡng). Có an toàn khi bề mặt vi phạm giả định không? | Cột `unsafe%` = P(không bị chặn VÀ phân loại sai); mục tiêu ≤ 5% ở mọi γ và mọi bề mặt. Xem phần kết quả giao thức. |
| 10 | g0_shared_stress.py | Estimator chia sẻ có bền khi độ cong đổi theo rank không (c₂(r)=c₂·(r/8)^γ)? | KHÔNG: lệch +0.4/−0.25 và phân loại về 0.33 khi γ=±0.3, trong khi `mask` vẫn pass ở c₂≥0.2. |

## Điều kiện và giả định cần nhớ
- σ = 0.0136 và c₂ = 0.047 lấy từ Macro-F1 của audit, chưa phải NLL. Pilot 0C phải đo lại c₂, σ của NLL; mọi kết luận về thiết kế phụ thuộc vào tỉ lệ c₂/σ.
- Mô phỏng giả định: nhiễu Gaussian độc lập, bề mặt parabol (có plateau/vách ở 06–08), điểm collapse có thể được đánh cờ, các rank dùng chung hệ số chặn.
- Phân loại dùng QDA hiệu chỉnh trên mô phỏng độc lập (không đặt ngưỡng cứng 0.75).

## Quy tắc phán quyết đề xuất cho đăng ký 0
1. G0 = xác suất phân loại đúng ≥ 0.9 trên **3 lớp** (Hκ=1, HμA, Hκ=1/2) VÀ độ phủ CI trong [0.93, 0.97], ≥ 2,000 lần lặp.
2. Phải đạt ở **mọi** bề mặt kiểm khi bề mặt hiệu chỉnh khác bề mặt kiểm (bước 08), không chỉ trên mô hình đã dùng để hiệu chỉnh.
3. Giả thuyết trung gian (0.82, 1), (0.8, 0.6): báo cáo độ nhạy, không tính vào cổng.

## Kết quả đã quan sát (trên máy tôi; hãy chạy lại và so sánh)
- Bề mặt lý tưởng + bootstrap tham số: mọi thiết kế từ 3 seed trở lên PASS.
- Có plateau + vách, mô hình đúng: PASS khi c₂ ≥ 0.2 (bước 2, 5 seed); c₂ ≈ 0.047 cần bước √2 và 5–8 seed.
- **Mô hình sai (bước 08): FAIL rộng.** Ước lượng đỉnh parabol bị lệch hệ thống (â +0.1…+0.4, b̂ −0.3…−0.47) và độ lệch phụ thuộc dạng bề mặt. Chỉ cấu hình "loại điểm collapse + c₂ = 0.5" qua được ở mọi bề mặt kiểm. Ngay cả khi biết đúng bề mặt (cột oracle), nhiều cấu hình ở c₂ ≤ 0.2 vẫn rớt.
- Bước 09–10 cho thấy estimator chia sẻ sửa được độ lệch này nhưng đổi lại nhạy với độ cong đổi theo rank.
- Hệ quả: G0 hiện **chưa** được chứng minh bền. Cần (a) c₂/σ của NLL từ pilot 0C, và (b) một estimator ít nhạy với hình dạng bề mặt (ví dụ mô hình bề mặt chia sẻ giữa các rank/p, dịch chuyển theo x*, thay vì lấy đỉnh parabol từng cell) trước khi chốt thiết kế Pha 1.

## Kết quả bước 09–10 (đã đối chiếu máy local và Kaggle, hạt giống khác nhau)
- Từ v3, mọi bộ phân loại dùng QDA bền với ngoại lệ. Một số kết luận ở v2 (dùng QDA thường) đã đổi: ví dụ `mask` không còn sụp về 0.33 ở c₂ ≤ 0.1 mà tăng đều theo c₂.
- `shared` pass G0 (hiệu chuẩn trên 4 bề mặt, kiểm trên 5 bề mặt khác) ở MỌI thiết kế và MỌI c₂ kể cả 0.047 (mixture tệ nhất 0.96–0.99). Hiệu chuẩn chỉ trên MỘT bề mặt giả định thì tệ nhất ở bước 2 là 0.86–0.87 khi c₂=0.047 ⇒ phải hiệu chuẩn trên họ bề mặt.
- `mask` pass từ c₂ ≥ 0.2 với bước √2 (c₂ ≥ 0.5 với bước 2); fail ở c₂ ≤ 0.1.
- Khi độ cong đổi theo rank (γ=±0.3), `shared` sụp về 0.33–0.45 ở mọi c₂; `mask` giữ ≥ 0.96 từ c₂ ≥ 0.2.
- Nghĩa là: nếu hình dạng bề mặt chia sẻ giữa các rank thì `shared` đủ ở cả c₂=0.047, nhưng ở c₂ thấp không có cách kiểm chứng giả định đó (xem bước 11).

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

## Đối chiếu local (Windows, hạt giống mặc định) và Kaggle (G0_SEED_OFFSET=1000)
- Local khớp từng số với môi trường Linux của người viết ở mọi bước ⇒ kết quả tái lập được giữa các máy.
- Kaggle (hạt giống khác): mọi kết luận PASS/FAIL ở bước 01–04, 08–10 giữ nguyên; số liệu chênh trong ±0.01–0.03. Ngoại lệ biên: bước 05, bước 2.5 với 3 seed có độ phủ 0.926/0.935 (Kaggle) so với 0.938/0.940 (local) ⇒ biên sát ngưỡng 0.93, không coi là pass chắc.
- Bước 06–07 ở v3 còn dùng QDA thường nên không ổn định giữa hai lần chạy (ví dụ 07: `mask`, bước √2, 5 seed, c₂=0.1: 0.828 local vs 0.969 Kaggle). v3.2 đổi sang bộ phân loại bền; hai hạt giống 0 và 1000 khớp trong ±0.01.
- Cột bias trước đây là trung bình nên bị ngoại lệ kéo (ví dụ +63); v3.2 dùng trung vị.

## Giới hạn đã biết của bộ kiểm toán
- Chưa mô phỏng: tập rank 5/7+128/256, ít mức p hơn, estimator GP, bước lưới 1.25 cùng bước 08, độ lệch một phía kiểu khác, nhiễu phụ thuộc vị trí LR.
- Bootstrap tham số (bước 05) dựa vào giả định parabol nên lạc quan khi bề mặt thật khác.
- Các con số 0.9 và [0.93, 0.97] là ngưỡng của v3; sai số Monte Carlo của độ phủ ~±0.005 ở 2,000 lần lặp mỗi giả thuyết.
