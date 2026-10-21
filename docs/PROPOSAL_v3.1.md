# Đề xuất kỹ thuật v3.1: chuyển giao LR giữa các rank LoRA

*Cập nhật 2026-10-21. Tác giả: Lê Quang Phát. Thay thế v3 (2026-10-20) ở những mục nêu dưới đây; mọi mục không được nhắc đến giữ nguyên như v3.*

Bản v3 gốc: `docs/proposal_v3_original_2026-10-20.docx` (đây là file đã gửi; tên "De_xuat_de_tai_PEFT_ban_chinh_sua_lan3" trong tin nhắn là một bản khác, thuộc hướng phân bổ rank, không dùng ở repo này).
Cách đọc: v3 (docx) là đặc tả đầy đủ; file này ghi **điều đã thay đổi vì Pha 0A đã chạy**, kèm lý do và số liệu. Số liệu 0A: `phase0/README.md`.

## 0. Tóm tắt đề tài (không đổi)

Đo bằng thiết kế đăng ký trước LR tối ưu của LoRA co giãn theo rank thế nào dưới ba tham số hóa hệ số co giãn γ_r = 2·(r/8)^(−p), p ∈ {0, 1/2, 1}; phân xử giữa Hκ=1 (σ_p = 1 − p), HμA (σ_p = (1 − p)/2) và Hκ=1/2 (σ_p = 1/2 − p); kiểm nguyên lý đẳng-bước (LR tối ưu cho cùng độ lớn cập nhật hiệu dụng E) và luật R-probe; đo regret và chi phí GPU so với tune riêng từng rank. Nối tiếp bài kiểm toán rank LoRA toàn cục trên XLM-R/UIT-VSFC (dự án trước, đã hoàn thành). Đối thủ lý thuyết trực tiếp: Chen–Villar–Hayou (ICML 2026, arXiv:2602.06204), và *LoRA Without Regret*. Ánh xạ μA sang cấu hình chuẩn của PEFT (V1) vẫn chưa xác minh; hàng HμA ở Bảng 1 là cách đọc tạm thời.

## Tình trạng thực hiện

| Hạng mục (v3) | Trạng thái |
|---|---|
| Đọc toàn văn Chen et al. và V1–V5 | Chưa có bằng chứng trong repo; **bắt buộc trước R0** |
| Đăng ký 0 | Chưa nộp (xem A7) |
| 0.1 hồi quy 5 rank | Chưa thấy kết quả |
| 0A mô phỏng nhận dạng | Đã chạy (v3.2); chưa đủ để phán G0 (A1, A2, A5) |
| 0B, 0C | Chưa làm |
| Code: tham số hóa P1/P2/P3 | Đã thêm `--alpha_mode p:<p>` (`code/rankcap.py`); test thuần Python đạt; test tích hợp peft chưa chạy được ở môi trường soạn thảo, cần `pytest` trên máy có torch |
| Code: log E_W, E_f, ρ | Chưa có |

## A1. Định nghĩa lại cổng G0 (thay mục 6.3, 13.1, Phụ lục A)

v3: G0 đạt khi P(phân loại đúng 3 lớp) ≥ 0,9 và độ phủ CI ∈ [0,93; 0,97].
Phát hiện: tiêu chí đó đạt được trên bề mặt parabol lý tưởng nhưng vô nghĩa khi mô hình bề mặt sai; độ phủ CI theo seed không đánh giá được ở 3 seed (0,82–0,92 chỉ vì có 10 cách lấy lại mẫu).
**v3.1:** G0 đạt khi, trên **mọi** bề mặt kiểm khác bề mặt hiệu chuẩn và γ ∈ {−0,3; −0,15; 0; +0,15; +0,3}: (i) `unsafe%` = P(không bị cờ và phân loại sai) ≤ 5%; (ii) độ chính xác khi được phép kết luận ≥ 0,9 tại γ = 0; (iii) tỉ lệ cờ tại γ = 0 xấp xỉ 5% (đúng thiết kế). Độ phủ CI chỉ báo cáo bằng bootstrap tham số, như chỉ số phụ. Trong phân tích dữ liệu thật: nếu cờ bật thì **không được kết luận về số mũ** (trạng thái CHƯA KẾT LUẬN, theo quy tắc mở rộng một lần của mục 13.1).

## A2. Thiết kế đã mô phỏng khác thiết kế đã lập ngân sách

| | Mô phỏng 0A | Ngân sách v3 (mục 13.3) |
|---|---|---|
| Rank | 7 rank {1,…,64} ở **cả ba** mức p | P1: 7 rank; P2, P3: chỉ 4 rank {2, 8, 32, 64} ở D1 |
| Lưới LR | 2 decade; bước 2 → 7 điểm, bước √2 → **14 điểm**, bước 2,5 → 6 điểm | 7 điểm LR, nhưng mục 5.5 nêu bước 0,15 dex (≈ √2) |
| Seed chọn | 5 và 8 (giao thức cuối); 3 chỉ có ở các bước đầu | 3 |
| Regret của R-est, R-known, R-μA | **không được mô phỏng** | tiêu chí "Go hẹp" của G0 dựa vào nó |

Hệ quả: 7 điểm LR chỉ tương ứng bước 2 (an toàn khi c₂ ≥ 0,2), còn bước √2 cần 14 điểm; 3 seed chọn chưa từng được xác nhận bởi giao thức cuối; thiết kế 4 rank cho P2/P3 và nhánh "Go hẹp" chưa có bằng chứng.
**Việc phải làm (0A-bis, trước R0):** mô phỏng đúng thiết kế sẽ chạy (7 rank P1 + 4 rank P2/P3), với seed chọn n ∈ {3, 5, 8}, và thêm đầu ra regret cho R-est, R-known, R-μA, R-probe.

## A3. Lưới LR và số seed chọn (thay mục 5.5, 13.3)

Số điểm LR L = số điểm trên 2 decade = 7 (bước 2), 14 (bước √2). Số seed chọn tối thiểu đã xác nhận: 5, và chỉ khi c₂ ≥ 0,2 (A5). Quy tắc dừng seed theo mục 7.6 (8 → 12 → 16) áp dụng cho cell xác nhận; ở cell khám phá, nếu cờ G0 bật thì mở rộng 5 → 8 một lần.

## A4. Bộ ước lượng (thay mục 5.5, 7.1)

v3 dùng mô hình bề mặt chia sẻ hình dạng để giảm số điểm LR (7 thay vì 9, −22% run). Bỏ lập luận đó: bộ ước lượng chia sẻ chỉ đáng tin khi hình dạng bề mặt thực sự chia sẻ giữa rank, và 0A cho thấy nó sụp ngay khi |γ| ≥ 0,15.
**v3.1:** luôn tính **hai** ước lượng, (a) parabol cửa sổ từng rank (`mask`, mục 7.1 của v3) và (b) bề mặt chia sẻ; phán quyết theo giao thức A1 (không kết luận nếu hai ước lượng bất đồng). Nếu 0C cho thấy γ̂ ≠ 0 rõ rệt, mở rộng ước lượng chia sẻ thêm tham số γ trước khi chốt R1 (chưa được mô phỏng).

## A5. Pilot 0C mở rộng và ánh xạ sang thiết kế (thay mục 6.1 bước 0.3)

0C vẫn là 54 lượt trên D1, nhưng phải xuất thêm: σ của NLL giữa seed, c₂ của NLL từng rank, và γ̂ (độ cong đổi theo rank). Thiết kế Pha 1 chọn theo c₂ (NLL), với giả định tỉ lệ c₂/σ là đại lượng quyết định (chỉ σ = 0,0136 được chạy ở giao thức cuối, nên giả định này chưa được kiểm độc lập):

| c₂ của NLL (σ = 0,0136) | Thiết kế Pha 1 | Căn cứ 0A |
|---|---|---|
| ≥ 0,2 và \|γ̂\| nhỏ | Bước 2 (7 điểm), 5 seed | an toàn ở mọi dòng bảng bước 11 |
| 0,1 đến dưới 0,2 | Bước √2 (14 điểm), 8 seed | chỉ c₂ = 0,1 được kiểm (an toàn); đoạn giữa chưa mô phỏng |
| dưới 0,1 | "Go hẹp": bỏ claim số mũ; giữ RQ3, RQ4 | c₂ = 0,047 không an toàn ở mọi thiết kế; **nhánh này chưa được kiểm bằng regret** (A2) |
| \|γ̂\| ≥ 0,15 | Mở rộng ước lượng chia sẻ thêm γ, mô phỏng lại, rồi mới chọn thiết kế | giao thức chặn gần hết ở c₂ ≥ 0,2 |

## A6. Ngân sách tầng M (thay bảng mục 13.3 và mục 5.1)

Cùng công thức v3 (Pha 0: 56; Pha 1: 4 cell-ngân sách × 7 rank × L × s + giai thừa P2/P3 trên D1 + 20; Pha 2: 2 cell nặng), thay L và s; thời gian theo hai mức: 4,2 phút/lượt (đo được: 249 s ở 6 epoch, XLM-R-base, T4 fp16, không đổi theo rank) và 8 phút/lượt (cận trên cho DeBERTa-v3, ViSoBERT, chưa đo).

| Kịch bản | Run (Pha 0 / 1 / 2) | Tổng run | GPU-giờ (4,2 → 8 phút) | Tuần ở 35 GPU-giờ/tuần |
|---|---|---|---|---|
| v3 gốc: L = 7, s = 3 (chưa được xác nhận) | 56 / 776 / 626 | 1.458 | 102 → 194 | 2,9 → 5,6 |
| A: bước 2, L = 7, s = 5 | 56 / 1.280 / 766 | 2.102 | 147 → 280 | 4,2 → 8,0 |
| B: bước √2, L = 14, s = 5 | 56 / 2.540 / 1.116 | 3.712 | 260 → 495 | 7,4 → 14,1 |
| C: bước √2, L = 14, s = 8 | 56 / 4.052 / 1.536 | 5.644 | 395 → 753 | 11,3 → 21,5 |

Các số này giữ nguyên ghi chú của v3: cell xác nhận nặng dùng cùng L và s ở đây; có thể giảm bằng lưới địa phương quanh LR dự đoán (như cell nhẹ), nhưng đó là thay đổi thiết kế phải qua đăng ký. Thứ tự cắt ở mục 13.3 của v3 vẫn giữ nguyên, thêm mục đầu: "dùng kịch bản A thay vì B hoặc C nếu 0C cho phép". Thời gian chạy không phụ thuộc rank nên chi phí GPU của một lượt là như nhau ở mọi rank; mô hình chi phí ở mục 9.1 chỉ cần đếm số lượt và tỉ lệ t_p/tổng số bước.

## A7. Đăng ký 0 chỉ khóa giao thức G0 v3.2, ngưỡng 0B, cổng G0/G0B và ma trận quyết định 0C (sửa mục 12)

v3 nói R0 khóa ngưỡng trước khi thấy kết quả 0A. Thực tế 0A đã được chạy và chỉnh qua nhiều phiên bản (v1 → v3.2; ngưỡng, bộ phân loại và giao thức đổi sau khi xem kết quả). Vì vậy: 0A **và 0A-bis** được ghi là **mô phỏng phát triển giao thức (exploratory)**, không khóa trong R0; R0 khóa giao thức v3.2, ngưỡng 0B, cổng G0/G0B và ma trận quyết định 0C; bài báo nêu rõ điều này trong phần hạn chế. R0 chỉ nộp **sau** 0A-bis và **trước** mọi lượt GPU của 0B, 0C.

## A8. Lộ trình cập nhật (thay mục 13.2 và Phụ lục C)

1. Đọc toàn văn Chen et al. (V1 đến V4), *LoRA Without Regret*, arXiv:2601.22708 (V5); sửa Bảng 1 nếu ánh xạ HμA đổi.
2. Chạy bước 0.1 (hồi quy σ̂₀ trên `data_prior/audit_tuning/tuning_full.csv`, bootstrap theo seed).
3. 0A-bis (CPU): thiết kế thật, n ∈ {3, 5, 8}, thêm regret; thêm γ vào ước lượng chia sẻ nếu cần.
4. 0B-T1 (CPU, numpy) và gắn log E_W, E_f, ρ vào `code/run_pilot.py`; chạy T2 trên D1.
5. Nộp R0 (xem `registrations/`).
6. Chạy 0C (54 lượt) với đầu ra A5; chốt kịch bản A, B hoặc C hoặc "Go hẹp" theo bảng A5 → G0, G0B.
7. R1, rồi Pha 1 → G1 → R2 → Pha 2 → G2, như v3.

Mốc thời gian phụ thuộc quota GPU; với 30 đến 40 GPU-giờ/tuần, Pha 1 + 2 mất 4 đến 21 tuần tùy kịch bản (bảng A6), chưa kể viết bài.

## A9. Những điều không đổi

Ma trận cell, giao thức cố định (mục 5.3, nay thêm `--alpha_mode p:<p>` cho P1, P3, P2), ước lượng η\* bằng parabol cửa sổ và bootstrap (7.1), mô hình phân cấp (7.2), các kiểm định H1–H5, endpoint chi phí và phát lại offline (mục 9), mục tiêu bài báo và bậc thang đóng góp L0–L4, phạm vi claim (1.5), danh sách "điều cần tránh" (11.2).
