# Lora-Rank-Lr-Scaling

**Chuyển giao learning rate giữa các rank LoRA: phép kiểm phân biệt có đăng ký trước, nguyên lý đẳng-bước và chi phí GPU**

*LR tối ưu của LoRA co giãn theo rank như thế nào dưới các cách đặt hệ số co giãn khác nhau, lý thuyết nào đúng, và một phép đo ngắn có thay được việc tune riêng từng rank không?*

Tác giả: Lê Quang Phát (Khoa Công nghệ Thông tin, Trường Đại học Quy Nhơn)
Trạng thái (2026-10-21): **Pha 0, đang làm.** Mô phỏng nhận dạng 0A đã chạy (chưa phán quyết được cổng G0); 0A-bis, 0B, 0C chưa làm; chưa có lượt GPU nào cho Pha 1. Chưa đăng ký trước.

## Bối cảnh

Bài kiểm toán trước (rank LoRA toàn cục, XLM-R/UIT-VSFC, repo `Peft-Rank-Audit`, đã hoàn thành) cho thấy LR tối ưu giảm khoảng 20 lần khi rank tăng 32 lần dưới α = 2r, và tương tác rank × epoch đảo dấu theo cách chọn LR. Từ 5 điểm rank và lưới LR thô, độ dốc log η\* theo log r ước lượng được khoảng 0,82 với CI 95% xấp xỉ [0,13; 1,51], bao trùm cả 0,5 lẫn 1, nên dữ liệu cũ không phân biệt được các giả thuyết về cách LR co giãn theo rank.
Repo này là dự án riêng, bắt đầu từ câu hỏi đó. Những gì giữ lại và bỏ đi: `docs/MIGRATION_from_peft_rank_audit.md`.

## Câu hỏi và giả thuyết

Gọi σ_p = −∂ log η\* / ∂ log r (dương nghĩa là LR tối ưu giảm khi rank tăng), dưới tham số hóa γ_r = 2·(r/8)^(−p) của hệ số co giãn ΔW = γ_r·BA. Cả ba tham số hóa cùng bằng 2 tại rank tham chiếu r = 8:

| Tham số hóa | p | α(r) | Ghi chú |
|---|---|---|---|
| P1 | 0 | 2r | như dự án trước; tham số hóa chính |
| P3 | 1/2 | ≈ 5,66·√r | rsLoRA |
| P2 | 1 | 16 | LoRA gốc, α cố định |

| Giả thuyết | Cơ chế (q, u, k) | σ_p | p = 0 | p = 1/2 | p = 1 |
|---|---|---|---|---|---|
| Hκ=1: các chiều rank cộng kết hợp | (1, 1, 1) | 1 − p | 1 | 1/2 | 0 |
| HμA: chế độ r^(−1/2) của Chen–Villar–Hayou (ICML 2026), *cách đọc tạm thời, cần xác minh* | (2, 1, 1) | (1 − p)/2 | 1/2 | 1/4 | 0 |
| Hκ=1/2: các chiều cộng như bước ngẫu nhiên | (1, 1, 1/2) | 1/2 − p | 1/2 | 0 | −1/2 |

Một tham số hóa không đủ (Hκ=1/2 trùng HμA ở p = 0), nên thiết kế là giai thừa ba mức p. Nguyên lý đẳng-bước (A1′): LR tối ưu ở mỗi rank cho cùng độ lớn cập nhật hiệu dụng E đo sau vài chục bước; luật R-probe suy LR từ phép đo đó mà không cần khớp σ.

Sáu câu hỏi: RQ1 phân xử giữa ba giả thuyết; RQ2 đẳng-bước và số mũ phản hồi (q, u, k); RQ3 quy tắc chuyển giao (R-known, R-μA, R-est, R-probe) có mất quá δ_R so với tune riêng, và chi phí GPU-giờ; RQ4 dự đoán ngoài mẫu (mô hình, kích thước, tác vụ, ngôn ngữ) và độ trôi theo ngân sách; RQ5 chọn rank tuần tự (tầng X); RQ6 can thiệp vào độ kết hợp (tầng S).

Bài **không** claim: phát hiện việc co giãn LR theo rank (đã có: LoRA, rsLoRA, μP, μA, *LoRA Without Regret*); rằng rank không quan trọng; kết quả cho decoder lớn; một luật phổ quát; khung lý thuyết chặt chẽ (đây là lập luận heuristic sinh giả thuyết); quan hệ nhân quả, trừ phần can thiệp L4 nếu làm.

**Bậc thang đóng góp.** L0 bộ dữ liệu bề mặt LR mở (kèm logits và log cơ chế); L1 phép kiểm phân biệt (σ_p đo được, lý thuyết nào được ủng hộ); L2 giải thích bằng cơ chế (đẳng-bước, bản đồ miền hợp lệ); L3 quy tắc thực hành với regret ≤ δ_R và GPU-giờ tiết kiệm đo được; L4 bằng chứng can thiệp. Kịch bản kết quả: A (R-probe/R-est thắng quy tắc lý thuyết), B (một lý thuyết đủ tốt), B′ (κ = 1 đúng ở mọi cell), C (không nhận dạng được hoặc không chuyển giao; bài kết quả âm). Không kịch bản nào được bảo đảm; xác suất rơi vào A chưa biết.

## Đề xuất hiện hành

Đặc tả đầy đủ là bản v3 (`docs/proposal_v3_original_2026-10-20.docx`); các thay đổi sau khi chạy Pha 0A nằm ở **`docs/PROPOSAL_v3.1.md`**. Tóm tắt thay đổi:

| # | Thay đổi | Vì sao |
|---|---|---|
| A1 | G0 đo bằng `unsafe%` ≤ 5% ở mọi bề mặt và mọi độ cong γ, kèm cờ khi hai ước lượng bất đồng; bỏ tiêu chí độ phủ CI theo seed | Tiêu chí cũ đạt trên bề mặt lý tưởng nhưng không bền khi mô hình sai |
| A2 | Mô phỏng lại đúng thiết kế đã lập ngân sách (0A-bis), thêm regret | 0A mô phỏng 7 rank ở cả 3 mức p và chưa tính regret; ngân sách chỉ có 4 rank cho P2/P3 |
| A3 | Seed chọn ≥ 5 (v3: 3); L = 7 điểm LR chỉ ứng với bước 2, bước √2 cần 14 điểm | Giao thức cuối chỉ kiểm n = 5 và 8 |
| A4 | Luôn tính hai ước lượng (parabol từng rank, bề mặt chia sẻ); bỏ lập luận "bề mặt chia sẻ cho phép bớt điểm LR" | Ước lượng chia sẻ sụp khi \|γ\| ≥ 0,15 |
| A5 | 0C đo thêm σ, c₂ của NLL và γ̂; thiết kế Pha 1 chọn theo c₂ | σ, c₂ trong mô phỏng lấy từ Macro-F1 chứ không phải NLL |
| A6 | Ngân sách tầng M: 1.458 run (v3, chưa xác nhận) → 2.102 / 3.712 / 5.644 run tùy kịch bản | Hệ quả của A3; thời gian đo thật 249 s/lượt (6 epoch) |
| A7 | Đăng ký 0 không còn "trước khi thấy kết quả" cho 0A; 0A ghi là mô phỏng phát triển giao thức | 0A đã chạy và chỉnh qua nhiều phiên bản |

## Kế hoạch và cổng

Thứ tự ưu tiên cố định G0 → G0B → G1 → G2; mỗi cổng có ba trạng thái (ĐẠT / TRƯỢT / CHƯA KẾT LUẬN), chỉ được mở rộng một lần.

| Pha | Nội dung | Trạng thái |
|---|---|---|
| 0A | Mô phỏng nhận dạng (CPU) | chạy xong v3.2; kết luận ở `phase0/README.md` |
| 0A-bis | Mô phỏng thiết kế thật + regret | chưa làm |
| 0B | Công cụ đo E_W, E_f, ρ: năm phép kiểm T1–T5 | chưa làm |
| 0C | Pilot NLL: D1, rank {2, 8, 64}, 6 LR, 3 seed (54 lượt) | chưa làm |
| 1 | Khám phá D1–D3 ở P1; giai thừa P2/P3 trên D1; đo cơ chế từ log quét; G1 | chưa bắt đầu |
| 2 | Xác nhận trên hai cell nặng khác họ mô hình và tác vụ (H2, H3); phát lại offline; G2 | chưa bắt đầu |
| S, X | Tầng mở rộng (can thiệp, rank 128/256, XLM-R-large, decoder) | chỉ mở khi G1 qua |

Ba lần đăng ký trước (R0, R1, R2) xem `registrations/`; chưa nộp lần nào, chưa gắn nền tảng.

Kết quả 0A tóm tắt: với σ = 0,0136, giao thức an toàn khi c₂ ≥ 0,2 (bước 2, 5 seed), an toàn ở c₂ = 0,1 chỉ với bước √2 và 8 seed, và **không an toàn** ở c₂ = 0,047 (giá trị Macro-F1 của audit cũ). Vì vậy việc kế tiếp quyết định là đo c₂ của NLL ở pilot 0C.

## Cấu trúc repo

```text
Lora-Rank-Lr-Scaling/
├── README.md, CHANGELOG.md, CITATION.cff, LICENSE, requirements.txt
├── docs/
│   ├── PROPOSAL_v3.1.md              đề xuất hiện hành (thay đổi so với v3)
│   ├── proposal_v3_original_2026-10-20.docx
│   ├── MIGRATION_from_peft_rank_audit.md
│   └── SETUP_KAGGLE.md
├── phase0/
│   ├── README.md                     trạng thái và kết quả Pha 0
│   ├── 0A_identification/            mô phỏng nhận dạng v3.2 + logs
│   ├── 0B_measurement/               kiểm định công cụ đo (đặc tả)
│   └── 0C_pilot/                     pilot NLL (đặc tả)
├── code/                             run_pilot.py, rankcap.py, chuẩn bị dữ liệu, test
├── data_prior/                       dữ liệu khởi điểm từ dự án trước (chỉ tham chiếu)
├── data/                             dữ liệu (không đưa lên git)
├── registrations/                    đăng ký trước (chưa có)
├── results/                          kết quả chạy mới (trống)
└── scripts/                          init_repo.ps1, push_cycle.ps1
```

## Cách chạy

Pha 0A (CPU, chỉ cần numpy và scipy):

```bash
cd phase0/0A_identification
python run_all.py quick      # kiểm tra môi trường, KHÔNG dùng để kết luận
python run_all.py            # đầy đủ, 60–90 phút
```

Huấn luyện (cần GPU; chi tiết ở `docs/SETUP_KAGGLE.md`). Ba tham số hóa chọn bằng `--alpha_mode`: `2r` hoặc `p:0` (P1), `p:0.5` (P3), `p:1` (P2).

```bash
pip install -r requirements.txt
python -m pytest code/test_rankcap.py code/test_run_pilot.py code/test_run_pilot_v3.py -q
```

Hai test mới (`test_gamma_alpha_parametrisations`, `test_alpha_mode_scaling_p_modes`): test đầu thuần Python và đã đạt khi soạn; test sau cần torch và peft nên **chưa chạy được** khi soạn, hãy chạy `pytest` trên máy có torch trước khi dùng `p:<p>` cho lượt thật.

## Dữ liệu và đạo đức nghiên cứu

Dữ liệu không được phân phối lại; xem `data/README.md`. Giấy phép dataset và mô hình phải kiểm tra trước khi phát hành dữ liệu dẫn xuất. Mọi phân tích sẽ gắn nhãn preregistered, post hoc hoặc exploratory.

## Tham khảo chính

Chen, Villar & Hayou, *Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning* (ICML 2026, arXiv:2602.06204), mới đọc tóm tắt; *LoRA Without Regret* (Thinking Machines); Hu et al., LoRA (ICLR 2022); Kalajdzievski, rsLoRA (2023); Hayou et al., LoRA+ (2024); Yang et al., μP. Danh sách còn lại và các điều phải xác minh (V1–V5) ở v3, mục 2.

## Giấy phép

MIT, xem `LICENSE`.
