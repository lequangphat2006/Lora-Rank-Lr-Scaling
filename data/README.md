# Dữ liệu

Dữ liệu **không** được đưa vào repo (xem `.gitignore`). Đặt các file đã chuẩn bị vào thư mục con tại đây
(mỗi task một thư mục, gồm `train.csv`, `calibration.csv`, `test.csv`, cột `text`, `label`).

| Cell (ứng viên) | Thư mục | Nguồn | Ghi chú |
|---|---|---|---|
| D1, D2 | `data/vsfc_sent/` | UIT-VSFC (sentiment) | Tạo bằng `python code/prepare_vsfc.py --train ... --val ... --test ... --out_dir data/vsfc_sent`. PhoBERT cần cột `text_seg` (`--text_col text_seg`, thư mục riêng). |
| D3, D4, H1–H3, L1–L3 | chưa chốt | xem `docs/PROPOSAL_v3.1.md` mục 5.2 | Phải kiểm tra cỡ test, lớp thiểu số, giấy phép, near-duplicate (`python code/check_near_dup.py --data_dir <thư mục>`) trước khi chốt. |

Train/calibration/test của UIT-VSFC sentiment dùng ở dự án trước: 12.780 / 1.598 / 1.599 mẫu; near-duplicate với train: 5,19% (calibration), 3,75% (test).
Chỉ dùng `test.csv` để báo cáo; `calibration.csv` dùng để chọn LR và fit temperature.
