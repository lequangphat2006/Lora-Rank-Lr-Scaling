# Từ Peft-Rank-Audit sang Lora-Rank-Lr-Scaling

Dự án cũ (bài kiểm toán rank LoRA toàn cục, XLM-R/UIT-VSFC) đã hoàn thành và giữ nguyên ở repo `Peft-Rank-Audit`. Repo này là dự án mới, **không** thừa kế lịch sử git, tag, đăng ký trước hay tác giả của repo cũ.

## Giữ lại

| Thành phần | Vị trí mới | Ghi chú |
|---|---|---|
| Bộ huấn luyện (sweep, resume, logits, calibration) | `code/run_pilot.py`, `code/rankcap.py` | Thêm `--alpha_mode p:<p>`; phần phân bổ rank theo lớp trong `rankcap.py` là di sản, không dùng |
| Chuẩn bị dữ liệu, near-duplicate | `code/prepare_vsfc.py`, `code/check_near_dup.py` | |
| Test | `code/test_*.py` | `test_rankcap.py` thêm test tham số hóa |
| Bộ kiểm toán Pha 0A v3.2 + log | `phase0/0A_identification/` | Từ `_inbox/kiem_toan_G0_v3_2`, dọn trùng lặp |
| Dữ liệu khởi điểm (69 lần tune, R1, smoke) | `data_prior/` | Chỉ tham chiếu; ghi nguồn commit |
| Đề xuất v3 (docx) | `docs/proposal_v3_original_2026-10-20.docx` | |

## Bỏ

Toàn bộ `.git` (kể cả mọi tag); mọi thông tin đăng ký trước cũ (DOI, URL, tài liệu liên quan); bản thảo, kế hoạch phân tích, nhật ký kết quả, hình, `docs/archive/`, `pivot_v2`, `roadmap_v2`, `lit_review`, `limitations`, `decisions`; kết quả R1/B1/B2/AdaLoRA/V1–V8, logits `.npz` (165 file) và `epochwise`; mã phân tích riêng của bài cũ (`analyze_*`, `make_figures`, `run_adalora`, `run_phase1a`, `power_*`, `merge_stage1c`); bản sao trùng và bản thử cũ trong `_inbox` (`g0_robust.py`, `g0_robust_v2.py`, các `g0_*.py` ở gốc `_inbox`); `scripts/pull_kaggle.ps1` (trỏ kernel cũ); `CHANGELOG.md` và `README.md` cũ.

## Lưu ý về bản sao trên Desktop

- `README.md` trong thư mục `peft-rank-audit` bạn gửi bị hỏng mã hóa (UTF-8 bị đọc lại thành cp1252, hiện ra "KiÃ¡Â»Æ’m toÃƒÂ¡n"). `CHANGELOG.md` cũng đang ở trạng thái `modified`. Nếu cần giữ bản cũ, khôi phục bằng `git checkout -- README.md CHANGELOG.md` trong thư mục cũ.
- Thư mục `_inbox` bị `.gitignore` chặn; hai commit lưu trữ bằng chứng Pha 0 đã bị revert. Bằng chứng 0A giờ nằm ở `phase0/` trong repo mới.
- Nên **đổi tên** thư mục cũ thành `peft-rank-audit_ARCHIVE` thay vì sửa tại chỗ, để `.git` cũ còn nguyên.

## Các bước (PowerShell)

```powershell
cd C:\Users\lequa\Desktop
Rename-Item peft-rank-audit peft-rank-audit_ARCHIVE
Expand-Archive Lora-Rank-Lr-Scaling.zip -DestinationPath .          # tạo Lora-Rank-Lr-Scaling\
Copy-Item -Recurse peft-rank-audit_ARCHIVE\data\vsfc_sent Lora-Rank-Lr-Scaling\data\vsfc_sent   # dữ liệu, bị .gitignore chặn
cd Lora-Rank-Lr-Scaling
.\scripts\init_repo.ps1
```

Nếu repo GitHub mới đã có commit khởi tạo (README hoặc LICENSE do GitHub tạo), `init_repo.ps1` sẽ dừng ở bước push; chạy `git pull origin main --allow-unrelated-histories` rồi push lại.
