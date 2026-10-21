# Chạy trên Kaggle

Mỗi phiên mới: bật GPU (T4 x2), tắt internet nếu dùng Dataset gắn sẵn, rồi:

```bash
pip uninstall -y torchao
git clone https://github.com/lequangphat2006/Lora-Rank-Lr-Scaling.git /kaggle/working/code_repo
cd /kaggle/working/code_repo && pip install -r requirements.txt
python -m pytest code/test_rankcap.py code/test_run_pilot.py code/test_run_pilot_v3.py -q
```

Chạy một lượt (ví dụ P3, rank 32, LR 2e-4), mỗi GPU một tiến trình, `--shard i/n` và `--out` riêng:

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python run_pilot.py --data_dir /kaggle/input/<dataset>/vsfc_sent \
  --model FacebookAI/xlm-roberta-base --cell D1 --sweep_global --alpha_mode p:0.5 \
  --ranks 2 8 64 --epochs 6 --lr_list 2e-4 3e-4 --seeds 500 501 502 --precision fp16 \
  --shard 0/2 --out /kaggle/working/results/pilot0c_gpu0.csv
```

`--alpha_mode`: `2r` (= `p:0`, P1), `p:0.5` (P3), `p:1` (P2, α = 16); `fixed:N` và `rslora` còn lại từ dự án trước. `run_id` có hậu tố `__ap0.5`, `__ap1` cho các chế độ khác `2r`. Trình huấn luyện đọc lại `--out` và bỏ qua lượt đã xong, nên chạy lại đúng lệnh sau khi bị ngắt. Dải seed cho từng vai trò (chọn LR, đánh giá, đo cơ chế, mô phỏng, baseline) chưa được chốt; chốt trong R1 và **không** dùng seed 200–215, 300–307, 400–402 của dự án trước.
Lưu `/kaggle/working/results` ra Dataset sau mỗi đợt; phiên tối đa ~12 giờ, kiểm tra hạn mức GPU tuần trước khi chạy.
