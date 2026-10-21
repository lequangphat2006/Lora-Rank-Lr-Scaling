# Khởi tạo repo mới (chạy MỘT lần trong thư mục Lora-Rank-Lr-Scaling, chưa có .git)
if (Test-Path .git) { Write-Host "Đã có .git, dừng."; exit 1 }
git init -b main
git remote add origin https://github.com/lequangphat2006/Lora-Rank-Lr-Scaling.git
git add -A
git status --short
git commit -m "chore: khởi tạo repo (Pha 0A v3.2, đề xuất v3.1)"
git push -u origin main
