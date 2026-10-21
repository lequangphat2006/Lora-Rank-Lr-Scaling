# Commit và push một đợt thay đổi
# Usage: .\scripts\push_cycle.ps1 "thông điệp commit"
$msg = $args[0]
if (-not $msg) { Write-Host 'Usage: push_cycle.ps1 "thông điệp commit"'; exit 1 }
git add -A
git commit -m $msg
git push origin main
