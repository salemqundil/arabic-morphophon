# Arabic Phonology Engine - Bootstrap Script
# تسريع إعداد البيئة والمشروع

$ErrorActionPreference = "Stop"

Write-Host "🚀 Arabic Phonology Engine - Bootstrap Setup" -ForegroundColor Green

# 1. إعداد Git
Write-Host "📝 Setting up Git configuration..." -ForegroundColor Yellow
$gitUserName = Read-Host "Enter your Git username"
$gitUserEmail = Read-Host "Enter your Git email"

git config --global user.name "$gitUserName"
git config --global user.email "$gitUserEmail"

# 2. إنشاء مستودع GitHub
Write-Host "📁 Creating GitHub repository..." -ForegroundColor Yellow
$repoName = "arabic-phonology-engine"
gh repo create $repoName --public --description "Dynamic Arabic Phonological Analysis Engine with Real-time Features"

# 3. إعداد SSH Key
Write-Host "🔐 Setting up SSH key..." -ForegroundColor Yellow
if (!(Test-Path "$env:USERPROFILE\.ssh\id_rsa")) {
    ssh-keygen -t rsa -b 4096 -C "$gitUserEmail" -f "$env:USERPROFILE\.ssh\id_rsa" -N ""
    Write-Host "Add this SSH key to GitHub:" -ForegroundColor Cyan
    Get-Content "$env:USERPROFILE\.ssh\id_rsa.pub"
    Read-Host "Press Enter after adding SSH key to GitHub"
}

# 4. تثبيت Poetry (اختياري)
Write-Host "📦 Installing Poetry..." -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri https://install.python-poetry.org | python -
} catch {
    Write-Host "Poetry installation skipped or already installed" -ForegroundColor Yellow
}

# 5. إضافة المكتبات الصوتية
Write-Host "🎵 Installing phonetic libraries..." -ForegroundColor Yellow
& "C:/Users/Administrator/new engine/.venv/Scripts/python.exe" -m pip install praat-parselmouth arabic-reshaper phonemizer librosa

# 6. إعداد ملفات المشروع
Write-Host "📄 Creating project configuration files..." -ForegroundColor Yellow

Write-Host "✅ Bootstrap setup completed!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Push code to GitHub: git push origin main" -ForegroundColor White
Write-Host "2. Setup GitHub Actions" -ForegroundColor White
Write-Host "3. Configure project documentation" -ForegroundColor White
