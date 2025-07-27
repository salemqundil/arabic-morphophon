# Arabic NLP Project PowerShell Startup
Write-Host "🚀 Arabic NLP Project - Safe Environment" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# Set encoding
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# Set environment variables
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO = "1"

# Activate virtual environment if exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "⚠️ Virtual environment not found" -ForegroundColor Yellow
    Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    & .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    Write-Host "✅ New virtual environment created" -ForegroundColor Green
}

# Run environment setup
if (Test-Path "setup_environment.py") {
    python setup_environment.py
}

# Navigate to safe workspace
if (Test-Path "safe_workspace") {
    Set-Location safe_workspace
} else {
    Write-Host "📁 Creating safe workspace..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Name "safe_workspace" -Force
    Set-Location safe_workspace
}

Write-Host "🎉 Ready for development!" -ForegroundColor Green
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host "Python version:" -ForegroundColor Cyan
python --version
