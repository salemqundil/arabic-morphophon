# Docker WSL Reset Script
# Run this script as Administrator to reset Docker WSL integration

# Stop Docker Desktop
Write-Host "Stopping Docker Desktop..." -ForegroundColor Yellow
Get-Process "Docker Desktop" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5

# Shut down all WSL instances
Write-Host "Shutting down WSL..." -ForegroundColor Yellow
wsl --shutdown

# Reset Docker Desktop WSL directories
$dockerWSLPath = "C:\Users\Administrator\AppData\Local\Docker\wsl"
if (Test-Path $dockerWSLPath) {
    Write-Host "Backing up Docker WSL data..." -ForegroundColor Yellow
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = "$env:USERPROFILE\Desktop\Docker_WSL_Backup_$timestamp"
    New-Item -Path $backupDir -ItemType Directory -Force

    # Try to copy files first (safer)
    try {
        Write-Host "Copying WSL data to backup location..." -ForegroundColor Green
        Copy-Item -Path "$dockerWSLPath\*" -Destination $backupDir -Recurse -Force
        Write-Host "WSL data backed up to $backupDir" -ForegroundColor Green
    }
    catch {
        Write-Host "Backup failed but continuing with reset..." -ForegroundColor Yellow
    }
}

# Check for WSL updates
Write-Host "Checking for WSL updates..." -ForegroundColor Green
wsl --update

# Set WSL 2 as default
Write-Host "Setting WSL 2 as default..." -ForegroundColor Green
wsl --set-default-version 2

# Start Docker Desktop
Write-Host "Starting Docker Desktop..." -ForegroundColor Green
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

Write-Host "`nDocker WSL integration has been reset." -ForegroundColor Cyan
Write-Host "Docker Desktop is restarting with fresh WSL configuration." -ForegroundColor Cyan
Write-Host "If issues persist, please restart your computer and try again." -ForegroundColor Cyan
Write-Host "Reset complete!" -ForegroundColor Green
