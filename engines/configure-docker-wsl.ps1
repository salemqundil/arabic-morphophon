# Docker WSL Configuration Script
# Run this script as Administrator to configure Docker with WSL 2

# Stop Docker Desktop if it's running
Write-Host "Stopping Docker Desktop if running..." -ForegroundColor Yellow
Get-Process "Docker Desktop" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5

# Create Docker config directory if it doesn't exist
Write-Host "Creating Docker configuration directory..." -ForegroundColor Green
$configDir = "C:\ProgramData\Docker\config"
if (-not (Test-Path $configDir)) {
    New-Item -Path $configDir -ItemType Directory -Force
}

# Copy the daemon.json file to the correct location
Write-Host "Installing Docker daemon configuration..." -ForegroundColor Green
$sourceDaemonJson = "$PSScriptRoot\docker-daemon.json"
$destDaemonJson = "C:\ProgramData\Docker\config\daemon.json"
Copy-Item -Path $sourceDaemonJson -Destination $destDaemonJson -Force

# Configure WSL
Write-Host "Configuring WSL..." -ForegroundColor Green

# Create .wslconfig file in the user's home directory
$wslConfigContent = @"
[wsl2]
memory=8GB
processors=4
swap=4GB
localhostForwarding=true
kernelCommandLine=sysctl.vm.max_map_count=262144
"@

$wslConfigPath = "$env:USERPROFILE\.wslconfig"
Set-Content -Path $wslConfigPath -Value $wslConfigContent -Force
Write-Host "WSL configuration file created at $wslConfigPath" -ForegroundColor Green

# Check if Docker WSL distro exists and is healthy
Write-Host "Checking Docker WSL integration..." -ForegroundColor Green
$dockerDistros = wsl --list | Where-Object { $_ -match "docker-desktop" }
if (-not $dockerDistros) {
    Write-Host "Docker WSL distros not found. You may need to reinstall Docker Desktop." -ForegroundColor Yellow
}

# Start Docker Desktop
Write-Host "Starting Docker Desktop..." -ForegroundColor Green
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

Write-Host "`nDocker has been configured with optimized WSL 2 settings." -ForegroundColor Cyan
Write-Host "If Docker doesn't start properly, please restart your computer and try again." -ForegroundColor Cyan
Write-Host "Configuration complete!" -ForegroundColor Green
