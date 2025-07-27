# GitHub Setup Script
# Run this script to configure GitHub connection properly

Write-Host "GitHub Connection Setup" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

# Check current Git configuration
Write-Host "`n1. Current Git Configuration:" -ForegroundColor Yellow
Write-Host "User Name: " -NoNewline; git config --global user.name
Write-Host "User Email: " -NoNewline; git config --global user.email

# Configure Git if needed
Write-Host "`n2. Configuring Git (if needed):" -ForegroundColor Yellow
$userName = Read-Host "Enter your Git username (or press Enter to keep current)"
if ($userName) {
    git config --global user.name $userName
    Write-Host "Username set to: $userName" -ForegroundColor Green
}

$userEmail = Read-Host "Enter your Git email (or press Enter to keep current)"
if ($userEmail) {
    git config --global user.email $userEmail
    Write-Host "Email set to: $userEmail" -ForegroundColor Green
}

# Check current remote
Write-Host "`n3. Current Remote Configuration:" -ForegroundColor Yellow
git remote -v

# Option to update remote URL
Write-Host "`n4. GitHub Repository Setup:" -ForegroundColor Yellow
$newRemote = Read-Host "Enter your actual GitHub repository URL (or press Enter to skip)"
if ($newRemote) {
    git remote set-url origin $newRemote
    Write-Host "Remote URL updated to: $newRemote" -ForegroundColor Green
}

# Test connection
Write-Host "`n5. Testing GitHub Connection:" -ForegroundColor Yellow
$testResult = git ls-remote origin 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ GitHub connection successful!" -ForegroundColor Green
} else {
    Write-Host "✗ GitHub connection failed:" -ForegroundColor Red
    Write-Host $testResult -ForegroundColor Red
    Write-Host "`nTroubleshooting steps:" -ForegroundColor Yellow
    Write-Host "1. Verify the repository URL is correct" -ForegroundColor Yellow
    Write-Host "2. Check if the repository exists on GitHub" -ForegroundColor Yellow
    Write-Host "3. Ensure you have access to the repository" -ForegroundColor Yellow
    Write-Host "4. Consider using SSH instead of HTTPS" -ForegroundColor Yellow
}

# Show current status
Write-Host "`n6. Current Repository Status:" -ForegroundColor Yellow
git status --short

Write-Host "`nSetup completed!" -ForegroundColor Green
