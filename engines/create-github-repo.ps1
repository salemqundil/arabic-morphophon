# Create New GitHub Repository Script

Write-Host "Create New GitHub Repository" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan

$repoName = Read-Host "Enter repository name (default: arabic-morphophon)"
if (-not $repoName) { $repoName = "arabic-morphophon" }

$githubUsername = Read-Host "Enter your GitHub username"
if (-not $githubUsername) {
    Write-Host "GitHub username is required!" -ForegroundColor Red
    exit 1
}

$description = Read-Host "Enter repository description (optional)"
if (-not $description) { $description = "Arabic Morphophonology Engine" }

$isPrivate = Read-Host "Should the repository be private? (y/N)"
$private = if ($isPrivate -eq 'y' -or $isPrivate -eq 'Y') { "true" } else { "false" }

Write-Host "`nRepository Configuration:" -ForegroundColor Yellow
Write-Host "Name: $repoName" -ForegroundColor White
Write-Host "Username: $githubUsername" -ForegroundColor White
Write-Host "Description: $description" -ForegroundColor White
Write-Host "Private: $private" -ForegroundColor White

$confirm = Read-Host "`nProceed with repository creation? (Y/n)"
if ($confirm -eq 'n' -or $confirm -eq 'N') {
    Write-Host "Repository creation cancelled." -ForegroundColor Yellow
    exit 0
}

# Create repository using GitHub API
$repoUrl = "https://github.com/$githubUsername/$repoName.git"

Write-Host "`nTo complete the setup:" -ForegroundColor Cyan
Write-Host "1. Go to https://github.com/new" -ForegroundColor Yellow
Write-Host "2. Repository name: $repoName" -ForegroundColor Yellow
Write-Host "3. Description: $description" -ForegroundColor Yellow
Write-Host "4. Set visibility: $(if ($private -eq 'true') { 'Private' } else { 'Public' })" -ForegroundColor Yellow
Write-Host "5. Click 'Create repository'" -ForegroundColor Yellow
Write-Host "`nThen run these commands:" -ForegroundColor Cyan
Write-Host "git remote set-url origin $repoUrl" -ForegroundColor White
Write-Host "git branch -M main" -ForegroundColor White
Write-Host "git push -u origin main" -ForegroundColor White

# Option to set the remote URL now
$setRemote = Read-Host "`nSet remote URL now? (Y/n)"
if ($setRemote -ne 'n' -and $setRemote -ne 'N') {
    git remote set-url origin $repoUrl
    Write-Host "Remote URL set to: $repoUrl" -ForegroundColor Green

    # Test if repository exists
    Write-Host "Testing connection..." -ForegroundColor Yellow
    $testResult = git ls-remote origin 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Repository exists and is accessible!" -ForegroundColor Green
    } else {
        Write-Host "Repository not yet accessible. Create it on GitHub first." -ForegroundColor Yellow
    }
}

Write-Host "`nSetup completed!" -ForegroundColor Green
