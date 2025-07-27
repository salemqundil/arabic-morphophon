#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Install npm dependencies for the project
.DESCRIPTION
    This script installs npm dependencies for all parts of the project
#>

Write-Host "Installing root-level npm dependencies..." -ForegroundColor Cyan
npm install

Write-Host "Installing frontend npm dependencies..." -ForegroundColor Cyan
Push-Location -Path frontend
npm install
Pop-Location

Write-Host "NPM dependencies installation completed!" -ForegroundColor Green
