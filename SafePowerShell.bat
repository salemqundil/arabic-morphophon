@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=1
cd /d "%~dp0safe_workspace"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& {[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; [Console]::InputEncoding=[System.Text.Encoding]::UTF8; Clear-Host; Write-Host 'Safe PowerShell Environment Ready!' -ForegroundColor Green; Write-Host 'Current Directory:' (Get-Location) -ForegroundColor Cyan}"
