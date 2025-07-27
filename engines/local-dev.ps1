#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Management script for local development environment
.DESCRIPTION
    This script provides commands for managing the local development environment
    for the Arabic NLP Engine project.
#>

param (
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Args = @()
)

$DockerComposeFile = "docker-compose.local.yml"
$EnvFile = "local/.env.local"
$EnvExampleFile = "local/.env.example"

function Start-LocalEnvironment {
    Write-Host "Starting local development environment..." -ForegroundColor Green

    # Check if environment file exists, if not create from example
    if (-Not (Test-Path $EnvFile)) {
        Write-Host "Environment file not found. Creating from example..." -ForegroundColor Yellow
        Copy-Item -Path $EnvExampleFile -Destination $EnvFile
        Write-Host "Created $EnvFile. You may want to edit this file before continuing." -ForegroundColor Yellow
    }

    # Start services
    docker compose -f $DockerComposeFile up -d

    Write-Host "Local development environment started successfully!" -ForegroundColor Green
    Write-Host "Access the services at:" -ForegroundColor Cyan
    Write-Host "  - API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  - API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "  - Frontend: http://localhost:3000" -ForegroundColor Cyan
    Write-Host "  - PgAdmin: http://localhost:5050" -ForegroundColor Cyan
    Write-Host "  - MailHog: http://localhost:8025" -ForegroundColor Cyan
}

function Stop-LocalEnvironment {
    Write-Host "Stopping local development environment..." -ForegroundColor Yellow
    docker compose -f $DockerComposeFile down
    Write-Host "Local development environment stopped." -ForegroundColor Green
}

function Restart-LocalEnvironment {
    Stop-LocalEnvironment
    Start-LocalEnvironment
}

function Remove-LocalEnvironment {
    Write-Host "Removing local development environment and volumes..." -ForegroundColor Red
    docker compose -f $DockerComposeFile down -v
    Write-Host "Local development environment removed." -ForegroundColor Green
}

function Show-Logs {
    param (
        [Parameter(Position = 0)]
        [string]$Service = ""
    )

    if ($Service -eq "") {
        docker compose -f $DockerComposeFile logs -f
    } else {
        docker compose -f $DockerComposeFile logs -f $Service
    }
}

function Invoke-BackendShell {
    docker compose -f $DockerComposeFile exec backend bash
}

function Invoke-FrontendShell {
    docker compose -f $DockerComposeFile exec frontend sh
}

function Invoke-DatabaseShell {
    docker compose -f $DockerComposeFile exec db psql -U $env:DB_USER -d $env:DB_NAME
}

function Show-Status {
    docker compose -f $DockerComposeFile ps
}

function Show-Help {
    Write-Host "Arabic NLP Engine - Local Development Management Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: ./local-dev.ps1 [command] [arguments]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host "  start         Start the local development environment" -ForegroundColor Green
    Write-Host "  stop          Stop the local development environment" -ForegroundColor Green
    Write-Host "  restart       Restart the local development environment" -ForegroundColor Green
    Write-Host "  remove        Remove the local environment and volumes" -ForegroundColor Green
    Write-Host "  logs [service]Show logs for all services or a specific service" -ForegroundColor Green
    Write-Host "  backend-shell Access a shell in the backend container" -ForegroundColor Green
    Write-Host "  frontend-shell Access a shell in the frontend container" -ForegroundColor Green
    Write-Host "  db-shell      Access the PostgreSQL shell" -ForegroundColor Green
    Write-Host "  status        Show the status of all services" -ForegroundColor Green
    Write-Host "  help          Show this help message" -ForegroundColor Green
    Write-Host ""
}

switch ($Command) {
    "start" { Start-LocalEnvironment }
    "stop" { Stop-LocalEnvironment }
    "restart" { Restart-LocalEnvironment }
    "remove" { Remove-LocalEnvironment }
    "logs" { Show-Logs $Args[0] }
    "backend-shell" { Invoke-BackendShell }
    "frontend-shell" { Invoke-FrontendShell }
    "db-shell" { Invoke-DatabaseShell }
    "status" { Show-Status }
    default { Show-Help }
}
