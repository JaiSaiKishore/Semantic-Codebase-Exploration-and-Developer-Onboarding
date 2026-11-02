$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Semantic Codebase RAG - Developer Onboarding Assistant" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Test-Path ".env")) {
    Write-Host "[ERROR] .env file not found! Please create .env with GROQ_API_KEY." -ForegroundColor Red
    exit 1
}

$pythonExe = if (Test-Path "venv\Scripts\python.exe") { ".\venv\Scripts\python.exe" } else { "python" }

Start-Job -ScriptBlock {
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:8000"
} | Out-Null

Write-Host "Starting server at http://localhost:8000 ..." -ForegroundColor Green
& $pythonExe server.py