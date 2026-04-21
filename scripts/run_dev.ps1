$ErrorActionPreference = "Stop"

Write-Host "Starting CySent backend and frontend..."

Start-Process powershell -ArgumentList '-NoExit', '-Command', 'Set-Location "' + $PSScriptRoot + '\.."; .\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload'
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'Set-Location "' + $PSScriptRoot + '\.."; npm --prefix frontend run dev -- --hostname 127.0.0.1 --port 3000'

Write-Host "Backend: http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:3000"
