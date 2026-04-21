@echo off
set ROOT=%~dp0..

start powershell -NoExit -Command "Set-Location '%ROOT%'; .\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload"
start powershell -NoExit -Command "Set-Location '%ROOT%'; npm --prefix frontend run dev -- --hostname 127.0.0.1 --port 3000"

echo Backend: http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:3000
