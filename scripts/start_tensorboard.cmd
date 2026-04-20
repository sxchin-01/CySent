@echo off
set TB=%~dp0..\.venv\Scripts\tensorboard.exe
if not exist "%TB%" (
  echo TensorBoard executable not found at "%TB%"
  exit /b 1
)
"%TB%" --logdir backend/train/artifacts/runs --port 6006
