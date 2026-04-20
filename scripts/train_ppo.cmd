@echo off
set PY=%~dp0..\.venv\Scripts\python.exe
if not exist "%PY%" (
  echo Python executable not found at "%PY%"
  exit /b 1
)
"%PY%" -m backend.train.train_ppo --timesteps 200000 --model-path backend/train/artifacts/cysent_ppo_tuned --n-envs 4 --seed 42 --max-steps 150
