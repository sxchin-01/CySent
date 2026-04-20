param(
    [int]$Timesteps = 200000,
    [string]$ModelPath = "backend/train/artifacts/cysent_ppo_tuned",
    [int]$NEnvs = 4,
    [int]$Seed = 42,
    [int]$MaxSteps = 150
)

$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$py = [System.IO.Path]::GetFullPath($py)

if (-not (Test-Path $py)) {
    Write-Error "Python executable not found at $py"
    exit 1
}

& $py -m backend.train.train_ppo --timesteps $Timesteps --model-path $ModelPath --n-envs $NEnvs --seed $Seed --max-steps $MaxSteps
