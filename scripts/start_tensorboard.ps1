param(
    [string]$LogDir = "backend/train/artifacts/runs",
    [int]$Port = 6006
)

$tb = Join-Path $PSScriptRoot "..\.venv\Scripts\tensorboard.exe"
$tb = [System.IO.Path]::GetFullPath($tb)

if (-not (Test-Path $tb)) {
    Write-Error "TensorBoard executable not found at $tb"
    exit 1
}

& $tb --logdir $LogDir --port $Port
