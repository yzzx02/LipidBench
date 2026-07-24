param(
    [string]$EnvironmentPath = ".venv-nvidia-cu128"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$environment = [System.IO.Path]::GetFullPath(
    (Join-Path $projectRoot $EnvironmentPath)
)
$python = Join-Path $environment "Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install 64-bit Python 3.12 first."
}

if (-not (Test-Path -LiteralPath $python)) {
    & py -3.12 -m venv $environment
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the Python 3.12 virtual environment."
    }
}

& $python -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update pip tooling."
}
& $python -m pip install -r (Join-Path $projectRoot "requirements-peak-common.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install common PeakTruthLab dependencies."
}
& $python -m pip install -r (
    Join-Path $projectRoot "requirements-peak-nvidia-cu128.txt"
)
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the CUDA 12.8 PyTorch dependencies."
}

& $python -c @"
import torch
import torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("NVIDIA CUDA device was not detected")
print("device", torch.cuda.get_device_name(0))
"@
if ($LASTEXITCODE -ne 0) {
    throw "CUDA environment verification failed."
}

Write-Output "NVIDIA environment ready: $environment"
