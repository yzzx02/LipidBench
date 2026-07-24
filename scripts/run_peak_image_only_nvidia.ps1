param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$OutputDirectory = "PeakTruthLab\results\nvidia_2000_500_b8_e5",
    [string]$PythonPath = ".venv-nvidia-cu128\Scripts\python.exe",
    [int]$Epochs = 5,
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $PythonPath))
$dataset = [System.IO.Path]::GetFullPath($DatasetRoot)
$output = [System.IO.Path]::GetFullPath(
    (Join-Path $projectRoot $OutputDirectory)
)
$trainingScript = Join-Path $projectRoot (
    "PeakTruthLab\scripts\detection\train_peak_image_only_staged.py"
)
$trainManifest = Join-Path $dataset "manifests\train.jsonl"
$valManifest = Join-Path $dataset "manifests\val.jsonl"
$imageRoot = Join-Path $dataset "eic_images_flat"

foreach ($requiredPath in @(
    $python,
    $trainingScript,
    $trainManifest,
    $valManifest,
    $imageRoot
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

& $python $trainingScript `
    --mode pilot `
    --train-manifest $trainManifest `
    --val-manifest $valManifest `
    --image-root $imageRoot `
    --output-dir $output `
    --train-limit 2000 `
    --val-limit 500 `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --amp `
    --pretrained `
    --image-size 480 `
    --rpn-train-proposals 128 `
    --rpn-test-proposals 64 `
    --overlay-count 8 `
    --num-workers 0

if ($LASTEXITCODE -ne 0) {
    throw "PeakTruthLab NVIDIA training failed with exit code $LASTEXITCODE."
}

Write-Output "Training completed: $output"
