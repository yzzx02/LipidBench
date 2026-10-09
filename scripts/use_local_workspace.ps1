param(
    [string]$LocalRoot = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $LocalRoot) {
    $LocalRoot = $env:CHROMAPEAK_LOCAL_ROOT
}
if (-not $LocalRoot) {
    $LocalRoot = Join-Path (Split-Path -Parent $projectRoot) ((Split-Path -Leaf $projectRoot) + "-local")
}
$env:CHROMAPEAK_LOCAL_ROOT = [System.IO.Path]::GetFullPath($LocalRoot)
if (-not $env:CHROMAPEAK_CACHE_ROOT) {
    $env:CHROMAPEAK_CACHE_ROOT = Join-Path $env:CHROMAPEAK_LOCAL_ROOT "cache"
}
$env:PYTHONPYCACHEPREFIX = Join-Path $env:CHROMAPEAK_CACHE_ROOT "python"
$env:PIP_CACHE_DIR = Join-Path $env:CHROMAPEAK_CACHE_ROOT "pip"
$env:TEMP = Join-Path $env:CHROMAPEAK_CACHE_ROOT "tmp"
$env:TMP = $env:TEMP
foreach ($directory in @($env:PYTHONPYCACHEPREFIX, $env:PIP_CACHE_DIR, $env:TEMP)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
Write-Output "Local storage: $env:CHROMAPEAK_LOCAL_ROOT"
Write-Output "Runtime cache: $env:CHROMAPEAK_CACHE_ROOT"
