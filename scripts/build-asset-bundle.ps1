<#
.SYNOPSIS
    Build MoeChat asset bundle (kernel source + wheels + models)
.DESCRIPTION
    Packages everything into moechat-assets-{version}-{cpu|cuda}.zip
    for cloud drive distribution.
    Usage: .\scripts\build-asset-bundle.ps1 [-Cuda] [-OutputDir ./dist]
#>

param(
    [string]$OutputDir = "./dist",
    [string]$Version = "",
    [switch]$Cuda = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptPath = $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptPath)

if (-not $Version) {
    $pyprojectFile = Join-Path $ProjectRoot "pyproject.toml"
    if (Test-Path $pyprojectFile) {
        $tomlContent = Get-Content $pyprojectFile -Raw
        if ($tomlContent -match 'version\s*=\s*"([^"]+)"') {
            $Version = $matches[1]
        }
    }
}
if (-not $Version) { $Version = "1.7.0" }

$uvExe = Get-Command "uv" -ErrorAction SilentlyContinue
if (-not $uvExe) {
    Write-Error "uv is required (https://docs.astral.sh/uv/)"
    exit 1
}

$OutputPath = Join-Path $ProjectRoot $OutputDir
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
}
$OutputPath = (Get-Item $OutputPath).FullName

$WorkDir = Join-Path $env:TEMP "moechat-asset-build"
if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
$WheelsDir = Join-Path $WorkDir "wheels"
New-Item -ItemType Directory -Path $WheelsDir -Force | Out-Null

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  MoeChat Asset Bundle Builder" -ForegroundColor Cyan
Write-Host "  Version: $Version" -ForegroundColor Cyan
Write-Host "  Project: $ProjectRoot" -ForegroundColor Cyan
if ($Cuda) { Write-Host "  Mode: CUDA" -ForegroundColor Yellow }
else { Write-Host "  Mode: CPU" -ForegroundColor Green }
Write-Host "============================================" -ForegroundColor Cyan

# Step 1: Download large wheels via pip download (from local .venv)
Write-Host "[1/4] Downloading wheels..." -ForegroundColor Yellow

$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path (Join-Path $VenvDir "Scripts") "python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    Set-Location -LiteralPath $ProjectRoot
    & $uvExe venv --seed --python 3.11 2>&1 | Out-Null
}

$pipArgs = @(
    "-m", "pip", "download"
    "--only-binary=:all:"
    "--python-version", "3.11"
    "--platform", "win_amd64"
    "--no-deps"
    "-d", $WheelsDir
)

if ($Cuda) {
    $pipArgs += "--index-url", "https://download.pytorch.org/whl/cu130"
    $pipArgs += "--extra-index-url", "https://mirrors.ustc.edu.cn/pypi/simple"
    $pipArgs += "torch==2.12.0+cu130"
    $pipArgs += "torchaudio==2.11.0+cu130"
} else {
    $pipArgs += "torch==2.12.0"
    $pipArgs += "torchaudio==2.11.0"
}
$pipArgs += "onnxruntime==1.25.0"

Write-Host "  Downloading torch/torchaudio/onnxruntime..." -ForegroundColor Yellow
& $VenvPython $pipArgs 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }

$wheelCount = (Get-ChildItem -Path "$WheelsDir/*.whl" -Name).Count
Write-Host "  Done: ${wheelCount} wheels" -ForegroundColor Green


# Step 3: Copy kernel source
Write-Host "[3/4] Copying kernel source..." -ForegroundColor Yellow

$excludeDirs = @(
    ".venv", "__pycache__", ".git", ".github", ".gitignore",
    ".opencode", ".vscode", "node_modules", "data", "wheels",
    "dist", "build", ".ruff_cache", ".python-version",
    "uv.lock", "config.yaml"
)

Get-ChildItem -Path $ProjectRoot -File -Recurse | Where-Object {
    $relative = $_.FullName.Substring($ProjectRoot.Length + 1)
    $parts = $relative -split "[\\/]"
    $shouldExclude = $false
    foreach ($part in $parts) {
        if ($part -in $excludeDirs) { $shouldExclude = $true; break }
    }
    if ($_.Extension -in ".pyc", ".pyo", ".pyd") { $shouldExclude = $true }
    -not $shouldExclude
} | ForEach-Object {
    $relative = $_.FullName.Substring($ProjectRoot.Length + 1)
    $dest = Join-Path $WorkDir $relative
    $destDir = Split-Path $dest -Parent
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
    Copy-Item -Path $_.FullName -Destination $dest -Force
}

Write-Host "  Kernel source copied" -ForegroundColor Green

# Step 4: Package
Write-Host "[4/4] Packaging assets..." -ForegroundColor Yellow

$suffix = if ($Cuda) { "cu130" } else { "cpu" }
$zipName = "moechat-assets-v${Version}-${suffix}.zip"
$zipPath = Join-Path $OutputPath $zipName

$PackageDir = Join-Path $env:TEMP "moechat-asset-pkg"
if (Test-Path $PackageDir) { Remove-Item -Recurse -Force $PackageDir }
New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null

Get-ChildItem -Path $WorkDir -Exclude "wheels", "data" | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $PackageDir -Recurse -Force
}

if (Test-Path $WheelsDir) {
    Copy-Item -Path $WheelsDir -Destination "$PackageDir/wheels" -Recurse -Force
}

$modelSource = Join-Path (Join-Path $ProjectRoot "data") "models"
if ($IncludeModels -and (Test-Path $modelSource)) {
    $modelsDir = Join-Path $PackageDir "data" "models"
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
    Copy-Item -Path "$modelSource/*" -Destination $modelsDir -Recurse -Force
}

$Version | Out-File -FilePath (Join-Path $PackageDir "version.txt") -Encoding utf8

$manifest = @{
    version = $Version
    type    = if ($Cuda) { "cuda" } else { "cpu" }
    wheels  = @(Get-ChildItem -Path "$PackageDir/wheels/*.whl" -Name)
    models  = @()
}
if (Test-Path "$PackageDir/data/models") {
    $manifest.models = (Get-ChildItem -Path "$PackageDir/data/models" -Directory -Name)
}
$manifest | ConvertTo-Json | Out-File -FilePath (Join-Path $PackageDir "manifest.json") -Encoding utf8

if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($PackageDir, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)

$zipSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)

Write-Host ""
Write-Host ("=" * 44) -ForegroundColor Green
Write-Host "  Build complete!" -ForegroundColor Green
Write-Host "  File: $zipPath" -ForegroundColor White
Write-Host "  Size: ${zipSize}MB" -ForegroundColor White
Write-Host "  Wheels: ${wheelCount}" -ForegroundColor White
if ($manifest.models.Count -gt 0) {
    $modelMsg = "  Models: " + $manifest.models.Count
    Write-Host $modelMsg -ForegroundColor White
}
Write-Host ("=" * 44) -ForegroundColor Green

Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $PackageDir -ErrorAction SilentlyContinue


