<#
.SYNOPSIS
    构建MoeChat资产包（内核源代码+可选依赖）
.DESCRIPTION
    将内核源代码打包到 moechat-assets-v{version}-{cpu|cu130}.zip
    （或 -lite 变体）。三种变体：

    -lite ：moechat-assets-v{ver}-lite.zip
        仅包含内核源代码，无依赖、无模型。桌面应用首次运行时在线安装依赖。

    -cpu  ：moechat-assets-v{ver}-cpu.zip
        内核源代码 + CPU 版 torch / torchaudio / onnxruntime wheels。

    -cu130：moechat-assets-v{ver}-cu130.zip
        内核源代码 + CUDA 13.0 版 torch / torchaudio wheels + onnxruntime。

    默认无参数时一键构建全部三种变体（lite + cpu + cu130）；
    也可用 -Lite / -Cpu / -Cuda 快捷开关任意组合，仅构建所选变体。
.PARAMETER OutputDir
    输出目录，默认 "./dist"。
.PARAMETER Version
    包版本。自动从 pyproject.toml 读取。
.PARAMETER Lite
    构建 Lite 变体（仅源码，无依赖）。可与 -Cpu / -Cuda 组合。
.PARAMETER Cpu
    构建 CPU 变体（源码 + CPU wheels）。可与 -Lite / -Cuda 组合。
.PARAMETER Cuda
    构建 CUDA 变体（cu130）。可与 -Lite / -Cpu 组合。
.EXAMPLE
    .\scripts\build-asset-bundle.ps1                       # 一键输出全部三种（lite + cpu + cu130）
    .\scripts\build-asset-bundle.ps1 -Lite                 # 仅 lite
    .\scripts\build-asset-bundle.ps1 -Cpu                  # 仅 cpu
    .\scripts\build-asset-bundle.ps1 -Cuda                 # 仅 CUDA
    .\scripts\build-asset-bundle.ps1 -Cpu -Cuda            # cpu + cuda
    .\scripts\build-asset-bundle.ps1 -Lite -Cpu -Cuda      # 等同默认：全部三种
#>

param(
    [string]$OutputDir = "./dist",
    [string]$Version = "",
    [switch]$Lite = $false,
    [switch]$Cpu = $false,
    [switch]$Cuda = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptPath = $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptPath)

# ── 解析版本号（优先 pyproject.toml）───────────────────
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

# ── 解析要构建的变体列表 ────────────────────────────────
# 默认（无参数）构建全部三种：lite + cpu + cuda。
# -Lite / -Cpu / -Cuda 快捷开关可任意组合，仅构建所选变体。
if ($Lite -or $Cpu -or $Cuda) {
    $Variants = @()
    if ($Lite) { $Variants += "lite" }
    if ($Cpu)  { $Variants += "cpu" }
    if ($Cuda) { $Variants += "cuda" }
} else {
    # 无选择参数：默认全部三种
    $Variants = @("lite", "cpu", "cuda")
}

# 完整版（cpu/cuda）依赖 uv/uvx 下载 wheels（uvx 提供 pip download）
$uvExe = Get-Command "uv" -ErrorAction SilentlyContinue
$uvxExe = Get-Command "uvx" -ErrorAction SilentlyContinue
if ($Variants -contains "cpu" -or $Variants -contains "cuda") {
    if (-not $uvExe -or -not $uvxExe) {
        Write-Error "uv and uvx are required (https://docs.astral.sh/uv/)"
        exit 1
    }
}

# 输出目录：绝对路径直接用；相对路径基于项目根解析（Join-Path 无法拼接绝对路径）
if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputPath = $OutputDir
} else {
    $OutputPath = Join-Path $ProjectRoot $OutputDir
}
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
}
$OutputPath = (Get-Item $OutputPath).FullName

# ── 构建单个变体的核心逻辑 ──────────────────────────────
function Invoke-Build-Variant {
    param([string]$Variant)

    # 变体语义映射：lite=精简版（仅源码）、cuda=CUDA 变体（cu130），其余为 cpu
    $isLite = ($Variant -eq "lite")
    $isCuda = ($Variant -eq "cuda")

    # 工作目录：最终 zip 根 = 内核运行目录（按变体独立，避免残留污染）
    $WorkDir = Join-Path $env:TEMP "moechat-asset-build-${Variant}"
    if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
    New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  MoeChat Asset Bundle Builder" -ForegroundColor Cyan
    Write-Host "  Version: $Version" -ForegroundColor Cyan
    Write-Host "  Project: $ProjectRoot" -ForegroundColor Cyan
    Write-Host "  Variant: $(if ($isLite) { 'N/A (无 wheels)' } elseif ($isCuda) { 'CUDA (cu130)' } else { 'CPU' })" -ForegroundColor $(if ($isCuda) { 'Yellow' } elseif ($isLite) { 'DarkYellow' } else { 'Green' })
    Write-Host "  Mode   : $(if ($isLite) { 'Lite (仅源码)' } else { 'Full (源码 + wheels)' })" -ForegroundColor $(if ($isLite) { 'Yellow' } else { 'Green' })
    Write-Host "============================================" -ForegroundColor Cyan

    # ── 步骤 1: 拷贝内核源码（排除运行时产物与数据）─────
    Write-Host "[1/2] Copying kernel source..." -ForegroundColor Yellow

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

    # ── 步骤 2: 下载并打包 wheels（完整版）───────────────
    Write-Host "[2/2] Packaging assets..." -ForegroundColor Yellow

    $wheelCount = 0
    if (-not $isLite) {
        # 每次构建使用独立 wheels 目录，避免变体间（cpu/cuda）互相污染
        $WheelsDir = Join-Path $env:TEMP "moechat-asset-wheels-${Variant}"
        if (Test-Path $WheelsDir) { Remove-Item -Recurse -Force $WheelsDir }
        New-Item -ItemType Directory -Path $WheelsDir -Force | Out-Null

        Write-Host "  Downloading wheels..." -ForegroundColor Yellow

        # 使用 uvx pip download（uvx 自动拉取 pip 运行，无需 venv 内安装 pip），
        # 按目标 Python 版本 / 平台直接抓取预编译 wheel
        $pipArgs = @(
            "pip", "download"
            "--only-binary=:all:"
            "--python-version", "3.11"
            "--platform", "win_amd64"
            "--no-deps"
            "-d", $WheelsDir
        )

        if ($isCuda) {
            $pipArgs += "--index-url", "https://download.pytorch.org/whl/cu130"
            $pipArgs += "--extra-index-url", "https://pypi.org/simple"
            $pipArgs += "torch==2.12.0+cu130"
            $pipArgs += "torchaudio==2.11.0+cu130"
        } else {
            $pipArgs += "torch==2.12.0"
            $pipArgs += "torchaudio==2.11.0"
        }

        & $uvxExe $pipArgs 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }

        $wheelCount = (Get-ChildItem -Path "$WheelsDir/*.whl" -Name).Count
        if ($wheelCount -gt 0) {
            Copy-Item -Path $WheelsDir -Destination "$WorkDir/wheels" -Recurse -Force
        }
        Write-Host "  Wheels: ${wheelCount}" -ForegroundColor Green
    } else {
        Write-Host "  Lite 模式：跳过 wheels（运行时在线安装依赖）" -ForegroundColor DarkYellow
    }

    # 写入版本号
    $Version | Out-File -FilePath (Join-Path $WorkDir "version.txt") -Encoding utf8

    # 写入清单（记录版本 / 类型 / wheels，便于诊断）
    $manifest = @{
        version = $Version
        type    = if ($isLite) { "lite" } elseif ($isCuda) { "cuda" } else { "cpu" }
        mode    = if ($isLite) { "lite" } else { "full" }
        wheels  = @()
    }
    if (-not $isLite) {
        $manifest.wheels = @(Get-ChildItem -Path "$WorkDir/wheels/*.whl" -Name)
    }
    $manifest | ConvertTo-Json | Out-File -FilePath (Join-Path $WorkDir "manifest.json") -Encoding utf8

    # 产物命名：精简版不带变体后缀（无 wheels，与 cpu/cuda 无关）；完整版按变体区分
    $zipName = if ($isLite) {
        "moechat-assets-v${Version}-lite.zip"
    } else {
        $suffix = if ($isCuda) { "cu130" } else { "cpu" }
        "moechat-assets-v${Version}-${suffix}.zip"
    }
    $zipPath = Join-Path $OutputPath $zipName

    if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem

    # 使用 ZipArchive 手动写入，条目名统一使用正斜杠 "/" 分隔符。
    # 背景：.NET 的 CreateFromDirectory 在 Windows 上会以反斜杠 "\" 写条目名，
    # 属于非标准 zip，桌面端 node-stream-zip 会将其判为恶意条目而拒绝解压
    # （"Malicious entry: api\asr_api.py"）。
    $zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -Path $WorkDir -Recurse -File | ForEach-Object {
            # 相对路径统一转为 "/" 分隔符（兼容 Windows 与跨平台解压）
            $relative = $_.FullName.Substring($WorkDir.Length + 1).Replace('\', '/')
            $entry = $zip.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
            $entryStream = $entry.Open()
            try {
                $fileStream = [System.IO.File]::OpenRead($_.FullName)
                try { $fileStream.CopyTo($entryStream) } finally { $fileStream.Dispose() }
            } finally { $entryStream.Dispose() }
        }
    } finally {
        $zip.Dispose()
    }

    $zipSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)

    Write-Host ""
    Write-Host ("=" * 44) -ForegroundColor Green
    Write-Host "  Build complete!" -ForegroundColor Green
    Write-Host "  File: $zipPath" -ForegroundColor White
    Write-Host "  Size: ${zipSize}MB" -ForegroundColor White
    Write-Host "  Wheels: ${wheelCount}" -ForegroundColor White
    Write-Host ("=" * 44) -ForegroundColor Green

    # 清理本次构建的临时目录
    Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $env:TEMP "moechat-asset-wheels-${Variant}") -ErrorAction SilentlyContinue
}

# ── 主流程：按变体列表逐个构建 ──────────────────────────
foreach ($variant in $Variants) {
    Invoke-Build-Variant -Variant $variant
}

Write-Host ""
Write-Host "All variants built: $($Variants -join ', ')" -ForegroundColor Cyan
