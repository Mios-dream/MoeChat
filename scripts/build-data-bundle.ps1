<#
.SYNOPSIS
    Build MoeChat data resource bundle (models + resources + motion + agents)
.DESCRIPTION
    Packages data resources into moechat-data-{version}.zip for distribution.

    The zip ROOT contains the data content directly (NOT wrapped in a data/
    folder): models/, resources/, agents/, worldbook/, motion dbs + manifest.json.
    The desktop app extracts this bundle with a "data/" prefix into
    {kernel}/data/ so the backend sees ./data/... relative to the kernel root.

    Agents are selectively included; only info.yaml and assets/ are kept per agent.

.PARAMETER OutputDir
    Output directory, default "./dist".
.PARAMETER Version
    Package version. Auto-read from pyproject.toml when empty.
.PARAMETER Agent
    Agent name(s) to include, e.g. -Agent "澪","香风智乃".
.PARAMETER AllAgents
    Include all agents found under data/agents.
.EXAMPLE
    .\scripts\build-data-bundle.ps1 -AllAgents            # 打包全部助手
    .\scripts\build-data-bundle.ps1 -Agent "澪"            # 仅打包指定助手
    .\scripts\build-data-bundle.ps1                        # 交互式选择助手
#>

param(
    [string]$OutputDir = "./dist",
    [string]$Version = "",
    [string[]]$Agent = @(),
    [switch]$AllAgents = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptPath = $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptPath)

# 解析版本号
if (-not $Version) {
    $pyprojectFile = Join-Path $ProjectRoot "pyproject.toml"
    if (Test-Path $pyprojectFile) {
        $tomlContent = Get-Content $pyprojectFile -Raw
        if ($tomlContent -match 'version\s*=\s*"([^"]+)"') {
            $Version = $matches[1]
        }
    }
}
if (-not $Version) { $Version = "2.0.0" }

$OutputPath = Join-Path $ProjectRoot $OutputDir
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
}
$OutputPath = (Get-Item $OutputPath).FullName

# 工作目录：最终 zip 根 = {kernel}/data 的内容（不含 data/ 外层）
$WorkDir = Join-Path $env:TEMP "moechat-data-build"
if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

$DataDir = Join-Path $ProjectRoot "data"
if (-not (Test-Path $DataDir)) {
    Write-Error "data directory not found: $DataDir"
    exit 1
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  MoeChat Data Bundle Builder" -ForegroundColor Cyan
Write-Host "  Version: $Version" -ForegroundColor Cyan
Write-Host "  Data   : $DataDir" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ============================================================
# 选择需要打包的助手
# ============================================================
$AgentsDir = Join-Path $DataDir "agents"
$selectedAgents = @()

if ($AllAgents) {
    # 打包所有助手
    if (Test-Path $AgentsDir) {
        $selectedAgents = Get-ChildItem -Path $AgentsDir -Directory | ForEach-Object { $_.Name }
    }
} elseif ($Agent.Count -gt 0) {
    # 使用参数指定的助手列表
    $selectedAgents = $Agent
} else {
    # 默认使用交互式选择助手
    if (Test-Path $AgentsDir) {
        $availableAgents = Get-ChildItem -Path $AgentsDir -Directory | ForEach-Object { $_.Name }
        if ($availableAgents.Count -gt 0) {
            Write-Host "`nAvailable agents:" -ForegroundColor Yellow
            for ($i = 0; $i -lt $availableAgents.Count; $i++) {
                Write-Host "  [$($i+1)] $($availableAgents[$i])" -ForegroundColor White
            }
            Write-Host "  [a] Select All" -ForegroundColor White
            Write-Host "  [Enter] Skip agents" -ForegroundColor White
            $input = Read-Host "`nSelect agents (comma-separated numbers, e.g. 1,2)"
            if ($input -eq "a" -or $input -eq "A") {
                $selectedAgents = $availableAgents
            } elseif ($input -ne "") {
                $indices = $input -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
                $selectedAgents = $indices | ForEach-Object {
                    $idx = [int]$_ - 1
                    if ($idx -ge 0 -and $idx -lt $availableAgents.Count) {
                        $availableAgents[$idx]
                    }
                }
            }
        }
    }
}

# ============================================================
# Step 1: 打包 models
# ============================================================
Write-Host "[1/5] Packing models..." -ForegroundColor Yellow
$modelsSource = Join-Path $DataDir "models"
if (Test-Path $modelsSource) {
    $modelsDest = Join-Path $WorkDir "models"
    New-Item -ItemType Directory -Path $modelsDest -Force | Out-Null
    Copy-Item -Path "$modelsSource/*" -Destination $modelsDest -Recurse -Force
    Write-Host "  Models copied" -ForegroundColor Green
} else {
    Write-Host "  Models directory not found, skipping" -ForegroundColor Gray
}

# ============================================================
# Step 2: 打包 resources
# ============================================================
Write-Host "[2/5] Packing resources..." -ForegroundColor Yellow
$resourcesSource = Join-Path $DataDir "resources"
if (Test-Path $resourcesSource) {
    $resourcesDest = Join-Path $WorkDir "resources"
    New-Item -ItemType Directory -Path $resourcesDest -Force | Out-Null
    Copy-Item -Path "$resourcesSource/*" -Destination $resourcesDest -Recurse -Force
    Write-Host "  Resources copied" -ForegroundColor Green
} else {
    Write-Host "  Resources directory not found, skipping" -ForegroundColor Gray
}


# ============================================================
# Step 3: 打包 motion 数据库
# ============================================================
Write-Host "[3/5] Packing motion databases..." -ForegroundColor Yellow
$motionFiles = @(
    "motion.db", "motion.db-shm", "motion.db-wal"
)
$hasMotion = $false
foreach ($file in $motionFiles) {
    $src = Join-Path $DataDir $file
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $WorkDir $file) -Force
        $hasMotion = $true
    }
}
if ($hasMotion) {
    Write-Host "  Motion databases copied" -ForegroundColor Green
} else {
    Write-Host "  Motion databases not found, skipping" -ForegroundColor Gray
}

# ============================================================
# Step 4: 打包助手（选择性，仅 info.yaml + assets）
# ============================================================
if ($selectedAgents.Count -gt 0) {
    Write-Host "[4/5] Packing agents (info.yaml + assets only)..." -ForegroundColor Yellow
    $agentsDest = Join-Path $WorkDir "agents"
    New-Item -ItemType Directory -Path $agentsDest -Force | Out-Null
    foreach ($agentName in $selectedAgents) {
        $agentSource = Join-Path $AgentsDir $agentName
        if (-not (Test-Path $agentSource)) {
            Write-Host "  Agent '$agentName' not found, skipping" -ForegroundColor Gray
            continue
        }
        $agentDest = Join-Path $agentsDest $agentName
        New-Item -ItemType Directory -Path $agentDest -Force | Out-Null

        # 复制 info.yaml
        $infoYaml = Join-Path $agentSource "info.yaml"
        if (Test-Path $infoYaml) {
            Copy-Item -Path $infoYaml -Destination (Join-Path $agentDest "info.yaml") -Force
        }

        # 复制 assets 目录
        $assetsSource = Join-Path $agentSource "assets"
        if (Test-Path $assetsSource) {
            $assetsDest = Join-Path $agentDest "assets"
            New-Item -ItemType Directory -Path $assetsDest -Force | Out-Null
            Copy-Item -Path "$assetsSource/*" -Destination $assetsDest -Recurse -Force
        }

        Write-Host "  Agent '$agentName' packed" -ForegroundColor Green
    }
} else {
    Write-Host "[4/5] Skipping agents (none selected)" -ForegroundColor Gray
}

# ============================================================
# Step 5: 打包成 zip
# ============================================================
Write-Host "Packaging..." -ForegroundColor Yellow

# 创建清单（models/resources/agents 仅记录目录名，便于诊断）
$manifest = @{
    version  = $Version
    type     = "data"
    models   = @()
    resources = @()
    agents   = @($selectedAgents)
}

$pkgModelsDir = Join-Path $WorkDir "models"
if (Test-Path $pkgModelsDir) {
    # 使用 @() 强制结果为数组，避免单个目录时被管道展开为标量字符串
    $manifest.models = @(Get-ChildItem -Path $pkgModelsDir -Directory | ForEach-Object { $_.Name })
}

$pkgResourcesDir = Join-Path $WorkDir "resources"
if (Test-Path $pkgResourcesDir) {
    # 使用 @() 强制结果为数组，避免单个目录时被管道展开为标量字符串
    $manifest.resources = @(Get-ChildItem -Path $pkgResourcesDir -Directory | ForEach-Object { $_.Name })
}

$manifest | ConvertTo-Json | Out-File -FilePath (Join-Path $WorkDir "manifest.json") -Encoding utf8

$zipName = "moechat-data-v${Version}.zip"
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

# 输出结果
Write-Host ""
Write-Host ("=" * 44) -ForegroundColor Green
Write-Host "  Build complete!" -ForegroundColor Green
Write-Host "  File: $zipPath" -ForegroundColor White
Write-Host "  Size: ${zipSize}MB" -ForegroundColor White
if ($manifest.models.Count -gt 0) {
    Write-Host ("  Models: " + $manifest.models.Count) -ForegroundColor White
}
if ($manifest.resources.Count -gt 0) {
    Write-Host ("  Resources: " + $manifest.resources.Count) -ForegroundColor White
}
if ($selectedAgents.Count -gt 0) {
    Write-Host ("  Agents: " + $selectedAgents.Count) -ForegroundColor White
}
Write-Host ("=" * 44) -ForegroundColor Green

# 清理临时目录
Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
