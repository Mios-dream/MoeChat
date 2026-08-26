<#
.SYNOPSIS
    Build MoeChat runtime asset bundle.
.DESCRIPTION
    The bundle contains source, whitelisted runtime assets, KWS, and selected assistants.
    When -Agent is omitted, assistants are selected interactively in the terminal.
.PARAMETER OutputDir
    Output directory, default ./dist.
.PARAMETER Version
    Version, read from pyproject.toml by default.
.PARAMETER Platform
    Target platform, windows or linux.
.PARAMETER Agent
    Assistant names; may be repeated.
.PARAMETER NoGlobalAssets
    Skip whitelisted global assets.
.PARAMETER NoMotion
    Skip motion database files.
.PARAMETER NoKws
    Skip KWS model.
.PARAMETER NoSource
    Skip backend source.
.EXAMPLE
    .\scripts\build-asset-bundle.ps1 -Agent "assistant-name"
.EXAMPLE
    .\scripts\build-asset-bundle.ps1
#>

param(
    [string]$OutputDir = "./dist",
    [string]$Version = "",
    [ValidateSet("windows", "linux", "")]
    [string]$Platform = "",
    [string[]]$Agent = @(),
    [switch]$NoGlobalAssets,
    [switch]$NoMotion,
    [switch]$NoKws,
    [switch]$NoSource
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Keep the script ASCII-compatible with Windows PowerShell 5.1 while displaying Chinese text.
function Convert-Cn {
    param([string]$Text)
    return [regex]::Replace($Text, '\\u([0-9a-fA-F]{4})', {
        param($Match)
        [char]([Convert]::ToInt32($Match.Groups[1].Value, 16))
    })
}

# ========================= Configuration =========================
$SourceExcludeDirectories = @(
    ".git", ".github", ".venv", ".vscode", ".opencode", ".ruff_cache",
    "__pycache__", "node_modules", "data", "dist", "build", "wheels","config.yaml","uv.lock"
)

# Global asset whitelist. data/resources is copied as-is, including required models.
$GlobalAssetWhitelist = @("resources")
$GlobalAssetExcludedDirectories = @()

# Motion database whitelist.
$MotionFileWhitelist = @("motion.db", "motion.db-shm", "motion.db-wal")

# Only this KWS model directory is copied from data/models.
$KwsModelName = "sherpa-onnx-kws-zipformer-zh-en-3M"

# Assistant whitelist: copy info.yaml and assets, including assistant model assets.
$AgentInfoFiles = @("info.yaml")
# ================================================================

$ScriptPath = $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptPath)
$ProjectRootFull = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd("\", "/")
$DataDir = Join-Path -Path $ProjectRoot -ChildPath "data"

if (-not $Version) {
    $pyprojectPath = Join-Path -Path $ProjectRoot -ChildPath "pyproject.toml"
    if (Test-Path -LiteralPath $pyprojectPath) {
        $toml = Get-Content -LiteralPath $pyprojectPath -Raw
        if ($toml -match 'version\s*=\s*"([^"]+)"') { $Version = $matches[1] }
    }
}
if (-not $Version) { $Version = "2.0.0" }

if (-not $Platform) {
    $Platform = if ($env:OS -match "Windows") { "windows" } else { "linux" }
}
$PlatformTag = if ($Platform -eq "windows") { "win" } else { "linux" }

if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "data directory not found: $DataDir"
}

if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputPath = $OutputDir
} else {
    $OutputPath = Join-Path -Path $ProjectRoot -ChildPath $OutputDir
}
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
$OutputPath = (Get-Item -LiteralPath $OutputPath).FullName

$TempRoot = [System.IO.Path]::GetTempPath()
$TempRootFull = [System.IO.Path]::GetFullPath($TempRoot).TrimEnd("\", "/")
if ($TempRootFull -eq $ProjectRootFull -or $TempRootFull.StartsWith("$ProjectRootFull\", [System.StringComparison]::OrdinalIgnoreCase)) {
    $TempRoot = Join-Path -Path ([System.IO.Path]::GetPathRoot($ProjectRootFull)) -ChildPath "MoeChat-asset-build-temp"
}
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string[]]$ExcludeDirectories = @()
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) { return @() }
    $sourceFull = (Get-Item -LiteralPath $Source).FullName
    $copied = @()
    Get-ChildItem -LiteralPath $sourceFull -File -Recurse -Force | ForEach-Object {
        $relative = $_.FullName.Substring($sourceFull.Length + 1)
        $parts = $relative -split "[\\/]"
        $excluded = $false
        foreach ($part in $parts) {
            if ($ExcludeDirectories -contains $part) { $excluded = $true; break }
        }
        if ($excluded) { return }
        $target = Join-Path -Path $Destination -ChildPath $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
        $copied += $relative.Replace("\", "/")
    }
    return $copied
}

function Copy-FileList {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot,
        [Parameter(Mandatory = $true)][string[]]$RelativePaths
    )
    $copied = @()
    foreach ($relative in $RelativePaths) {
        $source = Join-Path -Path $SourceRoot -ChildPath $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
        $target = Join-Path -Path $DestinationRoot -ChildPath $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $target -Force
        $copied += $relative.Replace("\", "/")
    }
    return $copied
}

function Add-DirectoryToZip {
    param(
        [Parameter(Mandatory = $true)][string]$SourceDir,
        [Parameter(Mandatory = $true)][string]$ZipPath
    )
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::Open($ZipPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        $sourceFull = (Get-Item -LiteralPath $SourceDir).FullName
        Get-ChildItem -LiteralPath $sourceFull -File -Recurse | ForEach-Object {
            $relative = $_.FullName.Substring($sourceFull.Length + 1).Replace("\", "/")
            $entry = $zip.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
            $entryStream = $entry.Open()
            try {
                $fileStream = [System.IO.File]::OpenRead($_.FullName)
                try { $fileStream.CopyTo($entryStream) } finally { $fileStream.Dispose() }
            } finally { $entryStream.Dispose() }
        }
    } finally { $zip.Dispose() }
}

function Select-Agents {
    param([string]$AgentsRoot)
    $available = @(Get-ChildItem -LiteralPath $AgentsRoot -Directory | Sort-Object Name)
    if ($available.Count -eq 0) { return @() }
    Write-Host (Convert-Cn "\u53ef\u9009\u52a9\u624b:") -ForegroundColor Cyan
    for ($i = 0; $i -lt $available.Count; $i++) {
        Write-Host ("  [{0}] {1}" -f ($i + 1), $available[$i].Name)
    }
    $answer = Read-Host (Convert-Cn "\u8f93\u5165\u7f16\u53f7(\u9017\u53f7\u5206\u9694,\u76f4\u63a5\u56de\u8f66\u8df3\u8fc7)")
    if ([string]::IsNullOrWhiteSpace($answer)) { return @() }
    $selected = @()
    foreach ($token in ($answer -split ",")) {
        $index = 0
        if ([int]::TryParse($token.Trim(), [ref]$index) -and $index -ge 1 -and $index -le $available.Count) {
            $selected += $available[$index - 1].Name
        } else {
            Write-Warning (Convert-Cn ("\u5ffd\u7565\u65e0\u6548\u52a9\u624b\u7f16\u53f7: {0}" -f $token))
        }
    }
    return @($selected | Select-Object -Unique)
}

$agentsRoot = Join-Path -Path $DataDir -ChildPath "agents"
if ($Agent.Count -gt 0) {
    $SelectedAgents = @($Agent)
} elseif (Test-Path -LiteralPath $agentsRoot -PathType Container) {
    $SelectedAgents = Select-Agents -AgentsRoot $agentsRoot
} else {
    $SelectedAgents = @()
}

$Include = @{
    Source = -not $NoSource
    GlobalAssets = -not $NoGlobalAssets
    Motion = -not $NoMotion
    Kws = -not $NoKws
    Agents = $true
}

$workDir = Join-Path -Path $TempRoot -ChildPath "moechat-runtime-assets-$([guid]::NewGuid().ToString('N'))"
$workDirFull = [System.IO.Path]::GetFullPath($workDir).TrimEnd("\", "/")
if ($workDirFull -eq $ProjectRootFull -or $workDirFull.StartsWith("$ProjectRootFull\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Asset work directory must be outside the project directory: $workDirFull"
}
New-Item -ItemType Directory -Path $workDir -Force | Out-Null

$manifest = [ordered]@{
    version = $Version
    build_id = Get-Date -Format "yyyyMMddHHmmssfff"
    platform = $Platform
    type = "runtime"
    variant = "lite"
    source_included = $Include.Source
    global_assets = @()
    motion_files = @()
    embedded_models = @()
    agents = @()
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host (Convert-Cn "  MoeChat \u8fd0\u884c\u65f6\u8d44\u6e90\u6253\u5305") -ForegroundColor Cyan
Write-Host (Convert-Cn ("  \u7248\u672c: {0}" -f $Version)) -ForegroundColor Cyan
Write-Host (Convert-Cn ("  \u5e73\u53f0: {0}" -f $Platform)) -ForegroundColor Cyan
Write-Host (Convert-Cn ("  \u5de5\u4f5c\u76ee\u5f55: {0}" -f $workDir)) -ForegroundColor DarkGray
Write-Host "============================================" -ForegroundColor Cyan

if ($Include.Source) {
    Write-Host (Convert-Cn "[1/5] \u6b63\u5728\u590d\u5236\u6e90\u7801...") -ForegroundColor Yellow
    $sourceFiles = Copy-Tree -Source $ProjectRoot -Destination $workDir -ExcludeDirectories $SourceExcludeDirectories
    Write-Host (Convert-Cn ("  \u6e90\u7801\u6587\u4ef6: {0}" -f @($sourceFiles).Count)) -ForegroundColor Gray
} else { Write-Host (Convert-Cn "[1/5] \u8df3\u8fc7\u6e90\u7801") -ForegroundColor DarkGray }

if ($Include.GlobalAssets) {
    Write-Host (Convert-Cn "[2/5] \u6b63\u5728\u590d\u5236\u5168\u5c40\u8d44\u4ea7...") -ForegroundColor Yellow
    foreach ($relativeRoot in $GlobalAssetWhitelist) {
        $source = Join-Path -Path $DataDir -ChildPath $relativeRoot
        $destination = Join-Path -Path $workDir -ChildPath ("data/{0}" -f $relativeRoot)
        $files = Copy-Tree -Source $source -Destination $destination -ExcludeDirectories $GlobalAssetExcludedDirectories
        $manifest.global_assets += @($files | ForEach-Object { "data/$relativeRoot/$_" })
    }
    Write-Host (Convert-Cn ("  \u5168\u5c40\u8d44\u4ea7\u6587\u4ef6: {0}" -f @($manifest.global_assets).Count)) -ForegroundColor Gray
} else { Write-Host (Convert-Cn "[2/5] \u8df3\u8fc7\u5168\u5c40\u8d44\u4ea7") -ForegroundColor DarkGray }

if ($Include.Motion) {
    Write-Host (Convert-Cn "[3/5] \u6b63\u5728\u590d\u5236\u52a8\u4f5c\u6570\u636e\u5e93...") -ForegroundColor Yellow
    $motionFiles = Copy-FileList -SourceRoot $DataDir -DestinationRoot (Join-Path $workDir "data") -RelativePaths $MotionFileWhitelist
    $manifest.motion_files = @($motionFiles)
} else { Write-Host (Convert-Cn "[3/5] \u8df3\u8fc7\u52a8\u4f5c\u6570\u636e\u5e93") -ForegroundColor DarkGray }

if ($Include.Kws) {
    Write-Host (Convert-Cn "[4/5] \u6b63\u5728\u590d\u5236\u6a21\u578b...") -ForegroundColor Yellow
    $kwsSource = Join-Path -Path (Join-Path -Path $DataDir -ChildPath "models") -ChildPath $KwsModelName
    if (-not (Test-Path -LiteralPath $kwsSource -PathType Container)) { throw (Convert-Cn ("\u6a21\u578b\u76ee\u5f55\u4e0d\u5b58\u5728: {0}" -f $kwsSource)) }
    $kwsDestination = Join-Path -Path (Join-Path -Path $workDir -ChildPath "data/models") -ChildPath $KwsModelName
    $kwsFiles = Copy-Tree -Source $kwsSource -Destination $kwsDestination
    $manifest.embedded_models = @($KwsModelName)
    Write-Host (Convert-Cn ("  \u6a21\u578b\u6587\u4ef6: {0}" -f @($kwsFiles).Count)) -ForegroundColor Gray
} else { Write-Host (Convert-Cn "[4/5] \u8df3\u8fc7\u6a21\u578b") -ForegroundColor DarkGray }

Write-Host (Convert-Cn "[5/5] \u6b63\u5728\u590d\u5236\u9009\u5b9a\u52a9\u624b...") -ForegroundColor Yellow
foreach ($agentName in $SelectedAgents) {
    $agentSource = Join-Path -Path $agentsRoot -ChildPath $agentName
    if (-not (Test-Path -LiteralPath $agentSource -PathType Container)) {
        Write-Warning (Convert-Cn ("\u627e\u4e0d\u5230\u52a9\u624b,\u8df3\u8fc7: {0}" -f $agentName))
        continue
    }
    $agentDestination = Join-Path -Path $workDir -ChildPath "data/agents/$agentName"
    $infoFiles = Copy-FileList -SourceRoot $agentSource -DestinationRoot $agentDestination -RelativePaths $AgentInfoFiles
    $assetsSource = Join-Path -Path $agentSource -ChildPath "assets"
    $assetsDestination = Join-Path -Path $agentDestination -ChildPath "assets"
    $assetFiles = Copy-Tree -Source $assetsSource -Destination $assetsDestination
    $manifest.agents += [ordered]@{
        name = $agentName
        info_files = @($infoFiles | ForEach-Object { "data/agents/$agentName/$_" })
        asset_files = @($assetFiles | ForEach-Object { "data/agents/$agentName/assets/$_" })
    }
}

$manifestPath = Join-Path -Path $workDir -ChildPath "manifest.json"
[System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))

$zipPath = Join-Path -Path $OutputPath -ChildPath "moechat-assets-v${Version}-${PlatformTag}-lite.zip"
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Add-DirectoryToZip -SourceDir $workDir -ZipPath $zipPath
$zipSize = [math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 1)
Write-Host ""; Write-Host (Convert-Cn ("\u5b8c\u6210: {0}" -f $zipPath)) -ForegroundColor Green
Write-Host (Convert-Cn ("\u5927\u5c0f: {0} MB | \u52a9\u624b: {1} | \u6a21\u578b: {2}" -f $zipSize, @($manifest.agents).Count, $Include.Kws)) -ForegroundColor Green
Remove-Item -LiteralPath $workDir -Recurse -Force -ErrorAction SilentlyContinue
