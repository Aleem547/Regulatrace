[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$repositoryUrl = 'https://github.com/Aleem547/Regulatrace.git'

function Invoke-ProjectGit {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArguments)
    & git -C $projectPath @GitArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Git failed. Resolve the displayed error and rerun; do not force-push.'
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git for Windows is required. Install it and reopen PowerShell.'
}

foreach ($relativePath in @(
    'src/regulatrace/api/app.py', 'frontend/src/App.tsx',
    'frontend/package-lock.json', 'alembic/env.py', 'pyproject.toml', 'UPSTREAM.md'
)) {
    if (-not (Test-Path (Join-Path $projectPath $relativePath) -PathType Leaf)) {
        throw "Incomplete extracted package: missing $relativePath"
    }
}

# Run only when you have permission to redistribute the credited upstream source.
# No clone, force push, credential storage or history deletion is performed.
if (-not (Test-Path (Join-Path $projectPath '.git'))) {
    Invoke-ProjectGit init -b main
}

$remoteNames = @(Invoke-ProjectGit remote)
if ($remoteNames -contains 'origin') {
    $existingUrl = (Invoke-ProjectGit remote get-url origin | Out-String).Trim()
    if ($existingUrl.TrimEnd('/') -ine $repositoryUrl) {
        throw 'This folder targets another repository. Extract a fresh package instead.'
    }
} else {
    Invoke-ProjectGit remote add origin $repositoryUrl
}

$currentBranch = (Invoke-ProjectGit branch --show-current | Out-String).Trim()
if ($currentBranch -ne 'main') {
    throw 'This folder is on another branch. Extract a fresh package instead.'
}

Invoke-ProjectGit add --all
$stagedPaths = @(Invoke-ProjectGit diff --cached --name-only)
foreach ($path in $stagedPaths) {
    $basename = Split-Path $path -Leaf
    if (($basename -match '^\.env($|\.)' -and $basename -ne '.env.example') -or
        $path -match '(^|/)(node_modules|\.venv|postgres_data|qdrant_storage)/' -or
        $basename -match '\.(pem|key)$') {
        throw "Private or generated file was staged: $path. Remove it before publishing."
    }
}
if ($stagedPaths.Count -gt 0) {
    Invoke-ProjectGit commit -m 'Import credited Regulatrace source and Regulatrace deployment changes'
}
Invoke-ProjectGit push -u origin main
Write-Host 'Source published to https://github.com/Aleem547/Regulatrace'
Write-Host 'Hosting deployment still requires private service settings and the regulatory corpus.'
