[CmdletBinding()]
param(
    [string]$RepositoryPath = (Join-Path $env:USERPROFILE 'Downloads\Compressed\Regulatrace_Deployment_Package\Regulatrace'),
    [string]$PackagePath
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($PackagePath)) {
    $PackagePath = $scriptDirectory
}
$resolvedPackage = Resolve-Path -LiteralPath $PackagePath -ErrorAction Stop
if ($resolvedPackage.Provider.Name -ne 'FileSystem' -or
    -not (Test-Path -LiteralPath $resolvedPackage.ProviderPath -PathType Container)) {
    throw 'PackagePath must be an existing Regulatrace full-stack folder.'
}
$package = $resolvedPackage.ProviderPath
$payload = Join-Path $scriptDirectory 'branding-payload'
$python = Join-Path $package '.fullstack-state\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python) -or
    -not (Test-Path -LiteralPath (Join-Path $package '.fullstack-state\secrets.json')) -or
    -not (Test-Path -LiteralPath (Join-Path $package 'corpus\manifest.json'))) {
    throw 'Run this updater from the existing configured Regulatrace_Fullstack_Package folder.'
}
$manifest = Get-Content (Join-Path $payload 'branding-files.json') -Raw | ConvertFrom-Json
$release = Get-Content (Join-Path $payload 'release.json') -Raw | ConvertFrom-Json
foreach ($file in $manifest.PSObject.Properties) {
    if ($file.Name -match '(^/|(^|/)\.\.(/|$)|(^|/)\.fullstack-state(/|$))') {
        throw 'The update contains an invalid file path.'
    }
    $source = Join-Path $payload $file.Name
    if (-not (Test-Path -LiteralPath $source) -or
        (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.Value) {
        throw 'An update file changed. Download and extract the update again.'
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $RepositoryPath '.git'))) {
    throw 'Supply -RepositoryPath with the existing Regulatrace Git folder.'
}
$remote = (& git -C $RepositoryPath remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $remote -notmatch '^(https://github\.com/Aleem547/Regulatrace(\.git)?/?|git@github\.com:Aleem547/Regulatrace\.git)$') {
    throw 'This updater only changes Aleem547/Regulatrace.'
}
$branch = (& git -C $RepositoryPath branch --show-current).Trim()
$changes = & git -C $RepositoryPath status --porcelain
if ($LASTEXITCODE -ne 0 -or $branch -ne 'main' -or $changes) {
    throw 'Commit existing changes and switch the Regulatrace repository to main before retrying.'
}
& git -C $RepositoryPath fetch origin main
if ($LASTEXITCODE -ne 0) { throw 'Could not check GitHub main.' }
& git -C $RepositoryPath merge --ff-only origin/main
if ($LASTEXITCODE -ne 0) { throw 'Reconcile the local and remote main branches before retrying.' }
$head = (& git -C $RepositoryPath rev-parse HEAD).Trim()
if ($head -ne $release.baseline_github_sha) {
    throw 'GitHub source changed after this update was prepared. Keep the current setup and request a refreshed update.'
}
& git -C $RepositoryPath apply --check (Join-Path $payload 'Regulatrace_Fullstack_Update.patch')
if ($LASTEXITCODE -ne 0) { throw 'The reviewed rename patch does not match this source checkout.' }
& $python (Join-Path $payload 'migrate_branding.py') --package $package --repository $RepositoryPath
if ($LASTEXITCODE -ne 0) { throw 'Private configuration migration stopped. Existing values have a private backup.' }
$backup = Join-Path $package '.fullstack-state\branding-backup\files'
foreach ($file in $manifest.PSObject.Properties) {
    $source = Join-Path $payload $file.Name
    $destination = Join-Path $package $file.Name
    $saved = Join-Path $backup $file.Name
    if ((Test-Path -LiteralPath $destination) -and -not (Test-Path -LiteralPath $saved)) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force | Out-Null
        Copy-Item -LiteralPath $destination -Destination $saved
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}
Write-Host 'Regulatrace files installed. Reusing the configured free services and saved credentials.'
& $python (Join-Path $package 'deploy_fullstack.py')
if ($LASTEXITCODE -ne 0) {
    throw 'The renamed deployment has not passed its live checks. Rerun this updater to resume.'
}
& (Join-Path $package 'Update-GitHub.ps1') -RepositoryPath $RepositoryPath
if ($LASTEXITCODE -ne 0) { throw 'The renamed deployment is verified; the GitHub publication needs review.' }
Write-Host 'Regulatrace rename deployed, verified and published to GitHub.'
