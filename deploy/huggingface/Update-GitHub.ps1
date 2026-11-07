param(
    [string]$RepositoryPath = (Join-Path $env:USERPROFILE 'Downloads\Compressed\Regulatrace_Deployment_Package\Regulatrace')
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$patchPath = Join-Path $PSScriptRoot 'Regulatrace_ZeroGPU_Update.patch'
$listPath = Join-Path $PSScriptRoot 'github-files.json'
if (-not (Test-Path (Join-Path $RepositoryPath '.git'))) {
    throw 'Supply -RepositoryPath with the existing Regulatrace Git folder. No clone is required.'
}
$remote = (& git -C $RepositoryPath remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $remote -notmatch '^(https://github\.com/Aleem547/Regulatrace(\.git)?/?|git@github\.com:Aleem547/Regulatrace\.git)$') {
    throw 'The repository must target Aleem547/Regulatrace.'
}
$branch = (& git -C $RepositoryPath branch --show-current).Trim()
if ($branch -ne 'main') { throw 'Switch the existing repository to main first.' }
$workingChanges = & git -C $RepositoryPath status --porcelain
if ($LASTEXITCODE -ne 0 -or $workingChanges) { throw 'Save or commit existing changes before applying this reviewed update.' }
& git -C $RepositoryPath fetch origin main
if ($LASTEXITCODE -ne 0) { throw 'Could not check the remote branch.' }
$localHead = (& git -C $RepositoryPath rev-parse HEAD).Trim()
$remoteHead = (& git -C $RepositoryPath rev-parse origin/main).Trim()
if ($localHead -ne $remoteHead) { throw 'Local main and origin/main differ. Reconcile them before applying this update.' }
& git -C $RepositoryPath apply --check $patchPath
if ($LASTEXITCODE -ne 0) { throw 'The patch does not match this source revision. No files were changed.' }
& git -C $RepositoryPath apply $patchPath
if ($LASTEXITCODE -ne 0) { throw 'Could not apply the source update.' }
$sourcePaths = @(Get-Content $listPath -Raw | ConvertFrom-Json)
foreach ($sourcePath in $sourcePaths) {
    if ($sourcePath -match '(^/|(^|/)\.\.(/|$)|(^|/)\.env|\.(pdf|pem|key)$)') {
        throw 'The source allowlist contains an unsafe path.'
    }
}
& git -C $RepositoryPath add -- @sourcePaths
if ($LASTEXITCODE -ne 0) { throw 'Could not stage the reviewed files.' }
& git -C $RepositoryPath commit -m 'feat: add free Regulatrace ZeroGPU document demo'
if ($LASTEXITCODE -ne 0) { throw 'Could not commit the reviewed update.' }
& git -C $RepositoryPath push origin main
if ($LASTEXITCODE -ne 0) { throw 'The commit is saved locally, but GitHub publication failed. No force push was attempted.' }
Write-Host 'Published the reviewed source update to Aleem547/Regulatrace.'
