param(
    [string]$RepositoryPath = (Join-Path $env:USERPROFILE 'Downloads\Compressed\Regulatrace_Deployment_Package\Regulatrace')
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$resultPath = Join-Path $PSScriptRoot 'DEPLOYMENT_RESULT.json'
$releasePath = Join-Path $PSScriptRoot 'release.json'
$patchPath = Join-Path $PSScriptRoot 'Regulatrace_Fullstack_Update.patch'
if (-not (Test-Path $resultPath)) { throw 'Complete Deploy-Regulatrace.ps1 first; it must pass hosted checks.' }
$result = Get-Content $resultPath -Raw | ConvertFrom-Json
if ($result.status -ne 'live_core_verified' -or $result.web_url -notmatch '^https://[a-z0-9-]+\.vercel\.app$') {
    throw 'A verified Vercel deployment result is required.'
}
foreach ($check in $result.verification.PSObject.Properties) {
    if ($check.Value -ne 'passed') { throw 'Every hosted verification check must pass first.' }
}
$release = Get-Content $releasePath -Raw | ConvertFrom-Json
if (($release.PSObject.Properties.Name -contains 'branding_migration') -and $release.branding_migration) {
    if (-not ($result.PSObject.Properties.Name -contains 'branding') -or
        $result.branding.python_package -ne 'regulatrace' -or
        $result.branding.configuration_prefix -ne 'REGULATRACE_') {
        throw 'Complete Rename-Regulatrace.ps1 before publishing the renamed source.'
    }
}
if ((Get-FileHash -Algorithm SHA256 $patchPath).Hash.ToLowerInvariant() -ne $release.patch_sha256) {
    throw 'The reviewed source patch changed. Extract the package again.'
}
if (-not (Test-Path (Join-Path $RepositoryPath '.git'))) {
    throw 'Supply -RepositoryPath with the existing Regulatrace Git folder.'
}
$remote = (& git -C $RepositoryPath remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $remote -notmatch '^(https://github\.com/Aleem547/Regulatrace(\.git)?/?|git@github\.com:Aleem547/Regulatrace\.git)$') {
    throw 'This script only publishes to Aleem547/Regulatrace.'
}
$branch = (& git -C $RepositoryPath branch --show-current).Trim()
if ($branch -ne 'main') { throw 'Switch the existing repository to main first.' }
$changes = & git -C $RepositoryPath status --porcelain
if ($LASTEXITCODE -ne 0 -or $changes) { throw 'Save or commit existing changes before publishing this update.' }
& git -C $RepositoryPath fetch origin main
if ($LASTEXITCODE -ne 0) { throw 'Could not check GitHub main.' }
& git -C $RepositoryPath merge --ff-only origin/main
if ($LASTEXITCODE -ne 0) { throw 'Local and remote main diverged. Reconcile them before retrying.' }
$head = (& git -C $RepositoryPath rev-parse HEAD).Trim()
$remoteHead = (& git -C $RepositoryPath rev-parse origin/main).Trim()
if ($head -ne $remoteHead) { throw 'Local main has extra commits. Reconcile them before publishing.' }
& git -C $RepositoryPath apply --check $patchPath
if ($LASTEXITCODE -ne 0) { throw 'The patch does not match current GitHub source. No update was applied.' }
& git -C $RepositoryPath apply $patchPath
if ($LASTEXITCODE -ne 0) { throw 'Could not apply the reviewed update.' }
$readmePath = Join-Path $RepositoryPath 'README.md'
$readme = [IO.File]::ReadAllText($readmePath)
$block = @"
<!-- REGULATRACE_DEPLOYMENT_START -->
**Live application:** [Open Regulatrace]($($result.web_url)).

**Document demo:** [Hugging Face](https://huggingface.co/spaces/abdul547/Regulatrace).

**Deployment status:** Core login, chat, stored history, corpus indexes and hosted checks passed.
See [deployment verification](docs/LIVE_DEPLOYMENT.json) and [deployment instructions](docs/FREE_FULLSTACK_DEPLOYMENT.md).
<!-- REGULATRACE_DEPLOYMENT_END -->
"@
$pattern = '(?s)<!-- REGULATRACE_DEPLOYMENT_START -->.*?<!-- REGULATRACE_DEPLOYMENT_END -->'
if (-not [regex]::IsMatch($readme, $pattern)) { throw 'The expected deployment section is missing.' }
$readme = [regex]::Replace($readme, $pattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $block })
[IO.File]::WriteAllText($readmePath, $readme, [Text.UTF8Encoding]::new($false))
Copy-Item $resultPath (Join-Path $RepositoryPath 'docs\LIVE_DEPLOYMENT.json')
$sourcePaths = @($release.github_files) + @('docs/LIVE_DEPLOYMENT.json')
foreach ($sourcePath in $sourcePaths) {
    $basename = Split-Path $sourcePath -Leaf
    if ($sourcePath -match '(^/|(^|/)\.\.(/|$)|\.(pdf|pem|key|safetensors)$)' -or
        ($basename -match '^\.env($|\.)' -and $basename -ne '.env.example')) {
        throw 'The source allowlist contains an unsafe path.'
    }
}
$migrationTool = Join-Path $PSScriptRoot 'migrate_branding.py'
if (($release.PSObject.Properties.Name -contains 'branding_migration') -and $release.branding_migration) {
    $python = Join-Path $PSScriptRoot '.fullstack-state\venv\Scripts\python.exe'
    & $python $migrationTool --package $PSScriptRoot --repository $RepositoryPath --env-only
    if ($LASTEXITCODE -ne 0) { throw 'Local configuration migration needs review before publishing.' }
}
& git -C $RepositoryPath add -A -- @sourcePaths
if ($LASTEXITCODE -ne 0) { throw 'Could not stage the reviewed source.' }
$commitMessage = 'feat: deploy Regulatrace with Vercel Neon and Qdrant'
if (($release.PSObject.Properties.Name -contains 'branding_migration') -and $release.branding_migration) {
    $commitMessage = 'refactor: rename application and configuration to Regulatrace'
}
& git -C $RepositoryPath commit -m $commitMessage
if ($LASTEXITCODE -ne 0) { throw 'Could not commit the reviewed update.' }
& git -C $RepositoryPath push origin main
if ($LASTEXITCODE -ne 0) { throw 'The commit is saved locally; GitHub push failed. No force push was attempted.' }
Write-Host "Published Regulatrace source and live README link: $($result.web_url)"
Write-Host 'To set the GitHub About website too: repository About > gear > Website.'
