$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Install Python 3.12 or newer and reopen PowerShell.' }
if (-not (Get-Command npx -ErrorAction SilentlyContinue)) { throw 'Install Node.js 22 LTS or newer and reopen PowerShell.' }
$venvPath = Join-Path $PSScriptRoot '.fullstack-state\venv'
if (-not (Test-Path (Join-Path $venvPath 'Scripts\python.exe'))) {
    & py -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the deployment environment.' }
}
$python = Join-Path $venvPath 'Scripts\python.exe'
& $python -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install deployment tools.' }
& $python (Join-Path $PSScriptRoot 'deploy_fullstack.py')
if ($LASTEXITCODE -ne 0) { throw 'Full deployment is not verified. Read the message above.' }
