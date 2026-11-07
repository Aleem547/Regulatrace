$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$publisherRoot = $PSScriptRoot
$publisherEnv = Join-Path $publisherRoot '.publisher-venv'
$publisherPython = Join-Path $publisherEnv 'Scripts\python.exe'

if (-not (Test-Path $publisherPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $publisherEnv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $publisherEnv
    } else {
        throw 'Install Python 3.10 or newer, then run this script again.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the upload environment.' }
}
& $publisherPython -m pip install --disable-pip-version-check 'huggingface-hub==1.33.0'
if ($LASTEXITCODE -ne 0) { throw 'Could not install the upload client.' }
& $publisherPython (Join-Path $publisherRoot 'publish_space.py')
if ($LASTEXITCODE -ne 0) { throw 'Space publication did not complete. See the message above.' }
