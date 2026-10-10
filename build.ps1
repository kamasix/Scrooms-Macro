[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$OutputName = 'ShroomsMacro',
    [string]$Python = (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Create a Python 3.12 environment and install requirements-dev.txt first. See README.md.'
}
$taskAssets = Join-Path $PSScriptRoot 'assets'
$taskLicenses = Join-Path $PSScriptRoot 'licenses'
$taskManifest = Join-Path $PSScriptRoot 'app.manifest'
$taskEntry = Join-Path $PSScriptRoot 'main.py'
$taskProjectLicense = Join-Path $PSScriptRoot 'LICENSE'
$taskNotices = Join-Path $PSScriptRoot 'THIRD_PARTY_NOTICES.md'
& $Python -m PyInstaller --noconfirm --onefile --windowed --name $OutputName --manifest $taskManifest --add-data "$taskAssets;assets" --add-data "$taskLicenses;licenses" --add-data "$taskProjectLicense;licenses" --add-data "$taskNotices;licenses" --distpath dist --workpath build --specpath build $taskEntry
if ($LASTEXITCODE -ne 0) { throw 'The executable build failed.' }
