# One-step launcher (ADR-021): installs what is missing, then runs `biblio start`.
# Downloads happen only when something is missing: uv (after asking), Python and the dependencies
# (`uv sync`); `biblio start` asks before downloading models or the LLM.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Test-Uv { $null -ne (Get-Command uv -ErrorAction SilentlyContinue) }

$uvHome = Join-Path $HOME ".local\bin"
if (-not (Test-Uv) -and (Test-Path (Join-Path $uvHome "uv.exe"))) { $env:Path = "$uvHome;$env:Path" }

if (-not (Test-Uv)) {
    Write-Host "uv, the tool that installs Python and this app's packages, is not installed."
    $answer = Read-Host "Install it now from astral.sh? [Y/n]"
    if ($answer -match '^(n|no)$') {
        Write-Host "Install uv from https://docs.astral.sh/uv/ and run this again."
        exit 1
    }
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$uvHome;$env:Path"
    if (-not (Test-Uv)) { Write-Host "uv could not be installed."; exit 1 }
}

Write-Host "Checking Python and the app's packages (downloads only what is missing)..."
uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run biblio start @args
exit $LASTEXITCODE
