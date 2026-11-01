# SignalSense AI launcher (CAMP-CRL method, Windows PowerShell). Simulation and advisory only: never connect to real signals.
#
#   .\start.ps1               install deps if needed, build dashboard if needed, start it at http://127.0.0.1:8000
#   .\start.ps1 -Test         also run the Python and frontend test suites first
#   .\start.ps1 -Rebuild      force a fresh frontend build
#   .\start.ps1 -Gui          instead of the dashboard, watch one episode in the SUMO GUI window
#                             (optionally -Controller camp_rule -Scenario balanced -Seed 1)
param(
    [switch]$Test, [switch]$Rebuild, [switch]$Gui,
    [string]$Controller = "camp_rule", [string]$Scenario = "balanced", [int]$Seed = 1, [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .\.venv\Scripts\python.exe)) { py -3.14 -m venv .venv }
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install -q -r requirements.txt

if ($Gui) {
    & $py -m camp_crl.cli run --controller $Controller --scenario $Scenario --seed $Seed --gui
    exit $LASTEXITCODE
}

if ($Rebuild -or -not (Test-Path .\frontend\dist\index.html)) {
    Push-Location frontend
    if (-not (Test-Path node_modules)) { npm ci }
    npm run build
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "frontend build failed" }
    Pop-Location
}

if ($Test) {
    & $py -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Python tests failed" }
    Push-Location frontend; npx vitest run; $code = $LASTEXITCODE; Pop-Location
    if ($code -ne 0) { throw "frontend tests failed" }
}

Write-Host "SignalSense AI dashboard: http://127.0.0.1:$Port   (Ctrl+C to stop)"
Start-Process "http://127.0.0.1:$Port"
& $py -m uvicorn camp_crl.api.app:app --host 127.0.0.1 --port $Port
