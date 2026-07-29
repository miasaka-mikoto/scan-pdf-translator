param([switch]$SkipInstall)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = (Join-Path $Root ".venv-cloud")
$Python = (Join-Path $Venv "Scripts\python.exe")
$LocalConfig = (Join-Path $Root "local-mode.json")

# After local installation, launch the app from its isolated environment so
# PaddleOCR-VL is importable when the user selects local/auto.
if (Test-Path $LocalConfig) {
    try {
        $LocalPython = (Get-Content -Raw $LocalConfig | ConvertFrom-Json).paddlePython
        if ($LocalPython -and (Test-Path $LocalPython)) { $Python = $LocalPython }
    } catch { }
}

if (-not (Test-Path $Python)) {
    $BasePython = if ($env:PDF_TRANSLATOR_PYTHON) { $env:PDF_TRANSLATOR_PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
    & $BasePython -m venv $Venv
}

if (-not $SkipInstall -and $Python -like "$Venv*") {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r (Join-Path $Root "requirements.txt")
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
        throw "Node.js/npm not found. Install Node.js LTS, then rerun start.ps1."
    }
    Push-Location $Root
    try { & npm.cmd install --omit=dev } finally { Pop-Location }
}

$env:PDF_TRANSLATOR_OUTPUT_DIR = (Join-Path -Path $Root -ChildPath 'outputs')
& $Python (Join-Path -Path $Root -ChildPath 'app.py')
