$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv-local\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "未安装本地扩展。请先执行 install-and-enable-local.ps1" }
$env:PDF_TRANSLATOR_LOCAL_CONFIG = Join-Path $Root "local-mode.json"
& $Python (Join-Path $Root "app.py")

