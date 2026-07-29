param([switch]$RemoveOllamaModel)
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv-local"
if (Test-Path $Venv) { Remove-Item -LiteralPath $Venv -Recurse -Force }
Remove-Item -LiteralPath (Join-Path $Root "local-mode.json") -Force -ErrorAction SilentlyContinue
if ($RemoveOllamaModel -and (Get-Command ollama -ErrorAction SilentlyContinue)) { ollama rm qwen3:8b }
Write-Host "本地扩展环境已移除。Ollama 程序及其他模型保持不变。"

