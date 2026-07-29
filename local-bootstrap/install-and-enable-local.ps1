param(
    [ValidateSet("auto", "cpu", "gpu")]
    [string]$Mode = "auto",
    [string]$TranslationModel = "qwen3:8b"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv-local"
$Python = Join-Path $Venv "Scripts\python.exe"
$Config = Join-Path $Root "local-mode.json"
$Log = Join-Path $PSScriptRoot "install.log"
$Hardware = Join-Path $PSScriptRoot "hardware-report.json"

function Write-Step([string]$Message) {
    $line = "[$(Get-Date -Format s)] $Message"
    $line | Tee-Object -FilePath $Log -Append
}

Write-Step "Detecting hardware"
$gpu = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
}
$useGpu = $Mode -eq "gpu" -or ($Mode -eq "auto" -and $null -ne $gpu)
$hardwareObject = [ordered]@{
    collectedAt = (Get-Date).ToString("o")
    windows = (Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty Caption)
    cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
    memoryGiB = [math]::Round(((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB), 1)
    nvidia = $gpu
    selectedDevice = $(if($useGpu){"gpu"}else{"cpu"})
}
$hardwareObject | ConvertTo-Json | Set-Content -Path $Hardware -Encoding UTF8

if (-not (Test-Path $Python)) {
    if (Test-Path $Venv) {
        Write-Step "Removing incomplete local environment"
        Remove-Item -LiteralPath $Venv -Recurse -Force
    }
    Write-Step "Creating isolated Python environment"
    python -m venv $Venv
}
if (-not (Test-Path $Python)) { throw "Local Python environment creation failed" }

Write-Step "Installing PaddleOCR-VL dependencies"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
if ($useGpu) {
    Write-Step "Installing PaddlePaddle GPU runtime"
    & $Python -m pip install "paddlepaddle-gpu>=3.2.1"
    if ($LASTEXITCODE -ne 0) {
        # The public Windows index does not currently provide every 3.x GPU
        # wheel. A working CPU backend is preferable to a false GPU success.
        Write-Step "GPU runtime unavailable; falling back to PaddlePaddle CPU"
        $useGpu = $false
        & $Python -m pip install "paddlepaddle>=3.2.1"
        if ($LASTEXITCODE -ne 0) { throw "PaddlePaddle CPU fallback failed" }
    }
} else {
    Write-Step "Installing PaddlePaddle CPU runtime"
    & $Python -m pip install "paddlepaddle>=3.2.1"
    if ($LASTEXITCODE -ne 0) { throw "PaddlePaddle CPU install failed" }
}
& $Python -m pip install "paddleocr[doc-parser]" "svglib>=2" "gradio>=5" "pypdf>=5" "reportlab>=4" "Pillow>=10" "opencv-python-headless>=4.10"
if ($LASTEXITCODE -ne 0) { throw "PaddleOCR-VL dependency install failed" }

Write-Step "Prefetching PaddleOCR-VL models"
& $Python -c "from paddleocr import PaddleOCRVL; PaddleOCRVL()"
if ($LASTEXITCODE -ne 0) { throw "PaddleOCR-VL model prefetch failed" }

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Step "Installing Ollama"
    irm https://ollama.com/install.ps1 | iex
}
Write-Step "Downloading local translation model $TranslationModel"
& ollama pull $TranslationModel
if ($LASTEXITCODE -ne 0) { throw "Ollama model download failed" }

Write-Step "Installing MathJax"
Push-Location $Root
try { if (-not (Test-Path "node_modules\mathjax-full")) { & npm.cmd install --omit=dev } } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "MathJax install failed" }

Write-Step "Writing install verification"
$checksums = [ordered]@{
    generatedAt = (Get-Date).ToString("o")
    packageLockSha256 = $(if(Test-Path (Join-Path $Root 'package-lock.json')){(Get-FileHash (Join-Path $Root 'package-lock.json') -Algorithm SHA256).Hash}else{$null})
    pythonRequirements = (& $Python -m pip freeze)
    ollamaModels = $(if(Get-Command ollama -ErrorAction SilentlyContinue){& ollama list}else{$null})
}
$checksums | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $PSScriptRoot "installed-checksums.json") -Encoding UTF8
$configObject = @{ enabled=$true; installedAt=(Get-Date).ToString("o"); device=$(if($useGpu){"gpu"}else{"cpu"}); translationModel=$TranslationModel; paddlePython=$Python; modelsCached=$true; hardwareReport=$Hardware; checksums=(Join-Path $PSScriptRoot "installed-checksums.json") }
$configObject | ConvertTo-Json | Set-Content -Path $Config -Encoding UTF8
Write-Step "Local mode installation completed"
