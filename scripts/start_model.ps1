param(
    [string]$BaseModel = "Qwen/Qwen3-14B",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$adapter = Join-Path $repoRoot "model\artifacts\final_adapter"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv. Create it and install model/requirements.txt first."
}
if (-not (Test-Path -LiteralPath (Join-Path $adapter "adapter_model.safetensors"))) {
    throw "Missing final SFT adapter. Run scripts/download_model.ps1 first."
}
if ([string]::IsNullOrWhiteSpace($env:QWEN_SERVER_API_KEY)) {
    throw "Set QWEN_SERVER_API_KEY before starting the model service."
}

$env:QWEN_BASE_MODEL = $BaseModel
$env:QWEN_ADAPTER_DIR = $adapter
$env:PYTHONPATH = $repoRoot
& $python -m uvicorn model.server:app --host $HostAddress --port $Port
