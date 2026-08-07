$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$agentRoot = Join-Path $repoRoot "agent"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$envFile = Join-Path $agentRoot ".env"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv. Create it and install agent/requirements.txt first."
}
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing agent/.env. Copy agent/.env.example and fill your own keys first."
}
$envText = Get-Content -LiteralPath $envFile -Raw
if ($envText -match 'DEEPSEEK_API_KEY=(your_|$)') {
    throw "agent/.env does not contain a usable DEEPSEEK_API_KEY."
}

$env:NO_PROXY = "127.0.0.1,localhost,::1"
$env:no_proxy = "127.0.0.1,localhost,::1"
Push-Location $agentRoot
try {
    docker compose up -d
    & $python -m scripts.init_personal_data
    & $python -m scripts.run_production_api
} finally {
    Pop-Location
}
