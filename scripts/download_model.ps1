param(
    [Parameter(Mandatory = $true)]
    [string]$Repo,
    [string]$Tag = "model-v1",
    [string]$Asset = "qwen3-14b-finance-sft-adapter-v1.tar.gz"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$artifactRoot = Join-Path $repoRoot "model\artifacts"
$archivePath = Join-Path $artifactRoot $Asset
$checksumPath = "$archivePath.sha256"
$baseUrl = "https://github.com/$Repo/releases/download/$Tag"

New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
Invoke-WebRequest -Uri "$baseUrl/$Asset" -OutFile $archivePath
Invoke-WebRequest -Uri "$baseUrl/$Asset.sha256" -OutFile $checksumPath

$expected = (Get-Content -LiteralPath $checksumPath -Raw).Trim().Split()[0].ToLowerInvariant()
$actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($expected -ne $actual) {
    throw "Model archive SHA-256 mismatch. Expected $expected, got $actual"
}

tar -xzf $archivePath -C $artifactRoot
if ($LASTEXITCODE -ne 0) {
    throw "Failed to extract model archive"
}

$required = @(
    "adapter_config.json",
    "adapter_model.safetensors",
    "embedding_patch.pt",
    "tokenizer.json",
    "tokenizer_config.json"
)
foreach ($name in $required) {
    $path = Join-Path $artifactRoot "final_adapter\$name"
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Incomplete model package: missing $name"
    }
}

Write-Host "Final SFT adapter installed at $artifactRoot\final_adapter"
