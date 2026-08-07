#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 OWNER/REPO [TAG]" >&2
  exit 2
fi

REPO="$1"
TAG="${2:-model-v1}"
ASSET="qwen3-14b-finance-sft-adapter-v1.tar.gz"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="$ROOT/model/artifacts"
BASE_URL="https://github.com/$REPO/releases/download/$TAG"

mkdir -p "$ARTIFACT_ROOT"
curl --fail --location "$BASE_URL/$ASSET" --output "$ARTIFACT_ROOT/$ASSET"
curl --fail --location "$BASE_URL/$ASSET.sha256" --output "$ARTIFACT_ROOT/$ASSET.sha256"
(
  cd "$ARTIFACT_ROOT"
  sha256sum --check "$ASSET.sha256"
  tar -xzf "$ASSET"
)

for name in adapter_config.json adapter_model.safetensors embedding_patch.pt tokenizer.json tokenizer_config.json; do
  test -f "$ARTIFACT_ROOT/final_adapter/$name" || {
    echo "Incomplete model package: missing $name" >&2
    exit 1
  }
done

echo "Final SFT adapter installed at $ARTIFACT_ROOT/final_adapter"
