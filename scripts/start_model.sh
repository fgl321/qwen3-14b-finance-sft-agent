#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
export QWEN_BASE_MODEL="${QWEN_BASE_MODEL:-Qwen/Qwen3-14B}"
export QWEN_ADAPTER_DIR="${QWEN_ADAPTER_DIR:-$ROOT/model/artifacts/final_adapter}"
export PYTHONPATH="$ROOT"

: "${QWEN_SERVER_API_KEY:?Set QWEN_SERVER_API_KEY before starting the model service}"
test -x "$PYTHON" || { echo "Missing .venv" >&2; exit 1; }
test -f "$QWEN_ADAPTER_DIR/adapter_model.safetensors" || {
  echo "Missing final SFT adapter. Run scripts/download_model.sh first." >&2
  exit 1
}

exec "$PYTHON" -m uvicorn model.server:app --host "${QWEN_HOST:-127.0.0.1}" --port "${QWEN_PORT:-8001}"
