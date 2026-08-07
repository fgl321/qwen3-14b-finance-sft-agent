#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_ROOT="$ROOT/agent"
PYTHON="$ROOT/.venv/bin/python"

test -x "$PYTHON" || { echo "Missing .venv" >&2; exit 1; }
test -f "$AGENT_ROOT/.env" || { echo "Missing agent/.env" >&2; exit 1; }
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

cd "$AGENT_ROOT"
docker compose up -d
"$PYTHON" -m scripts.init_personal_data
exec "$PYTHON" -m scripts.run_production_api
