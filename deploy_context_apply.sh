#!/usr/bin/env bash
set -euo pipefail

cd /home/yjq/qwen3-14b-finance-sft-agent

TS="$(date +%Y%m%d_%H%M%S)"
BK="/home/yjq/deploy_context_backup_${TS}"
mkdir -p "${BK}"

while IFS= read -r f; do
  [ -z "$f" ] && continue
  [ -f "$f" ] || continue
  mkdir -p "${BK}/$(dirname "$f")"
  cp -p "$f" "${BK}/$f"
done < deploy_context_manifest.txt

tar -xzf deploy_context.tgz
echo "context_${TS}" > /home/yjq/last_deploy_ts.txt

echo "backup_dir=${BK}"
echo "deployed_files=$(wc -l < deploy_context_manifest.txt)"
echo "last_deploy_ts=context_${TS}"
