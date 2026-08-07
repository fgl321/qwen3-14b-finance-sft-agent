#!/bin/bash
#SBATCH --job-name=finance_embed_gpu
#SBATCH --account=acc_gpu80g
#SBATCH --partition=gpu40g
#SBATCH --qos=gpu80g
#SBATCH --gres=gpu:40g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/home/yjq/qwen3-14b-finance-sft-agent/agent
#SBATCH --output=/home/yjq/qwen3-14b-finance-sft-agent/logs/embed_server_%j.log
#SBATCH --error=/home/yjq/qwen3-14b-finance-sft-agent/logs/embed_server_%j.err

set -euo pipefail
mkdir -p /home/yjq/qwen3-14b-finance-sft-agent/logs

export BGE_M3_MODEL_NAME=/home/yjq/models/bge-m3
export RAG_RERANK_MODEL=/home/yjq/models/bge-reranker-v2-m3
export EMBED_SERVER_DEVICE=cuda
export HF_HUB_CACHE=/home/yjq/.cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false

exec ../.venv/bin/python -m uvicorn app.rag.embed_server:app \
  --host 0.0.0.0 --port 9000
