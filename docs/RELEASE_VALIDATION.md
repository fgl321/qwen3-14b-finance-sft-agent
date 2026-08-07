# Release Validation

## Model artifact

- Asset: `qwen3-14b-finance-sft-adapter-v1.tar.gz`
- Size: 242,086,199 bytes
- SHA-256: `4447f31637905b5a51aaf6a99bf2c1397c21e0dbebd0d44fcb5981fca8af739d`
- Forbidden-string scan: 0 matches for API Key、SSH private key、internal cluster IP and personal absolute paths.
- `adapter_config.json` base model: `Qwen/Qwen3-14B`

## Independent self-hosted load

The release archive was extracted into a clean validation directory and loaded with the repository's `model/model_loader.py` rather than the existing online service.

- Slurm job: 1895
- State: COMPLETED
- Exit code: 0
- Runtime: 32 seconds
- GPU: NVIDIA A100 80GB PCIe
- Tokenizer size: 151,669
- Test prompt: `请用一句话解释市盈率的含义。`
- Output: `市盈率是股票价格与每股收益的比率，用于衡量股票的估值水平。`

## Agent tests

- Model loader tests: 2 passed
- Agent unit tests: 159 passed, 1 skipped
- Secret scan: passed
- Large-file scan: passed
- Python compile check: passed
