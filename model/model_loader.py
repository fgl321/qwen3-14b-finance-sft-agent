from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "embedding_patch.pt",
    "tokenizer.json",
    "tokenizer_config.json",
)


def validate_adapter_dir(adapter_dir: Path) -> None:
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Final SFT package is incomplete. Missing: " + ", ".join(missing)
        )


def apply_embedding_patch(model: Any, patch_path: Path) -> None:
    import torch

    patch = torch.load(patch_path, map_location="cpu", weights_only=True)
    token_ids = [int(value) for value in patch.get("token_ids", [])]
    if not token_ids:
        # The patch file is still part of the release contract. An empty list
        # means tokenizer alignment required no extra embedding rows.
        return

    required_size = max(token_ids) + 1
    current_size = int(model.get_input_embeddings().weight.shape[0])
    if required_size > current_size:
        model.resize_token_embeddings(required_size, mean_resizing=True)

    input_rows = patch.get("input_rows")
    output_rows = patch.get("output_rows")
    if input_rows is None or output_rows is None:
        raise ValueError("embedding_patch.pt is missing input_rows/output_rows")
    if len(token_ids) != int(input_rows.shape[0]) or len(token_ids) != int(output_rows.shape[0]):
        raise ValueError("embedding patch row count does not match token_ids")

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    ids = torch.tensor(token_ids, device=input_weight.device)
    with torch.no_grad():
        input_weight.index_copy_(0, ids, input_rows.to(input_weight.device, input_weight.dtype))
        output_weight.index_copy_(0, ids, output_rows.to(output_weight.device, output_weight.dtype))


def load_finance_model(
    base_model: str,
    adapter_dir: Path,
    *,
    device_map: str = "auto",
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = adapter_dir.resolve()
    validate_adapter_dir(adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        use_fast=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    apply_embedding_patch(model, adapter_dir / "embedding_patch.pt")
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer
