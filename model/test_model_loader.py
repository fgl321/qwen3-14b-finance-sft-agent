from pathlib import Path

import pytest

from model.model_loader import validate_adapter_dir


def test_incomplete_adapter_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Final SFT package is incomplete"):
        validate_adapter_dir(tmp_path)


def test_required_adapter_files_are_accepted(tmp_path: Path) -> None:
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "embedding_patch.pt",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (tmp_path / name).touch()
    validate_adapter_dir(tmp_path)
