from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


ASSET_NAME = "qwen3-14b-finance-sft-adapter-v1.tar.gz"
FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "embedding_patch.pt",
    "sft_metadata.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(source: Path, output_dir: Path) -> tuple[Path, Path]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("Final adapter is incomplete: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="finance-sft-release-", dir=output_dir) as temp_name:
        staging = Path(temp_name) / "final_adapter"
        staging.mkdir()
        for name in FILES:
            shutil.copy2(source / name, staging / name)

        adapter_config_path = staging / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_config["base_model_name_or_path"] = "Qwen/Qwen3-14B"
        adapter_config_path.write_text(
            json.dumps(adapter_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "package": "qwen3-14b-finance-sft-adapter-v1",
            "base_model": "Qwen/Qwen3-14B",
            "format": "peft_lora_with_expanded_tokenizer_and_embedding_patch",
            "files": [],
        }
        checksum_lines: list[str] = []
        for path in sorted(staging.iterdir()):
            if not path.is_file():
                continue
            digest = sha256_file(path)
            manifest["files"].append(
                {"name": path.name, "size": path.stat().st_size, "sha256": digest}
            )
            checksum_lines.append(f"{digest}  {path.name}")

        (staging / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_lines.append(
            f"{sha256_file(staging / 'MANIFEST.json')}  MANIFEST.json"
        )
        (staging / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )

        partial = output_dir / f".{ASSET_NAME}.{os.getpid()}.partial"
        with tarfile.open(partial, "w:gz") as archive:
            archive.add(staging, arcname="final_adapter")
        final_archive = output_dir / ASSET_NAME
        os.replace(partial, final_archive)

    archive_checksum = output_dir / f"{ASSET_NAME}.sha256"
    archive_checksum.write_text(
        f"{sha256_file(final_archive)}  {ASSET_NAME}\n", encoding="utf-8"
    )
    return final_archive, archive_checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    archive, checksum = package(args.source, args.output_dir)
    print(json.dumps({
        "archive": str(archive),
        "size": archive.stat().st_size,
        "sha256": checksum.read_text(encoding="utf-8").split()[0],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
