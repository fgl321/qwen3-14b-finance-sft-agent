from __future__ import annotations

import hashlib
from pathlib import Path


SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
}


def normalize_file_path(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    if not path.is_file():
        raise ValueError(f"路径不是文件：{path}")

    return path


def get_file_extension(file_path: str | Path) -> str:
    path = Path(file_path)
    return path.suffix.lower()


def validate_supported_document(file_path: str | Path) -> Path:
    path = normalize_file_path(file_path)
    extension = get_file_extension(path)

    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            "不支持的文件类型："
            f"{extension}。当前支持：{sorted(SUPPORTED_DOCUMENT_EXTENSIONS)}"
        )

    return path


def calculate_sha256(file_path: str | Path) -> str:
    path = normalize_file_path(file_path)

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def estimate_token_count(text: str) -> int:
    """
    简化版 token 估算。

    中文场景下不能简单按英文空格切词。
    这里先用保守估算：
    - 中文字符大约 1~2 个字符对应 1 个 token
    - 英文单词和数字按空格近似

    后面接入真正 tokenizer 后，可以替换这个函数。
    """
    if not text:
        return 0

    chinese_char_count = 0
    ascii_token_like_count = 0

    current_ascii_segment = []

    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            chinese_char_count += 1
            if current_ascii_segment:
                ascii_token_like_count += 1
                current_ascii_segment = []
        elif char.isascii() and (char.isalnum() or char in {"_", "-", "."}):
            current_ascii_segment.append(char)
        else:
            if current_ascii_segment:
                ascii_token_like_count += 1
                current_ascii_segment = []

    if current_ascii_segment:
        ascii_token_like_count += 1

    return max(1, int(chinese_char_count / 1.5) + ascii_token_like_count)


def clean_text(text: str) -> str:
    """
    做轻量清洗，不做过度改写。

    原则：
    - 保留原始语义
    - 去掉明显多余空白
    - 不删除金额、数字、标点
    """
    lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines).strip()
