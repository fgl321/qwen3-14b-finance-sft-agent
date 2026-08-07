from __future__ import annotations

import re
from pathlib import Path


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "data/uploads",
}
PATTERNS = [
    ("DeepSeek/OpenAI key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    (
        "Authorization bearer",
        re.compile(r"Authorization\s*[:=]\s*Bearer\s+\S+", re.I),
    ),
    (
        "filled API key",
        re.compile(r"(?:DEEPSEEK|OPENAI|API)_API_KEY\s*=\s*[^\s#]+", re.I),
    ),
]
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".env",
    ".example",
    ".ps1",
}


def should_skip(path: Path) -> bool:
    normalized = path.as_posix()
    return any(
        marker in path.parts or marker in normalized for marker in EXCLUDED_DIRS
    )


def main() -> None:
    root = Path.cwd()
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue
        if path.name == ".env":
            # .env 不应提交，但本地存在是正常的；只由 .gitignore 约束。
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".env.example",
            ".gitignore",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(0)
                if "your_" in value.lower() or "example" in value.lower():
                    continue
                findings.append(f"{path}: {name}")
                break
    if findings:
        print("发现疑似敏感信息：")
        print("\n".join(sorted(set(findings))))
        raise SystemExit(1)
    print("GitHub sensitive-information check passed.")


if __name__ == "__main__":
    main()
