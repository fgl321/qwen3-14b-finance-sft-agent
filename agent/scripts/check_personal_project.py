from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = [
    "app/memory/short_term_memory.py",
    "app/memory/long_term_memory.py",
    "app/memory/llm_fact_extractor.py",
    "app/rag/document_parser.py",
    "app/rag/document_lifecycle.py",
    "app/api/routes/personal_management.py",
    "app/api/routes/chat_graph_v2.py",
    "app/personal_bootstrap.py",
    "scripts/run_production_api.py",
    "scripts/init_personal_data.py",
    "scripts/test_stage_4_4_personal_http.py",
    "scripts/run_stage_4_4_acceptance.py",
    "scripts/check_git_secrets.py",
    "data/eval/personal_quality_cases.jsonl",
    ".env.example",
    ".gitignore",
    "README.md",
    "README_STAGE_4_4_LITE.md",
    "requirements-stage-4-4.txt",
]


def main() -> None:
    root = Path.cwd()
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]

    from app.memory.long_term_memory import DEFAULT_FACT_WHITELIST
    from app.personal_data.models import PERSONAL_DATA_VERSION
    from app.rag.document_parser import SUPPORTED_EXTENSIONS

    report = {
        "passed": not missing,
        "version": PERSONAL_DATA_VERSION,
        "missing_files": missing,
        "long_memory_fact_types": sorted(DEFAULT_FACT_WHITELIST),
        "rag_supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)
    print("Stage 4.4 Lite static project check passed.")


if __name__ == "__main__":
    main()
