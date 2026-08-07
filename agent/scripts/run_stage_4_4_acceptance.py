from __future__ import annotations

import argparse
import subprocess
import sys


NEW_TESTS = [
    "tests/test_stage_4_4_privacy.py",
    "tests/test_stage_4_4_short_memory.py",
    "tests/test_stage_4_4_long_memory_policy.py",
    "tests/test_stage_4_4_rag_lifecycle.py",
    "tests/test_stage_4_4_bootstrap.py",
    "tests/test_stage_4_4_chat_memory_integration.py",
]


def run(command: list[str]) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-http", action="store_true")
    parser.add_argument("--skip-stage-4-2-8", action="store_true")
    args = parser.parse_args()

    run([sys.executable, "-m", "compileall", "-q", "app", "scripts", "tests"])
    run([sys.executable, "-m", "scripts.check_personal_project"])
    run([sys.executable, "-m", "scripts.check_git_secrets"])
    if not args.skip_stage_4_2_8:
        run([sys.executable, "-m", "scripts.run_stage_4_2_8_acceptance"])
    run([sys.executable, "-m", "pytest", *NEW_TESTS, "-q"])
    if args.with_http:
        run([sys.executable, "-m", "scripts.test_stage_4_4_personal_http"])
    print("\nStage 4.4 Lite final acceptance passed.")


if __name__ == "__main__":
    main()
