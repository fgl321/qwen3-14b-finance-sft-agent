from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.test_stage_4_4_personal_http import main as http_acceptance


async def main() -> None:
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "stage_4_4_lite",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    try:
        await http_acceptance()
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        path = report_dir / "personal_quality_report.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"quality_report={path}")


if __name__ == "__main__":
    asyncio.run(main())
