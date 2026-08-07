from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app
from app.core.config import get_settings
from app.personal_bootstrap import install_personal_features


install_personal_features(app)


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        loop="app.core.uvicorn_loop:selector_loop_factory",
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
