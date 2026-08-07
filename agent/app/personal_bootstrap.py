from __future__ import annotations

from typing import Any

from app.api.routes.personal_management import router
from app.personal_data.models import PERSONAL_DATA_VERSION


def install_personal_features(app: Any) -> Any:
    """在不替换现有 app.main 的情况下安装 Stage 4.4 Lite 路由。"""
    if getattr(app.state, "personal_features_installed", False):
        return app
    app.include_router(router)
    app.state.personal_features_installed = True
    app.state.personal_data_version = PERSONAL_DATA_VERSION
    return app
