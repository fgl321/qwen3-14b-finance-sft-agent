from fastapi import FastAPI

from app.personal_bootstrap import install_personal_features


def test_personal_routes_are_installed_once() -> None:
    app = FastAPI()
    install_personal_features(app)
    install_personal_features(app)
    paths = app.openapi()["paths"]
    assert "/health/personal-data" in paths
    assert "/api/personal/short-memory" in paths
    assert "/api/personal/long-memory/facts" in paths
    assert "/api/personal/rag/documents/text" in paths
    assert "/personal-console" in paths
