from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from app.api.routes.chat_graph_v2 import _sse
from app.core.config import Settings
from app.core.request_boundary import (
    personal_request_identity,
    validate_public_identifier,
    validate_uploaded_document,
)


def test_personal_identity_is_owned_by_server_configuration() -> None:
    identity = personal_request_identity(
        Settings(personal_tenant_id="personal", personal_user_id="owner")
    )
    assert (identity.tenant_id, identity.user_id) == ("personal", "owner")


def test_boundary_rejects_unsafe_identifier_and_fake_pdf(tmp_path) -> None:
    with pytest.raises(HTTPException) as unsafe:
        validate_public_identifier("../../other-user", field_name="knowledge_base_id")
    assert unsafe.value.status_code == 400

    fake_pdf = tmp_path / "report.pdf"
    fake_pdf.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(HTTPException) as mismatch:
        validate_uploaded_document(
            fake_pdf,
            extension=".pdf",
            content_type="application/pdf",
        )
    assert mismatch.value.status_code == 415
    assert mismatch.value.detail["code"] == "DOCUMENT_SIGNATURE_MISMATCH"


def test_sse_frame_contains_named_json_event() -> None:
    frame = _sse({"event": "planner_progress", "status": "running"})
    lines = frame.strip().splitlines()
    assert lines[0] == "event: planner_progress"
    assert json.loads(lines[1].removeprefix("data: ")) == {
        "event": "planner_progress",
        "status": "running",
    }


def test_ensure_request_id_never_empty_and_unique() -> None:
    from app.api.routes.chat_graph_v2 import ensure_request_id

    first = ensure_request_id(None)
    second = ensure_request_id("")
    assert first.startswith("api-prod-")
    assert second.startswith("api-prod-")
    assert first != second
    assert ensure_request_id("abc-123") == "abc-123"
