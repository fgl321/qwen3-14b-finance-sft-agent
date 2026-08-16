from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from fastapi import HTTPException

from app.core.config import Settings


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


@dataclass(frozen=True, slots=True)
class RequestIdentity:
    """Server-owned identity for the personal, single-user deployment."""

    tenant_id: str
    user_id: str


def personal_request_identity(settings: Settings) -> RequestIdentity:
    """Return the only identity accepted by this personal deployment.

    The browser is intentionally not an authentication authority.  Keeping
    identity server-owned prevents a crafted request from crossing memory or
    document namespaces while avoiding a login screen for the demo.
    """

    tenant_id = settings.personal_tenant_id.strip()
    user_id = settings.personal_user_id.strip()
    if not _SAFE_IDENTIFIER.fullmatch(tenant_id):
        raise RuntimeError("PERSONAL_TENANT_ID is not a safe identifier.")
    if not _SAFE_IDENTIFIER.fullmatch(user_id):
        raise RuntimeError("PERSONAL_USER_ID is not a safe identifier.")
    return RequestIdentity(tenant_id=tenant_id, user_id=user_id)


def validate_public_identifier(value: str, *, field_name: str) -> str:
    clean = str(value).strip()
    if not _SAFE_IDENTIFIER.fullmatch(clean):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_IDENTIFIER",
                "message": f"{field_name} contains unsupported characters.",
            },
        )
    return clean


def validate_uploaded_document(
    path: Path,
    *,
    extension: str,
    content_type: str | None,
) -> None:
    """Validate a document by extension, MIME family and file signature.

    This is deliberately deterministic.  It is a boundary check, not an LLM
    judgement, and protects parsers from files merely renamed to an allowed
    extension.
    """

    declared_type = (content_type or "").split(";", 1)[0].strip().lower()
    allowed_mime = {
        ".pdf": {"application/pdf", "application/octet-stream", ""},
        ".docx": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
            "application/octet-stream",
            "",
        },
        ".png": {"image/png", "application/octet-stream", ""},
        ".jpg": {"image/jpeg", "application/octet-stream", ""},
        ".jpeg": {"image/jpeg", "application/octet-stream", ""},
        ".txt": {"text/plain", "application/octet-stream", ""},
        ".md": {"text/markdown", "text/plain", "application/octet-stream", ""},
        ".markdown": {
            "text/markdown",
            "text/plain",
            "application/octet-stream",
            "",
        },
        ".csv": {
            "text/csv",
            "application/vnd.ms-excel",
            "text/plain",
            "application/octet-stream",
            "",
        },
        ".json": {"application/json", "text/plain", "application/octet-stream", ""},
        ".jsonl": {
            "application/jsonl",
            "application/x-ndjson",
            "application/json",
            "text/plain",
            "application/octet-stream",
            "",
        },
    }
    if declared_type not in allowed_mime.get(extension, set()):
        raise HTTPException(
            status_code=415,
            detail={
                "code": "DOCUMENT_MIME_MISMATCH",
                "message": "The declared document type does not match its extension.",
            },
        )

    with path.open("rb") as stream:
        header = stream.read(8)

    signature_ok = True
    if extension == ".pdf":
        signature_ok = header.startswith(b"%PDF-")
    elif extension == ".png":
        signature_ok = header == b"\x89PNG\r\n\x1a\n"
    elif extension in {".jpg", ".jpeg"}:
        signature_ok = header.startswith(b"\xff\xd8\xff")
    elif extension == ".docx":
        try:
            with ZipFile(path) as archive:
                names = set(archive.namelist())
                signature_ok = "[Content_Types].xml" in names and any(
                    name.startswith("word/") for name in names
                )
        except (BadZipFile, OSError):
            signature_ok = False
    else:
        try:
            path.read_bytes()[:65_536].decode("utf-8-sig")
        except UnicodeDecodeError:
            signature_ok = False

    if not signature_ok:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "DOCUMENT_SIGNATURE_MISMATCH",
                "message": "The uploaded file content does not match its extension.",
            },
        )
