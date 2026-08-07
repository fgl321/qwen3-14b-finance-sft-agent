from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".csv",
    ".pdf",
    ".docx",
}


def normalize_document_text(text: str) -> str:
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    result: list[str] = []
    blank = False
    for line in lines:
        clean = line.strip()
        if not clean:
            if result and not blank:
                result.append("")
            blank = True
            continue
        blank = False
        result.append(clean)
    return "\n".join(result).strip()


def _parse_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return json.dumps(data, ensure_ascii=False, indent=2)


def _parse_jsonl(path: Path) -> str:
    records: list[Any] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return "\n".join(
        json.dumps(item, ensure_ascii=False) for item in records
    )


def _parse_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                "；".join(f"{key}: {value}" for key, value in row.items())
            )
    return "\n".join(rows)


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖环境
        raise RuntimeError(
            "解析 PDF 需要 pypdf：python -m pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _parse_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - 依赖环境
        raise RuntimeError(
            "解析 DOCX 需要 python-docx：python -m pip install python-docx"
        ) from exc
    document = Document(str(path))
    blocks = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            blocks.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(blocks)


def parse_document(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"暂不支持文档类型 {suffix or '[无扩展名]'}。")

    if suffix in {".txt", ".md", ".markdown"}:
        text = file_path.read_text(encoding="utf-8-sig")
    elif suffix == ".json":
        text = _parse_json(file_path)
    elif suffix == ".jsonl":
        text = _parse_jsonl(file_path)
    elif suffix == ".csv":
        text = _parse_csv(file_path)
    elif suffix == ".pdf":
        text = _parse_pdf(file_path)
    elif suffix == ".docx":
        text = _parse_docx(file_path)
    else:  # pragma: no cover
        raise ValueError(f"不支持的文档类型：{suffix}")

    normalized = normalize_document_text(text)
    if not normalized:
        raise ValueError("文档没有可提取的文本内容。")
    return normalized

# ---------------------------------------------------------------------------
# Legacy compatibility layer
# ---------------------------------------------------------------------------
# Stage 4.4 Lite uses the function-style parse_document() API above.  The
# original project also contains RagIngestionService, which imports and
# instantiates DocumentParser.  Keep both APIs so the new lifecycle service and
# the old knowledge upload route can coexist.

from dataclasses import asdict, dataclass, field
import tempfile
from collections.abc import Iterator


@dataclass(slots=True)
class ParsedPage:
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return self.to_dict()


class ParsedDocument(str):
    """String-compatible parsed result with legacy document attributes."""

    file_name: str
    source_type: str
    pages: list[ParsedPage]
    metadata: dict[str, Any]

    def __new__(
        cls,
        text: str,
        *,
        file_name: str,
        source_type: str,
        pages: list[ParsedPage] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ParsedDocument":
        obj = str.__new__(cls, text)
        obj.file_name = file_name
        obj.source_type = source_type
        obj.pages = pages or [ParsedPage(page_number=1, text=text)]
        obj.metadata = dict(metadata or {})
        return obj

    @property
    def text(self) -> str:
        return str(self)

    @property
    def content(self) -> str:
        return str(self)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def iter_pages(self) -> Iterator[ParsedPage]:
        return iter(self.pages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "source_type": self.source_type,
            "text": str(self),
            "content": str(self),
            "page_count": self.page_count,
            "pages": [page.to_dict() for page in self.pages],
            "metadata": dict(self.metadata),
        }

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return self.to_dict()


def _pages_from_path(path: Path, normalized_text: str) -> list[ParsedPage]:
    suffix = path.suffix.lower()
    if suffix != ".pdf":
        return [ParsedPage(page_number=1, text=normalized_text)]

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[ParsedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = normalize_document_text(page.extract_text() or "")
            if text:
                pages.append(ParsedPage(page_number=index, text=text))
        return pages or [ParsedPage(page_number=1, text=normalized_text)]
    except Exception:
        return [ParsedPage(page_number=1, text=normalized_text)]


class DocumentParser:
    """
    Backward-compatible parser facade.

    Supported calls include:
      parser.parse(Path("file.pdf"))
      parser.parse(file_path="file.pdf")
      parser.parse(file_name="file.txt", content=b"...")
      parser.parse_bytes(b"...", file_name="file.txt")

    The return value behaves like a normal string and also exposes legacy
    attributes such as .text, .content, .pages and .page_count.
    """

    supported_extensions = SUPPORTED_EXTENSIONS

    def __init__(self, settings: Any | None = None, **_: Any) -> None:
        self.settings = settings

    @staticmethod
    def _resolve_input(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Path | None, bytes | None, str | None]:
        path_value = (
            kwargs.pop("file_path", None)
            or kwargs.pop("path", None)
            or kwargs.pop("source_path", None)
        )
        file_name = (
            kwargs.pop("file_name", None)
            or kwargs.pop("filename", None)
            or kwargs.pop("name", None)
        )
        raw = (
            kwargs.pop("content", None)
            or kwargs.pop("file_bytes", None)
            or kwargs.pop("raw_bytes", None)
            or kwargs.pop("data", None)
        )

        if len(args) == 1:
            value = args[0]
            if isinstance(value, (bytes, bytearray, memoryview)):
                raw = bytes(value)
            else:
                path_value = value
        elif len(args) == 2:
            first, second = args
            if isinstance(first, (bytes, bytearray, memoryview)):
                raw = bytes(first)
                file_name = str(second)
            elif isinstance(second, (bytes, bytearray, memoryview)):
                file_name = str(first)
                raw = bytes(second)
            else:
                path_value = first
                file_name = str(second)
        elif len(args) > 2:
            raise TypeError("DocumentParser.parse 最多接受两个位置参数。")

        path = Path(path_value) if path_value is not None else None
        if raw is not None and not isinstance(raw, bytes):
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            else:
                raw = bytes(raw)
        return path, raw, str(file_name) if file_name else None

    def parse(self, *args: Any, **kwargs: Any) -> ParsedDocument:
        path, raw, file_name = self._resolve_input(args, dict(kwargs))
        cleanup_path: Path | None = None

        if raw is not None:
            final_name = file_name or "document.txt"
            suffix = Path(final_name).suffix.lower() or ".txt"
            if suffix not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"暂不支持文档类型 {suffix}。")
            handle = tempfile.NamedTemporaryFile(
                prefix="finance_agent_parse_",
                suffix=suffix,
                delete=False,
            )
            try:
                handle.write(raw)
                handle.flush()
            finally:
                handle.close()
            path = Path(handle.name)
            cleanup_path = path
        elif path is None:
            raise TypeError("必须提供文档路径，或 file_name + content。")

        assert path is not None
        try:
            text = parse_document(path)
            display_name = file_name or path.name
            source_type = path.suffix.lower().lstrip(".") or "text"
            pages = _pages_from_path(path, text)
            return ParsedDocument(
                text,
                file_name=display_name,
                source_type=source_type,
                pages=pages,
                metadata={"source_path": str(path)},
            )
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        return self.parse(file_path)

    def parse_path(self, file_path: str | Path) -> ParsedDocument:
        return self.parse(file_path)

    def parse_bytes(
        self,
        content: bytes,
        file_name: str = "document.txt",
    ) -> ParsedDocument:
        return self.parse(file_name=file_name, content=content)

    def parse_upload(
        self,
        file_name: str,
        content: bytes,
    ) -> ParsedDocument:
        return self.parse(file_name=file_name, content=content)

    def extract_text(self, *args: Any, **kwargs: Any) -> str:
        return str(self.parse(*args, **kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> ParsedDocument:
        return self.parse(*args, **kwargs)
