from __future__ import annotations

import uuid
import re
from dataclasses import dataclass

from app.rag.file_utils import clean_text, estimate_token_count
from app.rag.rag_types import ParsedDocument, RagChunk


@dataclass(frozen=True)
class ChunkingConfig:
    parent_token_limit: int = 900
    parent_overlap_tokens: int = 120
    child_token_limit: int = 280
    child_overlap_tokens: int = 60
    min_chunk_tokens: int = 20


class ParentChildChunker:
    """
    生产级 RAG 分块器：父子分块。

    父块 parent：
    - 保存更完整上下文
    - 后续用于给大模型阅读

    子块 child：
    - 更短、更精确
    - 后续主要用于检索命中

    命中流程：
    - 检索 child chunk
    - 根据 parent_id 回填 parent chunk
    - 用 parent chunk 作为回答证据
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()

        if self.config.parent_token_limit <= self.config.child_token_limit:
            raise ValueError("parent_token_limit 必须大于 child_token_limit")

        if self.config.parent_overlap_tokens >= self.config.parent_token_limit:
            raise ValueError("parent_overlap_tokens 必须小于 parent_token_limit")

        if self.config.child_overlap_tokens >= self.config.child_token_limit:
            raise ValueError("child_overlap_tokens 必须小于 child_token_limit")

    def chunk(
        self,
        parsed_document: ParsedDocument,
    ) -> list[RagChunk]:
        parent_chunks = self._build_parent_chunks(parsed_document)

        all_chunks: list[RagChunk] = []

        for parent_chunk in parent_chunks:
            all_chunks.append(parent_chunk)

            child_chunks = self._build_child_chunks(
                parent_chunk=parent_chunk,
            )

            all_chunks.extend(child_chunks)

        if not all_chunks:
            fallback_chunks = self._build_fallback_chunks(
                parsed_document=parsed_document,
            )
            all_chunks.extend(fallback_chunks)

        return all_chunks

    def _build_fallback_chunks(
        self,
        parsed_document: ParsedDocument,
    ) -> list[RagChunk]:
        """
        短文档（如 OCR 图片识别出的少量文字）保底：整篇作为一个父块入库，
        避免“有文字但 0 分块”导致无法检索。
        """
        pages = [
            page
            for page in (parsed_document.pages or [])
            if str(page.text or "").strip()
        ]
        if not pages:
            return []
        text = clean_text(
            "\n".join(str(page.text) for page in pages)
        )
        if not text:
            return []

        parent_index = 1
        parent_id = self._stable_chunk_id(
            document_id=parsed_document.meta.document_id,
            chunk_type="parent",
            index=parent_index,
            text=text,
        )
        parent_chunk = RagChunk(
            chunk_id=parent_id,
            parent_id=parent_id,
            document_id=parsed_document.meta.document_id,
            tenant_id=parsed_document.meta.tenant_id,
            owner_user_id=parsed_document.meta.owner_user_id,
            knowledge_base_id=parsed_document.meta.knowledge_base_id,
            visibility=parsed_document.meta.visibility,
            file_name=parsed_document.meta.file_name,
            page_start=pages[0].page_number,
            page_end=pages[-1].page_number,
            section_path=[],
            text=text,
            token_count_estimate=estimate_token_count(text),
            metadata={
                "chunk_type": "parent",
                "parent_index": parent_index,
                "file_sha256": parsed_document.meta.file_sha256,
                "source_type": parsed_document.meta.source_type,
                "document_version": parsed_document.meta.version,
                "fallback_short_document": True,
            },
        )
        children = self._build_child_chunks(parent_chunk)
        return [parent_chunk, *children]

    def _build_parent_chunks(
        self,
        parsed_document: ParsedDocument,
    ) -> list[RagChunk]:
        page_segments: list[tuple[int, str, list[str]]] = []

        for page in parsed_document.pages:
            current_path: list[str] = []
            for paragraph in self._split_page_into_paragraphs(page.text):
                heading = self._heading_of(paragraph)
                if heading is not None:
                    current_path = self._update_heading_path(
                        current_path,
                        heading,
                    )
                page_segments.append(
                    (page.page_number, paragraph, list(current_path))
                )

        windows = self._merge_segments_to_windows(
            segments=page_segments,
            token_limit=self.config.parent_token_limit,
            overlap_tokens=self.config.parent_overlap_tokens,
        )

        parent_chunks: list[RagChunk] = []

        for parent_index, window in enumerate(windows, start=1):
            text = clean_text("\n".join(item[1] for item in window))
            token_count = estimate_token_count(text)

            if token_count < self.config.min_chunk_tokens:
                continue

            page_numbers = [item[0] for item in window]
            page_start = min(page_numbers) if page_numbers else None
            page_end = max(page_numbers) if page_numbers else None
            section_path = list(window[0][2]) if window else []

            parent_id = self._stable_chunk_id(
                document_id=parsed_document.meta.document_id,
                chunk_type="parent",
                index=parent_index,
                text=text,
            )

            parent_chunks.append(
                RagChunk(
                    chunk_id=parent_id,
                    parent_id=parent_id,
                    document_id=parsed_document.meta.document_id,
                    tenant_id=parsed_document.meta.tenant_id,
                    owner_user_id=parsed_document.meta.owner_user_id,
                    knowledge_base_id=parsed_document.meta.knowledge_base_id,
                    visibility=parsed_document.meta.visibility,
                    file_name=parsed_document.meta.file_name,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                    text=text,
                    token_count_estimate=token_count,
                    metadata={
                        "chunk_type": "parent",
                        "parent_index": parent_index,
                        "file_sha256": parsed_document.meta.file_sha256,
                        "source_type": parsed_document.meta.source_type,
                        "document_version": parsed_document.meta.version,
                        "content_type": (
                            parsed_document.meta.content_type
                        ),
                        "scope": parsed_document.meta.scope,
                        "trust_level": (
                            parsed_document.meta.trust_level
                        ),
                        "generated_content": (
                            parsed_document.meta.generated_content
                        ),
                        "allow_rag_direct": (
                            parsed_document.meta.allow_rag_direct
                        ),
                    },
                )
            )

        return parent_chunks

    def _build_child_chunks(
        self,
        parent_chunk: RagChunk,
    ) -> list[RagChunk]:
        parent_index = int(parent_chunk.metadata.get("parent_index", 1))

        segments = [
            (
                parent_chunk.page_start or 1,
                paragraph,
                list(parent_chunk.section_path),
            )
            for paragraph in self._split_page_into_paragraphs(parent_chunk.text)
        ]

        windows = self._merge_segments_to_windows(
            segments=segments,
            token_limit=self.config.child_token_limit,
            overlap_tokens=self.config.child_overlap_tokens,
        )

        child_chunks: list[RagChunk] = []

        for child_index, window in enumerate(windows, start=1):
            text = clean_text("\n".join(item[1] for item in window))
            token_count = estimate_token_count(text)

            if token_count < self.config.min_chunk_tokens:
                continue

            child_id = self._stable_chunk_id(
                document_id=parent_chunk.document_id,
                chunk_type="child",
                index=child_index,
                text=f"{parent_chunk.chunk_id}:{text}",
            )

            child_chunks.append(
                RagChunk(
                    chunk_id=child_id,
                    parent_id=parent_chunk.parent_id,
                    document_id=parent_chunk.document_id,
                    tenant_id=parent_chunk.tenant_id,
                    owner_user_id=parent_chunk.owner_user_id,
                    knowledge_base_id=parent_chunk.knowledge_base_id,
                    visibility=parent_chunk.visibility,
                    file_name=parent_chunk.file_name,
                    page_start=parent_chunk.page_start,
                    page_end=parent_chunk.page_end,
                    section_path=parent_chunk.section_path,
                    text=text,
                    token_count_estimate=token_count,
                    metadata={
                        "chunk_type": "child",
                        "parent_index": parent_index,
                        "child_index": child_index,
                        "parent_chunk_id": parent_chunk.chunk_id,
                        "file_sha256": parent_chunk.metadata.get("file_sha256"),
                        "source_type": parent_chunk.metadata.get("source_type"),
                        "document_version": parent_chunk.metadata.get(
                            "document_version"
                        ),
                        "content_type": parent_chunk.metadata.get(
                            "content_type"
                        ),
                        "scope": parent_chunk.metadata.get("scope"),
                        "trust_level": parent_chunk.metadata.get(
                            "trust_level"
                        ),
                        "generated_content": (
                            parent_chunk.metadata.get(
                                "generated_content"
                            )
                        ),
                        "allow_rag_direct": (
                            parent_chunk.metadata.get(
                                "allow_rag_direct"
                            )
                        ),
                    },
                )
            )

        if not child_chunks:
            # 短文档兜底：保证每个父块至少有一个子块，
            # 否则纯父块文档无法被全局 child 检索命中。
            text = clean_text(parent_chunk.text)
            if text:
                child_id = self._stable_chunk_id(
                    document_id=parent_chunk.document_id,
                    chunk_type="child",
                    index=1,
                    text=f"{parent_chunk.chunk_id}:{text}",
                )
                child_chunks.append(
                    RagChunk(
                        chunk_id=child_id,
                        parent_id=parent_chunk.parent_id,
                        document_id=parent_chunk.document_id,
                        tenant_id=parent_chunk.tenant_id,
                        owner_user_id=parent_chunk.owner_user_id,
                        knowledge_base_id=(
                            parent_chunk.knowledge_base_id
                        ),
                        visibility=parent_chunk.visibility,
                        file_name=parent_chunk.file_name,
                        page_start=parent_chunk.page_start,
                        page_end=parent_chunk.page_end,
                        section_path=parent_chunk.section_path,
                        text=text,
                        token_count_estimate=(
                            estimate_token_count(text)
                        ),
                        metadata={
                            "chunk_type": "child",
                            "parent_index": parent_index,
                            "child_index": 1,
                            "parent_chunk_id": (
                                parent_chunk.chunk_id
                            ),
                            "file_sha256": (
                                parent_chunk.metadata.get(
                                    "file_sha256"
                                )
                            ),
                            "source_type": (
                                parent_chunk.metadata.get(
                                    "source_type"
                                )
                            ),
                            "document_version": (
                                parent_chunk.metadata.get(
                                    "document_version"
                                )
                            ),
                            "fallback_child": True,
                            "content_type": (
                                parent_chunk.metadata.get(
                                    "content_type"
                                )
                            ),
                            "scope": parent_chunk.metadata.get("scope"),
                            "trust_level": parent_chunk.metadata.get(
                                "trust_level"
                            ),
                            "generated_content": (
                                parent_chunk.metadata.get(
                                    "generated_content"
                                )
                            ),
                            "allow_rag_direct": (
                                parent_chunk.metadata.get(
                                    "allow_rag_direct"
                                )
                            ),
                        },
                    )
                )

        return child_chunks

    def _merge_segments_to_windows(
        self,
        *,
        segments: list[tuple[int, str, list[str]]],
        token_limit: int,
        overlap_tokens: int,
    ) -> list[list[tuple[int, str, list[str]]]]:
        windows: list[list[tuple[int, str, list[str]]]] = []
        current_window: list[tuple[int, str, list[str]]] = []
        current_tokens = 0

        for page_number, segment, section_path in segments:
            segment = clean_text(segment)
            if not segment:
                continue

            segment_tokens = estimate_token_count(segment)

            if segment_tokens > token_limit:
                hard_split_segments = self._split_long_segment(
                    page_number=page_number,
                    text=segment,
                    token_limit=token_limit,
                )
                fallback_path = (
                    current_window[0][2]
                    if current_window
                    else section_path
                )

                for hard_page_number, hard_segment in hard_split_segments:
                    hard_segment_tokens = estimate_token_count(hard_segment)

                    if current_window and current_tokens + hard_segment_tokens > token_limit:
                        windows.append(current_window)

                        current_window = self._build_overlap_window(
                            current_window=current_window,
                            overlap_tokens=overlap_tokens,
                        )
                        current_tokens = estimate_token_count(
                            "\n".join(item[1] for item in current_window)
                        )

                    current_window.append(
                        (
                            hard_page_number,
                            hard_segment,
                            list(fallback_path),
                        )
                    )
                    current_tokens += hard_segment_tokens

                continue

            if current_window and current_tokens + segment_tokens > token_limit:
                windows.append(current_window)

                current_window = self._build_overlap_window(
                    current_window=current_window,
                    overlap_tokens=overlap_tokens,
                )
                current_tokens = estimate_token_count(
                    "\n".join(item[1] for item in current_window)
                )

            current_window.append(
                (page_number, segment, list(section_path))
            )
            current_tokens += segment_tokens

        if current_window:
            windows.append(current_window)

        return windows

    def _build_overlap_window(
        self,
        *,
        current_window: list[tuple[int, str, list[str]]],
        overlap_tokens: int,
    ) -> list[tuple[int, str, list[str]]]:
        if overlap_tokens <= 0:
            return []

        overlap_window: list[tuple[int, str, list[str]]] = []
        total_tokens = 0

        for page_number, segment, section_path in reversed(current_window):
            segment_tokens = estimate_token_count(segment)

            if total_tokens + segment_tokens > overlap_tokens:
                break

            overlap_window.insert(
                0,
                (page_number, segment, list(section_path)),
            )
            total_tokens += segment_tokens

        return overlap_window

    @staticmethod
    def _split_page_into_paragraphs(text: str) -> list[str]:
        cleaned = clean_text(text)

        if not cleaned:
            return []

        return [
            item.strip()
            for item in cleaned.split("\n")
            if item.strip()
        ]

    @staticmethod
    def _split_long_segment(
        *,
        page_number: int,
        text: str,
        token_limit: int,
    ) -> list[tuple[int, str]]:
        if not text:
            return []

        approx_chars_per_token = 1.5
        char_limit = max(100, int(token_limit * approx_chars_per_token))

        result: list[tuple[int, str]] = []

        start = 0
        while start < len(text):
            end = start + char_limit
            piece = clean_text(text[start:end])

            if piece:
                result.append((page_number, piece))

            start = end

        return result

    @staticmethod
    def _heading_of(text: str) -> tuple[int, str] | None:
        """
        识别 Markdown / 中文编号标题。

        返回 (level, title)；不是标题返回 None。
        level 从 1 开始：Markdown # 为 1，中文“一、”为 1，
        数字“1.”为 2，依此类推。
        """
        cleaned = text.strip()
        if not cleaned:
            return None

        md = re.match(r"^(#{1,6})\s+(.+)$", cleaned)
        if md:
            return len(md.group(1)), md.group(2).strip()

        chinese = re.match(
            r"^([一二三四五六七八九十百]+)[、.．]\s*(.+)$",
            cleaned,
        )
        if chinese:
            return 1, chinese.group(2).strip()

        numbered = re.match(
            r"^(\d{1,3})[、.．]\s*(.+)$",
            cleaned,
        )
        if numbered:
            return 2, numbered.group(2).strip()

        return None

    @staticmethod
    def _update_heading_path(
        current_path: list[str],
        heading: tuple[int, str],
    ) -> list[str]:
        """
        维护当前标题路径。

        遇到同级或更高级标题时，截断当前路径后追加新标题。
        """
        level, title = heading
        level = max(1, level)
        path = list(current_path[: max(level - 1, 0)])
        path.append(title)
        return path

    @staticmethod
    def _stable_chunk_id(
        *,
        document_id: str,
        chunk_type: str,
        index: int,
        text: str,
    ) -> str:
        raw = f"{document_id}:{chunk_type}:{index}:{text}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))
