from __future__ import annotations

import hashlib
import math
import threading
from abc import ABC, abstractmethod
from typing import Any, Iterable

from pydantic import BaseModel, Field


class SparseEmbedding(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class TextEmbedding(BaseModel):
    dense: list[float]
    sparse: SparseEmbedding


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[TextEmbedding]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> TextEmbedding:
        raise NotImplementedError


class BgeM3EmbeddingProvider(EmbeddingProvider):
    """
    BGE-M3 真实向量模型。

    能力：
    - 输出 dense 语义向量
    - 输出 sparse 稀疏词权重
    - 当前项目先用于 dense 检索
    - sparse 会先写入 Qdrant，为后续混合召回做准备

    注意：
    - 第一次运行会加载模型。
    - 如果 BGE_M3_MODEL_NAME 是 BAAI/bge-m3，会尝试从线上下载。
    - 如果你已经手动下载模型，可以把 BGE_M3_MODEL_NAME 改成本地目录。
    """

    def __init__(
        self,
        *,
        model_name_or_path: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 4,
        max_length: int = 8192,
        device: str = "",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device.strip()
        # BGE-M3 推理线程安全：入库线程与检索线程可能并发调用，
        # 用锁串行化推理，避免 torch 并发推理竞态。
        self._inference_lock = threading.Lock()

        if self.device.lower() == "cpu":
            use_fp16 = False

        self.use_fp16 = use_fp16

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "未安装 FlagEmbedding。请先运行：pip install -U FlagEmbedding"
            ) from exc

        try:
            if self.device:
                self.model = BGEM3FlagModel(
                    self.model_name_or_path,
                    use_fp16=self.use_fp16,
                    device=self.device,
                )
            else:
                self.model = BGEM3FlagModel(
                    self.model_name_or_path,
                    use_fp16=self.use_fp16,
                )
        except TypeError:
            self.model = BGEM3FlagModel(
                self.model_name_or_path,
                use_fp16=self.use_fp16,
            )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[TextEmbedding]:
        return self._embed_texts(texts)

    def embed_query(
        self,
        text: str,
    ) -> TextEmbedding:
        return self._embed_texts([text])[0]

    def _embed_texts(
        self,
        texts: list[str],
    ) -> list[TextEmbedding]:
        normalized_texts = [
            text if text and text.strip() else "<empty>"
            for text in texts
        ]

        with self._inference_lock:
            output = self.model.encode(
                normalized_texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

        dense_vecs = output.get("dense_vecs")
        lexical_weights = output.get("lexical_weights") or []

        dense_list = self._to_python_list(dense_vecs)

        if len(dense_list) != len(normalized_texts):
            raise ValueError(
                "BGE-M3 dense 向量数量和文本数量不一致："
                f"dense={len(dense_list)}, texts={len(normalized_texts)}"
            )

        result: list[TextEmbedding] = []

        for index, dense_vector in enumerate(dense_list):
            sparse_raw = (
                lexical_weights[index]
                if index < len(lexical_weights)
                else {}
            )

            result.append(
                TextEmbedding(
                    dense=[
                        float(item)
                        for item in dense_vector
                    ],
                    sparse=self._convert_lexical_weights_to_sparse(
                        sparse_raw,
                    ),
                )
            )

        return result

    @staticmethod
    def _to_python_list(
        value: Any,
    ) -> list:
        if hasattr(value, "tolist"):
            return value.tolist()

        return list(value)

    @staticmethod
    def _convert_lexical_weights_to_sparse(
        lexical_weights: Any,
    ) -> SparseEmbedding:
        if not lexical_weights:
            return SparseEmbedding(
                indices=[],
                values=[],
            )

        if not isinstance(lexical_weights, dict):
            return SparseEmbedding(
                indices=[],
                values=[],
            )

        merged: dict[int, float] = {}

        for raw_index, raw_value in lexical_weights.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                index = BgeM3EmbeddingProvider._stable_sparse_index(
                    str(raw_index)
                )

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue

            if value == 0:
                continue

            merged[index] = merged.get(index, 0.0) + value

        if not merged:
            return SparseEmbedding(
                indices=[],
                values=[],
            )

        sorted_items = sorted(merged.items())

        return SparseEmbedding(
            indices=[
                item[0]
                for item in sorted_items
            ],
            values=[
                item[1]
                for item in sorted_items
            ],
        )

    @staticmethod
    def _stable_sparse_index(
        token: str,
    ) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    测试用 embedding。

    生产说明：
    - 它不是语义向量模型。
    - 它只用于验证 Qdrant 入库、检索、过滤、父子回填流程。
    - 真实检索请使用 BgeM3EmbeddingProvider。
    """

    def __init__(
        self,
        dense_size: int = 1024,
        sparse_mod: int = 100_000,
    ) -> None:
        self.dense_size = dense_size
        self.sparse_mod = sparse_mod

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[TextEmbedding]:
        return [
            self._embed_text(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> TextEmbedding:
        return self._embed_text(text)

    def _embed_text(
        self,
        text: str,
    ) -> TextEmbedding:
        dense = self._dense_hash_embedding(text)
        sparse = self._sparse_keyword_embedding(text)

        return TextEmbedding(
            dense=dense,
            sparse=sparse,
        )

    def _dense_hash_embedding(
        self,
        text: str,
    ) -> list[float]:
        vector = [0.0] * self.dense_size

        tokens = self._tokenize_for_fake_embedding(text)

        if not tokens:
            tokens = ["<empty>"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()

            for offset in range(0, len(digest), 4):
                index = int.from_bytes(
                    digest[offset: offset + 4],
                    byteorder="big",
                    signed=False,
                ) % self.dense_size

                vector[index] += 1.0

        return self._normalize(vector)

    def _sparse_keyword_embedding(
        self,
        text: str,
    ) -> SparseEmbedding:
        token_counts: dict[int, float] = {}

        for token in self._tokenize_for_fake_embedding(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.sparse_mod
            token_counts[index] = token_counts.get(index, 0.0) + 1.0

        if not token_counts:
            return SparseEmbedding(indices=[], values=[])

        sorted_items = sorted(token_counts.items())

        return SparseEmbedding(
            indices=[item[0] for item in sorted_items],
            values=[item[1] for item in sorted_items],
        )

    @staticmethod
    def _tokenize_for_fake_embedding(
        text: str,
    ) -> list[str]:
        tokens: list[str] = []
        current_ascii: list[str] = []

        for char in text.lower():
            if "\u4e00" <= char <= "\u9fff":
                if current_ascii:
                    tokens.append("".join(current_ascii))
                    current_ascii = []
                tokens.append(char)
            elif char.isascii() and char.isalnum():
                current_ascii.append(char)
            else:
                if current_ascii:
                    tokens.append("".join(current_ascii))
                    current_ascii = []

        if current_ascii:
            tokens.append("".join(current_ascii))

        return tokens

    @staticmethod
    def _normalize(
        vector: Iterable[float],
    ) -> list[float]:
        values = list(vector)
        norm = math.sqrt(sum(item * item for item in values))

        if norm == 0:
            return values

        return [
            item / norm
            for item in values
        ]
