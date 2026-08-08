"""LLM JSON 输出解析公共工具。

项目中多个模块（Planner、Guard、Synthesizer、Router 等）
需要从 LLM 返回文本中提取 JSON 对象或解析 tool call arguments。
这里提供统一的实现，避免各模块重复。
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_arguments(raw: Any) -> dict[str, Any]:
    """把 LLM tool call 的 arguments 统一转为 dict。

    兼容：
    - 已经是 dict 的直接返回
    - JSON 字符串尝试解析
    - None / 空串返回空 dict
    """
    if raw is None:
        return {}

    if isinstance(raw, dict):
        return raw

    if not isinstance(raw, str):
        raise ValueError(
            f"arguments 格式错误，期望 dict 或 str，实际: {type(raw).__name__}"
        )

    stripped = raw.strip()
    if not stripped:
        return {}

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "arguments 不是合法 JSON。"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "arguments 顶层必须是 JSON 对象。"
        )

    return payload


def extract_json_object(text: str) -> dict[str, Any]:
    """从 LLM 返回文本中提取 JSON 对象。

    兼容：
    - 纯 JSON
    - ```json ... ``` 或 ``` ... ``` 代码块包裹的 JSON
    - JSON 前后有额外文字（从第一个 '{' 截到最后一个 '}'）
    """
    stripped = text.strip()

    if not stripped:
        raise ValueError("LLM 返回了空内容，无法提取 JSON。")

    # 1) 尝试直接解析
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    # 2) 去掉 markdown 代码块
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 3) 直接解析清洗后的文本
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    # 4) 从第一个 '{' 截到最后一个 '}' 再解析
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(
            "LLM 返回内容中没有找到 JSON 对象。"
        )

    segment = cleaned[start : end + 1]
    try:
        payload = json.loads(segment)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM 返回内容中的 JSON 无法解析。"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "LLM 返回 JSON 顶层必须是对象。"
        )

    return payload
