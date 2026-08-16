from __future__ import annotations

import json
from typing import Any

from app.rag.rag_types import SourceAuthorityContract


def normalize_authority(value: Any) -> SourceAuthorityContract | None:
    """Normalize a SourceAuthorityContract or its JSON dict form."""

    if value is None or isinstance(value, SourceAuthorityContract):
        return value

    if isinstance(value, dict):
        allowed_fields = set(SourceAuthorityContract.model_fields)
        payload = {
            str(key): item
            for key, item in value.items()
            if str(key) in allowed_fields
        }
        try:
            return SourceAuthorityContract.model_validate(payload)
        except Exception:
            return None

    return None


def source_authority_contract_message(
    authority: Any,
) -> str:
    """Render the SourceAuthorityContract as a machine-checkable prompt block.

    The contract is passed verbatim as JSON; the prose rules only explain the
    enforcement semantics so the synthesizer/guard do not need keyword parsing
    to discover which sources are allowed.
    """

    contract = normalize_authority(authority)
    if contract is None:
        return ""

    serialized = json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    )

    rules: list[str] = []
    if contract.general_model_knowledge == "forbidden":
        rules.append(
            "- general_model_knowledge=forbidden: 不得输出没有用户事实、"
            "已验证工具结果或文档引用支撑的金融规则/制度/数字结论；"
            "也不得以“通用金融建议/通用金融知识”名义输出此类内容；"
            "无法获得支撑的结论只能明确说明当前证据不足，"
            "不能陈述该规则本身。"
        )
    if contract.domain_heuristics == "forbidden":
        rules.append(
            "- domain_heuristics=forbidden: 不得把经验法则（如三一定律、"
            "80 定律、4321 定律）当作用户结论的依据；"
            "此类经验规则只能在没有文档证据时明确标注为通用金融建议。"
        )
    if contract.selected_documents == "forbidden":
        rules.append(
            "- selected_documents=forbidden: 不得引用任何上传文档；"
            "不得声称“根据文档/条款”。"
        )
    if contract.memory == "forbidden":
        rules.append(
            "- memory=forbidden: 不得使用长期记忆或历史对话中的个人事实；"
            "只能使用本轮用户消息明确提供的事实。"
        )
    if contract.current_user_facts == "forbidden":
        rules.append(
            "- current_user_facts=forbidden: 不得使用用户本轮提供的个人事实。"
        )
    if contract.web == "forbidden":
        rules.append(
            "- web=forbidden: 不得使用联网检索的信息。"
        )
    if contract.deterministic_derivation == "forbidden":
        rules.append(
            "- deterministic_derivation=forbidden: 不得使用确定性计算工具，"
            "也不得声称存在已验证的工具计算结果。"
        )

    if not rules:
        rules.append("- 所有来源均可按契约允许。")

    return (
        "<source_authority_contract>\n"
        f"{serialized}\n"
        "</source_authority_contract>\n"
        "执行规则：\n"
        + "\n".join(rules)
    )
