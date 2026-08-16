from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextBudgetConfig:
    """Centralized context budgets for Synthesis / Guard prompts.

    Values are token budgets (deterministic estimator).  These budgets are
    structural: they never delete logical evidence requirements, requirement
    observations, or the delivery contract items.
    """

    history_tokens: int = 4000
    memory_tokens: int = 3000
    contract_tokens: int = 6000
    tool_tokens: int = 4000
    evidence_tokens: int = 22000
    guard_context_tokens: int = 10000

    max_history_messages: int = 8
    max_non_continuation_messages: int = 2

    per_evidence_chars: int = 600
    per_citation_chars: int = 500
    per_tool_output_chars: int = 1200
    max_tool_results: int = 24

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name.endswith("tokens") or name.startswith("max_") or name.startswith("per_"):
                value = getattr(self, name)
                if int(value) <= 0:
                    raise ValueError(
                        f"{name} must be positive, got {value}"
                    )


DEFAULT_CONTEXT_BUDGET = ContextBudgetConfig()


_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x3000, 0x303F),
    (0xFF00, 0xFFEF),
)


def estimate_tokens(text: str) -> int:
    """Deterministic conservative token estimate.

    CJK characters are roughly 1 token per 1.5 chars; other characters are
    roughly 1 token per 4 chars.  The estimator is intentionally simple so
    budgets are reproducible in tests and observability.
    """

    if not text:
        return 0

    cjk_count = 0
    for char in text:
        codepoint = ord(char)
        for start, end in _CJK_RANGES:
            if start <= codepoint <= end:
                cjk_count += 1
                break

    other_count = len(text) - cjk_count
    return max(
        1,
        math.ceil(cjk_count / 1.5)
        + math.ceil(other_count / 4),
    )


def trim_text(
    text: str,
    budget_tokens: int,
    *,
    suffix: str = "...[truncated]",
) -> str:
    """Trim a plain text string to a token budget."""

    if not text:
        return ""

    if estimate_tokens(text) <= budget_tokens:
        return text

    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= budget_tokens:
            low = middle
        else:
            high = middle - 1

    return f"{text[:low].rstrip()}{suffix}"


def _normalize_hash_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        "",
        str(value or ""),
    ).casefold()


def content_hash(value: Any) -> str:
    normalized = _normalize_hash_text(value)
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]


def dedupe_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate messages by content hash, keeping the latest occurrence."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message in reversed(list(messages)):
        message_dict = dict(message)
        digest = (
            key(message_dict)
            if key is not None
            else content_hash(
                f"{message_dict.get('role') or ''}:"
                f"{message_dict.get('content') or ''}"
            )
        )
        if digest in seen:
            continue
        seen.add(digest)
        result.append(message_dict)
    result.reverse()
    return result


def select_history_messages(
    messages: Sequence[Mapping[str, Any]],
    user_message: str,
    *,
    budget: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select recent, deduplicated history for the planner.

    Self-contained questions keep at most the immediate previous turns;
    explicit continuations keep more recent turns.  The current user message
    is never duplicated into history.
    """

    stats: dict[str, Any] = {
        "input_count": len(messages),
        "continuation": None,
    }

    current_digest = content_hash(user_message)
    filtered = [
        message
        for message in messages
        if content_hash(message.get("content")) != current_digest
    ]
    stats["deduplicated_current_count"] = (
        len(messages) - len(filtered)
    )

    unique = dedupe_messages(filtered)
    max_count = budget.max_history_messages
    selected = unique[-max_count:] if max_count > 0 else []

    budget_tokens = budget.history_tokens

    compacted: list[dict[str, Any]] = []
    used_tokens = 0
    for message in selected:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        tokens = estimate_tokens(content) + 4
        if used_tokens + tokens > budget_tokens:
            break
        compacted.append(
            {
                "role": role,
                "content": content,
            }
        )
        used_tokens += tokens

    stats["selected_count"] = len(compacted)
    stats["selected_tokens"] = used_tokens
    stats["budget_tokens"] = budget_tokens

    return compacted, stats


def trim_context_summary(
    context_summary: str,
    *,
    budget: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET,
) -> tuple[str, dict[str, Any]]:
    """Trim memory/context summary to the memory budget."""

    original_tokens = estimate_tokens(context_summary)
    trimmed = trim_text(
        context_summary,
        budget.memory_tokens,
    )
    return trimmed, {
        "original_tokens": original_tokens,
        "trimmed_tokens": estimate_tokens(trimmed),
        "budget_tokens": budget.memory_tokens,
        "truncated": trimmed != context_summary,
    }


def compact_json(
    value: Any,
    *,
    max_chars: int = 1200,
    depth: int = 0,
) -> Any:
    """Compact structured tool output without losing keys or numbers.

    Long string fields are truncated and long lists are capped; numeric
    values are never altered because they are correctness-critical.
    """

    if depth > 6:
        return "[max_depth]"

    if isinstance(value, str):
        return value[:max_chars] + "...[truncated]" if len(value) > max_chars else value

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result[str(key)] = compact_json(
                item,
                max_chars=max_chars,
                depth=depth + 1,
            )
        return result

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        items = [
            compact_json(
                item,
                max_chars=max_chars,
                depth=depth + 1,
            )
            for item in value[:40]
        ]
        if len(value) > 40:
            items.append("[truncated]")
        return items

    return value


def compact_tool_results(
    results: Sequence[Mapping[str, Any]],
    *,
    budget: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compact tool result payloads for Synthesis/Guard prompts."""

    items: list[dict[str, Any]] = []
    total_tokens = 0
    for result in results[: budget.max_tool_results]:
        item = dict(result)
        if isinstance(item.get("output"), (dict, list, str)):
            item["output"] = compact_json(
                item["output"],
                max_chars=budget.per_tool_output_chars,
            )
        serialized = json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        tokens = estimate_tokens(serialized)
        if total_tokens + tokens > budget.tool_tokens:
            break
        items.append(item)
        total_tokens += tokens

    return items, {
        "selected_count": len(items),
        "selected_tokens": total_tokens,
        "budget_tokens": budget.tool_tokens,
        "dropped_count": max(
            0,
            len(results) - len(items),
        ),
    }


_CITATION_TEXT_FIELDS = (
    "text",
    "quote",
    "evidence_excerpt",
)


def compact_citation(
    citation: Mapping[str, Any],
    *,
    max_chars: int = 500,
    include_text: bool = True,
) -> dict[str, Any]:
    """Keep citation identity + evidence text compactly."""

    item = dict(citation)
    metadata = dict(item.get("metadata") or {})
    keep_metadata_fields = {
        "requirement_id",
        "requirement_ids",
        "support_level",
        "supported_requirement_ids",
        "evidence_confidence",
        "evidence_excerpt",
    }
    compact_metadata = {
        key: value
        for key, value in metadata.items()
        if key in keep_metadata_fields
    }

    compact: dict[str, Any] = {
        key: item[key]
        for key in (
            "citation_id",
            "document_id",
            "file_name",
            "page_start",
            "page_end",
            "chunk_id",
            "score",
            "score_type",
        )
        if key in item
    }
    compact["metadata"] = compact_metadata

    scores = item.get("scores")
    if isinstance(scores, Mapping):
        compact["scores"] = {
            key: scores.get(key)
            for key in (
                "display_score",
                "display_score_source",
                "evidence_confidence",
            )
            if key in scores
        }

    if include_text:
        for field in _CITATION_TEXT_FIELDS:
            raw = item.get(field)
            if raw is None:
                continue
            text = str(raw)
            if len(text) > max_chars:
                text = text[:max_chars] + "...[truncated]"
            compact[field] = text

        if (
            "text" not in compact
            and "quote" not in compact
            and compact_metadata.get("evidence_excerpt")
        ):
            excerpt = str(compact_metadata["evidence_excerpt"])
            if len(excerpt) > max_chars:
                excerpt = excerpt[:max_chars] + "...[truncated]"
            compact["text"] = excerpt

    return compact


_SUPPORT_RANK = {
    "direct_support": 3,
    "partial_support": 2,
    "background_support": 1,
}


def select_evidence_citations(
    citations: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    budget: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the strongest, non-duplicated evidence per requirement.

    For every logical requirement observation:
    - keep up to 2 strongest direct-support citations, else 1 partial,
      else 1 background;
    - a citation that supports several requirements is emitted once with
      ``supported_requirement_ids`` metadata;
    - requirement observations are never deleted.
    """

    citation_by_id: dict[str, dict[str, Any]] = {}
    for citation in citations:
        citation_id = str(citation.get("citation_id") or "")
        if citation_id:
            citation_by_id[citation_id] = dict(citation)

    selected_ids: list[str] = []
    supported_by_id: dict[str, list[str]] = {}
    total_requirements = 0

    for observation in observations:
        requirement_id = str(
            observation.get("requirement_id") or ""
        )
        if not requirement_id:
            continue
        total_requirements += 1
        metadata = dict(observation.get("metadata") or {})
        requirement_citations = list(
            observation.get("citation_ids")
            or metadata.get("citation_ids")
            or []
        )
        ranked = sorted(
            (
                citation_by_id.get(str(citation_id))
                for citation_id in requirement_citations
                if str(citation_id) in citation_by_id
            ),
            key=lambda citation: (
                _SUPPORT_RANK.get(
                    str(
                        (
                            citation.get("metadata")
                            or {}
                        ).get("support_level")
                        or ""
                    ),
                    0,
                ),
                float(citation.get("score") or 0),
            ),
            reverse=True,
        )
        if not ranked:
            continue
        best = ranked[0]
        best_level = str(
            (best.get("metadata") or {}).get(
                "support_level"
            )
            or ""
        )
        keep_count = (
            2 if best_level == "direct_support" else 1
        )
        for citation in ranked[:keep_count]:
            citation_id = str(
                citation.get("citation_id") or ""
            )
            if not citation_id:
                continue
            if citation_id not in selected_ids:
                selected_ids.append(citation_id)
            supported_by_id.setdefault(
                citation_id,
                [],
            ).append(requirement_id)

    compacted: list[dict[str, Any]] = []
    total_chars = 0
    char_budget = budget.evidence_tokens * 2
    for citation_id in selected_ids:
        citation = citation_by_id[citation_id]
        item = compact_citation(
            citation,
            max_chars=budget.per_evidence_chars,
        )
        item.setdefault("metadata", {})[
            "supported_requirement_ids"
        ] = supported_by_id.get(citation_id, [])
        serialized = json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if total_chars + len(serialized) > char_budget:
            break
        compacted.append(item)
        total_chars += len(serialized)

    return compacted, {
        "requirement_observation_count": (
            total_requirements
        ),
        "input_citation_count": len(citations),
        "selected_citation_count": len(compacted),
        "dropped_citation_count": max(
            0,
            len(citations) - len(compacted),
        ),
        "selected_chars": total_chars,
        "budget_tokens": budget.evidence_tokens,
    }


def build_evidence_context(
    rag: Mapping[str, Any] | None,
    *,
    budget: ContextBudgetConfig = DEFAULT_CONTEXT_BUDGET,
) -> tuple[str, dict[str, Any]]:
    """Render compact evidence context from a RAG result.

    Keeps provisional handling when evidence assessment was unavailable:
    in that case only a bounded number of unassessed candidates are shown
    and are explicitly labelled as provisional.
    """

    if not rag:
        return "", {}

    snippets: list[str] = []
    assessment_status = str(
        ((rag.get("stage_status") or {}).get(
            "evidence_assessment_status"
        ))
        or ""
    )
    provisional = assessment_status in {
        "protocol_failed",
        "service_failed",
    }

    if provisional:
        snippets.append(
            "[RAG_PROTOCOL_STATUS] 检索和候选召回已完成，但证据充分性审核未完成。"
            "以下内容仅为 provisional_unassessed 候选，不是已验证引用；"
            "禁止用模型通用知识替代用户要求的文档制度事实，也禁止引用这些候选。"
        )
        for index, chunk in enumerate(
            (rag.get("retrieved_chunks") or [])[:6],
            1,
        ):
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            source = str(
                chunk.get("document_name")
                or chunk.get("document_id")
                or "document"
            )
            snippets.append(
                f"[Provisional Evidence {index} | {source}] "
                f"{text[:budget.per_evidence_chars]}"
            )
        joined = "\n\n".join(snippets)
        return joined, {
            "provisional": True,
            "candidate_count": min(
                6,
                len(rag.get("retrieved_chunks") or []),
            ),
            "tokens": estimate_tokens(joined),
            "budget_tokens": budget.evidence_tokens,
        }

    observations = list(
        rag.get("requirement_coverage") or []
    )
    citations, selection_stats = select_evidence_citations(
        list(rag.get("citations") or []),
        observations,
        budget=budget,
    )

    for citation in citations:
        text = str(
            citation.get("quote")
            or citation.get("text")
            or (citation.get("metadata") or {}).get(
                "evidence_excerpt"
            )
            or ""
        ).strip()
        if not text:
            continue
        source = str(
            citation.get("file_name")
            or citation.get("document_id")
            or "document"
        )
        requirement_ids = ",".join(
            (citation.get("metadata") or {}).get(
                "supported_requirement_ids"
            )
            or []
        )
        support = (
            citation.get("metadata") or {}
        ).get("support_level")
        label = (
            f"[Evidence {citation.get('citation_id')} | "
            f"{source}"
            + (f" | req:{requirement_ids}" if requirement_ids else "")
            + (f" | {support}" if support else "")
            + "]"
        )
        snippets.append(f"{label} {text}")

    joined = "\n\n".join(snippets)
    if estimate_tokens(joined) > budget.evidence_tokens:
        joined = trim_text(
            joined,
            budget.evidence_tokens,
        )

    return joined, {
        "provisional": False,
        **selection_stats,
        "tokens": estimate_tokens(joined),
        "budget_tokens": budget.evidence_tokens,
        "truncated": (
            estimate_tokens(joined)
            >= budget.evidence_tokens
        ),
    }
