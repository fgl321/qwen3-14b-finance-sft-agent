from __future__ import annotations

from app.rag.source_classifier import SourceClassifier


def test_missing_api_key_falls_back_to_default() -> None:
    classifier = SourceClassifier(settings=None)
    metadata = classifier.classify(
        text="普通文档内容",
        file_name="test.md",
    )
    assert metadata.generated_content is False
    assert metadata.allow_rag_direct is True
    assert metadata.trust_level == "unverified"


def test_payload_generated_content_forces_ineligible() -> None:
    classifier = SourceClassifier()
    metadata = classifier._payload_to_metadata(
        {
            "content_type": "agent_debug_log",
            "trust_level": "unverified",
            "generated_content": True,
            "allow_rag_direct": True,
            "reason": "Agent 运行日志",
        }
    )
    assert metadata.generated_content is True
    assert metadata.allow_rag_direct is False
    assert metadata.content_type == "agent_debug_log"


def test_payload_financial_knowledge_is_verified() -> None:
    classifier = SourceClassifier()
    metadata = classifier._payload_to_metadata(
        {
            "content_type": "financial_knowledge",
            "trust_level": "verified",
            "generated_content": False,
            "allow_rag_direct": True,
            "reason": "正式金融知识",
        }
    )
    assert metadata.trust_level == "verified"
    assert metadata.allow_rag_direct is True
    assert metadata.content_type == "financial_knowledge"


def test_invalid_content_type_normalized_to_other() -> None:
    classifier = SourceClassifier()
    metadata = classifier._payload_to_metadata(
        {
            "content_type": "weird_type",
            "trust_level": "unknown",
            "generated_content": False,
            "allow_rag_direct": True,
            "reason": "test",
        }
    )
    assert metadata.content_type == "other"
    assert metadata.trust_level == "unverified"


def test_classify_uses_llm_payload(monkeypatch) -> None:
    classifier = SourceClassifier()

    def fake_llm(self, text: str, file_name: str) -> dict:
        return {
            "content_type": "agent_debug_log",
            "trust_level": "unverified",
            "generated_content": True,
            "allow_rag_direct": False,
            "reason": "检测到运行日志结构",
        }

    monkeypatch.setattr(
        SourceClassifier,
        "_llm_classify_sync",
        fake_llm,
    )
    metadata = classifier.classify(
        text='{"request_id": "x", "final_answer": "y"}',
        file_name="原始响应.txt",
    )
    assert metadata.generated_content is True
    assert metadata.allow_rag_direct is False
