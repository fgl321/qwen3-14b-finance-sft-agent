from app.agent_graph.quality_gate import build_quality_gate_result


def test_quality_gate_should_fallback_for_general_finance_question_without_evidence():
    state = {
        "user_message": "什么是紧急备用金？",
        "final_answer": "我在当前知识库中没有找到足够依据回答这个问题，因此不能基于知识库给出确定回答。",
        "executed_tools": [
            {
                "ok": True,
                "tool_name": "search_knowledge_base",
                "arguments": {
                    "query": "什么是紧急备用金",
                },
                "result": {
                    "answer": "我在当前知识库中没有找到足够依据回答这个问题，因此不能基于知识库给出确定回答。",
                    "retrieved_count": 0,
                    "evidence_assessment": {
                        "sufficient": False,
                        "confidence": "high",
                        "reason": "知识库没有检索到相关证据。",
                    },
                    "citations": [],
                },
            }
        ],
    }

    result = build_quality_gate_result(state)

    assert result["needs_general_finance_fallback"] is True
    assert result["is_kb_specific"] is False
    assert result["answer_is_no_evidence"] is True
    assert result["has_no_evidence_rag_tool_result"] is True


def test_quality_gate_should_not_fallback_when_user_explicitly_asks_for_knowledge_base():
    state = {
        "user_message": "请根据知识库解释什么是紧急备用金。",
        "final_answer": "我在当前知识库中没有找到足够依据回答这个问题，因此不能基于知识库给出确定回答。",
        "executed_tools": [
            {
                "ok": True,
                "tool_name": "search_knowledge_base",
                "arguments": {
                    "query": "什么是紧急备用金",
                },
                "result": {
                    "answer": "我在当前知识库中没有找到足够依据回答这个问题，因此不能基于知识库给出确定回答。",
                    "retrieved_count": 0,
                    "evidence_assessment": {
                        "sufficient": False,
                        "confidence": "high",
                        "reason": "知识库没有检索到相关证据。",
                    },
                    "citations": [],
                },
            }
        ],
    }

    result = build_quality_gate_result(state)

    assert result["needs_general_finance_fallback"] is False
    assert result["is_kb_specific"] is True
    assert result["answer_is_no_evidence"] is True
    assert result["has_no_evidence_rag_tool_result"] is True


def test_quality_gate_should_not_fallback_for_normal_answer():
    state = {
        "user_message": "什么是紧急备用金？",
        "final_answer": "紧急备用金是用于应对突发支出的现金储备。",
        "executed_tools": [],
    }

    result = build_quality_gate_result(state)

    assert result["needs_general_finance_fallback"] is False
    assert result["is_kb_specific"] is False
    assert result["answer_is_no_evidence"] is False
    assert result["has_no_evidence_rag_tool_result"] is False
