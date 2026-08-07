from app.agent_graph.production_dependencies import build_production_graph_dependencies
from app.llm.qwen_client import _parse_json_object


class DummyClient:
    pass


def test_qwen_json_parser_handles_fenced_output():
    assert _parse_json_object('```json\n{"answer":"ok"}\n```') == {"answer": "ok"}


def test_only_synthesizer_uses_qwen_client():
    deepseek = DummyClient()
    qwen = DummyClient()
    dependencies = build_production_graph_dependencies(
        llm_client=deepseek,
        synthesis_llm_client=qwen,
    )
    assert dependencies.agent_loop.planner.llm_client is deepseek
    assert dependencies.agent_loop.reviewer.llm_client is deepseek
    assert dependencies.final_response_pipeline.synthesizer.llm_client is qwen
    assert dependencies.final_response_pipeline.output_guard.llm_client is deepseek
