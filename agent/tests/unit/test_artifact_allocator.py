from __future__ import annotations

from app.agent_graph.conversation_state import (
    allocate_artifact_handle,
    default_conversation_state,
    materialize_new_artifacts,
    validate_referential_integrity,
)


def test_allocator_assigns_real_handles() -> None:
    state = default_conversation_state()
    assert allocate_artifact_handle(state, "conclusion") == (
        "CONCLUSION_1"
    )
    assert allocate_artifact_handle(state, "conclusion") == (
        "CONCLUSION_2"
    )
    assert allocate_artifact_handle(state, "claim") == "CLAIM_1"
    assert allocate_artifact_handle(state, "calc") == "CALC_1"


def test_materialize_new_artifacts_uses_python_handles() -> None:
    state = default_conversation_state()
    materialized = materialize_new_artifacts(
        state,
        {
            "new_artifacts": [
                {
                    "local_key": "main_conclusion",
                    "artifact_type": "conclusion",
                    "text": "仅凭该广告无法证明本金安全",
                    "handle": "CONCLUSION_999",
                },
                {
                    "local_key": "calc_remaining",
                    "artifact_type": "calc",
                    "text": "90−20=70",
                    "operation": "subtract",
                    "inputs": {
                        "cash": 900000,
                        "down_payment": 200000,
                    },
                },
            ],
            "focus_candidate": {
                "artifact_local_key": "main_conclusion"
            },
        },
    )
    artifacts = materialized["artifacts"]
    assert artifacts[0]["handle"] == "CONCLUSION_1"
    assert artifacts[1]["handle"] == "CALC_1"
    assert artifacts[1]["output"] == 700000.0
    assert artifacts[1]["verification_status"] == "verified"
    assert materialized["focus"]["handle"] == "CONCLUSION_1"
    assert materialized["focus"]["local_key"] == (
        "main_conclusion"
    )


def test_referential_integrity_gate() -> None:
    violations = validate_referential_integrity(
        used_fact_refs=["cash", "unknown_fact"],
        used_derivation_ids=["CALC_1", "CALC_99"],
        used_result_artifact_refs=["RESULT_1.CLAIM_1"],
        used_citation_ids=["2", "999"],
        canonical_fact_fields=["cash"],
        known_derivation_ids=["CALC_1"],
        known_sub_artifact_ids=["RESULT_1.CLAIM_1"],
        allowed_citation_ids=["1", "2"],
    )
    assert "fact:unknown_fact" in violations
    assert "derivation:CALC_99" in violations
    assert "citation:999" in violations
    assert "sub_artifact:RESULT_1.CLAIM_1" not in violations


def test_referential_integrity_rejects_bare_calc_ref() -> None:
    violations = validate_referential_integrity(
        used_fact_refs=[],
        used_derivation_ids=[],
        used_result_artifact_refs=[".CALC_1"],
        used_citation_ids=[],
        canonical_fact_fields=[],
        known_derivation_ids=[],
        known_sub_artifact_ids=["RESULT_2.CALC_1"],
        allowed_citation_ids=[],
    )
    assert "sub_artifact:.CALC_1" in violations


def test_referential_integrity_accepts_qualified_result_ref() -> None:
    violations = validate_referential_integrity(
        used_fact_refs=[],
        used_derivation_ids=[],
        used_result_artifact_refs=["RESULT_2.CALC_1"],
        used_citation_ids=[],
        canonical_fact_fields=[],
        known_derivation_ids=[],
        known_sub_artifact_ids=["RESULT_2.CALC_1"],
        allowed_citation_ids=[],
    )
    assert "sub_artifact:RESULT_2.CALC_1" not in violations
