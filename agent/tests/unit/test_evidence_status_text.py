from __future__ import annotations

from app.rag.rag_types import (
    EVIDENCE_STATUS_USER_TEXT,
    RequirementObservation,
)


def test_evidence_status_user_text_has_all_frozen_statuses() -> None:
    frozen = {
        "direct_support",
        "partial_support",
        "background_support",
        "insufficient_evidence",
        "not_observed",
        "technical_unavailable",
        "conflict",
        "assessment_protocol_failed",
    }
    assert frozen <= set(EVIDENCE_STATUS_USER_TEXT)


def test_evidence_status_texts_are_distinct() -> None:
    texts = set(EVIDENCE_STATUS_USER_TEXT.values())
    assert len(texts) == len(EVIDENCE_STATUS_USER_TEXT)


def test_evidence_absence_never_conflated_with_technical_failure() -> None:
    insufficient = EVIDENCE_STATUS_USER_TEXT["insufficient_evidence"]
    not_observed = EVIDENCE_STATUS_USER_TEXT["not_observed"]
    technical = EVIDENCE_STATUS_USER_TEXT["technical_unavailable"]
    assert insufficient != not_observed
    assert insufficient != technical
    assert not_observed != technical


def test_requirement_observation_accepts_background_support() -> None:
    observation = RequirementObservation(
        requirement_id="T1:E1",
        task_id="T1",
        status="background_support",
    )
    assert observation.status == "background_support"
