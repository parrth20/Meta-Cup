from __future__ import annotations

from grader import grade_episode, text_matches
from server.environment import RunbookOpsEnvironment


def test_grader_score_bounds() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")

    scenario = env.scenario_map["easy_auth_token_expiry"]
    snapshot = env.state()
    grade = grade_episode(scenario, snapshot)

    assert 0.0 <= grade.score <= 1.0


def test_alias_matching_for_root_cause_and_mitigation() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")

    scenario = env.scenario_map["easy_auth_token_expiry"]
    snapshot = env.state().model_copy(
        update={
            "selected_severity": "SEV-2",
            "assigned_team": "auth-oncall",
            "submitted_root_cause": "expired auth signing key",
            "submitted_mitigation": "perform key rotation and reload auth issuer",
            "inspected_evidence_ids": scenario.required_evidence_ids,
            "done": True,
            "resolution_attempted": True,
            "premature_resolution": False,
        }
    )

    grade = grade_episode(scenario, snapshot)
    assert grade.components["root_cause"] == 1.0
    assert grade.components["mitigation"] == 1.0
    assert grade.components["safe_resolution"] == 1.0


def test_paraphrase_matching_for_root_cause_and_mitigation() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="hard_auth_multi_signal_conflict")

    scenario = env.scenario_map["hard_auth_multi_signal_conflict"]
    snapshot = env.state().model_copy(
        update={
            "selected_severity": "SEV-1",
            "assigned_team": "auth-oncall",
            "submitted_root_cause": "OAuth nonce checks failed because one auth node had time drift after ntp stopped",
            "submitted_mitigation": "isolate the skewed host, restart time-sync service, then return it after validation",
            "inspected_evidence_ids": scenario.required_evidence_ids,
            "done": True,
            "resolution_attempted": True,
            "premature_resolution": False,
        }
    )

    grade = grade_episode(scenario, snapshot)
    assert grade.components["root_cause"] == 1.0
    assert grade.components["mitigation"] == 1.0


def test_negated_statement_does_not_match() -> None:
    assert not text_matches(
        "This was not a clock skew issue, it was a network ACL problem",
        "Clock skew on auth node caused OAuth nonce validation failures",
        ["time skew on auth node broke nonce checks"],
    )


def test_safe_resolution_requires_enough_evidence() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="hard_auth_multi_signal_conflict")
    scenario = env.scenario_map["hard_auth_multi_signal_conflict"]
    snapshot = env.state().model_copy(
        update={
            "selected_severity": "SEV-1",
            "assigned_team": "auth-oncall",
            "submitted_root_cause": scenario.true_root_cause,
            "submitted_mitigation": scenario.true_mitigation,
            "inspected_evidence_ids": scenario.required_evidence_ids[:2],
            "done": True,
            "resolution_attempted": True,
            "premature_resolution": True,
        }
    )

    grade = grade_episode(scenario, snapshot)
    assert grade.components["safe_resolution"] == 0.0
