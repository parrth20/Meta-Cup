from __future__ import annotations

from grader import grade_episode, text_matches
from server.environment import RunbookOpsEnvironment


def test_grader_score_bounds() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")

    scenario = env.scenario_map["easy_auth_token_expiry"]
    snapshot = env.state()
    grade = grade_episode(scenario, snapshot)

    assert 0.0 < grade.score < 1.0


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
            "steps_taken": 8,
            "resolution_attempted": True,
            "premature_resolution": False,
            "terminal_reason": "resolved_safely",
        }
    )

    grade = grade_episode(scenario, snapshot)
    assert grade.components["root_cause"] == 1.0
    assert grade.components["mitigation"] == 1.0
    assert grade.components["safe_resolution"] > 0.9


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
            "steps_taken": 10,
            "resolution_attempted": True,
            "premature_resolution": False,
            "terminal_reason": "resolved_safely",
        }
    )

    grade = grade_episode(scenario, snapshot)
    assert grade.components["root_cause"] >= 0.72
    assert grade.components["mitigation"] >= 0.72


def test_negated_statement_does_not_match() -> None:
    assert not text_matches(
        "This was not a clock skew issue, it was a network ACL problem",
        "Clock skew on auth node caused OAuth nonce validation failures",
        ["time skew on auth node broke nonce checks"],
    )


def test_matching_works_when_canonical_contains_negation() -> None:
    assert text_matches(
        "secret rotated but gateway did not reload new version",
        "Secret rotated but gateway did not reload new version",
        ["stale secret version on gateway"],
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
            "steps_taken": 4,
            "resolution_attempted": True,
            "premature_resolution": True,
            "terminal_reason": "unsafe_resolution",
        }
    )

    grade = grade_episode(scenario, snapshot)
    assert grade.components["safe_resolution"] < 0.6


def test_perfect_episode_score_is_strictly_less_than_one() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")
    scenario = env.scenario_map["easy_auth_token_expiry"]
    snapshot = env.state().model_copy(
        update={
            "selected_severity": "SEV-2",
            "assigned_team": "auth-oncall",
            "submitted_root_cause": scenario.true_root_cause,
            "submitted_mitigation": scenario.true_mitigation,
            "inspected_evidence_ids": scenario.required_evidence_ids,
            "done": True,
            "steps_taken": 8,
            "resolution_attempted": True,
            "premature_resolution": False,
            "terminal_reason": "resolved_safely",
        }
    )

    grade = grade_episode(scenario, snapshot)
    assert 0.0 < grade.score < 1.0
    assert 0.95 < grade.score < 0.9999


def test_empty_episode_score_is_strictly_greater_than_zero() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")
    scenario = env.scenario_map["easy_auth_token_expiry"]
    snapshot = env.state()

    grade = grade_episode(scenario, snapshot)
    assert 0.0 < grade.score < 1.0
    assert grade.score < 0.05


def test_perfect_solutions_vary_across_scenarios() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")
    easy = env.scenario_map["easy_auth_token_expiry"]
    hard = env.scenario_map["hard_auth_multi_signal_conflict"]

    easy_snapshot = env.state().model_copy(
        update={
            "scenario_id": easy.scenario_id,
            "title": easy.title,
            "difficulty": easy.difficulty,
            "service": easy.service,
            "steps_taken": 8,
            "max_steps": easy.max_steps,
            "steps_remaining": 0,
            "done": True,
            "selected_severity": easy.true_severity.value,
            "assigned_team": easy.true_owner_team.value,
            "submitted_root_cause": easy.true_root_cause,
            "submitted_mitigation": easy.true_mitigation,
            "inspected_evidence_ids": easy.required_evidence_ids,
            "discovered_relevant_evidence_ids": easy.required_evidence_ids,
            "known_facts": [],
            "action_history": [],
            "resolution_attempted": True,
            "premature_resolution": False,
            "terminal_reason": "resolved_safely",
            "last_action_result": "resolved",
            "total_reward": 0.0,
        }
    )
    hard_snapshot = easy_snapshot.model_copy(
        update={
            "scenario_id": hard.scenario_id,
            "title": hard.title,
            "difficulty": hard.difficulty,
            "service": hard.service,
            "steps_taken": 10,
            "max_steps": hard.max_steps,
            "steps_remaining": 2,
            "selected_severity": hard.true_severity.value,
            "assigned_team": hard.true_owner_team.value,
            "submitted_root_cause": hard.true_root_cause,
            "submitted_mitigation": hard.true_mitigation,
            "inspected_evidence_ids": hard.required_evidence_ids,
            "discovered_relevant_evidence_ids": hard.required_evidence_ids,
        }
    )

    easy_grade = grade_episode(easy, easy_snapshot)
    hard_grade = grade_episode(hard, hard_snapshot)
    assert easy_grade.score != hard_grade.score


def test_irrelevant_overinspection_reduces_evidence_score() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="medium_email_provider_vs_config")
    scenario = env.scenario_map["medium_email_provider_vs_config"]
    required_only = env.state().model_copy(
        update={
            "steps_taken": 9,
            "selected_severity": scenario.true_severity.value,
            "assigned_team": scenario.true_owner_team.value,
            "submitted_root_cause": scenario.true_root_cause,
            "submitted_mitigation": scenario.true_mitigation,
            "inspected_evidence_ids": scenario.required_evidence_ids,
            "discovered_relevant_evidence_ids": scenario.required_evidence_ids,
            "done": True,
            "resolution_attempted": True,
            "premature_resolution": False,
            "terminal_reason": "resolved_safely",
        }
    )
    noisy = required_only.model_copy(
        update={
            "inspected_evidence_ids": scenario.required_evidence_ids + ["me1_log_provider_status_green"],
            "discovered_relevant_evidence_ids": scenario.required_evidence_ids,
        }
    )

    clean_grade = grade_episode(scenario, required_only)
    noisy_grade = grade_episode(scenario, noisy)
    assert noisy_grade.components["evidence_coverage"] < clean_grade.components["evidence_coverage"]
