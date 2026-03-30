from __future__ import annotations

from models import Action, ActionType
from server.environment import RunbookOpsEnvironment


def test_reset_initializes_clean_state() -> None:
    env = RunbookOpsEnvironment()
    obs = env.reset(scenario_id="easy_auth_token_expiry")

    assert obs.scenario_id == "easy_auth_token_expiry"
    assert obs.steps_taken == 0
    assert obs.done is False
    assert obs.known_facts == []


def test_step_is_deterministic_for_same_state() -> None:
    env = RunbookOpsEnvironment()

    env.reset(scenario_id="easy_search_cache_stale")
    result_a = env.step(Action(action_type=ActionType.INSPECT_ALERT, target="es1_alert_freshness_lag"))

    env.reset(scenario_id="easy_search_cache_stale")
    result_b = env.step(Action(action_type=ActionType.INSPECT_ALERT, target="es1_alert_freshness_lag"))

    assert result_a.reward == result_b.reward
    assert result_a.info.message == result_b.info.message


def test_invalid_target_is_handled_gracefully() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")

    result = env.step(Action(action_type=ActionType.INSPECT_LOG, target="unknown_log_id"))
    assert result.info.invalid_action is True
    assert result.reward < 0


def test_reward_progression_and_duplicate_penalty() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")

    first = env.step(Action(action_type=ActionType.INSPECT_ALERT, target="ea1_alert_401_spike"))
    duplicate = env.step(Action(action_type=ActionType.INSPECT_ALERT, target="ea1_alert_401_spike"))

    assert first.reward > 0
    assert duplicate.reward < 0


def test_locked_evidence_requires_prerequisite_inspection() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")

    locked = env.step(Action(action_type=ActionType.INSPECT_LOG, target="ea1_log_jwt_expired"))
    assert locked.info.invalid_action is True
    assert "locked" in locked.info.message.lower()

    env.step(Action(action_type=ActionType.INSPECT_ALERT, target="ea1_alert_401_spike"))
    unlocked = env.step(Action(action_type=ActionType.INSPECT_LOG, target="ea1_log_jwt_expired"))
    assert unlocked.info.invalid_action is False
    assert unlocked.reward > 0


def test_coverage_milestone_reward_applied_once() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")

    step1 = env.step(Action(action_type=ActionType.INSPECT_ALERT, target="ea1_alert_401_spike"))
    step2 = env.step(Action(action_type=ActionType.INSPECT_LOG, target="ea1_log_jwt_expired"))
    step3 = env.step(Action(action_type=ActionType.INSPECT_RUNBOOK, target="ea1_runbook_key_rotation"))

    assert "milestone" in step2.info.message.lower()
    assert "milestone" in step3.info.message.lower()

    duplicate = env.step(Action(action_type=ActionType.INSPECT_RUNBOOK, target="ea1_runbook_key_rotation"))
    assert "milestone" not in duplicate.info.message.lower()
    assert step1.reward > duplicate.reward


def test_walkthrough_scores_high() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="easy_auth_token_expiry")

    actions = [
        Action(action_type=ActionType.INSPECT_ALERT, target="ea1_alert_401_spike"),
        Action(action_type=ActionType.INSPECT_LOG, target="ea1_log_jwt_expired"),
        Action(action_type=ActionType.SET_SEVERITY, content="SEV-2"),
        Action(action_type=ActionType.ASSIGN_TEAM, content="auth-oncall"),
        Action(
            action_type=ActionType.SUBMIT_ROOT_CAUSE,
            content="expired auth signing key",
        ),
        Action(
            action_type=ActionType.SUBMIT_MITIGATION,
            content="perform key rotation and reload auth issuer",
        ),
    ]

    for action in actions:
        result = env.step(action)
        if result.done:
            break

    grade = env.grade_current_episode()
    assert grade.score >= 0.9


def test_premature_resolution_penalty_and_terminal_reason() -> None:
    env = RunbookOpsEnvironment()
    env.reset(scenario_id="medium_checkout_flag_rollout")

    result = env.step(Action(action_type=ActionType.RESOLVE_INCIDENT))
    assert result.done is True
    assert result.reward <= -0.2
    assert result.info.terminal_reason == "unsafe_resolution"
