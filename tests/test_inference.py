from __future__ import annotations

from inference import _fallback_action, _parse_json_action
from models import ActionType, Difficulty, Observation


def _base_observation() -> Observation:
    return Observation(
        scenario_id="hard_auth_multi_signal_conflict",
        title="Auth nonce mismatches from skewed node",
        difficulty=Difficulty.HARD,
        service="auth",
        incident_summary="Users fail login after nonce validation errors.",
        visible_alerts=[],
        visible_logs=[],
        visible_runbooks=[],
        visible_timeline_notes=[],
        known_facts=[],
        last_action_result="",
        selected_severity=None,
        assigned_team=None,
        submitted_root_cause=None,
        submitted_mitigation=None,
        steps_taken=0,
        steps_remaining=12,
        done=False,
        action_history_summary=[],
        available_action_types=list(ActionType),
    )


def test_parse_json_action_with_strict_json() -> None:
    payload = _parse_json_action('{"action_type":"inspect_alert","target":"a1"}')
    assert payload == {"action_type": "inspect_alert", "target": "a1"}


def test_parse_json_action_with_code_fence_and_trailing_text() -> None:
    raw = """
```json
{"action_type":"assign_team","content":"auth-oncall"}
```
extra explanation
"""
    payload = _parse_json_action(raw)
    assert payload == {"action_type": "assign_team", "content": "auth-oncall"}


def test_parse_json_action_with_single_quotes() -> None:
    payload = _parse_json_action("{'action_type': 'set_severity', 'content': 'SEV-1'}")
    assert payload == {"action_type": "set_severity", "content": "SEV-1"}


def test_parse_function_call_action() -> None:
    payload = _parse_json_action("submit_root_cause('clock skew on auth node')")
    assert payload == {"action_type": "submit_root_cause", "content": "clock skew on auth node"}


def test_parse_key_value_action() -> None:
    payload = _parse_json_action("action_type=inspect_log, target=ha1_log_clock_skew_node7")
    assert payload == {"action_type": "inspect_log", "target": "ha1_log_clock_skew_node7"}


def test_parse_alias_action_name() -> None:
    payload = _parse_json_action("resolve()")
    assert payload == {"action_type": "resolve_incident"}


def test_fallback_avoids_early_resolution_when_fact_count_low() -> None:
    observation = _base_observation().model_copy(
        update={
            "selected_severity": "SEV-1",
            "assigned_team": "auth-oncall",
            "submitted_root_cause": "clock skew",
            "submitted_mitigation": "restore ntp",
            "known_facts": ["fact 1", "fact 2"],
        }
    )
    action = _fallback_action(observation)
    assert action.action_type == ActionType.ADD_NOTE
