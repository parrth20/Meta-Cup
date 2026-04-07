from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from inference import _fallback_action, _parse_json_action
from models import ActionType, Difficulty, EvidenceType, Observation, PublicEvidence


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


def test_fallback_prefers_more_evidence_before_resolution_when_available() -> None:
    observation = _base_observation().model_copy(
        update={
            "selected_severity": "SEV-1",
            "assigned_team": "auth-oncall",
            "submitted_root_cause": "clock skew",
            "submitted_mitigation": "restore ntp",
            "known_facts": ["fact 1", "fact 2"],
            "visible_alerts": [
                PublicEvidence(
                    id="ha1_alert_nonce_mismatch",
                    type=EvidenceType.ALERT,
                    title="Nonce mismatch spike",
                    content="Mismatch errors increasing",
                    tags=["auth"],
                )
            ],
        }
    )
    action = _fallback_action(observation)
    assert action.action_type == ActionType.INSPECT_ALERT


def test_default_model_name_is_present_for_validator_safety() -> None:
    import inference

    assert inference.MODEL_NAME
    assert isinstance(inference.MODEL_NAME, str)


def test_hf_token_symbol_exists_without_default_requirement() -> None:
    import inference

    assert hasattr(inference, "HF_TOKEN")


def test_inference_stdout_uses_structured_markers() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in ("MODEL_NAME", "HF_TOKEN", "API_KEY", "OPENAI_API_KEY", "RUNBOOKOPS_BASE_URL"):
        env.pop(key, None)
    env["RESULT_PATH"] = str(artifacts_dir / "test_validator_stdout.json")

    completed = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines[0].startswith("START ")
    assert lines[-1].startswith("END ")
    step_lines = [line for line in lines if line.startswith("STEP ")]
    assert len(step_lines) == 15

    start_payload = json.loads(lines[0].split(" ", 1)[1])
    end_payload = json.loads(lines[-1].split(" ", 1)[1])
    assert start_payload["model_name"]
    assert end_payload["scenario_count"] == 15
