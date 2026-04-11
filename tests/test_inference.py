from __future__ import annotations

import json
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

from client import LocalRunbookOpsClient
from inference import _fallback_action, _parse_json_action, _planned_action
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


def test_openenv_env_url_alias_is_supported(monkeypatch) -> None:
    import inference

    monkeypatch.delenv("RUNBOOKOPS_BASE_URL", raising=False)
    monkeypatch.setenv("OPENENV_ENV_URL", "https://example-openenv.test")
    reloaded = importlib.reload(inference)
    try:
        assert reloaded.OPENENV_ENV_URL == "https://example-openenv.test"
        assert reloaded.ENV_BASE_URL == "https://example-openenv.test"
    finally:
        monkeypatch.delenv("OPENENV_ENV_URL", raising=False)
        importlib.reload(inference)


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
    assert all(
        line.startswith("[START] ") or line.startswith("[STEP] ") or line.startswith("[END] ")
        for line in lines
    )
    start_lines = [line for line in lines if line.startswith("[START] ")]
    step_lines = [line for line in lines if line.startswith("[STEP] ")]
    end_lines = [line for line in lines if line.startswith("[END] ")]

    assert len(start_lines) == 15
    assert len(end_lines) == 15
    assert len(step_lines) >= 15
    assert "task=easy_auth_token_expiry" in start_lines[0]
    assert "env=runbookops" in start_lines[0]
    assert "model=" in start_lines[0]
    assert "action=" in step_lines[0]
    assert "reward=" in step_lines[0]
    assert "done=" in step_lines[0]
    assert "error=" in step_lines[0]
    assert "success=" in end_lines[-1]
    assert "steps=" in end_lines[-1]
    assert "rewards=" in end_lines[-1]
    assert lines.index(start_lines[0]) < lines.index(step_lines[0]) < lines.index(end_lines[0])


def test_structured_lines_use_validator_friendly_values() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ("MODEL_NAME", "HF_TOKEN", "API_KEY", "OPENAI_API_KEY", "RUNBOOKOPS_BASE_URL"):
        env.pop(key, None)
    env["RESULT_PATH"] = str(root / "artifacts" / "test_validator_values.json")

    completed = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    first_step = next(line for line in lines if line.startswith("[STEP] "))
    first_end = next(line for line in lines if line.startswith("[END] "))

    assert re.search(r"reward=-?\d+\.\d{2}\b", first_step)
    assert re.search(r"done=(true|false)\b", first_step)
    assert re.search(r"error=([^\s]+|null)\b", first_step)
    assert re.search(r"success=(true|false)\b", first_end)
    assert re.search(r"steps=\d+\b", first_end)
    assert re.search(r"rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*|)\b", first_end)


def test_planner_resolves_without_late_duplicate_inspects_on_hard_case() -> None:
    client = LocalRunbookOpsClient()
    observation = client.reset("hard_search_index_pipeline_failure")
    actions: list[str] = []

    for _ in range(12):
        action = _planned_action(observation)
        actions.append(action.action_type.value)
        result = client.step(action)
        observation = result.observation
        if result.done:
            break

    assert actions[-1] == "resolve_incident"
    assert actions.count("inspect_alert") == 1
    assert actions.count("inspect_log") == 2
