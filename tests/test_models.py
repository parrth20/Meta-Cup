from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from models import Action, ActionType, Scenario


def test_action_requires_target_for_inspect() -> None:
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.INSPECT_ALERT)


def test_action_requires_content_for_setters() -> None:
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.SET_SEVERITY)


def test_action_valid_shape() -> None:
    action = Action(action_type=ActionType.ASSIGN_TEAM, content="auth-oncall")
    assert action.content == "auth-oncall"


def test_scenario_model_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "scenarios" / "easy" / "easy_auth_token_expiry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenario = Scenario.model_validate(payload)

    assert scenario.scenario_id == "easy_auth_token_expiry"
    assert scenario.true_severity.value.startswith("SEV-")
    assert len(scenario.required_evidence_ids) >= 3
