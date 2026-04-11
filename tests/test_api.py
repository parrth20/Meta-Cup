from __future__ import annotations

import json

from fastapi.testclient import TestClient

from server.app import app, env
from server.environment import RunbookOpsEnvironment


client = TestClient(app)


def test_root_html_landing_page() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RunbookOps" in response.text
    assert "CaseOps Benchmark" in response.text


def test_root_json_payload_for_api_clients() -> None:
    response = client.get("/", headers={"accept": "application/json"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "RunbookOps"
    assert payload["docs"] == "/docs"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["scenarios_loaded"] == 15


def test_reset_step_state_grade_flow() -> None:
    reset_response = client.post("/reset", json={"scenario_id": "easy_auth_token_expiry"})
    assert reset_response.status_code == 200

    step_response = client.post(
        "/step",
        json={"action_type": "inspect_alert", "target": "ea1_alert_401_spike"},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert "reward" in step_payload

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["scenario_id"] == "easy_auth_token_expiry"

    grade_response = client.post("/grade")
    assert grade_response.status_code == 200
    grade_payload = grade_response.json()
    assert 0.0 <= grade_payload["score"] <= 1.0


def test_invalid_action_returns_invalid_flag() -> None:
    client.post("/reset", json={"scenario_id": "easy_auth_token_expiry"})
    response = client.post(
        "/step",
        json={"action_type": "inspect_log", "target": "does_not_exist"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["info"]["invalid_action"] is True


def test_state_before_reset_returns_400() -> None:
    env._state = None  # reset singleton env state for this precondition test
    response = client.get("/state")
    assert response.status_code == 400


def test_grade_scenario_mismatch_returns_400() -> None:
    client.post("/reset", json={"scenario_id": "easy_auth_token_expiry"})
    response = client.post("/grade", json={"scenario_id": "easy_search_cache_stale"})
    assert response.status_code == 400


def test_initial_observation_does_not_expose_hidden_truth_strings() -> None:
    env = RunbookOpsEnvironment()
    scenario = env.scenario_map["easy_auth_token_expiry"]
    observation = env.reset(scenario_id=scenario.scenario_id)
    serialized = json.dumps(observation.model_dump()).lower()

    assert scenario.true_root_cause.lower() not in serialized
    assert scenario.true_mitigation.lower() not in serialized


def test_score_alias_matches_grade_endpoint() -> None:
    client.post("/reset", json={"scenario_id": "easy_auth_token_expiry"})
    grade_resp = client.post("/grade")
    score_resp = client.post("/score")

    assert grade_resp.status_code == 200
    assert score_resp.status_code == 200
    assert 0.0 <= grade_resp.json()["score"] <= 1.0
    assert 0.0 <= score_resp.json()["score"] <= 1.0
