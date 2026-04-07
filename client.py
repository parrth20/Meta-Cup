from __future__ import annotations

from typing import Any, Optional

import requests

from models import Action, GraderResult, InternalStateSnapshot, Observation, ScenarioSummary, StepResult
from server.environment import RunbookOpsEnvironment


class RunbookOpsClient:
    """Lightweight HTTP client for RunbookOps FastAPI endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, json_payload: Optional[dict[str, Any]] = None) -> Any:
        response = self._session.request(
            method=method,
            url=self._url(path),
            json=json_payload,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{method} {path} failed ({response.status_code}): {detail}")

        if not response.text:
            return None
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def reset(self, scenario_id: Optional[str] = None, difficulty: Optional[str] = None) -> Observation:
        payload: dict[str, Any] = {}
        if scenario_id:
            payload["scenario_id"] = scenario_id
        if difficulty:
            payload["difficulty"] = difficulty
        data = self._request("POST", "/reset", payload or None)
        return Observation.model_validate(data)

    def step(self, action: Action | dict[str, Any]) -> StepResult:
        payload = action.model_dump() if isinstance(action, Action) else action
        data = self._request("POST", "/step", payload)
        return StepResult.model_validate(data)

    def state(self) -> InternalStateSnapshot:
        data = self._request("GET", "/state")
        return InternalStateSnapshot.model_validate(data)

    def tasks(self) -> dict[str, dict[str, Any]]:
        return self._request("GET", "/tasks")

    def scenarios(self) -> list[ScenarioSummary]:
        data = self._request("GET", "/scenarios")
        return [ScenarioSummary.model_validate(item) for item in data]

    def grade(self, scenario_id: Optional[str] = None) -> GraderResult:
        payload = {"scenario_id": scenario_id} if scenario_id else None
        data = self._request("POST", "/grade", payload)
        return GraderResult.model_validate(data)


class LocalRunbookOpsClient:
    """In-process client used when no HTTP endpoint is configured."""

    def __init__(self, scenarios_dir: Optional[str] = None) -> None:
        self._env = RunbookOpsEnvironment(scenarios_dir=scenarios_dir)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "environment": "RunbookOps",
            "scenarios_loaded": len(self._env.scenarios),
            "transport": "local",
        }

    def reset(self, scenario_id: Optional[str] = None, difficulty: Optional[str] = None) -> Observation:
        return self._env.reset(scenario_id=scenario_id, difficulty=difficulty)

    def step(self, action: Action | dict[str, Any]) -> StepResult:
        payload = action if isinstance(action, Action) else Action.model_validate(action)
        return self._env.step(payload)

    def state(self) -> InternalStateSnapshot:
        return self._env.state()

    def tasks(self) -> dict[str, dict[str, Any]]:
        return self._env.list_tasks()

    def scenarios(self) -> list[ScenarioSummary]:
        return self._env.list_scenarios()

    def grade(self, scenario_id: Optional[str] = None) -> GraderResult:
        snapshot = self._env.state()
        if scenario_id and snapshot.scenario_id != scenario_id:
            raise RuntimeError(
                "Active scenario does not match scenario_id in request. Call reset() first."
            )
        return self._env.grade_current_episode()
