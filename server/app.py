from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException

from models import (
    Action,
    GradeRequest,
    GraderResult,
    InternalStateSnapshot,
    Observation,
    ResetRequest,
    ScenarioSummary,
    StepResult,
)
from server.environment import RunbookOpsEnvironment

app = FastAPI(
    title="RunbookOps API",
    description="Deterministic incident triage and runbook operations environment",
    version="0.1.0",
)

env = RunbookOpsEnvironment()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "environment": "RunbookOps",
        "scenarios_loaded": len(env.scenarios),
    }


@app.post("/reset", response_model=Observation)
def reset(request: Optional[ResetRequest] = None) -> Observation:
    request = request or ResetRequest()
    try:
        return env.reset(scenario_id=request.scenario_id, difficulty=request.difficulty)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    try:
        return env.step(action)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=InternalStateSnapshot)
def state() -> InternalStateSnapshot:
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tasks")
def tasks() -> dict[str, dict[str, object]]:
    return env.list_tasks()


@app.get("/scenarios", response_model=list[ScenarioSummary])
def scenarios() -> list[ScenarioSummary]:
    return env.list_scenarios()


@app.post("/grade", response_model=GraderResult)
def grade(request: Optional[GradeRequest] = None) -> GraderResult:
    request = request or GradeRequest()
    try:
        if request.scenario_id and env.state().scenario_id != request.scenario_id:
            raise ValueError(
                "Active scenario does not match scenario_id in request. Call /reset first."
            )
        return env.grade_current_episode()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/score", response_model=GraderResult)
def score(request: Optional[GradeRequest] = None) -> GraderResult:
    return grade(request)
