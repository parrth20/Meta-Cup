from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

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
    description="Deterministic operational case handling benchmark for evidence-based triage and safe resolution",
    version="0.1.0",
)

env = RunbookOpsEnvironment()


def _root_payload() -> dict[str, object]:
    return {
        "name": "RunbookOps",
        "status": "ok",
        "tagline": "CaseOps benchmark for operational case handling",
        "docs": "/docs",
        "health": "/health",
    }


def _landing_page_html() -> str:
    difficulty_counts = env.list_tasks()
    total_scenarios = len(env.scenarios)
    easy_count = difficulty_counts["easy"]["scenario_count"]
    medium_count = difficulty_counts["medium"]["scenario_count"]
    hard_count = difficulty_counts["hard"]["scenario_count"]
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RunbookOps - CaseOps Benchmark</title>
    <style>
      :root {{
        --bg: #08111f;
        --panel: rgba(10, 20, 38, 0.72);
        --panel-strong: rgba(12, 24, 44, 0.88);
        --text: #ebf2ff;
        --muted: #b7c5df;
        --line: rgba(150, 180, 220, 0.18);
        --accent: #5eead4;
        --accent-strong: #38bdf8;
        --warm: #f59e0b;
        --success: #86efac;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        min-height: 100vh;
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 32%),
          radial-gradient(circle at top right, rgba(94, 234, 212, 0.16), transparent 30%),
          linear-gradient(180deg, #07101d 0%, #0b1324 48%, #111a2f 100%);
      }}
      a {{ color: inherit; text-decoration: none; }}
      .shell {{
        width: min(1180px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 40px 0 56px;
      }}
      .hero {{
        display: grid;
        grid-template-columns: 1.4fr 0.9fr;
        gap: 24px;
        align-items: stretch;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 28px;
        backdrop-filter: blur(18px);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
      }}
      .hero-copy {{
        padding: 36px;
      }}
      .eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(94, 234, 212, 0.12);
        border: 1px solid rgba(94, 234, 212, 0.25);
        color: var(--accent);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
      }}
      h1 {{
        margin: 18px 0 14px;
        font-size: clamp(38px, 6vw, 64px);
        line-height: 0.94;
        letter-spacing: -0.04em;
      }}
      .lede {{
        max-width: 62ch;
        margin: 0 0 24px;
        color: var(--muted);
        font-size: 17px;
        line-height: 1.65;
      }}
      .cta-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 24px 0 28px;
      }}
      .button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        min-width: 150px;
        padding: 14px 18px;
        border-radius: 16px;
        font-weight: 700;
        border: 1px solid transparent;
      }}
      .button.primary {{
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
        color: #08111f;
      }}
      .button.secondary {{
        background: rgba(255, 255, 255, 0.04);
        border-color: var(--line);
        color: var(--text);
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }}
      .metric {{
        padding: 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
      }}
      .metric .value {{
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.04em;
      }}
      .metric .label {{
        margin-top: 4px;
        color: var(--muted);
        font-size: 13px;
      }}
      .hero-side {{
        padding: 28px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }}
      .badge {{
        display: inline-flex;
        width: fit-content;
        padding: 7px 11px;
        border-radius: 999px;
        background: rgba(134, 239, 172, 0.14);
        border: 1px solid rgba(134, 239, 172, 0.26);
        color: var(--success);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}
      .card-title {{
        margin: 0;
        font-size: 22px;
        font-weight: 800;
      }}
      .bullet-list {{
        margin: 0;
        padding-left: 18px;
        color: var(--muted);
        line-height: 1.7;
      }}
      .surface {{
        margin-top: 24px;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
      }}
      .surface-card {{
        padding: 22px;
      }}
      .surface-card h3 {{
        margin: 0 0 10px;
        font-size: 18px;
      }}
      .surface-card p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.65;
        font-size: 14px;
      }}
      .route-list {{
        margin-top: 12px;
        display: grid;
        gap: 10px;
      }}
      .route {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
      }}
      .route code {{
        color: var(--accent);
        font-size: 13px;
      }}
      .route span {{
        color: var(--muted);
        font-size: 13px;
      }}
      @media (max-width: 980px) {{
        .hero, .surface {{
          grid-template-columns: 1fr;
        }}
        .metrics {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
      }}
      @media (max-width: 640px) {{
        .shell {{
          width: min(100vw - 20px, 1180px);
          padding-top: 20px;
        }}
        .hero-copy, .hero-side, .surface-card {{
          padding: 22px;
        }}
        .metrics {{
          grid-template-columns: 1fr 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <div class="panel hero-copy">
          <div class="eyebrow">OpenEnv RL Challenge</div>
          <h1>RunbookOps<br />CaseOps Benchmark</h1>
          <p class="lede">
            A deterministic benchmark for operational case handling. Agents must gather evidence,
            classify severity, route ownership, identify cause, choose a safe resolution, and close
            customer-impact cases without shallow shortcuts.
          </p>
          <div class="cta-row">
            <a class="button primary" href="/docs">Open API Docs</a>
            <a class="button secondary" href="/scenarios">Browse Scenarios</a>
            <a class="button secondary" href="/health">Health Check</a>
          </div>
          <div class="metrics">
            <div class="metric">
              <div class="value">{total_scenarios}</div>
              <div class="label">total cases</div>
            </div>
            <div class="metric">
              <div class="value">{easy_count}</div>
              <div class="label">easy</div>
            </div>
            <div class="metric">
              <div class="value">{medium_count}</div>
              <div class="label">medium</div>
            </div>
            <div class="metric">
              <div class="value">{hard_count}</div>
              <div class="label">hard</div>
            </div>
          </div>
        </div>
        <aside class="panel hero-side">
          <div class="badge">Deterministic and Offline</div>
          <h2 class="card-title">Why this benchmark matters</h2>
          <ul class="bullet-list">
            <li>Real operational work instead of a toy game loop</li>
            <li>No LLM-as-judge scoring or hidden internet dependencies</li>
            <li>Evidence, routing, diagnosis, and safe closure all matter</li>
            <li>Continuous deterministic grading with meaningful score variance</li>
          </ul>
          <div class="route-list">
            <a class="route" href="/reset"><code>POST /reset</code><span>start a case</span></a>
            <a class="route" href="/step"><code>POST /step</code><span>advance one action</span></a>
            <a class="route" href="/grade"><code>POST /grade</code><span>score current episode</span></a>
            <a class="route" href="/tasks"><code>GET /tasks</code><span>list task groups</span></a>
          </div>
        </aside>
      </section>

      <section class="surface">
        <article class="panel surface-card">
          <h3>Evidence-Based Resolution</h3>
          <p>
            Cases expose alerts, logs, workflow playbooks, and timeline notes gradually. The
            agent must collect enough evidence before proposing cause and mitigation.
          </p>
        </article>
        <article class="panel surface-card">
          <h3>Judge-Friendly Structure</h3>
          <p>
            Typed models, FastAPI endpoints, Docker deployment, deterministic grading, and a
            baseline inference runner aligned with the OpenEnv submission contract.
          </p>
        </article>
        <article class="panel surface-card">
          <h3>Broader Than Infra Ops</h3>
          <p>
            RunbookOps is framed as operational case handling across access failures, order issues,
            payment exceptions, message delivery, search freshness, and integration regressions.
          </p>
        </article>
      </section>
    </main>
  </body>
</html>"""


@app.get("/", response_model=None)
def root(request: Request):
    accepts = request.headers.get("accept", "")
    if "application/json" in accepts:
        return JSONResponse(_root_payload())
    return HTMLResponse(_landing_page_html())


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


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
