# RunbookOps: CaseOps Benchmark Summary

## One-Line Summary

RunbookOps is a deterministic OpenEnv-style environment that evaluates whether an agent can handle realistic operational cases safely under partial information.

## Why It Matters

Operational case handling is a high-value real-world agent task with concrete success criteria. RunbookOps goes beyond toy environments by testing evidence-driven reasoning, ownership routing, mitigation choice, and safe closure behavior across customer-impact workflows.

## What Is Included

- 15 fully synthetic, offline scenarios.
  - 5 easy, 5 medium, 5 hard.
- Coverage across account access, order completion, payments, message delivery, catalog freshness, and partner integration cases.
- Typed action/observation/state models (Pydantic).
- Deterministic environment API:
  - `reset()`
  - `step(action)`
  - `state()`
- Deterministic, continuous rubric-based grader with validator-safe published score in `(0.0, 1.0)`.
- Score variance comes from evidence quality, investigation selectivity, and safe closure behavior rather than randomization or LLM judging.
- Dense trajectory reward with partial-progress signals.
- FastAPI server endpoints for local and container deployment.
- Baseline `inference.py` using OpenAI client and required env vars.
- Validator-aligned structured stdout: exact `[START]`, `[STEP]`, `[END]` lines.
- Heuristic fallback derives decisions from visible evidence, not a scenario-id answer map.
- Dockerfile and test suite.

## Judge Quick Path (2-3 minutes)

1. Start API and open `/docs`.
2. Call `POST /reset` with `easy_auth_token_expiry`.
3. Call `POST /step` with one inspect action, then `GET /state`.
4. Call `POST /grade` and verify score is strictly inside `(0, 1)`.
5. Run `python3 -m pytest` for deterministic validation coverage.

## Judging Alignment

- Real-world utility: evidence-based operational case handling.
- Task/grader quality: explicit objectives, deterministic grading, difficulty progression.
- Environment design: stateful evidence unlocking, safety-aware transitions, reward shaping.
- Spec-readiness: typed models, OpenEnv-style manifest, deployable API + Docker.

## Quick Verification

1. Open `/docs` and call `POST /reset`, `POST /step`, `GET /state`, `POST /grade`.
2. Run `pytest`.
3. Run `python3 scripts/smoke_test.py`.
4. Build/run Docker and verify `/health`.

## Key Files

- `server/environment.py`
- `grader.py`
- `openenv.yaml`
- `inference.py`
- `tests/`
