# RunbookOps

RunbookOps is a deterministic OpenEnv-style environment for incident triage and runbook-driven resolution in a fictional SaaS company. Each episode simulates a production incident where an agent must gather evidence, classify severity, route ownership, identify root cause, choose mitigation, and close safely.

For fast review:

1. Read `PROJECT_SUMMARY.md` (judge-facing overview).
2. Read this README (execution + design).
3. Read `TECHNICAL_NOTES.md` (determinism, grading, reward logic).

## Why Judges Should Care

Most agent benchmarks reward broad reasoning. RunbookOps evaluates operational reliability: can the agent follow a realistic incident workflow under partial information without unsafe shortcuts?

- Real-world domain: on-call incident response.
- Offline and reproducible: no internet or external SaaS dependencies.
- Deterministic scoring: no LLM-as-judge.
- Safety-sensitive behavior: premature resolution is penalized.

## Round 1 Alignment

- Real-world utility: incident triage and runbook operations.
- OpenEnv-like API: typed `Action`, `Observation`, `StepResult`, plus `reset()`, `step()`, `state()`.
- 3 difficulty tiers with deterministic graders: `easy`, `medium`, `hard`.
- Dense trajectory reward + independent final rubric score.
- FastAPI server + Dockerfile + baseline inference script.

## Architecture

### Runtime Flow

1. Scenario JSON defines hidden ground truth and evidence graph.
2. `RunbookOpsEnvironment` loads scenarios and creates episode state.
3. `reset()` returns partial initial observation.
4. `step(action)` applies deterministic transition + reward shaping.
5. `grade()` computes final score in `[0,1]` from rubric components.
6. FastAPI exposes the environment through `/reset`, `/step`, `/state`, `/grade`.

### Core Files

- `models.py`: Pydantic models, enums, and API payload types.
- `server/environment.py`: deterministic dynamics and rewards.
- `grader.py`: deterministic rubric-based scorer.
- `server/app.py`: FastAPI endpoints.
- `inference.py`: OpenAI-client baseline agent runner.

## Task Families

15 total scenarios across three tiers.

| Tier | Count | Typical Steps | Characteristics |
|---|---:|---:|---|
| easy | 5 | 4-8 | Single dominant cause, minimal red herrings |
| medium | 5 | 6-10 | Multiple plausible causes, cross-source synthesis needed |
| hard | 5 | 8-12 | Conflicting signals, multiple false leads, shallow closure punished |

Example services covered: `auth`, `checkout`, `payments`, `email`, `search`, `notifications`, `platform`.

## Action Space

| Field | Type | Required | Notes |
|---|---|---|---|
| `action_type` | enum | yes | one of inspect/set/assign/submit/resolve actions |
| `target` | string | inspect only | evidence id |
| `content` | string | setter/submit/note actions | free text or enum-like values |

Supported actions:

- `inspect_alert`
- `inspect_log`
- `inspect_runbook`
- `inspect_timeline_note`
- `set_severity` (`SEV-1`, `SEV-2`, `SEV-3`)
- `assign_team`
- `submit_root_cause`
- `submit_mitigation`
- `add_note`
- `resolve_incident`

## Observation Space

| Field Group | Description |
|---|---|
| Incident context | scenario id, title, difficulty, service, incident summary |
| Visible evidence | unlocked alerts/logs/runbooks/timeline notes |
| Agent working memory | known facts + action history summary |
| Decision state | selected severity, assigned team, submitted root cause/mitigation |
| Episode status | steps taken/remaining, done flag, last action result |

Hidden ground truth values are not exposed in observations.

## Reward Design (Trajectory Signal)

Reward is separate from final grader score.

- `+0.03` first-time relevant evidence inspect
- `+0.01` first-time neutral inspect
- `+0.015 / +0.02` evidence coverage milestones
- `-0.02` duplicate relevant inspect
- `-0.03` repeated irrelevant/invalid inspect
- `+0.10` correct severity
- `+0.10` correct owner team
- `+0.20` correct root cause
- `+0.20` correct mitigation
- `+0.20` safe resolution
- `-0.20` unsafe resolution baseline penalty
- additional penalties for wrong root cause, very low evidence, repetitive loops
- `-0.005` living penalty each step

## Grader Design (Final Score)

Deterministic rubric in `grader.py`:

| Component | Weight |
|---|---:|
| severity correctness | 0.15 |
| owner team correctness | 0.15 |
| root cause correctness | 0.30 |
| mitigation correctness | 0.25 |
| evidence coverage | 0.10 |
| safe resolution behavior | 0.05 |

Text matching is deterministic (normalization, controlled synonyms, overlap/fuzzy thresholds, negation conflict checks).

## Sample Episode Walkthrough

Scenario: `easy_auth_token_expiry`

1. `inspect_alert` -> `ea1_alert_401_spike`
2. `inspect_log` -> `ea1_log_jwt_expired`
3. `inspect_runbook` -> `ea1_runbook_key_rotation`
4. `set_severity` -> `SEV-2`
5. `assign_team` -> `auth-oncall`
6. `submit_root_cause` -> `expired auth signing key`
7. `submit_mitigation` -> `rotate signing key and restart issuer`
8. `resolve_incident`

Expected: terminal reason `resolved_safely`, score close to `1.0`.

## API Endpoints

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /scenarios`
- `POST /grade`
- `POST /score` (alias)

OpenAPI docs: `http://127.0.0.1:8000/docs`

## Local Setup

Recommended runtime: Python `3.11+`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

If editable install is blocked by your Python setup, use:

```bash
python3 -m pip install '.[dev]'
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Tests and Smoke Validation

```bash
python3 -m pytest
python3 scripts/smoke_test.py
python3 scripts/export_task_summary.py
```

## Baseline Inference

`inference.py` uses the OpenAI Python client and runs across all scenarios.

Required env vars:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `API_KEY`
- `OPENAI_API_KEY`
- `RUNBOOKOPS_BASE_URL`
- `MAX_STEPS`, `TEMPERATURE`, `MAX_TOKENS`, `RESULT_PATH`

Run:

```bash
python3 inference.py
```

Output:

- per-scenario table
- per-difficulty aggregate table
- overall mean score
- JSON summary file (default: `baseline_results.json`)

## Docker

Build:

```bash
docker build -t runbookops:latest .
```

Run:

```bash
docker run --rm -p 7860:7860 runbookops:latest
```

Check:

- `http://127.0.0.1:7860/health`
- `http://127.0.0.1:7860/docs`

## Repository Layout

```text
runbookops/
├── PROJECT_SUMMARY.md
├── TECHNICAL_NOTES.md
├── README.md
├── openenv.yaml
├── inference.py
├── models.py
├── grader.py
├── client.py
├── scenarios/
├── server/
├── tests/
└── scripts/
```

## Submission Checklist



- [ ] All 15 scenarios load (`5 easy / 5 medium / 5 hard`).
- [ ] `python3 -m pytest` passes.
- [ ] `scripts/smoke_test.py` resolves a scenario and returns a valid grade.
- [ ] `/health`, `/reset`, `/step`, `/state`, `/grade` work from `/docs`.
- [ ] `inference.py` runs with required env vars and writes score summary JSON.
- [ ] Docker image builds and serves `/health`.
- [ ] `openenv.yaml` and implementation behavior are aligned.

## Limitations

- Single active in-memory episode per API instance.
- Text matching is deterministic but still bounded by handcrafted synonym rules.

## Future Work

- Multi-session API with explicit episode ids.
- Escalation/escalation-policy simulation (handoffs and approvals).
- Expanded adversarial scenario suite for anti-gaming evaluation.
