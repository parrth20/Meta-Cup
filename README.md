---
title: RunbookOps - CaseOps Benchmark
emoji: "🛠️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
base_path: /docs
tags:
  - openenv
pinned: false
---

# runbookops-caseop

RunbookOps is a deterministic OpenEnv-style benchmark for **operational case handling**. Each episode is a realistic service issue that affects customer access, orders, payments, messages, catalog freshness, or partner integrations. The agent must gather evidence, assess severity, route ownership, identify cause, choose a safe resolution, and close the case responsibly.

For fast review:

1. Read `PROJECT_SUMMARY.md` (judge-facing overview).
2. Read this README (execution + design).
3. Read `TECHNICAL_NOTES.md` (determinism, grading, reward logic).

## Why Judges Should Care

Most agent benchmarks reward broad reasoning. RunbookOps evaluates whether an agent can handle **real operational work** under partial information without unsafe shortcuts.

- Real-world domain: customer-impact service issue handling.
- Offline and reproducible: no internet or external SaaS dependencies.
- Deterministic scoring: no LLM-as-judge.
- Safety-sensitive behavior: premature closure is penalized.

## Why This Matters

Teams increasingly want agents that can do more than chat: they need agents that can work through messy operational cases, synthesize evidence, avoid shallow guesses, and recommend safe next actions.

RunbookOps is designed around that broader audience:

- account access cases
- order completion exceptions
- payment routing and settlement issues
- message delivery failures
- catalog freshness and search quality cases
- integration and configuration regressions

## Round 1 Alignment

- Real-world utility: evidence-based operational case handling.
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
5. `grade()` computes final score from rubric components and returns a validator-safe value strictly inside `(0,1)`.
6. FastAPI exposes the environment through `/reset`, `/step`, `/state`, `/grade`.

### Core Files

- `models.py`: Pydantic models, enums, and API payload types.
- `server/environment.py`: deterministic dynamics and rewards.
- `grader.py`: deterministic rubric-based scorer.
- `server/app.py`: FastAPI endpoints.
- `inference.py`: OpenAI-client baseline agent runner.

## Task Families

15 total operational cases across three tiers.

| Tier | Count | Typical Steps | Characteristics |
|---|---:|---:|---|
| easy | 5 | 4-8 | One dominant cause, low ambiguity, quick evidence synthesis |
| medium | 5 | 6-10 | Two plausible explanations, cross-source synthesis needed |
| hard | 5 | 8-12 | Conflicting signals, multiple false leads, shallow closure punished |

Example services covered: `auth`, `checkout`, `payments`, `email`, `search`, `notifications`, `platform`.

Representative case themes:

- account access failures after credential or rollout changes
- order completion issues during policy or dependency changes
- payment exceptions caused by routing or experiment leaks
- message delivery failures with configuration/provider ambiguity
- catalog freshness cases after propagation or schema issues

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
| Case context | scenario id, title, difficulty, service, case summary |
| Visible evidence | unlocked alerts/logs/runbooks/timeline notes |
| Agent working memory | known facts + action history summary |
| Decision state | selected severity, assigned team, submitted root cause/mitigation |
| Episode status | steps taken/remaining, done flag, last action result |

Hidden ground truth values are not exposed in observations.

Internal schema note: the environment keeps the action name `inspect_runbook`, but judge-facing documentation treats these snippets as **workflow playbooks** for broader readability.

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

Implementation note: while rubric components remain intuitive `0.0-1.0` signals, the final published task score is epsilon-clamped into `(0,1)` to satisfy strict validator parsing rules that reject exact boundary values.

## Sample Episode Walkthrough

Scenario: `easy_auth_token_expiry`

Plain-language framing: an account access case opens after a scheduled credential rollover. Customers suddenly cannot sign in, and the agent needs to work through the evidence before closing the case.

1. `inspect_alert` -> `ea1_alert_401_spike`
2. `inspect_log` -> `ea1_log_jwt_expired`
3. `inspect_runbook` -> `ea1_runbook_key_rotation`
4. `set_severity` -> `SEV-2`
5. `assign_team` -> `auth-oncall`
6. `submit_root_cause` -> `expired auth signing key`
7. `submit_mitigation` -> `rotate signing key and restart issuer`
8. `resolve_incident`

Expected: terminal reason `resolved_safely`, score close to `1.0`.

## What Makes This Realistic

- Evidence appears gradually instead of all at once.
- Some cases contain misleading but plausible false leads.
- Correct labels alone are not enough; the agent must gather enough evidence.
- Safe closure matters: resolving too early is penalized.
- Final scores are deterministic and reproducible across runs.

## Why This Is Broader Than Infra Ops

Although the evidence includes alerts, logs, and runbooks, the benchmark is not just about infrastructure firefighting. It models the wider workflow that operations, support, platform, and reliability teams handle every day: customer-impact cases that require evidence review, routing, diagnosis, mitigation, and safe closure.

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

Important for `/reset` in Swagger UI:
- Replace the default `"scenario_id": "string"` placeholder with a real id (for example `easy_auth_token_expiry`).
- Use `GET /scenarios` first to list valid ids.

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
- `LOCAL_IMAGE_NAME`
- `RUNBOOKOPS_BASE_URL`
- `OPENENV_ENV_URL`
- `MAX_STEPS`, `TEMPERATURE`, `MAX_TOKENS`, `RESULT_PATH`

Run:

```bash
python3 inference.py
```

Validator-safe behavior:

- If `RUNBOOKOPS_BASE_URL` is unset, `OPENENV_ENV_URL` can be used as an optional alias for the environment endpoint.
- If neither environment URL is set, or the configured endpoint is unreachable, `inference.py` falls back to the local in-process environment.
- If `HF_TOKEN` or other API credentials are missing, `inference.py` falls back to a deterministic planner-only baseline instead of exiting with a non-zero status.
- When credentials are present, the script initializes the OpenAI client and records `inference_mode: "openai_client"` in the output JSON.
- Stdout uses the exact validator-facing bracketed format with `[START]`, `[STEP]`, and `[END]` records on single lines.
- The deterministic safety fallback reasons from visible evidence tokens and service/runbook context; it does not rely on a scenario-id answer lookup table.

Round 1 recommended run (saves reproducible artifact):

```bash
mkdir -p artifacts
export RUNBOOKOPS_BASE_URL="https://<your-space>.hf.space"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="<your_token>"
export RESULT_PATH="artifacts/inference_live_$(date +%Y%m%d_%H%M%S).json"
python3 inference.py | tee artifacts/inference_live_stdout.txt
```

Output:

- one `[START] task=... env=runbookops model=...` line per scenario
- multiple `[STEP] step=... action=... reward=0.00 done=true|false error=...` lines across each episode
- one `[END] success=true|false steps=... rewards=r1,r2,...,rn` line per scenario
- JSON summary file (default: `baseline_results.json`)

Exact stdout contract used by the baseline:

```text
[START] task=easy_auth_token_expiry env=runbookops model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action=inspect_alert('ea1_alert_401_spike') reward=0.03 done=false error=null
[STEP] step=2 action=inspect_log('ea1_log_jwt_expired') reward=0.04 done=false error=null
[END] success=true steps=8 rewards=0.03,0.04,0.10,0.10,0.20,0.20,0.20,0.00
```

Environment variable expectations:

- `API_BASE_URL`: required by the hackathon contract, includes a safe default.
- `MODEL_NAME`: required by the hackathon contract, includes a safe default.
- `HF_TOKEN`: mandatory secret for real LLM-backed runs, with no default.
- `LOCAL_IMAGE_NAME`: optional and only used if a containerized local model workflow is introduced later.
- `RUNBOOKOPS_BASE_URL`: optional preferred environment endpoint variable for this project.
- `OPENENV_ENV_URL`: optional alias for environment endpoint compatibility and convenience.

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

## Validator Notes

This repo is tuned to avoid the specific Round 1 failure modes called out in the hackathon guide:

- `inference.py` is at the repository root.
- `API_BASE_URL` and `MODEL_NAME` are read with defaults.
- `HF_TOKEN` is read without a default.
- `OPENENV_ENV_URL` is supported only as an optional alias, not as a required contract variable.
- All LLM-backed calls go through `from openai import OpenAI`.
- Stdout is limited to exact `[START]`, `[STEP]`, and `[END]` records.
- Published task scores are always strictly inside `(0,1)`.
- The Hugging Face Space can be pointed at `/docs` by default via README front matter `base_path: /docs`.

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

- [x] All 15 scenarios load (`5 easy / 5 medium / 5 hard`).
- [x] `python3 -m pytest` passes.
- [x] `scripts/smoke_test.py` resolves a scenario and returns a valid grade.
- [x] `/health`, `/reset`, `/step`, `/state`, `/grade` work from `/docs`.
- [x] `inference.py` runs with required env vars and writes score summary JSON.
- [x] `inference.py` emits exact `[START]`, `[STEP]`, `[END]` structured stdout lines.
- [x] Published task scores are strictly inside `(0,1)` to satisfy validator parsing.
- [x] Docker image builds and serves `/health`.
- [x] `openenv.yaml` and implementation behavior are aligned.

## Limitations

- Single active in-memory episode per API instance.
- Text matching is deterministic but still bounded by handcrafted synonym rules.

## Future Work

- Multi-session API with explicit episode ids.
- Escalation/escalation-policy simulation (handoffs and approvals).
- Expanded adversarial scenario suite for anti-gaming evaluation.
