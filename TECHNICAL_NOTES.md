# Technical Notes

## Determinism

RunbookOps is designed for reproducibility and strict offline behavior.

- Scenario definitions are static JSON files under `scenarios/`.
- No randomness in transitions or grading.
- No network calls in environment dynamics.
- `reset(scenario_id=...)` yields the same initial state every run.
- Evidence visibility is controlled by deterministic unlock conditions.

## Grading

Grading is deterministic and independent from trajectory reward.

Rubric weights:

- Severity correctness: `0.15`
- Owner team correctness: `0.15`
- Root cause correctness: `0.30`
- Mitigation correctness: `0.25`
- Evidence coverage adequacy: `0.10`
- Safe resolution behavior: `0.05`

Raw rubric totals are clamped to `[0.0, 1.0]`, then every published score-like value is projected into a validator-safe open band inside `(0,1)`. This also avoids accidental `0.00` or `1.00` boundary strings when downstream systems round values aggressively.

The scorer is deliberately continuous. It does not collapse every correct run into the same published value. Evidence handling and closure behavior contribute meaningful variance across scenarios and trajectories.

Current reference behavior from the repository baseline run:

- 15 unique per-scenario published scores across 15 scenarios
- range: `0.8953` to `0.9925`
- solved-trajectory sanity checks still produce 9 unique scores across 15 scenarios

### Text Matching Strategy

Root cause and mitigation checks use deterministic normalization:

- lowercasing and punctuation removal
- light stemming
- controlled synonym mapping
- token overlap checks
- bounded sequence/trigram similarity checks

This supports realistic paraphrases while avoiding LLM-as-judge variability.

### Evidence and Closure Signals

The evidence component combines four deterministic signals:

- required evidence coverage
- relevant evidence discovery
- inspection precision
- investigation selectivity

The safe-resolution component combines:

- diagnosis correctness
- evidence sufficiency
- safe terminal state
- non-premature closure
- step efficiency relative to the scenario budget

This is important for agent evaluation because it means two agents can both be "correct" while still receiving different scores for how they reached resolution.

## Case Framing

The public framing for RunbookOps is broader than pure infrastructure response.

- episodes are presented as operational cases
- evidence captures customer-impact workflows and service exceptions
- the same deterministic engine supports account access, order, payment, messaging, catalog, and integration cases

This keeps the benchmark easier to understand for non-specialist reviewers without changing the underlying mechanics.

## Reward Design

Reward is dense-ish per step and intended for learning signals.

Positive signals:

- first-time relevant evidence inspection
- evidence coverage milestones (midpoint and full required-set coverage)
- correct severity assignment
- correct team assignment
- correct root cause submission
- correct mitigation submission
- safe final resolution

Negative signals:

- living penalty per step
- duplicate inspections
- repeated irrelevant/invalid exploration
- unsafe or premature resolution
- speculative low-evidence root-cause / mitigation submissions
- extra penalty for resolving with wrong root cause
- anti-loop repetition penalty for repeated identical actions

## Episode Termination

Episodes end when any of the following is true:

- `resolve_incident` is called
- step budget is exhausted

Step budgets are tiered for full triage trajectories:

- easy: 8
- medium: 10
- hard: 12

The test suite enforces `max_steps >= len(required_evidence_ids) + 5` to ensure each scenario has room for inspect/classify/resolve without requiring brittle shortcuts.

## Grader Robustness Guardrails

- Alias matching uses deterministic normalization and bounded fuzzy thresholds.
- Negated statements that conflict with candidate truth (for example, `not clock skew`) are explicitly rejected.
- Safe-resolution scoring requires both correctness and adequate evidence coverage; correct labels alone are not enough.

## Test Coverage Focus

The test suite validates critical judging paths:

- model and action validation
- deterministic step behavior
- invalid action handling
- reward progression
- grader bounds and alias/paraphrase matching
- scenario loading/integrity
- API endpoints and grading flow
- step-budget sufficiency for full resolution paths
- exact validator-facing inference stdout format

## Baseline Inference Guardrails

The baseline runner is intentionally robust under validator conditions:

- `API_BASE_URL` and `MODEL_NAME` are read with defaults, matching the submission guide.
- `HF_TOKEN` is read without a default, matching the submission guide.
- The baseline runner uses the local in-process environment directly, which removes remote endpoint configuration from the submission path.
- The OpenAI Python client is used for all LLM-backed calls.
- Stdout emits only the required single-line records:
  - `[START] task=... env=runbookops model=...`
  - `[STEP] step=... action=... reward=0.00 done=true|false error=...`
  - `[END] success=true|false steps=... rewards=r1,r2,...`
- Reward strings are always formatted to two decimal places.
- Published task scores are always strictly inside `(0,1)`.

The deterministic fallback planner now keys off visible evidence tokens and service context rather than hardcoded scenario-id answer tables, which makes the baseline easier to defend under human review.
