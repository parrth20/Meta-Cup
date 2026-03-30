from __future__ import annotations

import ast
import json
import os
import re
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from openai import OpenAI

from client import RunbookOpsClient
from models import Action, ActionType, Observation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
ENV_BASE_URL = os.getenv("RUNBOOKOPS_BASE_URL", "http://localhost:8000")
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
RESULT_PATH = os.getenv("RESULT_PATH", "baseline_results.json")
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
ACTION_ALIASES = {
    "inspect_timeline": "inspect_timeline_note",
    "inspect_timeline_event": "inspect_timeline_note",
    "set_priority": "set_severity",
    "set_owner_team": "assign_team",
    "submit_cause": "submit_root_cause",
    "submit_fix": "submit_mitigation",
    "resolve": "resolve_incident",
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an incident responder agent in RunbookOps.
    You must output exactly one JSON object and nothing else.

    JSON schema:
    {
      "action_type": "inspect_alert|inspect_log|inspect_runbook|inspect_timeline_note|set_severity|assign_team|submit_root_cause|submit_mitigation|add_note|resolve_incident",
      "target": "optional evidence id for inspect actions",
      "content": "optional text for set/assign/submit/add_note"
    }

    Rules:
    - Use inspect_* actions first to gather relevant evidence.
    - Set severity to one of SEV-1, SEV-2, SEV-3.
    - assign_team must be one of:
      auth-oncall, checkout-oncall, payments-oncall, email-ops,
      search-infra, notifications-ops, analytics-data, platform-ops.
    - Resolve only when severity, team, root cause, and mitigation are set.
    - Do not include explanations.
    """
).strip()


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(value)


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _canonical_action_name(raw_name: str | None) -> str:
    if not raw_name:
        return ""
    normalized = re.sub(r"[^a-z_]", "", raw_name.strip().lower())
    return ACTION_ALIASES.get(normalized, normalized)


def _extract_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    quote_char = ""
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in {'"', "'"}:
            if not in_string:
                in_string = True
                quote_char = ch
            elif quote_char == ch:
                in_string = False
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _decode_object(candidate: str) -> Optional[Any]:
    payload = candidate.strip()
    if not payload:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(payload)
        except Exception:
            continue
    return None


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    action_name = payload.get("action_type") or payload.get("action")
    if action_name is not None:
        normalized["action_type"] = _canonical_action_name(str(action_name))

    target = payload.get("target")
    if target is not None:
        normalized["target"] = str(target)

    content = payload.get("content")
    if content is None and payload.get("value") is not None:
        content = payload.get("value")
    if content is not None:
        normalized["content"] = str(content)

    return normalized


def _parse_function_call_action(text: str) -> Optional[dict[str, Any]]:
    candidate = re.sub(r"^\s*(action|next action)\s*[:\-]\s*", "", text.strip(), flags=re.IGNORECASE)
    try:
        expr = ast.parse(candidate, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        return None

    action_type = _canonical_action_name(expr.func.id)
    args: list[Any] = []
    for arg in expr.args:
        try:
            args.append(ast.literal_eval(arg))
        except Exception:
            return None

    payload: dict[str, Any] = {"action_type": action_type}
    inspect_actions = {
        "inspect_alert",
        "inspect_log",
        "inspect_runbook",
        "inspect_timeline_note",
    }
    content_actions = {
        "set_severity",
        "assign_team",
        "submit_root_cause",
        "submit_mitigation",
        "add_note",
    }
    if action_type in inspect_actions and args:
        payload["target"] = str(args[0])
    elif action_type in content_actions and args:
        payload["content"] = str(args[0])
    return payload


def _parse_key_value_action(text: str) -> Optional[dict[str, Any]]:
    pairs: dict[str, str] = {}
    chunks = re.split(r"[\n,]+", text)
    for chunk in chunks:
        match = re.match(
            r"\s*(action_type|action|target|content|value)\s*[:=]\s*(.+?)\s*$",
            chunk,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        key, value = match.groups()
        extracted = value.strip().strip('"').strip("'")
        pairs[key.lower()] = extracted
    if not pairs:
        return None
    return _normalize_payload(pairs)


def _parse_json_action(raw: str) -> Optional[dict[str, Any]]:
    cleaned = _strip_code_fences(raw)
    candidates = [cleaned]

    balanced = _extract_balanced_object(cleaned)
    if balanced:
        candidates.append(balanced)

    compact = cleaned.replace("\n", " ").strip()
    if compact:
        candidates.append(compact)

    for candidate in candidates:
        decoded = _decode_object(candidate)
        if isinstance(decoded, dict):
            payload = _normalize_payload(decoded)
            if payload:
                return payload
        if isinstance(decoded, list) and decoded and isinstance(decoded[0], dict):
            payload = _normalize_payload(decoded[0])
            if payload:
                return payload

    fn_payload = _parse_function_call_action(cleaned)
    if fn_payload:
        return fn_payload

    return _parse_key_value_action(cleaned)


def _history_targets(observation: Observation) -> set[str]:
    ids: set[str] = set()
    for line in observation.action_history_summary:
        found = re.findall(r"target=([a-zA-Z0-9_\-]+)", line)
        ids.update(found)
    return ids


def _fallback_action(observation: Observation) -> Action:
    inspected = _history_targets(observation)
    min_facts_before_resolution = {
        "easy": 3,
        "medium": 4,
        "hard": 5,
    }.get(observation.difficulty.value, 4)

    for bucket, action_type in [
        (observation.visible_alerts, ActionType.INSPECT_ALERT),
        (observation.visible_logs, ActionType.INSPECT_LOG),
        (observation.visible_runbooks, ActionType.INSPECT_RUNBOOK),
        (observation.visible_timeline_notes, ActionType.INSPECT_TIMELINE_NOTE),
    ]:
        for item in bucket:
            if item.id not in inspected:
                return Action(action_type=action_type, target=item.id)

    if not observation.selected_severity:
        severity_by_difficulty = {
            "easy": "SEV-2",
            "medium": "SEV-2",
            "hard": "SEV-1",
        }
        value = severity_by_difficulty.get(observation.difficulty.value, "SEV-2")
        return Action(action_type=ActionType.SET_SEVERITY, content=value)

    if not observation.assigned_team:
        team_by_service = {
            "auth": "auth-oncall",
            "checkout": "checkout-oncall",
            "payments": "payments-oncall",
            "email": "email-ops",
            "search": "search-infra",
            "notifications": "notifications-ops",
            "analytics": "analytics-data",
            "platform": "platform-ops",
        }
        team = team_by_service.get(observation.service, "platform-ops")
        return Action(action_type=ActionType.ASSIGN_TEAM, content=team)

    if not observation.submitted_root_cause:
        hint = observation.known_facts[-1] if observation.known_facts else "Likely configuration regression in service pipeline"
        return Action(action_type=ActionType.SUBMIT_ROOT_CAUSE, content=hint)

    if not observation.submitted_mitigation:
        hint = "Rollback latest risky change and restart affected workers safely"
        return Action(action_type=ActionType.SUBMIT_MITIGATION, content=hint)

    if len(observation.known_facts) < min_facts_before_resolution:
        return Action(
            action_type=ActionType.ADD_NOTE,
            content="Need one more evidence item before safe resolution.",
        )

    return Action(action_type=ActionType.RESOLVE_INCIDENT)


def _build_user_prompt(observation: Observation) -> str:
    def format_evidence(items: list[Any]) -> list[str]:
        lines = []
        for item in items:
            lines.append(f"- {item.id}: {item.title} | {item.content}")
        return lines or ["- none"]

    prompt_lines = [
        f"Scenario: {observation.scenario_id} ({observation.difficulty.value})",
        f"Service: {observation.service}",
        f"Incident summary: {observation.incident_summary}",
        f"Last action result: {observation.last_action_result}",
        f"Steps taken: {observation.steps_taken} | Steps remaining: {observation.steps_remaining}",
        f"Selected severity: {observation.selected_severity}",
        f"Assigned team: {observation.assigned_team}",
        f"Submitted root cause: {observation.submitted_root_cause}",
        f"Submitted mitigation: {observation.submitted_mitigation}",
        "Known facts:",
        *(f"- {fact}" for fact in (observation.known_facts[-5:] or ["none"])),
        "Visible alerts:",
        *format_evidence(observation.visible_alerts),
        "Visible logs:",
        *format_evidence(observation.visible_logs),
        "Visible runbooks:",
        *format_evidence(observation.visible_runbooks),
        "Visible timeline notes:",
        *format_evidence(observation.visible_timeline_notes),
    ]
    return "\n".join(prompt_lines)


def _choose_action(model: OpenAI, observation: Observation) -> Action:
    user_prompt = _build_user_prompt(observation)

    try:
        completion = model.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_content = _to_text(completion.choices[0].message.content)
    except Exception:
        return _fallback_action(observation)

    payload = _parse_json_action(raw_content)
    if not payload:
        return _fallback_action(observation)

    try:
        return Action.model_validate(payload)
    except Exception:
        return _fallback_action(observation)


def run_episode(client: RunbookOpsClient, model: OpenAI, scenario_id: str) -> dict[str, Any]:
    observation = client.reset(scenario_id=scenario_id)

    for _ in range(MAX_STEPS):
        if observation.done:
            break
        action = _choose_action(model, observation)
        step_result = client.step(action)
        observation = step_result.observation
        if step_result.done:
            break

    grade = client.grade(scenario_id=scenario_id)
    state = client.state()

    return {
        "scenario_id": scenario_id,
        "difficulty": grade.difficulty.value,
        "score": grade.score,
        "components": grade.components,
        "steps_taken": state.steps_taken,
        "total_reward": state.total_reward,
        "terminal_reason": state.terminal_reason,
    }


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    separator = " | ".join("-" * width for width in widths)
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def main() -> None:
    if not MODEL_NAME:
        raise SystemExit("MODEL_NAME is required.")
    if not API_KEY:
        raise SystemExit("HF_TOKEN or API_KEY or OPENAI_API_KEY is required.")

    client = RunbookOpsClient(base_url=ENV_BASE_URL)
    health = client.health()
    print(f"Environment status: {health}")

    model = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    scenarios = sorted(
        client.scenarios(),
        key=lambda item: (DIFFICULTY_ORDER[item.difficulty.value], item.scenario_id),
    )
    episode_results: list[dict[str, Any]] = []

    for scenario in scenarios:
        result = run_episode(client=client, model=model, scenario_id=scenario.scenario_id)
        episode_results.append(result)
    scenario_rows = [
        [
            row["difficulty"],
            row["scenario_id"],
            f"{float(row['score']):.4f}",
            str(row["steps_taken"]),
            f"{float(row['total_reward']):.4f}",
            str(row["terminal_reason"] or "none"),
        ]
        for row in episode_results
    ]

    print("\nPer-scenario results")
    _print_table(
        headers=["difficulty", "scenario_id", "score", "steps", "reward", "terminal_reason"],
        rows=scenario_rows,
    )

    by_difficulty: dict[str, list[float]] = defaultdict(list)
    for row in episode_results:
        by_difficulty[row["difficulty"]].append(float(row["score"]))

    overall = mean([row["score"] for row in episode_results]) if episode_results else 0.0

    print("\nAggregate scores")
    aggregate_rows: list[list[str]] = []
    for difficulty in ["easy", "medium", "hard"]:
        scores = by_difficulty.get(difficulty, [])
        if scores:
            aggregate_rows.append(
                [
                    difficulty,
                    str(len(scores)),
                    f"{min(scores):.4f}",
                    f"{mean(scores):.4f}",
                    f"{max(scores):.4f}",
                ]
            )
    _print_table(headers=["difficulty", "count", "min", "mean", "max"], rows=aggregate_rows)
    print(f"\nOverall mean score: {overall:.4f} across {len(episode_results)} scenarios")

    terminal_counts: dict[str, int] = defaultdict(int)
    for row in episode_results:
        terminal_counts[str(row.get("terminal_reason") or "none")] += 1
    terminal_rows = [[reason, str(count)] for reason, count in sorted(terminal_counts.items())]
    print("\nTerminal reasons")
    _print_table(headers=["terminal_reason", "count"], rows=terminal_rows)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "environment_base_url": ENV_BASE_URL,
        "max_steps": MAX_STEPS,
        "results": episode_results,
        "aggregates": {
            difficulty: {
                "count": len(scores),
                "min_score": round(min(scores), 4),
                "mean_score": round(mean(scores), 4),
                "max_score": round(max(scores), 4),
            }
            for difficulty, scores in by_difficulty.items()
        },
        "overall_mean_score": round(overall, 4),
    }

    output_path = Path(RESULT_PATH)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved summary: {output_path}")


if __name__ == "__main__":
    main()
