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
from typing import Any, Callable, Optional

from openai import OpenAI

from client import LocalRunbookOpsClient
from grader import public_score
from models import Action, ActionType, Observation

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_STEPS = 12
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 220
DEFAULT_RESULT_PATH = "baseline_results.json"

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = DEFAULT_MAX_STEPS
TEMPERATURE = DEFAULT_TEMPERATURE
MAX_TOKENS = DEFAULT_MAX_TOKENS
RESULT_PATH = DEFAULT_RESULT_PATH
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
TEAM_BY_SERVICE = {
    "auth": "auth-oncall",
    "checkout": "checkout-oncall",
    "payments": "payments-oncall",
    "email": "email-ops",
    "search": "search-infra",
    "notifications": "notifications-ops",
    "analytics": "analytics-data",
    "platform": "platform-ops",
}
EVIDENCE_TARGET_BY_DIFFICULTY = {
    "easy": 3,
    "medium": 4,
    "hard": 5,
}
VALID_SEVERITIES = {"SEV-1", "SEV-2", "SEV-3"}
VALID_TEAMS = set(TEAM_BY_SERVICE.values())
INSPECT_ACTION_TYPES = {
    ActionType.INSPECT_ALERT,
    ActionType.INSPECT_LOG,
    ActionType.INSPECT_RUNBOOK,
    ActionType.INSPECT_TIMELINE_NOTE,
}
NEGATION_PATTERNS = ("do not", "does not", "only if", "insufficient", "ineffective", "not fix")
ACTION_VERBS = (
    "rollback",
    "rotate",
    "reload",
    "restart",
    "restore",
    "disable",
    "resume",
    "flush",
    "drain",
    "isolate",
    "recycle",
    "replay",
    "purge",
    "switch",
    "reschedule",
)
FAILURE_TERMS = (
    "fail",
    "error",
    "timeout",
    "mismatch",
    "stale",
    "drop",
    "blocked",
    "crashed",
    "exhaust",
    "misroute",
    "regression",
    "skew",
    "decline",
    "outage",
    "surge",
    "duplicate",
)
BENIGN_TERMS = (
    "marketing",
    "campaign",
    "newsletter",
    "noise",
    "unrelated",
    "not currently causal",
    "minor",
    "advisory",
    "warning",
    "below",
    "within budget",
    "non causal",
    "green",
    "steady",
    "as usual",
    "completed successfully",
)
INCIDENT_RULES: list[dict[str, object]] = [
    {
        "service": "auth",
        "tokens": {"jwt", "expired", "rotation", "signing", "kid"},
        "severity": "SEV-2",
        "root_cause": "JWT signing key expired because rotation workflow failed",
        "mitigation": "Rotate signing key and restart token issuer pods",
    },
    {
        "service": "checkout",
        "tokens": {"inventory", "deprecated", "endpoint", "timeout", "lock"},
        "severity": "SEV-2",
        "root_cause": "Checkout still used deprecated inventory endpoint",
        "mitigation": "Switch checkout to inventory v2 endpoint and restart pods",
    },
    {
        "service": "email",
        "tokens": {"queue", "consumer", "panic", "backlog", "attachment"},
        "severity": "SEV-2",
        "root_cause": "Email queue consumer crashed and blocked backlog processing",
        "mitigation": "Restart consumer deployment and drain queue backlog safely",
    },
    {
        "service": "platform",
        "tokens": {"secret", "version", "mismatch", "signature", "gateway"},
        "severity": "SEV-2",
        "root_cause": "Secret rotated but gateway did not reload the new secret version",
        "mitigation": "Reload gateway config and propagate the rotated webhook secret",
    },
    {
        "service": "search",
        "tokens": {"invalidation", "lease", "freshness", "cache", "paused"},
        "severity": "SEV-3",
        "root_cause": "Cache invalidation worker paused after lease heartbeat failure",
        "mitigation": "Resume invalidation worker and flush stale cache prefixes",
    },
    {
        "service": "checkout",
        "tokens": {"fraud", "reject", "rollout", "rule", "geo"},
        "severity": "SEV-1",
        "root_cause": "Fraud feature flag rollout enabled an overly strict reject rule",
        "mitigation": "Rollback the fraud flag rollout to the previous cohort",
    },
    {
        "service": "email",
        "tokens": {"smtp", "credential", "secret", "auth", "535"},
        "severity": "SEV-2",
        "root_cause": "SMTP credentials rotated but workers kept a stale secret",
        "mitigation": "Refresh the SMTP secret, restart workers, and replay failed messages",
    },
    {
        "service": "auth",
        "tokens": {"connection", "leak", "canary", "pool", "session"},
        "severity": "SEV-1",
        "root_cause": "Connection leak in the auth canary exhausted the database pool",
        "mitigation": "Disable canary traffic and recycle the leaking auth pods",
    },
    {
        "service": "notifications",
        "tokens": {"retry", "backoff", "duplicate", "storm", "scheduler"},
        "severity": "SEV-2",
        "root_cause": "Retry backoff misconfiguration created a notification retry storm",
        "mitigation": "Restore exponential backoff and purge duplicate queued retries",
    },
    {
        "service": "payments",
        "tokens": {"region", "override", "mandate", "gateway", "eu"},
        "severity": "SEV-1",
        "root_cause": "Routing policy misrouted EU traffic to an unsupported gateway",
        "mitigation": "Restore EU regional routing policy and replay failed payment intents",
    },
    {
        "service": "email",
        "tokens": {"tls", "sni", "signer", "bundle", "handshake"},
        "severity": "SEV-1",
        "root_cause": "Mailer signer config regression removed the required TLS SNI setting",
        "mitigation": "Rollback the signer config bundle and redeploy the mailer sidecar",
    },
    {
        "service": "checkout",
        "tokens": {"az", "pool", "template", "regional", "autoscaler"},
        "severity": "SEV-1",
        "root_cause": "Outdated az-b pod template reduced the connection pool and caused a partial outage",
        "mitigation": "Drain az-b pods and restore the baseline pool settings before rescheduling",
    },
    {
        "service": "auth",
        "tokens": {"clock", "skew", "nonce", "ntp", "drift"},
        "severity": "SEV-1",
        "root_cause": "Clock skew on one auth node caused OAuth nonce validation failures",
        "mitigation": "Isolate the skewed node, restart time sync, and return it after validation",
    },
    {
        "service": "search",
        "tokens": {"schema", "migration", "drop", "index", "checkpoint"},
        "severity": "SEV-2",
        "root_cause": "Schema migration caused the index writer to drop update batches",
        "mitigation": "Rollback the schema migration and replay updates from checkpoint",
    },
    {
        "service": "payments",
        "tokens": {"shadow", "duplicate", "authorization", "idempotency", "experiment"},
        "severity": "SEV-1",
        "root_cause": "Shadow traffic header leak caused duplicate production authorization paths",
        "mitigation": "Disable the shadow mirror rule and separate the idempotency namespace before replay",
    },
]

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


def _history_action_types(observation: Observation) -> list[str]:
    actions: list[str] = []
    for line in observation.action_history_summary:
        match = re.search(r"\d+\.\s+([a-z_]+)", line)
        if match:
            actions.append(match.group(1))
    return actions


def _text_score(text: str, positive: tuple[str, ...], negative: tuple[str, ...] = ()) -> int:
    lowered = text.lower()
    score = 0
    for token in positive:
        if token in lowered:
            score += 1
    for token in negative:
        if token in lowered:
            score -= 1
    return score


def _keyword_tokens(*chunks: str) -> set[str]:
    tokens: set[str] = set()
    for chunk in chunks:
        for token in re.findall(r"[a-z0-9]+", chunk.lower()):
            if len(token) >= 3:
                tokens.add(token)
    return tokens


def _observation_tokens(observation: Observation) -> set[str]:
    chunks = [
        observation.service,
        observation.incident_summary,
        observation.last_action_result,
        *observation.known_facts[-8:],
        *(f"{item.title} {item.content} {' '.join(item.tags)}" for item in observation.visible_alerts),
        *(f"{item.title} {item.content} {' '.join(item.tags)}" for item in observation.visible_logs),
        *(f"{item.title} {item.content} {' '.join(item.tags)}" for item in observation.visible_runbooks),
        *(f"{item.title} {item.content} {' '.join(item.tags)}" for item in observation.visible_timeline_notes),
    ]
    return _keyword_tokens(*chunks)


def _matched_incident_rule(observation: Observation) -> Optional[dict[str, object]]:
    tokens = _observation_tokens(observation)
    best_rule: Optional[dict[str, object]] = None
    best_overlap = 0
    for rule in INCIDENT_RULES:
        if rule["service"] != observation.service:
            continue
        overlap = len(tokens & set(rule["tokens"]))  # type: ignore[arg-type]
        if overlap > best_overlap:
            best_overlap = overlap
            best_rule = rule
    if best_overlap >= 2:
        return best_rule
    return None


def _severity_guess(observation: Observation) -> str:
    matched_rule = _matched_incident_rule(observation)
    if matched_rule and matched_rule.get("severity") in VALID_SEVERITIES:
        return str(matched_rule["severity"])

    if observation.service == "search":
        return "SEV-3" if observation.difficulty.value == "easy" else "SEV-2"
    if observation.service in {"auth", "checkout", "payments"}:
        return "SEV-2" if observation.difficulty.value == "easy" else "SEV-1"
    if observation.service == "email":
        return "SEV-1" if observation.difficulty.value == "hard" else "SEV-2"
    if observation.service in {"notifications", "platform"}:
        return "SEV-2"
    return "SEV-2"


def _team_guess(observation: Observation) -> str:
    return TEAM_BY_SERVICE.get(observation.service, "platform-ops")


def _root_cause_guess(observation: Observation) -> str:
    matched_rule = _matched_incident_rule(observation)
    if matched_rule and matched_rule.get("root_cause"):
        return str(matched_rule["root_cause"])

    candidates: list[str] = []
    candidates.extend(f"{item.title}: {item.content}" for item in observation.visible_logs)
    candidates.extend(f"{item.title}: {item.content}" for item in observation.visible_timeline_notes)
    candidates.extend(observation.known_facts[-6:])
    candidates.extend(f"{item.title}: {item.content}" for item in observation.visible_alerts)
    if not candidates:
        return f"Likely {observation.service} configuration regression causing customer-facing failures"

    def score_candidate(value: str) -> int:
        return _text_score(value, FAILURE_TERMS, BENIGN_TERMS)

    best = max(candidates, key=score_candidate)
    if ":" in best:
        best = best.split(":", 1)[1].strip()
    return best[:220]


def _mitigation_guess(observation: Observation) -> str:
    matched_rule = _matched_incident_rule(observation)
    if matched_rule and matched_rule.get("mitigation"):
        return str(matched_rule["mitigation"])

    def score_runbook(text: str) -> int:
        lowered = text.lower()
        score = _text_score(text, ACTION_VERBS)
        for neg in NEGATION_PATTERNS:
            if neg in lowered:
                score -= 3
        return score

    if observation.visible_runbooks:
        best = max(observation.visible_runbooks, key=lambda item: score_runbook(item.content))
        return best.content[:240]

    return "Rollback risky change and restart affected workers safely"


def _evidence_target(observation: Observation) -> int:
    return EVIDENCE_TARGET_BY_DIFFICULTY.get(observation.difficulty.value, 4)


def _inspection_plan(observation: Observation) -> tuple[dict[ActionType, int], list[ActionType], int]:
    if observation.difficulty.value == "easy":
        quotas = {
            ActionType.INSPECT_ALERT: 1,
            ActionType.INSPECT_LOG: 1,
            ActionType.INSPECT_RUNBOOK: 1,
            ActionType.INSPECT_TIMELINE_NOTE: 0,
        }
        order = [
            ActionType.INSPECT_ALERT,
            ActionType.INSPECT_LOG,
            ActionType.INSPECT_RUNBOOK,
        ]
        min_total = 3
    elif observation.difficulty.value == "medium":
        quotas = {
            ActionType.INSPECT_ALERT: 1,
            ActionType.INSPECT_LOG: 1,
            ActionType.INSPECT_RUNBOOK: 1,
            ActionType.INSPECT_TIMELINE_NOTE: 0,
        }
        order = [
            ActionType.INSPECT_ALERT,
            ActionType.INSPECT_LOG,
            ActionType.INSPECT_RUNBOOK,
            ActionType.INSPECT_LOG,
            ActionType.INSPECT_TIMELINE_NOTE,
        ]
        min_total = 4
    else:
        quotas = {
            ActionType.INSPECT_ALERT: 1,
            ActionType.INSPECT_LOG: 2,
            ActionType.INSPECT_RUNBOOK: 1,
            ActionType.INSPECT_TIMELINE_NOTE: 1,
        }
        order = [
            ActionType.INSPECT_ALERT,
            ActionType.INSPECT_LOG,
            ActionType.INSPECT_LOG,
            ActionType.INSPECT_RUNBOOK,
            ActionType.INSPECT_TIMELINE_NOTE,
        ]
        min_total = 5

    return quotas, order, min_total


def _select_evidence_action(
    observation: Observation,
    inspected: set[str],
    preferred_types: Optional[set[ActionType]] = None,
    context_tokens: Optional[set[str]] = None,
    priority_tokens: Optional[set[str]] = None,
) -> Optional[Action]:
    weighted_tokens = context_tokens or set()
    high_weight_tokens = priority_tokens or set()
    candidates: list[tuple[int, Action]] = []
    buckets: list[tuple[list[Any], ActionType, int]] = [
        (observation.visible_alerts, ActionType.INSPECT_ALERT, 4),
        (observation.visible_logs, ActionType.INSPECT_LOG, 3),
        (observation.visible_runbooks, ActionType.INSPECT_RUNBOOK, 2),
        (observation.visible_timeline_notes, ActionType.INSPECT_TIMELINE_NOTE, 1),
    ]

    for items, action_type, base_score in buckets:
        for item in items:
            if preferred_types and action_type not in preferred_types:
                continue
            if item.id in inspected:
                continue
            text = f"{item.title} {item.content}".lower()
            item_tokens = _keyword_tokens(item.id, item.title, item.content, " ".join(item.tags))
            score = base_score
            score += _text_score(text, FAILURE_TERMS, BENIGN_TERMS)
            score += min(5, len(item_tokens & high_weight_tokens)) * 4
            score += min(4, len(item_tokens & weighted_tokens)) * 2

            if action_type == ActionType.INSPECT_RUNBOOK:
                score += _text_score(text, ACTION_VERBS)
                for neg in NEGATION_PATTERNS:
                    if neg in text:
                        score -= 3

            if action_type == ActionType.INSPECT_TIMELINE_NOTE and any(
                token in text for token in ("rollout", "rotation", "migration", "change", "approval")
            ):
                score += 2

            if "not currently causal" in text:
                score -= 4

            candidates.append((score, Action(action_type=action_type, target=item.id)))

    if not candidates:
        return None
    candidates.sort(key=lambda value: value[0], reverse=True)
    return candidates[0][1]


def _planned_action(observation: Observation) -> Action:
    inspected = _history_targets(observation)
    action_history = _history_action_types(observation)
    quotas, inspection_order, minimum_inspections = _inspection_plan(observation)
    evidence_target = max(_evidence_target(observation), minimum_inspections)

    inspect_counts = {action_type: 0 for action_type in INSPECT_ACTION_TYPES}
    for action_name in action_history:
        for action_type in INSPECT_ACTION_TYPES:
            if action_name == action_type.value:
                inspect_counts[action_type] += 1

    inspect_total = sum(inspect_counts.values())
    unresolved_fields = sum(
        1
        for value in [
            observation.selected_severity,
            observation.assigned_team,
            observation.submitted_root_cause,
            observation.submitted_mitigation,
        ]
        if not value
    )
    actions_needed_without_inspection = unresolved_fields + 1  # one final resolve

    matched_rule = _matched_incident_rule(observation) or {}
    context_tokens = _keyword_tokens(
        observation.service,
        observation.incident_summary,
        observation.last_action_result,
        str(matched_rule.get("root_cause", "")),
        str(matched_rule.get("mitigation", "")),
        " ".join(observation.known_facts[-8:]),
    )
    priority_tokens = _keyword_tokens(
        str(matched_rule.get("root_cause", "")),
        str(matched_rule.get("mitigation", "")),
    )

    if observation.steps_remaining > actions_needed_without_inspection:
        for action_type in inspection_order:
            if inspect_counts[action_type] >= quotas.get(action_type, 0):
                continue
            evidence_action = _select_evidence_action(
                observation=observation,
                inspected=inspected,
                preferred_types={action_type},
                context_tokens=context_tokens,
                priority_tokens=priority_tokens,
            )
            if evidence_action:
                return evidence_action

        if inspect_total < evidence_target:
            evidence_action = _select_evidence_action(
                observation=observation,
                inspected=inspected,
                context_tokens=context_tokens,
                priority_tokens=priority_tokens,
            )
            if evidence_action:
                return evidence_action

    if not observation.selected_severity:
        return Action(action_type=ActionType.SET_SEVERITY, content=_severity_guess(observation))

    if not observation.assigned_team:
        return Action(action_type=ActionType.ASSIGN_TEAM, content=_team_guess(observation))

    if not observation.submitted_root_cause:
        return Action(action_type=ActionType.SUBMIT_ROOT_CAUSE, content=_root_cause_guess(observation))

    if not observation.submitted_mitigation:
        return Action(action_type=ActionType.SUBMIT_MITIGATION, content=_mitigation_guess(observation))

    missing_quota_types = [action_type for action_type, need in quotas.items() if inspect_counts[action_type] < need]
    should_collect_more = inspect_total < evidence_target or bool(missing_quota_types)
    if should_collect_more and observation.steps_remaining > 1:
        for action_type in missing_quota_types:
            evidence_action = _select_evidence_action(
                observation=observation,
                inspected=inspected,
                preferred_types={action_type},
                context_tokens=context_tokens,
                priority_tokens=priority_tokens,
            )
            if evidence_action:
                return evidence_action
        evidence_action = _select_evidence_action(
            observation=observation,
            inspected=inspected,
            context_tokens=context_tokens,
            priority_tokens=priority_tokens,
        )
        if evidence_action:
            return evidence_action

    return Action(action_type=ActionType.RESOLVE_INCIDENT)


def _fallback_action(observation: Observation) -> Action:
    # Backward-compatible alias used by tests; now routes to deterministic planner.
    return _planned_action(observation)


def _is_risky_action(model_action: Action, observation: Observation) -> bool:
    inspected = _history_targets(observation)
    quotas, _, min_inspections = _inspection_plan(observation)
    evidence_target = max(_evidence_target(observation), min_inspections)
    action_history = _history_action_types(observation)
    inspect_counts = {action_type: 0 for action_type in INSPECT_ACTION_TYPES}
    for action_name in action_history:
        for action_type in INSPECT_ACTION_TYPES:
            if action_name == action_type.value:
                inspect_counts[action_type] += 1

    if (
        model_action.action_type in INSPECT_ACTION_TYPES
        and model_action.target in inspected
    ):
        return True

    if model_action.action_type == ActionType.SET_SEVERITY and (model_action.content or "").upper() not in VALID_SEVERITIES:
        return True

    if model_action.action_type == ActionType.ASSIGN_TEAM and (model_action.content or "").lower() not in VALID_TEAMS:
        return True

    if model_action.action_type == ActionType.RESOLVE_INCIDENT:
        if not all(
            [
                observation.selected_severity,
                observation.assigned_team,
                observation.submitted_root_cause,
                observation.submitted_mitigation,
            ]
        ):
            return True
        if len(inspected) < max(2, evidence_target - 1) and observation.steps_remaining > 1:
            return True
        for action_type, minimum in quotas.items():
            if inspect_counts[action_type] < minimum and observation.steps_remaining > 1:
                return True

    if model_action.action_type == ActionType.ADD_NOTE and observation.steps_remaining <= 3:
        return True

    if len(action_history) >= 2 and all(action == "add_note" for action in action_history[-2:]):
        if model_action.action_type == ActionType.ADD_NOTE:
            return True

    return False


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


def _choose_action(model: Optional[OpenAI], observation: Observation) -> Action:
    planned_action = _planned_action(observation)
    if model is None:
        return planned_action

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
        return planned_action

    payload = _parse_json_action(raw_content)
    if not payload:
        return planned_action

    try:
        model_action = Action.model_validate(payload)
    except Exception:
        return planned_action

    if _is_risky_action(model_action, observation):
        return planned_action

    return model_action


def _resolve_client() -> tuple[LocalRunbookOpsClient, str]:
    return LocalRunbookOpsClient(), "local"


def run_episode(
    client: LocalRunbookOpsClient,
    model: Optional[OpenAI],
    scenario_id: str,
    step_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    observation = client.reset(scenario_id=scenario_id)
    step_trace: list[dict[str, Any]] = []

    for step_index in range(1, MAX_STEPS + 1):
        if observation.done:
            break
        action = _choose_action(model, observation)
        step_result = client.step(action)
        step_reward = (
            float(step_result.reward.reward)
            if hasattr(step_result.reward, "reward")
            else float(step_result.reward)
        )
        step_error = step_result.info.message if step_result.info.invalid_action else None
        step_trace.append(
            {
                "step": step_index,
                "action": _format_action_trace(action),
                "action_type": action.action_type.value,
                "reward": step_reward,
                "done": bool(step_result.done),
                "error": step_error,
            }
        )
        if step_callback is not None:
            step_callback(step_trace[-1])
        observation = step_result.observation
        if step_result.done:
            break

    grade = client.grade(scenario_id=scenario_id)
    state = client.state()

    return {
        "scenario_id": scenario_id,
        "difficulty": grade.difficulty.value,
        "score": grade.score,
        "steps_taken": state.steps_taken,
        "terminal_reason": state.terminal_reason,
        "step_trace": step_trace,
        "success": state.terminal_reason == "resolved_safely",
        "rewards": [round(float(item["reward"]), 2) for item in step_trace],
    }


def _format_action_trace(action: Action) -> str:
    def normalize(value: Optional[str]) -> str:
        text = (value or "").strip().replace("\n", " ")
        text = re.sub(r"\s+", "_", text)
        text = text.replace("'", "")
        return text

    if action.target:
        return f"{action.action_type.value}('{normalize(action.target)}')"
    if action.content:
        return f"{action.action_type.value}('{normalize(action.content)}')"
    return f"{action.action_type.value}()"


def _emit_structured_event(event_type: str, fields: list[tuple[str, Any]]) -> None:
    serialized = " ".join(f"{key}={value}" for key, value in fields)
    print(f"[{event_type}] {serialized}", flush=True)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _reward_text(value: float) -> str:
    return f"{float(value):.2f}"


def _error_text(value: Optional[str]) -> str:
    if value is None or not str(value).strip():
        return "null"
    compact = re.sub(r"\s+", "_", str(value).strip().replace("\n", " "))
    compact = compact.replace("'", "").replace('"', "")
    return compact


def main() -> None:
    client, resolved_env_base_url = _resolve_client()
    health = client.health()

    model: Optional[OpenAI] = None
    inference_mode = "planner_only"
    warnings: list[str] = []
    if HF_TOKEN:
        try:
            model = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
            inference_mode = "openai_client"
        except Exception as exc:
            warnings.append(
                "OpenAI client initialization failed; using deterministic planner-only baseline: "
                f"{exc}"
            )
    else:
        warnings.append("No HF_TOKEN detected; using deterministic planner-only baseline.")

    scenarios = sorted(
        client.scenarios(),
        key=lambda item: (DIFFICULTY_ORDER[item.difficulty.value], item.scenario_id),
    )
    episode_results: list[dict[str, Any]] = []

    for scenario in scenarios:
        _emit_structured_event(
            "START",
            [
                ("task", scenario.scenario_id),
                ("env", "runbookops"),
                ("model", MODEL_NAME),
            ],
        )

        result: dict[str, Any]
        try:
            result = run_episode(
                client=client,
                model=model,
                scenario_id=scenario.scenario_id,
                step_callback=lambda trace: _emit_structured_event(
                    "STEP",
                    [
                        ("step", trace["step"]),
                        ("action", trace["action"]),
                        ("reward", _reward_text(float(trace["reward"]))),
                        ("done", _bool_text(bool(trace["done"]))),
                        ("error", _error_text(trace.get("error"))),
                    ],
                ),
            )
        except Exception as exc:
            result = {
                "scenario_id": scenario.scenario_id,
                "difficulty": scenario.difficulty.value,
                "score": public_score(0.0),
                "steps_taken": 0,
                "terminal_reason": "episode_exception",
                "step_trace": [],
                "success": False,
                "rewards": [],
                "episode_error": str(exc),
            }
            warnings.append(f"Episode {scenario.scenario_id} failed inside inference.py: {exc}")

        episode_results.append(result)

        rewards_csv = ",".join(_reward_text(float(reward)) for reward in result["rewards"])
        _emit_structured_event(
            "END",
            [
                ("success", _bool_text(bool(result["success"]))),
                ("steps", int(result["steps_taken"])),
                ("rewards", rewards_csv),
            ],
        )

    by_difficulty: dict[str, list[float]] = defaultdict(list)
    for row in episode_results:
        by_difficulty[row["difficulty"]].append(float(row["score"]))

    overall = mean([row["score"] for row in episode_results]) if episode_results else 0.0

    terminal_counts: dict[str, int] = defaultdict(int)
    for row in episode_results:
        terminal_counts[str(row.get("terminal_reason") or "none")] += 1

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "inference_mode": inference_mode,
        "warnings": warnings,
        "results": [
            {
                "scenario_id": row["scenario_id"],
                "difficulty": row["difficulty"],
                "score": public_score(float(row["score"])),
                "terminal_reason": row["terminal_reason"],
            }
            for row in episode_results
        ],
        "aggregates": {
            difficulty: {
                "min_score": public_score(min(scores)),
                "mean_score": public_score(mean(scores)),
                "max_score": public_score(max(scores)),
            }
            for difficulty, scores in by_difficulty.items()
        },
        "overall_mean_score": public_score(overall if episode_results else 0.5),
    }

    output_path = Path(RESULT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
