from __future__ import annotations

import re
from difflib import SequenceMatcher
from statistics import mean
from typing import Iterable

from models import Difficulty, GraderResult, InternalStateSnapshot, Scenario, TaskSummary

GRADER_WEIGHTS: dict[str, float] = {
    "severity": 0.15,
    "owner_team": 0.15,
    "root_cause": 0.30,
    "mitigation": 0.25,
    "evidence_coverage": 0.10,
    "safe_resolution": 0.05,
}

STRICT_SCORE_EPSILON = 0.0001


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def clamp_open01(value: float, epsilon: float = STRICT_SCORE_EPSILON) -> float:
    return max(epsilon, min(1.0 - epsilon, value))


def public_score(value: float) -> float:
    return round(clamp_open01(clamp01(value)), 4)


TOKEN_SYNONYMS: dict[str, str] = {
    "recycle": "restart",
    "redeploy": "restart",
    "reload": "restart",
    "bounce": "restart",
    "restore": "restart",
    "rollback": "rollback",
    "revert": "rollback",
    "backout": "rollback",
    "rotation": "rotate",
    "rotated": "rotate",
    "refresh": "rotate",
    "credentials": "credential",
    "secrets": "secret",
    "keys": "key",
    "misrouted": "misroute",
    "misrouting": "misroute",
    "timeouts": "timeout",
    "failed": "fail",
    "failing": "fail",
    "failures": "fail",
    "caused": "cause",
    "causing": "cause",
    "configuration": "config",
    "configs": "config",
    "deployment": "deploy",
    "deployments": "deploy",
    "workers": "worker",
    "pods": "pod",
    "host": "node",
    "machine": "node",
    "time": "clock",
    "drift": "skew",
    "check": "validation",
    "checks": "validation",
    "sync": "synchronize",
    "synchronization": "synchronize",
    "skewed": "skew",
    "stopped": "stop",
    "ntp": "time",
    "oauth": "auth",
    "smtp": "mail",
    "tls": "encryption",
    "sni": "hostname",
    "qps": "traffic",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "after",
    "before",
    "from",
    "because",
    "by",
    "is",
    "was",
    "were",
    "be",
    "that",
    "this",
    "as",
    "it",
}

NEGATION_TOKENS = {"not", "no", "never", "without", "wrong", "incorrect"}


def _stem(token: str) -> str:
    if len(token) <= 4:
        return token
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    raw_tokens = [token for token in normalized.split() if token and token not in STOPWORDS]
    tokens: list[str] = []
    for token in raw_tokens:
        mapped = TOKEN_SYNONYMS.get(token, token)
        tokens.append(_stem(mapped))
    return tokens


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(_tokenize(text))


def _char_trigrams(text: str) -> set[str]:
    compact = text.replace(" ", "")
    if not compact:
        return set()
    if len(compact) < 3:
        return {compact}
    return {compact[index : index + 3] for index in range(len(compact) - 2)}


def _has_negation_conflict(submitted_tokens: list[str], candidate_tokens: set[str]) -> bool:
    if not submitted_tokens or not candidate_tokens:
        return False
    # Do not treat legitimate negative phrasing as conflict when the reference
    # itself contains negation (e.g., "did not reload").
    if candidate_tokens & NEGATION_TOKENS:
        return False
    for index, token in enumerate(submitted_tokens):
        if token in NEGATION_TOKENS:
            window = submitted_tokens[index + 1 : index + 4]
            if set(window) & candidate_tokens:
                return True
    return False


def text_similarity_score(submitted: str | None, canonical: str, aliases: list[str] | None = None) -> float:
    normalized_submitted = normalize_text(submitted)
    if not normalized_submitted:
        return 0.0
    submitted_tokens_ordered = _tokenize(submitted)
    submitted_tokens = set(submitted_tokens_ordered)

    candidate_texts = [canonical, *(aliases or [])]
    normalized_candidates = [normalize_text(value) for value in candidate_texts if normalize_text(value)]
    if not normalized_candidates:
        return 0.0

    best_score = 0.0
    for candidate in normalized_candidates:
        candidate_tokens = set(candidate.split())
        if _has_negation_conflict(submitted_tokens_ordered, candidate_tokens):
            continue

        if normalized_submitted == candidate:
            return 1.0
        if normalized_submitted in candidate or candidate in normalized_submitted:
            best_score = max(best_score, 0.97)
            continue

        if submitted_tokens and candidate_tokens:
            overlap = submitted_tokens & candidate_tokens
            recall = len(overlap) / len(candidate_tokens)
            precision = len(overlap) / len(submitted_tokens)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            best_score = max(best_score, 0.55 * f1 + 0.30 * recall + 0.15 * precision)

        overlap_count = len(submitted_tokens & candidate_tokens)
        overlap_gate = overlap_count >= 2 or len(candidate_tokens) <= 2

        sequence_ratio = SequenceMatcher(None, normalized_submitted, candidate).ratio()
        if overlap_gate:
            best_score = max(best_score, 0.92 * sequence_ratio)

        sub_trigrams = _char_trigrams(normalized_submitted)
        cand_trigrams = _char_trigrams(candidate)
        if sub_trigrams and cand_trigrams:
            trigram_overlap = len(sub_trigrams & cand_trigrams) / len(sub_trigrams | cand_trigrams)
            if overlap_gate:
                best_score = max(best_score, 0.88 * trigram_overlap)

    return clamp01(best_score)


def text_matches(submitted: str | None, canonical: str, aliases: list[str] | None = None) -> bool:
    return text_similarity_score(submitted, canonical, aliases) >= 0.72


def severity_similarity(submitted: str | None, truth: str) -> float:
    if not submitted:
        return 0.0
    order = {"SEV-1": 1, "SEV-2": 2, "SEV-3": 3}
    if submitted == truth:
        return 1.0
    if submitted not in order or truth not in order:
        return 0.0
    distance = abs(order[submitted] - order[truth])
    if distance == 1:
        return 0.35
    return 0.1


def team_similarity(submitted: str | None, truth: str) -> float:
    if not submitted:
        return 0.0
    if submitted == truth:
        return 1.0

    submitted_tokens = set(submitted.split("-"))
    truth_tokens = set(truth.split("-"))
    overlap = len(submitted_tokens & truth_tokens) / max(1, len(truth_tokens))
    suffix_match = 0.25 if submitted.split("-")[-1] == truth.split("-")[-1] else 0.0
    return clamp01(max(overlap, suffix_match))


def _minimum_viable_steps(scenario: Scenario) -> int:
    return min(scenario.max_steps, len(scenario.required_evidence_ids) + 5)


def _step_efficiency(scenario: Scenario, steps_taken: int) -> float:
    if steps_taken <= 0:
        return 0.0
    minimum = _minimum_viable_steps(scenario)
    if steps_taken <= minimum:
        return 1.0
    slack = max(1, scenario.max_steps - minimum)
    overage = steps_taken - minimum
    return clamp01(1.0 - (overage / slack))


def grade_episode(scenario: Scenario, state: InternalStateSnapshot) -> GraderResult:
    severity_component = severity_similarity(state.selected_severity, scenario.true_severity.value)
    team_component = team_similarity(state.assigned_team, scenario.true_owner_team.value)

    root_cause_component = text_similarity_score(
        state.submitted_root_cause,
        scenario.true_root_cause,
        scenario.true_root_cause_aliases,
    )

    mitigation_component = text_similarity_score(
        state.submitted_mitigation,
        scenario.true_mitigation,
        scenario.true_mitigation_aliases,
    )

    required_ids = set(scenario.required_evidence_ids)
    inspected_ids = set(state.inspected_evidence_ids)
    required_coverage = len(required_ids & inspected_ids) / len(required_ids) if required_ids else 1.0

    all_relevant_ids = {
        item.id
        for item in scenario.all_evidence
        if item.relevant
    }
    discovered_relevant_ids = set(state.discovered_relevant_evidence_ids)
    if not discovered_relevant_ids:
        discovered_relevant_ids = inspected_ids & all_relevant_ids

    relevant_discovery = (
        len(discovered_relevant_ids & all_relevant_ids) / len(all_relevant_ids)
        if all_relevant_ids
        else 1.0
    )
    inspection_precision = (
        len(inspected_ids & all_relevant_ids) / len(inspected_ids)
        if inspected_ids
        else 0.0
    )
    targeting_quality = (
        1.0 - (len(inspected_ids) / max(1, len(scenario.all_evidence)))
        if inspected_ids
        else 0.0
    )
    evidence_component = clamp01(
        0.45 * required_coverage
        + 0.25 * relevant_discovery
        + 0.15 * inspection_precision
        + 0.15 * targeting_quality
    )

    has_required_fields = all(
        [
            severity_component >= 0.99,
            team_component >= 0.99,
            root_cause_component >= 0.85,
            mitigation_component >= 0.85,
        ]
    )
    enough_evidence = required_coverage >= 0.75
    efficiency_component = _step_efficiency(scenario, state.steps_taken)
    resolution_correctness = mean(
        [
            severity_component,
            team_component,
            root_cause_component,
            mitigation_component,
        ]
    )
    safe_resolution_component = clamp01(
        (
            0.35 * resolution_correctness
            + 0.25 * required_coverage
            + 0.20 * (1.0 if state.terminal_reason == "resolved_safely" else 0.0)
            + 0.10 * (0.0 if state.premature_resolution else 1.0)
            + 0.10 * efficiency_component
        )
        * (1.0 if state.resolution_attempted else 0.35)
    )

    components = {
        "severity": clamp01(severity_component),
        "owner_team": clamp01(team_component),
        "root_cause": clamp01(root_cause_component),
        "mitigation": clamp01(mitigation_component),
        "evidence_coverage": clamp01(evidence_component),
        "safe_resolution": clamp01(safe_resolution_component),
    }

    weighted_score = sum(components[key] * GRADER_WEIGHTS[key] for key in GRADER_WEIGHTS)
    weighted_score = clamp_open01(clamp01(weighted_score))

    details: list[str] = [
        f"Severity component: {public_score(components['severity']):.4f}",
        f"Owner team component: {public_score(components['owner_team']):.4f}",
        f"Root cause component: {public_score(components['root_cause']):.4f}",
        f"Mitigation component: {public_score(components['mitigation']):.4f}",
        f"Evidence coverage component: {public_score(components['evidence_coverage']):.4f}",
        f"Safe resolution component: {public_score(components['safe_resolution']):.4f}",
        f"Required evidence coverage: {public_score(required_coverage):.4f}",
        f"Relevant evidence discovery: {public_score(relevant_discovery):.4f}",
        f"Inspection precision: {public_score(inspection_precision):.4f}",
        f"Targeting quality: {public_score(targeting_quality):.4f}",
        f"Step efficiency: {public_score(efficiency_component):.4f}",
        f"Safe resolution gate: {'passed' if (state.done and state.resolution_attempted and not state.premature_resolution and has_required_fields and enough_evidence) else 'not passed'}",
    ]

    return GraderResult(
        scenario_id=scenario.scenario_id,
        difficulty=scenario.difficulty,
        score=public_score(weighted_score),
        components={key: public_score(value) for key, value in components.items()},
        weights=GRADER_WEIGHTS,
        details=details,
    )


def aggregate_task_scores(results: Iterable[GraderResult]) -> list[TaskSummary]:
    by_difficulty: dict[Difficulty, list[float]] = {
        Difficulty.EASY: [],
        Difficulty.MEDIUM: [],
        Difficulty.HARD: [],
    }

    for result in results:
        by_difficulty[result.difficulty].append(result.score)

    summaries: list[TaskSummary] = []
    for difficulty, scores in by_difficulty.items():
        if not scores:
            continue
        summaries.append(
            TaskSummary(
                difficulty=difficulty,
                scenario_count=len(scores),
                average_score=public_score(mean(scores)),
                min_score=public_score(min(scores)),
                max_score=public_score(max(scores)),
            )
        )

    return summaries
