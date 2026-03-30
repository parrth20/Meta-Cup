from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from grader import grade_episode, text_matches
from models import (
    Action,
    ActionType,
    Difficulty,
    EvidenceItem,
    EvidenceType,
    GraderResult,
    InternalStateSnapshot,
    Observation,
    PublicEvidence,
    Scenario,
    ScenarioSummary,
    SeverityLevel,
    StepInfo,
    StepResult,
    TeamName,
)


@dataclass
class EpisodeRuntimeState:
    scenario: Scenario
    steps_taken: int = 0
    total_reward: float = 0.0
    done: bool = False
    selected_severity: Optional[str] = None
    assigned_team: Optional[str] = None
    submitted_root_cause: Optional[str] = None
    submitted_mitigation: Optional[str] = None
    inspected_evidence_ids: set[str] = field(default_factory=set)
    discovered_relevant_evidence_ids: set[str] = field(default_factory=set)
    known_facts: list[str] = field(default_factory=list)
    action_history: list[Action] = field(default_factory=list)
    action_trace: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    coverage_milestones_awarded: set[float] = field(default_factory=set)
    resolution_attempted: bool = False
    premature_resolution: bool = False
    terminal_reason: Optional[str] = None
    last_action_result: str = "Episode initialized. Gather evidence before resolving."
    action_counts: Counter[str] = field(default_factory=Counter)


class RunbookOpsEnvironment:
    """Deterministic incident response simulation environment."""

    LIVING_PENALTY = -0.005
    COVERAGE_MILESTONES: tuple[float, ...] = (0.5, 1.0)

    def __init__(self, scenarios_dir: str | Path | None = None) -> None:
        self.scenarios_dir = Path(scenarios_dir or Path(__file__).resolve().parents[1] / "scenarios")
        self.scenarios: list[Scenario] = self._load_scenarios()
        self.scenario_map: dict[str, Scenario] = {scenario.scenario_id: scenario for scenario in self.scenarios}

        self._difficulty_rotation: dict[Difficulty, int] = {
            Difficulty.EASY: 0,
            Difficulty.MEDIUM: 0,
            Difficulty.HARD: 0,
        }
        self._global_rotation: int = 0
        self._state: Optional[EpisodeRuntimeState] = None

    def _load_scenarios(self) -> list[Scenario]:
        if not self.scenarios_dir.exists():
            raise FileNotFoundError(f"Scenarios directory not found: {self.scenarios_dir}")

        scenarios: list[Scenario] = []
        for path in sorted(self.scenarios_dir.rglob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            scenario = Scenario.model_validate(raw)
            scenarios.append(scenario)

        if not scenarios:
            raise ValueError("No scenario files were loaded.")

        dedup = {scenario.scenario_id for scenario in scenarios}
        if len(dedup) != len(scenarios):
            raise ValueError("Duplicate scenario_id detected in scenario files.")

        return sorted(scenarios, key=lambda item: item.scenario_id)

    def _select_scenario(
        self,
        scenario_id: Optional[str] = None,
        difficulty: Optional[Difficulty] = None,
    ) -> Scenario:
        if scenario_id:
            if scenario_id not in self.scenario_map:
                raise ValueError(f"Unknown scenario_id: {scenario_id}")
            return self.scenario_map[scenario_id]

        if difficulty:
            by_difficulty = [item for item in self.scenarios if item.difficulty == difficulty]
            if not by_difficulty:
                raise ValueError(f"No scenarios available for difficulty={difficulty.value}")
            cursor = self._difficulty_rotation[difficulty] % len(by_difficulty)
            self._difficulty_rotation[difficulty] += 1
            return by_difficulty[cursor]

        cursor = self._global_rotation % len(self.scenarios)
        self._global_rotation += 1
        return self.scenarios[cursor]

    def reset(
        self,
        scenario_id: Optional[str] = None,
        difficulty: Optional[Difficulty] = None,
    ) -> Observation:
        scenario = self._select_scenario(scenario_id=scenario_id, difficulty=difficulty)

        self._state = EpisodeRuntimeState(scenario=scenario)
        return self._build_observation(self._state)

    def _require_state(self) -> EpisodeRuntimeState:
        if self._state is None:
            raise RuntimeError("Environment has not been reset. Call reset() before step().")
        return self._state

    def _unlock_satisfied(self, condition: Optional[str], state: EpisodeRuntimeState) -> bool:
        if not condition:
            return True

        # Syntax: "e1&e2|e3" means (e1 AND e2) OR e3.
        clauses = [clause.strip() for clause in condition.split("|") if clause.strip()]
        for clause in clauses:
            required_ids = [item.strip() for item in clause.split("&") if item.strip()]
            if required_ids and all(item in state.inspected_evidence_ids for item in required_ids):
                return True

        return False

    def _is_visible(self, evidence: EvidenceItem, state: EpisodeRuntimeState) -> bool:
        if evidence.id in state.scenario.initial_visible_evidence_ids:
            return True
        if evidence.id in state.inspected_evidence_ids:
            return True
        return self._unlock_satisfied(evidence.unlock_condition, state)

    def _public_evidence(self, evidence: EvidenceItem) -> PublicEvidence:
        return PublicEvidence(
            id=evidence.id,
            type=evidence.type,
            title=evidence.title,
            content=evidence.content,
            tags=evidence.tags,
        )

    def _visible_by_type(self, state: EpisodeRuntimeState, evidence_type: EvidenceType) -> list[PublicEvidence]:
        visible = [
            item
            for item in state.scenario.all_evidence
            if item.type == evidence_type and self._is_visible(item, state)
        ]
        return [self._public_evidence(item) for item in visible]

    def _build_observation(self, state: EpisodeRuntimeState) -> Observation:
        available_actions = [
            ActionType.INSPECT_ALERT,
            ActionType.INSPECT_LOG,
            ActionType.INSPECT_RUNBOOK,
            ActionType.INSPECT_TIMELINE_NOTE,
            ActionType.SET_SEVERITY,
            ActionType.ASSIGN_TEAM,
            ActionType.SUBMIT_ROOT_CAUSE,
            ActionType.SUBMIT_MITIGATION,
            ActionType.ADD_NOTE,
            ActionType.RESOLVE_INCIDENT,
        ]

        return Observation(
            scenario_id=state.scenario.scenario_id,
            title=state.scenario.title,
            difficulty=state.scenario.difficulty,
            service=state.scenario.service,
            incident_summary=state.scenario.incident_summary,
            visible_alerts=self._visible_by_type(state, EvidenceType.ALERT),
            visible_logs=self._visible_by_type(state, EvidenceType.LOG),
            visible_runbooks=self._visible_by_type(state, EvidenceType.RUNBOOK),
            visible_timeline_notes=self._visible_by_type(state, EvidenceType.TIMELINE),
            known_facts=list(state.known_facts),
            last_action_result=state.last_action_result,
            selected_severity=state.selected_severity,
            assigned_team=state.assigned_team,
            submitted_root_cause=state.submitted_root_cause,
            submitted_mitigation=state.submitted_mitigation,
            steps_taken=state.steps_taken,
            steps_remaining=max(0, state.scenario.max_steps - state.steps_taken),
            done=state.done,
            action_history_summary=state.action_trace[-8:],
            available_action_types=available_actions,
        )

    def state(self) -> InternalStateSnapshot:
        state = self._require_state()
        return InternalStateSnapshot(
            scenario_id=state.scenario.scenario_id,
            title=state.scenario.title,
            difficulty=state.scenario.difficulty,
            service=state.scenario.service,
            steps_taken=state.steps_taken,
            max_steps=state.scenario.max_steps,
            steps_remaining=max(0, state.scenario.max_steps - state.steps_taken),
            done=state.done,
            total_reward=round(state.total_reward, 4),
            selected_severity=state.selected_severity,
            assigned_team=state.assigned_team,
            submitted_root_cause=state.submitted_root_cause,
            submitted_mitigation=state.submitted_mitigation,
            inspected_evidence_ids=sorted(state.inspected_evidence_ids),
            discovered_relevant_evidence_ids=sorted(state.discovered_relevant_evidence_ids),
            known_facts=list(state.known_facts),
            action_history=[action.model_dump() for action in state.action_history],
            resolution_attempted=state.resolution_attempted,
            premature_resolution=state.premature_resolution,
            terminal_reason=state.terminal_reason,
            last_action_result=state.last_action_result,
        )

    def _is_correct_root_cause(self, state: EpisodeRuntimeState) -> bool:
        return text_matches(
            state.submitted_root_cause,
            state.scenario.true_root_cause,
            state.scenario.true_root_cause_aliases,
        )

    def _is_correct_mitigation(self, state: EpisodeRuntimeState) -> bool:
        return text_matches(
            state.submitted_mitigation,
            state.scenario.true_mitigation,
            state.scenario.true_mitigation_aliases,
        )

    def _evidence_coverage_ratio(self, state: EpisodeRuntimeState) -> float:
        required = set(state.scenario.required_evidence_ids)
        if not required:
            return 1.0
        return len(required & state.inspected_evidence_ids) / len(required)

    def _apply_coverage_milestone_reward(self, state: EpisodeRuntimeState) -> tuple[float, str]:
        coverage = self._evidence_coverage_ratio(state)
        bonus = 0.0
        notes: list[str] = []
        for milestone in self.COVERAGE_MILESTONES:
            if coverage >= milestone and milestone not in state.coverage_milestones_awarded:
                state.coverage_milestones_awarded.add(milestone)
                if milestone >= 1.0:
                    bonus += 0.02
                    notes.append("required evidence set complete")
                else:
                    bonus += 0.015
                    notes.append("evidence coverage reached 50%")
        return bonus, "; ".join(notes)

    def _handle_inspect(
        self,
        state: EpisodeRuntimeState,
        target_type: EvidenceType,
        target_id: Optional[str],
    ) -> tuple[float, str, bool, Optional[str]]:
        if not target_id:
            return -0.03, "Inspect action missing target evidence id.", True, None

        evidence = state.scenario.evidence_map.get(target_id)
        if evidence is None or evidence.type != target_type:
            return -0.03, f"Evidence {target_id} is not available for {target_type.value}.", True, None

        if not self._is_visible(evidence, state):
            return -0.03, f"Evidence {target_id} is locked. Inspect prerequisites first.", True, None

        if target_id in state.inspected_evidence_ids:
            if evidence.relevant:
                return -0.02, f"Evidence {target_id} already inspected.", False, target_id
            return -0.03, f"Repeated irrelevant search on {target_id}.", False, target_id

        state.inspected_evidence_ids.add(target_id)
        if evidence.relevant:
            state.discovered_relevant_evidence_ids.add(target_id)
            state.known_facts.append(f"{evidence.title}: {evidence.content}")
            return 0.03, f"Relevant evidence captured from {target_id}.", False, target_id

        state.known_facts.append(f"{evidence.title}: not currently causal.")
        return 0.01, f"Neutral evidence inspected from {target_id}.", False, target_id

    def _step_set_severity(self, state: EpisodeRuntimeState, value: Optional[str]) -> tuple[float, str, bool]:
        if not value:
            return -0.03, "set_severity requires content like SEV-1/SEV-2/SEV-3.", True

        normalized = value.strip().upper()
        valid_values = {level.value for level in SeverityLevel}
        if normalized not in valid_values:
            return -0.03, f"Invalid severity {value}. Use one of {sorted(valid_values)}.", True

        previous = state.selected_severity
        state.selected_severity = normalized

        if normalized == state.scenario.true_severity.value:
            if previous != normalized:
                return 0.10, f"Severity set correctly to {normalized}.", False
            return -0.005, f"Severity already set to correct value {normalized}.", False

        late_stage = state.steps_taken >= max(3, state.scenario.max_steps // 2)
        if late_stage and previous is not None:
            return -0.04, f"Severity changed late to incorrect value {normalized}.", False
        return -0.02, f"Severity set to {normalized}, currently inconsistent with evidence.", False

    def _step_assign_team(self, state: EpisodeRuntimeState, value: Optional[str]) -> tuple[float, str, bool]:
        if not value:
            return -0.03, "assign_team requires content with a team name.", True

        normalized = value.strip().lower()
        valid_teams = {team.value for team in TeamName}
        if normalized not in valid_teams:
            return -0.03, f"Unknown team {value}. Allowed teams: {sorted(valid_teams)}.", True

        previous = state.assigned_team
        state.assigned_team = normalized

        if normalized == state.scenario.true_owner_team.value:
            if previous != normalized:
                return 0.10, f"Team assignment correct: {normalized}.", False
            return -0.005, f"Team already assigned to {normalized}.", False

        return -0.02, f"Team assignment {normalized} does not match ownership signals.", False

    def _step_submit_root_cause(self, state: EpisodeRuntimeState, value: Optional[str]) -> tuple[float, str, bool]:
        if not value:
            return -0.03, "submit_root_cause requires free-text content.", True

        submitted = value.strip()
        previously_correct = self._is_correct_root_cause(state)
        state.submitted_root_cause = submitted
        currently_correct = self._is_correct_root_cause(state)
        coverage = self._evidence_coverage_ratio(state)
        speculative_penalty = -0.02 if coverage < 0.34 else 0.0

        if currently_correct and not previously_correct:
            reward = 0.20 + speculative_penalty
            if speculative_penalty < 0:
                return reward, "Root cause is correct but submitted before enough evidence was collected.", False
            return reward, "Root cause matches scenario ground truth.", False
        if currently_correct and previously_correct:
            return -0.005, "Root cause already correct; no new progress.", False
        penalty = -0.04 + speculative_penalty
        return penalty, "Submitted root cause is not supported by the current evidence.", False

    def _step_submit_mitigation(self, state: EpisodeRuntimeState, value: Optional[str]) -> tuple[float, str, bool]:
        if not value:
            return -0.03, "submit_mitigation requires free-text content.", True

        submitted = value.strip()
        previously_correct = self._is_correct_mitigation(state)
        state.submitted_mitigation = submitted
        currently_correct = self._is_correct_mitigation(state)
        coverage = self._evidence_coverage_ratio(state)
        speculative_penalty = -0.02 if coverage < 0.34 else 0.0

        if currently_correct and not previously_correct:
            reward = 0.20 + speculative_penalty
            if speculative_penalty < 0:
                return reward, "Mitigation is correct but submitted before enough evidence was collected.", False
            return reward, "Mitigation matches scenario ground truth.", False
        if currently_correct and previously_correct:
            return -0.005, "Mitigation already correct; no new progress.", False
        penalty = -0.04 + speculative_penalty
        return penalty, "Submitted mitigation is unsafe or mismatched.", False

    def _step_resolve(self, state: EpisodeRuntimeState) -> tuple[float, str, bool]:
        state.resolution_attempted = True

        severity_ok = state.selected_severity == state.scenario.true_severity.value
        team_ok = state.assigned_team == state.scenario.true_owner_team.value
        root_ok = self._is_correct_root_cause(state)
        mitigation_ok = self._is_correct_mitigation(state)
        coverage = self._evidence_coverage_ratio(state)

        if severity_ok and team_ok and root_ok and mitigation_ok and coverage >= 0.75:
            state.done = True
            state.premature_resolution = False
            state.terminal_reason = "resolved_safely"
            return 0.20, "Incident resolved safely with sufficient evidence.", False

        penalty = -0.20
        if coverage < 0.5:
            penalty -= 0.05
        if not root_ok:
            penalty -= 0.10

        missing_fields: list[str] = []
        if not severity_ok:
            missing_fields.append("severity")
        if not team_ok:
            missing_fields.append("owner_team")
        if not root_ok:
            missing_fields.append("root_cause")
        if not mitigation_ok:
            missing_fields.append("mitigation")
        if coverage < 0.75:
            missing_fields.append("evidence_coverage")

        state.done = True
        state.premature_resolution = True
        state.terminal_reason = "unsafe_resolution"
        return penalty, f"Unsafe resolution. Missing/correctness gaps: {', '.join(missing_fields)}.", False

    def _finalize_step(
        self,
        state: EpisodeRuntimeState,
        action: Action,
        step_reward: float,
        message: str,
        invalid_action: bool,
        inspected_evidence_id: Optional[str],
    ) -> StepResult:
        if not state.done and state.steps_taken >= state.scenario.max_steps:
            state.done = True
            state.terminal_reason = "max_steps_exceeded"
            message = f"{message} Step budget exhausted."

        state.total_reward += step_reward
        state.last_action_result = message
        state.action_history.append(action)
        state.action_trace.append(
            f"{state.steps_taken:02d}. {action.action_type.value}"
            + (f" target={action.target}" if action.target else "")
            + (f" content={action.content}" if action.content else "")
            + f" -> {message}"
        )

        observation = self._build_observation(state)
        info = StepInfo(
            message=message,
            invalid_action=invalid_action,
            terminal_reason=state.terminal_reason,
            inspected_evidence_id=inspected_evidence_id,
        )
        return StepResult(
            observation=observation,
            reward=round(step_reward, 4),
            done=state.done,
            info=info,
        )

    def step(self, action: Action) -> StepResult:
        state = self._require_state()

        if state.done:
            return StepResult(
                observation=self._build_observation(state),
                reward=0.0,
                done=True,
                info=StepInfo(
                    message="Episode already finished. Call reset() to start a new incident.",
                    invalid_action=True,
                    terminal_reason=state.terminal_reason,
                ),
            )

        state.steps_taken += 1
        step_reward = self.LIVING_PENALTY
        invalid_action = False
        inspected_evidence_id: Optional[str] = None

        action_key = (
            f"{action.action_type.value}|{action.target or ''}|{(action.content or '').strip().lower()}"
        )
        state.action_counts[action_key] += 1

        if action.action_type == ActionType.INSPECT_ALERT:
            delta, message, invalid_action, inspected_evidence_id = self._handle_inspect(
                state,
                EvidenceType.ALERT,
                action.target,
            )
            step_reward += delta
            milestone_bonus, milestone_note = self._apply_coverage_milestone_reward(state)
            step_reward += milestone_bonus
            if milestone_note:
                message = f"{message} Milestone: {milestone_note}."

        elif action.action_type == ActionType.INSPECT_LOG:
            delta, message, invalid_action, inspected_evidence_id = self._handle_inspect(
                state,
                EvidenceType.LOG,
                action.target,
            )
            step_reward += delta
            milestone_bonus, milestone_note = self._apply_coverage_milestone_reward(state)
            step_reward += milestone_bonus
            if milestone_note:
                message = f"{message} Milestone: {milestone_note}."

        elif action.action_type == ActionType.INSPECT_RUNBOOK:
            delta, message, invalid_action, inspected_evidence_id = self._handle_inspect(
                state,
                EvidenceType.RUNBOOK,
                action.target,
            )
            step_reward += delta
            milestone_bonus, milestone_note = self._apply_coverage_milestone_reward(state)
            step_reward += milestone_bonus
            if milestone_note:
                message = f"{message} Milestone: {milestone_note}."

        elif action.action_type == ActionType.INSPECT_TIMELINE_NOTE:
            delta, message, invalid_action, inspected_evidence_id = self._handle_inspect(
                state,
                EvidenceType.TIMELINE,
                action.target,
            )
            step_reward += delta
            milestone_bonus, milestone_note = self._apply_coverage_milestone_reward(state)
            step_reward += milestone_bonus
            if milestone_note:
                message = f"{message} Milestone: {milestone_note}."

        elif action.action_type == ActionType.SET_SEVERITY:
            delta, message, invalid_action = self._step_set_severity(state, action.content)
            step_reward += delta

        elif action.action_type == ActionType.ASSIGN_TEAM:
            delta, message, invalid_action = self._step_assign_team(state, action.content)
            step_reward += delta

        elif action.action_type == ActionType.SUBMIT_ROOT_CAUSE:
            delta, message, invalid_action = self._step_submit_root_cause(state, action.content)
            step_reward += delta

        elif action.action_type == ActionType.SUBMIT_MITIGATION:
            delta, message, invalid_action = self._step_submit_mitigation(state, action.content)
            step_reward += delta

        elif action.action_type == ActionType.ADD_NOTE:
            state.notes.append((action.content or "").strip())
            if len((action.content or "").strip()) >= 10:
                step_reward += 0.005
                message = "Operational note recorded."
            else:
                step_reward -= 0.005
                message = "Note is too short to be useful."

        elif action.action_type == ActionType.RESOLVE_INCIDENT:
            delta, message, invalid_action = self._step_resolve(state)
            step_reward += delta

        else:
            invalid_action = True
            message = f"Unsupported action type: {action.action_type.value}"
            step_reward -= 0.03

        if state.action_counts[action_key] > 1 and action.action_type in {
            ActionType.ADD_NOTE,
            ActionType.SET_SEVERITY,
            ActionType.ASSIGN_TEAM,
            ActionType.SUBMIT_ROOT_CAUSE,
            ActionType.SUBMIT_MITIGATION,
        }:
            step_reward -= 0.01

        if state.action_counts[action_key] >= 3:
            step_reward -= 0.02
            message = f"{message} Repetition penalty applied."

        step_reward = max(-1.0, min(1.0, step_reward))
        return self._finalize_step(
            state=state,
            action=action,
            step_reward=step_reward,
            message=message,
            invalid_action=invalid_action,
            inspected_evidence_id=inspected_evidence_id,
        )

    def grade_current_episode(self) -> GraderResult:
        state = self._require_state()
        snapshot = self.state()
        return grade_episode(state.scenario, snapshot)

    def list_scenarios(self) -> list[ScenarioSummary]:
        return [
            ScenarioSummary(
                scenario_id=item.scenario_id,
                title=item.title,
                difficulty=item.difficulty,
                service=item.service,
                max_steps=item.max_steps,
            )
            for item in self.scenarios
        ]

    def list_tasks(self) -> dict[str, dict[str, object]]:
        task_data: dict[str, dict[str, object]] = {}
        for difficulty in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
            scenario_ids = [item.scenario_id for item in self.scenarios if item.difficulty == difficulty]
            task_data[difficulty.value] = {
                "difficulty": difficulty.value,
                "scenario_count": len(scenario_ids),
                "scenario_ids": scenario_ids,
            }
        return task_data
