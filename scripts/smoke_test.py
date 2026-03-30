from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Action, ActionType
from server.environment import RunbookOpsEnvironment


def main() -> None:
    env = RunbookOpsEnvironment()
    observation = env.reset(scenario_id="hard_auth_multi_signal_conflict")
    print(f"Scenario: {observation.scenario_id}")

    actions = [
        Action(action_type=ActionType.INSPECT_ALERT, target="ha1_alert_nonce_mismatch"),
        Action(action_type=ActionType.INSPECT_LOG, target="ha1_log_clock_skew_node7"),
        Action(action_type=ActionType.INSPECT_LOG, target="ha1_log_ntp_daemon_stopped"),
        Action(action_type=ActionType.INSPECT_TIMELINE_NOTE, target="ha1_timeline_image_rollout"),
        Action(action_type=ActionType.INSPECT_RUNBOOK, target="ha1_runbook_time_sync_recover"),
        Action(action_type=ActionType.SET_SEVERITY, content="SEV-1"),
        Action(action_type=ActionType.ASSIGN_TEAM, content="auth-oncall"),
        Action(
            action_type=ActionType.SUBMIT_ROOT_CAUSE,
            content="time sync failure triggered nonce mismatches",
        ),
        Action(
            action_type=ActionType.SUBMIT_MITIGATION,
            content="restart time sync service and recover nonce checks",
        ),
        Action(action_type=ActionType.RESOLVE_INCIDENT),
    ]

    for index, action in enumerate(actions, start=1):
        result = env.step(action)
        print(
            f"step={index} action={action.action_type.value} "
            f"reward={result.reward:.4f} done={result.done}"
        )
        if result.done:
            break

    grade = env.grade_current_episode()
    print(f"score={grade.score:.4f}")


if __name__ == "__main__":
    main()
