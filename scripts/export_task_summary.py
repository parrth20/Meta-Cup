from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.environment import RunbookOpsEnvironment


def build_summary() -> dict[str, object]:
    env = RunbookOpsEnvironment()
    scenarios = env.list_scenarios()
    tasks = env.list_tasks()

    by_difficulty: dict[str, list[dict[str, object]]] = {"easy": [], "medium": [], "hard": []}
    for scenario in scenarios:
        by_difficulty[scenario.difficulty.value].append(
            {
                "scenario_id": scenario.scenario_id,
                "title": scenario.title,
                "service": scenario.service,
                "max_steps": scenario.max_steps,
            }
        )

    return {
        "task_counts": {key: value["scenario_count"] for key, value in tasks.items()},
        "scenarios": by_difficulty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RunbookOps task summary")
    parser.add_argument(
        "--output",
        default="task_summary.json",
        help="Output file path (default: task_summary.json)",
    )
    args = parser.parse_args()

    summary = build_summary()
    output_path = Path(args.output)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote task summary to {output_path}")


if __name__ == "__main__":
    main()
