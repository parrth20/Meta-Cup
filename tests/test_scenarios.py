from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from models import Scenario


def test_all_scenarios_load_and_count() -> None:
    root = Path(__file__).resolve().parents[1] / "scenarios"
    files = sorted(root.rglob("*.json"))
    assert len(files) == 15

    by_tier = {
        "easy": list((root / "easy").glob("*.json")),
        "medium": list((root / "medium").glob("*.json")),
        "hard": list((root / "hard").glob("*.json")),
    }
    assert len(by_tier["easy"]) == 5
    assert len(by_tier["medium"]) == 5
    assert len(by_tier["hard"]) == 5


def test_required_evidence_exists() -> None:
    root = Path(__file__).resolve().parents[1] / "scenarios"
    ids: set[str] = set()

    for file_path in sorted(root.rglob("*.json")):
        scenario = Scenario.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
        scenario_evidence_ids = {item.id for item in scenario.all_evidence}

        for required_id in scenario.required_evidence_ids:
            assert required_id in scenario_evidence_ids

        assert scenario.scenario_id not in ids
        ids.add(scenario.scenario_id)


def test_step_budget_allows_full_resolution_path() -> None:
    root = Path(__file__).resolve().parents[1] / "scenarios"

    for file_path in sorted(root.rglob("*.json")):
        scenario = Scenario.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
        minimum_actions = len(scenario.required_evidence_ids) + 5
        assert scenario.max_steps >= minimum_actions


def test_initial_visible_evidence_ids_exist_in_scenario() -> None:
    root = Path(__file__).resolve().parents[1] / "scenarios"
    for file_path in sorted(root.rglob("*.json")):
        scenario = Scenario.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
        all_ids = {item.id for item in scenario.all_evidence}
        for evidence_id in scenario.initial_visible_evidence_ids:
            assert evidence_id in all_ids


def test_difficulty_tiers_show_progressive_complexity() -> None:
    root = Path(__file__).resolve().parents[1] / "scenarios"
    by_tier: dict[str, list[Scenario]] = {"easy": [], "medium": [], "hard": []}
    for file_path in sorted(root.rglob("*.json")):
        scenario = Scenario.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
        by_tier[scenario.difficulty.value].append(scenario)

    easy_required = mean(len(s.required_evidence_ids) for s in by_tier["easy"])
    medium_required = mean(len(s.required_evidence_ids) for s in by_tier["medium"])
    hard_required = mean(len(s.required_evidence_ids) for s in by_tier["hard"])
    assert easy_required < medium_required < hard_required

    easy_red_herring = mean(len(s.red_herrings) for s in by_tier["easy"])
    medium_red_herring = mean(len(s.red_herrings) for s in by_tier["medium"])
    hard_red_herring = mean(len(s.red_herrings) for s in by_tier["hard"])
    assert easy_red_herring <= medium_red_herring <= hard_red_herring


def test_scenario_titles_and_root_causes_are_unique() -> None:
    root = Path(__file__).resolve().parents[1] / "scenarios"
    titles: set[str] = set()
    root_causes: set[str] = set()
    for file_path in sorted(root.rglob("*.json")):
        scenario = Scenario.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
        normalized_title = scenario.title.strip().lower()
        normalized_cause = scenario.true_root_cause.strip().lower()
        assert normalized_title not in titles
        assert normalized_cause not in root_causes
        titles.add(normalized_title)
        root_causes.add(normalized_cause)


def test_summary_does_not_directly_leak_hidden_answers() -> None:
    root = Path(__file__).resolve().parents[1] / "scenarios"
    for file_path in sorted(root.rglob("*.json")):
        scenario = Scenario.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
        summary = scenario.incident_summary.lower()
        assert scenario.true_root_cause.lower() not in summary
        assert scenario.true_mitigation.lower() not in summary
