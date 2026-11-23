"""P2 GEPA read-only audit projection and privacy regression tests."""

import json
from pathlib import Path

from app.services.planner_evolution_insights import project_journal


def _event(folder: Path, seq: int, kind: str, payload: dict):
    journal = folder / "journal"
    journal.mkdir(exist_ok=True)
    (journal / f"event-{seq:05d}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sequence": seq,
                "kind": kind,
                "recorded_at": "2026-11-21T00:00:00+00:00",
                "payload": payload,
            }
        ),
        encoding="utf-8",
    )


def test_observations_and_reflection_are_projected_without_sensitive_text(tmp_path):
    secret = "PRIVATE-QUERY-AND-PROMPT-NOT-FOR-THE-CLIENT"
    _event(
        tmp_path,
        1,
        "started",
        {"dataset": {"cases": [{"query": secret}]}, "baseline_override": secret},
    )
    _event(
        tmp_path,
        2,
        "observation",
        {
            "prompt_sha256": "a" * 64,
            "prompt_override": secret,
            "observation": {
                "case_id": "sample01",
                "split": "train",
                "required_count": 4,
                "covered_indices": [1, 2, 3],
                "task_queries": [secret],
                "planner_tokens": [25, 5],
                "judge_tokens": [15, 3],
            },
        },
    )
    _event(tmp_path, 3, "reflection_returned", {"response_sha256": "b" * 64, "answer": secret})
    _event(
        tmp_path,
        4,
        "candidate_frozen",
        {"override": secret, "sha256": "c" * 64, "accepted_for_holdout": True},
    )
    audit = project_journal(tmp_path)
    assert audit["available"] is True
    assert audit["event_count"] == 4
    assert audit["candidates"][0]["splits"]["train"]["mean_coverage"] == 0.75
    assert audit["observed_tokens"]["planner_input"] == 25
    assert audit["reflection_count"] == 1
    assert audit["pareto_parent_selection"] == "not_recorded"
    assert secret not in json.dumps(audit)
    assert "sample01" not in json.dumps(audit)


def test_invalid_event_sequence_fails_closed(tmp_path):
    _event(tmp_path, 2, "started", {})
    result = project_journal(tmp_path)
    assert result["available"] is False
    assert result["error"] == "journal_sequence_inconsistent"


def test_unmeasured_not_counted_as_zero(tmp_path):
    _event(
        tmp_path,
        1,
        "observation",
        {
            "prompt_sha256": "a" * 64,
            "observation": {
                "case_id": "caseA",
                "split": "validation",
                "required_count": 4,
                "covered_indices": [],
                "error": "TimeoutError",
            },
        },
    )
    result = project_journal(tmp_path)
    row = result["candidates"][0]
    assert row["samples"] == 1 and row["measured"] == 0 and row["errors"] == 1
    assert row["splits"]["validation"]["mean_coverage"] is None
