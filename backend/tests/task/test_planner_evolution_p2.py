"""P2 web-job audit authorization and frozen holdout anti-reselection."""

import json
import uuid

import pytest

from app.services.planner_evolution_jobs import (
    _has_consumed_holdout,
    _holdout_fingerprint,
    claim_next,
    create_job,
    job_audit,
)


def _sample_dataset(training_label: str):
    return {
        "schema_version": 1,
        "dataset_id": "test",
        "provenance": "reviewed",
        "cases": [
            {
                "case_id": "train-a",
                "group": "train",
                "split": "train",
                "query": training_label,
                "required_aspects": ["a", "b"],
            },
            {
                "case_id": "holdout-a",
                "group": "holdout",
                "split": "holdout",
                "query": "held-out question",
                "required_aspects": ["a", "b"],
            },
        ],
    }


def test_frozen_holdout_is_not_reset_by_training_backflow():
    assert _holdout_fingerprint(_sample_dataset("version 1")) == _holdout_fingerprint(
        _sample_dataset("version 2")
    )
    changed = _sample_dataset("version 1")
    changed["cases"][1]["required_aspects"][0] = "different"
    assert _holdout_fingerprint(changed) != _holdout_fingerprint(_sample_dataset("version 1"))


def test_consumed_live_holdout_blocks_future_web_live_admission(tmp_path):
    fingerprint = "h" * 64
    root = tmp_path
    experiment_id = "gepa-" + "a" * 24
    req = {"job_id": experiment_id, "mode": "live", "holdout_sha256": fingerprint}
    folder = root / "planner-evolution" / "jobs" / experiment_id
    folder.mkdir(parents=True)
    record = [(folder, req, {"state": "quality_rejected"})]
    assert _has_consumed_holdout(root, record, fingerprint)
    assert not _has_consumed_holdout(root, record, "x" * 64)


def test_failed_before_holdout_can_be_retried_and_after_holdout_cannot(tmp_path):
    root = tmp_path
    fingerprint = "h" * 64
    experiment_id = "gepa-" + "a" * 24
    folder = root / "planner-evolution" / "jobs" / experiment_id
    folder.mkdir(parents=True)
    req = {"job_id": experiment_id, "mode": "live", "holdout_sha256": fingerprint}
    rows = [(folder, req, {"state": "failed"})]
    assert not _has_consumed_holdout(root, rows, fingerprint)
    journal = root / "planner-evolution" / "experiments" / experiment_id / "journal"
    journal.mkdir(parents=True)
    (journal / "event-00001.json").write_text(
        json.dumps(
            {
                "sequence": 1,
                "kind": "observation",
                "recorded_at": "2026-11-21T00:00:00Z",
                "payload": {
                    "prompt_sha256": "a" * 64,
                    "observation": {
                        "case_id": "holdout-a",
                        "split": "holdout",
                        "required_count": 2,
                        "covered_indices": [1],
                        "task_queries": ["one"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    assert _has_consumed_holdout(root, rows, fingerprint)


def test_audit_is_owner_scoped_and_no_model_calls(tmp_path):
    row = create_job(
        tmp_path,
        "alice",
        request_id=str(uuid.uuid4()),
        mode="demo",
        max_evals=64,
        proposals=4,
        seed=42,
        consent=None,
    )
    assert row["state"] == "queued"
    with pytest.raises(KeyError):
        job_audit(tmp_path, "bob", row["job_id"])
    own = job_audit(tmp_path, "alice", row["job_id"])
    assert own["journal"]["available"] is False
    assert own["full_report_ab"] == "manual_not_started_by_gepa_worker"
    assert claim_next(tmp_path, "demo") is not None


def test_no_validation_improvement_does_not_consume_holdout(tmp_path):
    root = tmp_path
    fingerprint = "h" * 64
    experiment_id = "gepa-" + "b" * 24
    folder = root / "planner-evolution" / "jobs" / experiment_id
    folder.mkdir(parents=True)
    req = {"job_id": experiment_id, "mode": "live", "holdout_sha256": fingerprint}
    journal = root / "planner-evolution" / "experiments" / experiment_id / "journal"
    journal.mkdir(parents=True)
    (journal / "event-00001.json").write_text(
        json.dumps(
            {
                "sequence": 1,
                "kind": "candidate_frozen",
                "recorded_at": "2026-11-21T00:00:00Z",
                "payload": {"sha256": "a" * 64, "accepted_for_holdout": False},
            }
        ),
        encoding="utf-8",
    )
    assert not _has_consumed_holdout(
        root, [(folder, req, {"state": "quality_rejected"})], fingerprint
    )


def test_result_binding_rejects_mismatched_prompt_and_diff(monkeypatch):
    import hashlib
    import sys
    import types
    from difflib import unified_diff

    from app.services.evaluation_closure import input_hash
    from app.services.planner_evolution_jobs import EvolutionJobError, _verify_result_binding

    base, next_prompt = "Example baseline", "Example baseline\nExtra rule"
    fake_prompts = types.ModuleType("research_engine.prompts")
    fake_prompts.PLANNER_PROMPT_V2 = base
    fake_long_run = types.ModuleType("evals.agent_eval_ext.long_run")
    fake_long_run.sha256_text = lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    monkeypatch.setitem(sys.modules, "research_engine.prompts", fake_prompts)
    monkeypatch.setitem(sys.modules, "evals.agent_eval_ext.long_run", fake_long_run)
    dataset = {"schema_version": 1, "cases": []}
    result = {
        "schema_version": 1,
        "evidence_kind": "scripted_demo",
        "dataset": dataset,
        "dataset_sha256": input_hash(dataset),
        "baseline": {"override": None, "sha256": fake_long_run.sha256_text(base)},
        "candidate": {"override": next_prompt, "sha256": fake_long_run.sha256_text(next_prompt)},
        "prompt_diff": "\n".join(
            unified_diff(
                base.splitlines(),
                next_prompt.splitlines(),
                fromfile="baseline",
                tofile="candidate",
                lineterm="",
            )
        ),
    }
    _verify_result_binding({"mode": "demo"}, result)
    result["candidate"]["sha256"] = "0" * 64
    with pytest.raises(EvolutionJobError, match="result_candidate_hash_mismatch"):
        _verify_result_binding({"mode": "demo"}, result)
    result["candidate"]["sha256"] = fake_long_run.sha256_text(next_prompt)
    result["prompt_diff"] = "+ fake improved"
    with pytest.raises(EvolutionJobError, match="result_prompt_diff_mismatch"):
        _verify_result_binding({"mode": "demo"}, result)
