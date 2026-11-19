"""Offline coverage reconstruction, input honesty and artifact boundary checks."""

import json
from copy import deepcopy

import pytest

from evals.agent_eval_ext.audit import report_audit, sha256_file
from evals.agent_eval_ext.judge_batch_audit import analyze_report, main


def report(claims, *, case_id="x", completed=True):
    return {"id": case_id, "completed": completed, "claim_verdicts": claims}


def verdict(supported=True):
    return {"judged": True, "supported": supported}


def test_partial_full_and_no_cases():
    sample = {
        "results": [
            report([verdict(False), *[{"judged": False} for _ in range(7)]]),
            {"id": "failed", "completed": False, "error": "planner failed"},
        ]
    }
    original = deepcopy(sample)
    result = analyze_report(sample)
    assert sample == original
    assert (result["total_claims"], result["judged_claims"], result["unjudged_claims"]) == (8, 1, 7)
    assert (result["full_unjudged_batches"], result["partial_batches"]) == (1, 1)
    assert result["reports"][1]["pipeline_error"] == "planner failed"
    batches = result["reports"][0]["problem_batches"]
    assert batches[0]["unjudged_positions"] == [2, 3, 4]
    assert batches[1]["claim_positions"] == [5, 6, 7, 8]
    assert all(b["root_cause"] == "unknown" for b in batches)
    assert result["actual_requests_observed"] is False


def test_unsupported_is_still_judged():
    output = analyze_report({"results": [report([verdict(False)])]})
    assert output["judge_coverage"] == 1.0
    assert output["incomplete_batches"] == 0
    assert output["complete_batches"] == 1


def test_all_claims_counted_when_only_short_last_batch():
    output = analyze_report({"results": [report([verdict() for _ in range(5)])]})
    assert output["total_claims"] == 5
    assert output["total_batches"] == output["complete_batches"] == 2
    assert output["incomplete_batches"] == 0


def test_empty_reports_are_unmeasured():
    output = analyze_report({"results": []})
    assert output["judge_coverage"] is None
    assert output["total_batches"] == 0


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "4", None])
def test_reject_invalid_batch_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        analyze_report({"results": []}, batch_size=size)


def test_missing_completed_records_do_not_claim_full_coverage():
    run = {
        "results": [
            report([verdict()]),
            {"id": "missing", "completed": True},
            report([], case_id="explicit-empty"),
        ]
    }
    output = analyze_report(run)
    assert output["judge_coverage"] is None
    assert output["cases_missing_claim_records"] == ["missing"]
    assert [r["claim_records_status"] for r in output["reports"]] == ["present", "missing", "empty"]
    assert output["warnings"]


@pytest.mark.parametrize(
    "invalid",
    [
        {"judged": True},
        {"judged": True, "supported": None},
        {"judged": True, "supported": 1},
        {"judged": "true", "supported": True},
    ],
)
def test_measurement_contract_matches_existing_audit(invalid):
    run = {"results": [report([verdict(False), invalid])]}
    batches = analyze_report(run)
    audit = report_audit(run)
    assert batches["judged_claims"] == 1
    assert batches["judge_coverage"] == audit["judge"]["coverage"] == 0.5
    assert batches["unjudged_claims"] == audit["judge"]["unjudged"] == 1
    if invalid["judged"] is True:
        assert batches["reports"][0]["invalid_verdict_positions"] == [2]


def test_custom_size_short_batch_and_positions_reset_between_reports():
    run = {
        "results": [
            report([verdict(), verdict(), {"judged": False}]),
            report([{"judged": False}], case_id="second"),
        ],
        "candidate": {"results": []},
    }
    output = analyze_report(run, batch_size=2)
    first = output["reports"][0]["problem_batches"][0]
    second = output["reports"][1]["problem_batches"][0]
    assert first["batch_number"] == 2 and first["claim_positions"] == [3]
    assert second["batch_number"] == 1 and second["claim_positions"] == [1]
    assert first["is_full_batch"] is False
    assert output["total_batches"] == 3
    assert output["full_unjudged_batches"] == 2
    assert any("candidate" in w for w in output["warnings"])


@pytest.mark.parametrize(
    "run",
    [
        {},
        {"mode": "memory", "results": []},
        {"results": [report([None])]},
        {"results": [report({"judged": True})]},
        {"results": [report([]), report([])]},
    ],
)
def test_reject_malformed_report_inputs(run):
    with pytest.raises(ValueError):
        analyze_report(run)


@pytest.mark.parametrize("entry", ["standalone", "cli"])
def test_cli_preserves_input_and_refuses_overwrite(entry, monkeypatch, tmp_path):
    from evals.agent_eval_ext import cli

    source, output = tmp_path / "source.json", tmp_path / "analysis.json"
    source.write_text(json.dumps({"results": [report([verdict(False)])]}), encoding="utf-8-sig")
    original = source.read_bytes()
    args = ["audit"] + (["judge-batches"] if entry == "cli" else [])
    args += ["--input", str(source), "--out", str(output), "--batch-size", "4"]
    monkeypatch.setattr("sys.argv", args)
    run = cli.main if entry == "cli" else main
    run()
    saved = output.read_bytes()
    data = json.loads(saved)
    assert data["input_sha256"] == sha256_file(source)
    assert data["extension_schema_version"] == 1
    assert data["judge_coverage"] == 1.0
    with pytest.raises(FileExistsError):
        run()
    assert source.read_bytes() == original and output.read_bytes() == saved
