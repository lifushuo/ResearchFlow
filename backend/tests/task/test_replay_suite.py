"""Suite integrity tests with synthetic child files and a real four-process graph run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from evals.agent_eval_ext import replay_suite as suite
from evals.agent_eval_ext.replay import load_case

FIXTURE = {
    "schema_version": 1,
    "case_id": "planner-invalid",
    "query": "test request",
    "fault": {"kind": "planner_invalid_schema"},
    "expected": {"status": "failed", "planner_calls": 2},
}


def prepare(tmp_path: Path, count=2):
    names = []
    for i in range(count):
        name = f"case{i}.json"
        item = {**FIXTURE, "case_id": f"case{i}"}
        (tmp_path / name).write_text(json.dumps(item), encoding="utf-8")
        names.append(name)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": names}), encoding="utf-8")
    return manifest


def make_result(case_id, digest, passed=True):
    return {
        "case_id": case_id,
        "case_file_sha256": digest,
        "replay_type": "controlled_simulation",
        "historical_replay": False,
        "passed": passed,
        "status": "failed",
        "fault_kind": "planner_invalid_schema",
    }


def write_child(command, *, passed=True):
    """Produce a valid child fixture; this does not claim to execute an Agent."""
    source = Path(command[command.index("--case") + 1])
    output = Path(command[command.index("--out") + 1])
    case, digest = load_case(source)
    run_id, trace_id = "run-" + case.case_id, "trace-" + case.case_id
    traces = output.parent / "traces"
    traces.mkdir()
    trace = traces / "unit.jsonl"
    calls = 2 if passed else 1
    events = []
    for n in range(calls + 1):
        for status in ("start", "error" if n < calls else "business_error"):
            events.append(
                {
                    "event_version": 2,
                    "seq": len(events) + 1,
                    "trace_id": trace_id,
                    "run_id": run_id,
                    "case_id": case.case_id,
                    "thread_id": run_id,
                    "span_id": str(n),
                    "status": status,
                    "kind": "llm" if n < calls else "graph_node",
                    "name": "llm.planner" if n < calls else "failer",
                    "metadata": {"role": "planner"},
                }
            )
    trace.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    bad = traces / "unit.badcase.json"
    bad.write_text(
        json.dumps({"case_id": case.case_id, "run_id": run_id, "trace_id": trace_id}),
        encoding="utf-8",
    )
    result = {
        **make_result(case.case_id, digest, passed),
        "schema_version": 1,
        "llm_mode": "fake",
        "run_id": run_id,
        "planner_calls": calls,
        "case_spec_sha256": hashlib.sha256(
            json.dumps(case.as_dict(), sort_keys=True).encode()
        ).hexdigest(),
        "assertions": {
            "status": True,
            "planner_calls": calls == 2,
            "error_contains": None,
            "trace_available": True,
            "terminal_trace": True,
        },
        "trace_path": "traces/unit.jsonl",
        "bad_case_path": "traces/unit.badcase.json",
        "trace_status": "recorded",
        "trace_inspection": suite.inspect_trace(trace),
    }
    output.write_text(json.dumps(result), encoding="utf-8")
    return result


def test_suite_validates_fixture_and_digest(tmp_path):
    items = suite.load_manifest(prepare(tmp_path))
    assert [x[0] for x in items] == ["case0", "case1"]
    assert items[0][2] == hashlib.sha256(items[0][1].read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "cases": ["case0.json"]},
        {"schema_version": 1, "cases": []},
        {"schema_version": 1, "cases": ["../case0.json"]},
        {"schema_version": 1, "cases": ["case0.json", "case0.json"]},
        {"schema_version": True, "cases": ["case0.json"]},
    ],
)
def test_suite_rejects_invalid_manifests(tmp_path, payload):
    prepare(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        suite.load_manifest(tmp_path / "manifest.json")


def test_suite_isolated_invocations_and_pass(tmp_path, monkeypatch):
    manifest = prepare(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        write_child(command)
        assert kwargs["env"]["LLM_MODE"] == "fake"
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(suite.subprocess, "run", fake_run)
    output = suite.run_suite(manifest, tmp_path / "result")
    assert output["all_passed"] and output["passed_count"] == 2
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert (tmp_path / "result" / "suite.json").exists()
    with pytest.raises(FileExistsError):
        suite.run_suite(manifest, tmp_path / "result")


def test_partial_missing_result_fails_closed(tmp_path, monkeypatch):
    manifest = prepare(tmp_path, count=1)
    monkeypatch.setattr(
        suite.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0)
    )
    output = suite.run_suite(manifest, tmp_path / "out")
    assert not output["all_passed"]
    assert output["cases"][0]["harness_failure"] == "result_missing"


def test_result_identity_check(tmp_path, monkeypatch):
    manifest = prepare(tmp_path, count=1)

    def fake_run(command, **kwargs):
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps(make_result("other-case", "bad")), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(suite.subprocess, "run", fake_run)
    result = suite.run_suite(manifest, tmp_path / "out")
    assert result["cases"][0]["harness_failure"] == "result_identity_mismatch"


def test_suite_timeout_does_not_mark_success(tmp_path, monkeypatch):
    manifest = prepare(tmp_path, count=1)

    def fake_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="demo", timeout=1)

    monkeypatch.setattr(suite.subprocess, "run", fake_timeout)
    output = suite.run_suite(manifest, tmp_path / "out", timeout_s=1)
    assert output["cases"][0]["harness_failure"] == "subprocess_timeout"


def example_suite(passed_by_case, hash="a" * 64):
    return {
        "kind": "researchflow_replay_suite",
        "schema_version": 1,
        "manifest_sha256": "f" * 64,
        "replay_type": "controlled_simulation",
        "historical_replay": False,
        "total": len(passed_by_case),
        "passed_count": sum(v is True for v in passed_by_case.values()),
        "all_passed": all(v is True for v in passed_by_case.values()),
        "cases": [
            {
                "case_id": k,
                "case_sha256": hash,
                "passed": v,
                "result_passed": v,
                "result_verified": True,
                "process_exit_code": 0 if v is True else 1,
                "harness_failure": None,
            }
            for k, v in passed_by_case.items()
        ],
    }


def test_compare_detects_regression_and_recovery():
    b = example_suite({"a": True, "b": False, "c": True})
    c = example_suite({"a": False, "b": True, "c": True})
    cmp = suite.compare_suites(b, c)
    assert cmp["comparable"]
    assert cmp["regressions"] == 1 and cmp["recoveries"] == 1
    assert not cmp["quality_gate_passed"]


def test_compare_rejects_different_case_bytes_and_sets():
    b = example_suite({"a": True})
    c = example_suite({"a": True}, hash="b" * 64)
    assert suite.compare_suites(b, c)["reason"] == "case_bytes_mismatch"
    assert suite.compare_suites(b, example_suite({"b": True}))["reason"] == "case_set_mismatch"


def test_compare_all_good_and_no_historical_claim():
    s = example_suite({"a": True, "b": True})
    cmp = suite.compare_suites(s, s)
    assert cmp["quality_gate_passed"] and cmp["regressions"] == 0
    assert cmp["code_revision_verified"] is False


def test_refuse_unknown_timeouts(tmp_path):
    manifest = prepare(tmp_path, count=1)
    with pytest.raises(ValueError):
        suite.run_suite(manifest, tmp_path / "out", timeout_s=0)
    assert not (tmp_path / "out").exists()


def test_cli_validate_has_no_side_effects(tmp_path, capsys):
    manifest = prepare(tmp_path)
    assert suite.main(["validate", "--manifest", str(manifest)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["cases"]) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "case0.json",
        "case1.json",
        "manifest.json",
    ]


def test_compare_rejects_nonboolean_verdicts():
    a = example_suite({"a": True})
    b = example_suite({"a": "true"})
    with pytest.raises(ValueError):
        suite.compare_suites(a, b)


@pytest.mark.parametrize(
    "damage",
    [
        "passed",
        "assertions",
        "trace_missing",
        "trace_escape",
        "trace_identity",
        "bad_case_identity",
        "snapshot",
        "count",
    ],
)
def test_inconsistent_child_artifacts_never_pass(tmp_path, monkeypatch, damage):
    manifest = prepare(tmp_path, count=1)

    def child(command, **kwargs):
        detail = write_child(command)
        output = Path(command[command.index("--out") + 1])
        if damage == "passed":
            detail["passed"] = "true"
        elif damage == "assertions":
            detail["assertions"]["status"] = False
        elif damage == "trace_missing":
            (output.parent / detail["trace_path"]).unlink()
        elif damage == "trace_escape":
            detail["trace_path"] = "../outside.jsonl"
        elif damage == "trace_identity":
            path = output.parent / detail["trace_path"]
            events = [json.loads(line) for line in path.read_text().splitlines()]
            for event in events:
                event["run_id"] = "wrong-run"
            path.write_text("\n".join(json.dumps(e) for e in events))
            detail["trace_inspection"] = suite.inspect_trace(path)
        elif damage == "bad_case_identity":
            (output.parent / detail["bad_case_path"]).write_text('{"case_id":"wrong"}')
        elif damage == "snapshot":
            Path(command[command.index("--case") + 1]).write_text("changed bytes")
        else:
            detail["planner_calls"] = 1
            detail["assertions"]["planner_calls"] = False
            detail["passed"] = False
        output.write_text(json.dumps(detail))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(suite.subprocess, "run", child)
    result = suite.run_suite(manifest, tmp_path / "out")
    assert result["all_passed"] is False and result["passed_count"] == 0
    assert result["cases"][0]["harness_failure"]


def test_timeout_preserved_even_if_valid_result_was_written(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        write_child(command)
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(suite.subprocess, "run", timeout)
    output = suite.run_suite(prepare(tmp_path, 1), tmp_path / "out")
    entry = output["cases"][0]
    assert entry["result_verified"] and not entry["passed"]
    assert entry["harness_failure"] == "subprocess_timeout"


def test_nonzero_process_cannot_pass_even_with_valid_result(tmp_path, monkeypatch):
    def crashed(command, **kwargs):
        write_child(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(suite.subprocess, "run", crashed)
    result = suite.run_suite(prepare(tmp_path, 1), tmp_path / "out")
    assert result["cases"][0]["harness_failure"] == "process_result_disagreement"
    assert not result["all_passed"]


def test_input_bytes_are_frozen_before_first_child(tmp_path, monkeypatch):
    manifest = prepare(tmp_path)
    original_manifest = manifest.read_bytes()
    original_case = (tmp_path / "case1.json").read_bytes()
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            manifest.write_text("not JSON anymore")
            (tmp_path / "case1.json").write_text("changed original")
        write_child(command)
        assert kwargs["cwd"] == suite._BACKEND
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(suite.subprocess, "run", child)
    output = suite.run_suite(manifest, tmp_path / "out")
    assert output["all_passed"]
    assert output["manifest_sha256"] == hashlib.sha256(original_manifest).hexdigest()
    assert (tmp_path / "out/inputs/case1.json").read_bytes() == original_case


def test_invalid_last_case_prevents_every_subprocess(tmp_path, monkeypatch):
    manifest = prepare(tmp_path)
    (tmp_path / "case1.json").write_text("invalid")
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: pytest.fail("child spawned"))
    with pytest.raises(ValueError):
        suite.run_suite(manifest, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_case_symlink_cannot_escape_manifest_directory(tmp_path):
    source = tmp_path / "outside.json"
    source.write_text(json.dumps(FIXTURE))
    folder = tmp_path / "inside"
    folder.mkdir()
    (folder / "link.json").symlink_to(source)
    (folder / "manifest.json").write_text('{"schema_version":1,"cases":["link.json"]}')
    with pytest.raises(ValueError, match="outside"):
        suite.load_manifest(folder / "manifest.json")


def test_clean_child_environment_blocks_secrets_and_telemetry(monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "secret")
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "secret")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("RF_JUDGE_BATCH_TRACE_DIR", "/private")
    cleaned = suite._clean_environment()
    assert "CUSTOM_API_KEY" not in cleaned and "EVAL_JUDGE_API_KEY" not in cleaned
    assert "DATABASE_URL" not in cleaned and "RF_JUDGE_BATCH_TRACE_DIR" not in cleaned
    assert cleaned["LLM_MODE"] == "fake" and cleaned["LANGCHAIN_TRACING_V2"] == "false"


@pytest.mark.parametrize("digest", [None, "", "same", 123])
def test_compare_requires_real_case_fingerprints(digest):
    doc = example_suite({"a": True}, hash=digest)
    with pytest.raises(ValueError, match="fingerprint"):
        suite.compare_suites(doc, doc)


@pytest.mark.parametrize(
    "mutation", ["empty", "count", "total", "all_passed", "execution", "historical"]
)
def test_comparison_rejects_inconsistent_suite_metadata(mutation):
    baseline = example_suite({"a": True})
    candidate = deepcopy(baseline)
    if mutation == "empty":
        candidate["cases"] = []
    elif mutation == "count":
        candidate["passed_count"] = 0
    elif mutation == "total":
        candidate["total"] = True
    elif mutation == "all_passed":
        candidate["all_passed"] = False
    elif mutation == "execution":
        candidate["cases"][0]["process_exit_code"] = 1
    else:
        candidate["historical_replay"] = True
    with pytest.raises(ValueError):
        suite.compare_suites(baseline, candidate)


def test_failed_candidate_never_passes_gate_even_without_new_regression():
    doc = example_suite({"a": False})
    output = suite.compare_suites(doc, doc)
    assert output["regressions"] == 0 and output["quality_gate_passed"] is False


def test_compare_cli_gate_exit_and_exclusive_output(tmp_path, monkeypatch):
    manifest = prepare(tmp_path, 1)
    passed = [True]

    def child(command, **kwargs):
        write_child(command, passed=passed[0])
        return subprocess.CompletedProcess(command, 0 if passed[0] else 1)

    monkeypatch.setattr(suite.subprocess, "run", child)
    suite.run_suite(manifest, tmp_path / "baseline")
    passed[0] = False
    suite.run_suite(manifest, tmp_path / "candidate")
    baseline, candidate, out = (
        tmp_path / name
        for name in ("baseline/suite.json", "candidate/suite.json", "new/compare.json")
    )
    args = [
        "compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--out",
        str(out),
    ]
    assert suite.main(args) == 1
    before = out.read_bytes()
    assert json.loads(before)["regressions"] == 1
    with pytest.raises(FileExistsError):
        suite.main(args)
    assert out.read_bytes() == before


def test_comparison_cli_rechecks_saved_artifact_bytes(tmp_path, monkeypatch, capsys):
    def child(command, **kwargs):
        write_child(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(suite.subprocess, "run", child)
    result = suite.run_suite(prepare(tmp_path, 1), tmp_path / "out")
    doc = tmp_path / "out/suite.json"
    args = ["compare", "--baseline", str(doc), "--candidate", str(doc)]
    assert suite.main(args) == 0
    capsys.readouterr()
    trace = tmp_path / "out" / result["cases"][0]["case_id"] / "traces/unit.jsonl"
    trace.write_text("")
    assert suite.main(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["artifact_integrity_passed"] is False
    assert output["quality_gate_passed"] is False and output["regressions"] is None


def test_real_suite_executes_four_separate_interpreters(tmp_path, monkeypatch):
    from evals.agent_eval_ext import replay
    from research_engine import graph

    def parent_only_factory(*args):
        raise AssertionError("parent monkeypatch must not reach child interpreters")

    monkeypatch.setattr(graph, "get_llm", parent_only_factory)
    assert replay._REPLAY_LOCK.acquire(blocking=False)
    try:
        manifest = Path(__file__).parents[2] / "evals/agent_eval_ext/fixtures/replay_suite_v1.json"
        result = suite.run_suite(
            manifest, tmp_path / "real-suite", revision="unverified-unit-label"
        )
    finally:
        replay._REPLAY_LOCK.release()
    assert result["all_passed"] and result["passed_count"] == 4
    expected = {
        "planner-healthy-gate": ("awaiting_approval", 1),
        "planner-invalid-schema": ("failed", 2),
        "planner-rate-limit": ("failed", 1),
        "planner-upstream-522": ("failed", 2),
    }
    for row in result["cases"]:
        detail = json.loads((tmp_path / "real-suite" / row["result_path"]).read_text())
        assert (detail["status"], detail["planner_calls"]) == expected[row["case_id"]]
        assert row["result_verified"] is True
    assert result["revision_verified"] is False
