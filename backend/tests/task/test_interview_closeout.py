"""Offline tests for the additive interview closeout report; no model calls."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals.agent_eval_ext.interview_closeout import (
    _metric_display,
    _validate_tool,
    _validate_unified,
    build_interview_summary,
    generate,
    main,
    render_html,
)


@pytest.fixture
def run() -> dict:
    missing = {
        "status": "unmeasured",
        "reason": "artifact_not_supplied",
        "metrics": {},
        "cases": [],
    }
    return {
        "schema_version": 1,
        "kind": "researchflow_unified_evaluation",
        "generated_at": "2026-11-11T00:00:00Z",
        "not_a_single_score": True,
        "cross_source_pairing_verified": False,
        "dataset": {
            "dataset_id": "test",
            "dataset_version": 1,
            "split": "regression",
            "wrapper_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "cases": [
                {"case_id": "healthy", "case_sha256": "1" * 64},
                {"case_id": "bad", "case_sha256": "2" * 64},
            ],
        },
        "dimensions": {
            "outcome": {
                "status": "measured",
                "sample_count": 2,
                "source": {"basename": "report.json", "sha256": "a" * 64},
                "metrics": {
                    "completion_rate": {
                        "status": "measured",
                        "value": 0.5,
                        "unit": "ratio",
                        "n": 2,
                        "note": "",
                    },
                    "api_cost_usd": {
                        "status": "unmeasured",
                        "value": None,
                        "unit": "usd",
                        "n": None,
                        "note": "",
                    },
                },
                "cases": [
                    {
                        "case_id": "one",
                        "test_result": "unmeasured",
                        "agent_status": "completed",
                        "completion": True,
                    },
                    {
                        "case_id": "two",
                        "test_result": "unmeasured",
                        "agent_status": "failed",
                        "completion": False,
                    },
                ],
            },
            "reliability": {
                "status": "measured",
                "sample_count": 2,
                "contract": "controlled_fault_handling_not_task_completion",
                "integrity_verified": True,
                "manifest_sha256": "b" * 64,
                "source": {"basename": "suite.json", "sha256": "c" * 64},
                "metrics": {
                    "case_pass_rate": {
                        "status": "measured",
                        "value": 0.5,
                        "unit": "ratio",
                        "n": 2,
                        "note": "",
                    }
                },
                "cases": [
                    {
                        "case_id": "healthy",
                        "case_sha256": "1" * 64,
                        "test_result": "pass",
                        "agent_status": "awaiting_approval",
                        "result_verified": True,
                    },
                    {
                        "case_id": "bad",
                        "case_sha256": "2" * 64,
                        "test_result": "fail",
                        "agent_status": "failed",
                        "result_verified": True,
                    },
                ],
            },
            "retrieval": deepcopy(missing),
            "grounding": deepcopy(missing),
        },
    }


@pytest.fixture
def tool() -> dict:
    from evals.agent_eval_ext.tool_safety import evaluate, load_gold, load_observed

    fixtures = Path(__file__).parents[2] / "evals/agent_eval_ext/fixtures"
    gold, digest = load_gold(fixtures / "tool_gold_v1.json")
    observed, observed_digest = load_observed(fixtures / "tool_observed_pass_v1.json")
    return evaluate(gold, observed, gold_sha256=digest, observed_sha256=observed_digest)


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_report_only_generates_html_and_triage(run, tmp_path):
    candidate = _write(tmp_path / "unified.json", run)
    result = generate(
        unified_path=candidate,
        html_out=tmp_path / "report.html",
        summary_out=tmp_path / "summary.json",
        triage_out=tmp_path / "triage.json",
    )
    assert result["triage_count"] == 2
    assert result["paid_requests"] == 0
    html = (tmp_path / "report.html").read_text()
    assert "任务完成率" in html
    assert "50.00%" in html
    assert "Judge" in html
    assert "未测量" in html
    assert "不是统一总分" in html
    triage = json.loads((tmp_path / "triage.json").read_text())
    assert [c["case_id"] for c in triage["drafts"]] == ["two", "bad"]
    assert all(c["root_cause"] == "unknown" for c in triage["drafts"])
    assert triage["auto_promoted_to_regression_set"] is False
    assert json.loads(candidate.read_text()) == run


def test_badcase_does_not_treat_expected_failed_agent_as_bad_case(run, tmp_path):
    run["dimensions"]["reliability"]["cases"][1]["test_result"] = "pass"
    run["dimensions"]["reliability"]["metrics"]["case_pass_rate"]["value"] = 1.0
    path = _write(tmp_path / "c.json", run)
    result = generate(
        unified_path=path, html_out=tmp_path / "x.html", triage_out=tmp_path / "b.json"
    )
    assert result["triage_count"] == 1  # historical research failure only


def test_no_badcases_still_does_not_claim_perfect_agent(run, tmp_path):
    run["dimensions"]["outcome"]["cases"][1]["completion"] = True
    run["dimensions"]["outcome"]["metrics"]["completion_rate"]["value"] = 1.0
    run["dimensions"]["reliability"]["cases"][1]["test_result"] = "pass"
    run["dimensions"]["reliability"]["metrics"]["case_pass_rate"]["value"] = 1.0
    path = _write(tmp_path / "c.json", run)
    generate(unified_path=path, html_out=tmp_path / "x.html")
    assert "不能据此认定全部系统无故障" in (tmp_path / "x.html").read_text()


def test_tool_controlled_fixture_is_labeled_not_real(run, tool, tmp_path):
    src = _write(tmp_path / "candidate.json", run)
    tool_path = _write(tmp_path / "tool.json", tool)
    generate(
        unified_path=src,
        tool_path=tool_path,
        html_out=tmp_path / "out.html",
        summary_out=tmp_path / "sum.json",
    )
    html = (tmp_path / "out.html").read_text()
    assert "受控合成契约" in html
    assert "工具选择准确率（合成）" in html
    data = json.loads((tmp_path / "sum.json").read_text())
    assert data["tool_fixture"]["actual_agent_executed"] is False
    assert data["sources"][1]["sha256"] == hashlib.sha256(tool_path.read_bytes()).hexdigest()


def test_tool_rejects_forged_live_provenance(run, tool, tmp_path):
    tool["actual_agent_executed"] = True
    c = _write(tmp_path / "c.json", run)
    t = _write(tmp_path / "t.json", tool)
    with pytest.raises(ValueError, match="provenance"):
        generate(unified_path=c, tool_path=t, html_out=tmp_path / "out.html")
    assert not (tmp_path / "out.html").exists()


@pytest.mark.parametrize(
    "field, bad",
    [
        ("kind", "another"),
        ("schema_version", 2),
        ("not_a_single_score", False),
        ("cross_source_pairing_verified", True),
    ],
)
def test_wrong_unified_top_level(run, field, bad):
    run[field] = bad
    with pytest.raises(ValueError):
        _validate_unified(run)


@pytest.mark.parametrize("status", ["fake", "", None, 123])
def test_wrong_dimension_status(run, status):
    run["dimensions"]["outcome"]["status"] = status
    with pytest.raises(ValueError):
        _validate_unified(run)


@pytest.mark.parametrize(
    "entry",
    [
        {"status": "unmeasured", "value": 0},
        {"status": "measured", "value": None},
        {"status": "measured", "value": True},
        {"status": "measured", "value": 2, "unit": "ratio"},
        {"status": "measured", "value": 0.5, "n": -1},
        {"status": "invalid", "value": float("inf")},
    ],
)
def test_invalid_metric_never_renders_as_real(run, entry):
    run["dimensions"]["outcome"]["metrics"]["completion_rate"] = entry
    with pytest.raises(ValueError):
        _validate_unified(run)


def test_xss_escapes_untrusted_metric_note(run, tmp_path):
    run["dimensions"]["outcome"]["metrics"]["completion_rate"]["note"] = (
        '<script>alert("x")</script>'
    )
    src = _write(tmp_path / "u.json", run)
    generate(unified_path=src, html_out=tmp_path / "r.html")
    h = (tmp_path / "r.html").read_text()
    assert "&lt;script&gt;" in h
    assert '<script>alert("x")</script>' not in h


def test_refuse_input_overwrite(run, tmp_path):
    src = _write(tmp_path / "input.json", run)
    with pytest.raises(ValueError, match="distinct"):
        generate(unified_path=src, html_out=src)
    assert json.loads(src.read_text()) == run


def test_refuse_duplicate_targets(run, tmp_path):
    src = _write(tmp_path / "input.json", run)
    with pytest.raises(ValueError, match="unique"):
        generate(
            unified_path=src, html_out=tmp_path / "same.json", triage_out=tmp_path / "same.json"
        )


def test_refuse_existing_output_before_any_writes(run, tmp_path):
    src = _write(tmp_path / "input.json", run)
    existing = tmp_path / "triage.json"
    existing.write_text("locked")
    with pytest.raises(FileExistsError):
        generate(unified_path=src, html_out=tmp_path / "html.html", triage_out=existing)
    assert existing.read_text() == "locked"
    assert not (tmp_path / "html.html").exists()


def test_reject_nonstandard_numeric_json(run, tmp_path):
    src = _write(tmp_path / "input.json", run)
    src.write_text(src.read_text().replace('"value": 0.5', '"value": NaN'))
    assert main(["--unified", str(src), "--html", str(tmp_path / "html.html")]) == 2
    assert not (tmp_path / "html.html").exists()


def test_cli_success_and_no_overwrite(run, tmp_path):
    src = _write(tmp_path / "input.json", run)
    target = tmp_path / "html.html"
    assert main(["--unified", str(src), "--html", str(target)]) == 0
    assert main(["--unified", str(src), "--html", str(target)]) == 2


def test_comparison_preserves_replay_scope(run, tmp_path):
    candidate = deepcopy(run)
    candidate["dimensions"]["reliability"]["cases"][0]["test_result"] = "fail"
    candidate["dimensions"]["reliability"]["metrics"]["case_pass_rate"]["value"] = 0.0
    old = _write(tmp_path / "old.json", run)
    new = _write(tmp_path / "new.json", candidate)
    generate(
        unified_path=new,
        baseline_path=old,
        html_out=tmp_path / "r.html",
        summary_out=tmp_path / "s.json",
    )
    summary = json.loads((tmp_path / "s.json").read_text())
    assert summary["comparison"]["comparable"] is True
    assert summary["comparison"]["regressions"] == 1
    assert summary["comparison"]["quality_gate_passed"] is False
    assert summary["comparison_scope"].startswith("Replay only")
    assert summary["datasets_not_paired"] is True


def test_incomparable_dataset_does_not_gate(run, tmp_path):
    other = deepcopy(run)
    other["dataset"]["wrapper_sha256"] = "d" * 64
    src = _write(tmp_path / "candidate.json", run)
    old = _write(tmp_path / "baseline.json", other)
    summary, _ = build_interview_summary(run, unified_path=src, baseline=other, baseline_path=old)
    assert summary["comparison"]["comparable"] is False
    assert summary["comparison"]["quality_gate_passed"] is None


def test_reject_bad_legacy_source_claim(run):
    run["dimensions"]["outcome"]["metrics"]["api_cost_usd"]["value"] = 0.0
    with pytest.raises(ValueError):
        _validate_unified(run)


def test_render_no_external_scripts_or_assets(run, tmp_path):
    path = _write(tmp_path / "run.json", run)
    summary, triage = build_interview_summary(run, unified_path=path)
    rendered = render_html(summary, triage)
    assert "<script" not in rendered.lower()
    assert "href=" not in rendered.lower()
    assert "@media print" in rendered


def test_metric_display_valid_and_invalid():
    assert _metric_display({"status": "measured", "value": 1.0, "unit": "ratio"}) == "100.00%"
    assert _metric_display({"status": "invalid", "value": 0.3, "unit": "ratio"}) == "30.00%（无效）"
    assert _metric_display({"status": "unmeasured", "value": None}) == "未测量"


def test_tool_missing_measurement_not_promoted(run, tool, tmp_path):
    from evals.agent_eval_ext.tool_safety import _summary

    for row in tool["cases"]:
        row.update(status="unmeasured", pass_=None)
        row["pass"] = None
        row["metrics"] = None
        row["stage"] = None
        row["reason"] = "no_observation"
    tool.update(_summary(tool["cases"]))
    src = _write(tmp_path / "u.json", run)
    t = _write(tmp_path / "t.json", tool)
    result, _ = build_interview_summary(run, unified_path=src, tool=tool, tool_path=t)
    assert result["tool_fixture"]["metrics"]["contract_pass_rate"]["status"] == "unmeasured"


def test_outcome_not_confused_with_fault_contract(run, tmp_path):
    path = _write(tmp_path / "u.json", run)
    summary, _ = build_interview_summary(run, unified_path=path)
    assert summary["dimensions"]["outcome"]["metrics"]["completion_rate"]["value"] == 0.5
    assert summary["dimensions"]["reliability"]["metrics"]["case_pass_rate"]["value"] == 0.5
    assert summary["dimensions"]["reliability"]["cases"][0]["agent_status"] == "awaiting_approval"


def test_tool_total_mismatch_rejected(tool):
    tool["cases_total"] = 9
    with pytest.raises(ValueError, match="summary"):
        _validate_tool(tool)


def test_reject_oversize_file(run, tmp_path):
    src = _write(tmp_path / "u.json", run)
    with src.open("ab") as f:
        f.write(b" " * (21 * 1024 * 1024))
    assert main(["--unified", str(src), "--html", str(tmp_path / "x.html")]) == 2
    assert not (tmp_path / "x.html").exists()


def test_schema_bool_cannot_pass_as_int(run, tool):
    run["schema_version"] = True
    tool["schema_version"] = True
    with pytest.raises(ValueError):
        _validate_unified(run)
    with pytest.raises(ValueError):
        _validate_tool(tool)


def test_judge_counts_visible_without_conflating_coverage(run, tmp_path):
    run["dimensions"]["outcome"]["judge_counts"] = {"yes": 3, "no": 1, "unjudged": 2}
    path = _write(tmp_path / "u.json", run)
    generate(unified_path=path, html_out=tmp_path / "report.html")
    output = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "未判定=2" in output
    assert "分母（已判定）=4" in output


def test_invalid_missing_value_stays_invalid(run, tmp_path):
    run["dimensions"]["grounding"] = {
        "status": "invalid",
        "cases": [],
        "metrics": {
            "false_accept": {"status": "invalid", "value": None, "unit": "ratio", "n": None}
        },
    }
    src = _write(tmp_path / "u.json", run)
    generate(unified_path=src, html_out=tmp_path / "r.html")
    assert "未测量（无效）" in (tmp_path / "r.html").read_text(encoding="utf-8")


def test_duplicate_case_id_is_not_rendered(run):
    run["dimensions"]["outcome"]["cases"][1]["case_id"] = "one"
    with pytest.raises(ValueError, match="duplicate"):
        _validate_unified(run)


def test_wrong_metrics_do_not_become_pretty_scores(run):
    run["dimensions"]["reliability"]["metrics"]["case_pass_rate"]["value"] = 1
    with pytest.raises(ValueError):
        _validate_unified(run)


def test_summary_cannot_hash_a_different_input(run, tmp_path):
    src = _write(tmp_path / "input.json", run)
    changed = deepcopy(run)
    changed["dimensions"]["outcome"]["cases"][0]["case_id"] = "other"
    with pytest.raises(ValueError, match="differs"):
        build_interview_summary(changed, unified_path=src)


def test_tool_gate_is_checked_again_before_display(run, tool, tmp_path):
    candidate = _write(tmp_path / "u.json", run)
    tool["gate_passed"] = False
    source = _write(tmp_path / "tool.json", tool)
    with pytest.raises(ValueError):
        generate(unified_path=candidate, tool_path=source, html_out=tmp_path / "r.html")


def test_csp_forbids_scripts_and_network(run, tmp_path):
    source = _write(tmp_path / "u.json", run)
    generate(unified_path=source, html_out=tmp_path / "r.html")
    assert "default-src 'none'" in (tmp_path / "r.html").read_text(encoding="utf-8")


def test_judge_support_cannot_hide_unsupported_claims(run):
    outcome = run["dimensions"]["outcome"]
    outcome["judge_counts"] = {"yes": 3, "no": 1, "unjudged": 2}
    outcome["metrics"]["citation_support_among_judged"] = {
        "status": "measured",
        "value": 1.0,
        "unit": "ratio",
        "n": 4,
    }
    with pytest.raises(ValueError, match="Judge metric"):
        _validate_unified(run)
