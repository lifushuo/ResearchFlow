"""v0.8 offline scorer tests; these are NOT agent-run accuracy claims."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evals.agent_eval_ext import tool_safety as ts
from evals.agent_eval_ext.tool_safety_cli import main

ROOT = Path(__file__).resolve().parents[2] / "evals" / "agent_eval_ext" / "fixtures"


def files():
    gold, sha = ts.load_gold(ROOT / "tool_gold_v1.json")
    good, good_sha = ts.load_observed(ROOT / "tool_observed_pass_v1.json")
    return gold, sha, good, good_sha


def score(obs=None):
    gold, sha, good, good_sha = files()
    ob = good if obs is None else obs
    return ts.evaluate(gold, ob, gold_sha256=sha, observed_sha256=good_sha)


def test_fixture_pass():
    s = score()
    assert s["gate_passed"] is True
    assert s["cases_measured"] == 4
    assert s["metrics"]["contract_pass_rate"]["value"] == 1.0
    assert s["actual_agent_executed"] is False


def test_negative_fixture_fails():
    gold, sha = ts.load_gold(ROOT / "tool_gold_v1.json")
    bad, digest = ts.load_observed(ROOT / "tool_observed_fail_v1.json")
    s = ts.evaluate(gold, bad, gold_sha256=sha, observed_sha256=digest)
    assert s["gate_passed"] is False
    assert s["metrics"]["forbidden_attempt_count"]["value"] == 1
    assert s["cases"][0]["metrics"]["selection_correct"] is True
    assert s["cases"][0]["metrics"]["arguments_correct"] is False


def test_args_not_persisted():
    result = json.dumps(score(), ensure_ascii=False)
    assert "https://example.org/a" not in result
    assert "AI agent memory" not in result
    assert "expression" not in result


def test_strict_wrong_order():
    _, _, obs, _ = files()
    obs["cases"][0]["calls"].reverse()
    assert score(obs)["cases"][0]["pass"] is False


def test_unordered_ok_reversed():
    _, _, obs, _ = files()
    obs["cases"][1]["calls"].reverse()
    assert score(obs)["cases"][1]["pass"] is True


def test_unordered_duplicate_cannot_substitute():
    _, _, obs, _ = files()
    obs["cases"][1]["calls"][1] = obs["cases"][1]["calls"][0]
    assert score(obs)["cases"][1]["metrics"]["arguments_correct"] is False


def test_unordered_reject_extra():
    _, _, obs, _ = files()
    obs["cases"][1]["calls"].append(copy.deepcopy(obs["cases"][1]["calls"][0]))
    assert score(obs)["cases"][1]["pass"] is False


def test_subset_tolerates_allowed_extra():
    assert score()["cases"][2]["pass"] is True


def test_subset_reject_unknown_extra():
    _, _, obs, _ = files()
    obs["cases"][2]["calls"].append({"name": "other_tool", "args": {}})
    assert score(obs)["cases"][2]["metrics"]["forbidden_attempts"] == 1


def test_empty_allowed():
    assert score()["cases"][3]["pass"] is True


def test_empty_expected_with_tool_fails():
    _, _, obs, _ = files()
    obs["cases"][3]["calls"].append({"name": "calculate", "args": {}})
    assert score(obs)["cases"][3]["pass"] is False


def test_missing_observed_unmeasured():
    _, _, obs, _ = files()
    obs["cases"].pop()
    r = score(obs)
    assert r["cases"][-1]["status"] == "unmeasured"
    assert r["cases"][-1]["pass"] is None
    assert r["gate_passed"] is False
    assert r["metrics"]["case_coverage"]["value"] == 0.75


def test_empty_observed_all_unmeasured():
    _, _, obs, _ = files()
    obs["cases"] = []
    r = score(obs)
    assert r["metrics"]["contract_pass_rate"]["value"] is None
    assert r["metrics"]["contract_pass_rate"]["status"] == "unmeasured"


def test_unknown_observation_rejected():
    _, _, obs, _ = files()
    obs["cases"][0]["case_id"] = "unknown"
    with pytest.raises(ValueError, match="unexpected"):
        score(obs)


def test_gold_sha_mismatch():
    _, _, obs, _ = files()
    obs["gold_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact gold"):
        score(obs)


def test_gold_duplicate_id(tmp_path):
    gold, _ = ts.load_gold(ROOT / "tool_gold_v1.json")
    gold["cases"][1]["case_id"] = gold["cases"][0]["case_id"]
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(gold))
    with pytest.raises(ValueError, match="duplicate"):
        ts.load_gold(p)


def test_gold_unknown_key(tmp_path):
    gold, _ = ts.load_gold(ROOT / "tool_gold_v1.json")
    gold["cases"][0]["evil"] = "x"
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(gold))
    with pytest.raises(ValueError, match="missing/unknown"):
        ts.load_gold(p)


def test_gold_bool_version_rejected(tmp_path):
    gold, _ = ts.load_gold(ROOT / "tool_gold_v1.json")
    gold["dataset_version"] = True
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(gold))
    with pytest.raises(ValueError, match="dataset_version"):
        ts.load_gold(p)


def test_gold_expected_not_allowed(tmp_path):
    gold, _ = ts.load_gold(ROOT / "tool_gold_v1.json")
    gold["cases"][0]["allowed_tools"] = []
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(gold))
    with pytest.raises(ValueError, match="outside"):
        ts.load_gold(p)


def test_gold_allowed_and_forbidden_overlap(tmp_path):
    gold, _ = ts.load_gold(ROOT / "tool_gold_v1.json")
    gold["cases"][0]["forbidden_tools"].append("web_search")
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(gold))
    with pytest.raises(ValueError, match="overlap"):
        ts.load_gold(p)


def test_observed_unverified_provenance_rejected(tmp_path):
    _, _, obs, _ = files()
    obs["evidence_kind"] = "actual_langgraph"
    p = tmp_path / "obs.json"
    p.write_text(json.dumps(obs))
    with pytest.raises(ValueError, match="controlled_fixture only"):
        ts.load_observed(p)


def test_observed_duplicate_id(tmp_path):
    _, _, obs, _ = files()
    obs["cases"][1]["case_id"] = obs["cases"][0]["case_id"]
    p = tmp_path / "obs.json"
    p.write_text(json.dumps(obs))
    with pytest.raises(ValueError, match="duplicate"):
        ts.load_observed(p)


def test_observed_non_json_arg(tmp_path):
    _, _, obs, _ = files()
    obs["cases"][0]["calls"][0]["args"]["query"] = float("nan")
    p = tmp_path / "obs.json"
    p.write_text(json.dumps(obs))
    with pytest.raises(ValueError, match="JSON-serializable"):
        ts.load_observed(p)


def test_compare_same_pass():
    a = score()
    r = ts.compare(a, copy.deepcopy(a))
    assert r["comparable"] and r["gate_passed"] and r["regressions"] == 0


def test_compare_detect_regression():
    old = score()
    _, _, obs, _ = files()
    obs["cases"][0]["calls"][0]["args"]["query"] = "wrong"
    new = score(obs)
    r = ts.compare(old, new)
    assert r["regressions"] == 1 and r["gate_passed"] is False


def test_compare_reject_gold_change():
    old = score()
    new = copy.deepcopy(old)
    new["gold_sha256"] = "f" * 64
    assert ts.compare(old, new)["comparable"] is False


def test_compare_unmeasured_is_regression():
    old = score()
    _, _, obs, _ = files()
    obs["cases"].pop(0)
    new = score(obs)
    assert ts.compare(old, new)["regressions"] == 1


def test_cli_pass_and_no_overwrite(tmp_path):
    result = tmp_path / "pass.json"
    good = ROOT / "tool_observed_pass_v1.json"
    gold = ROOT / "tool_gold_v1.json"
    cmd = ["score", "--gold", str(gold), "--observed", str(good), "--out", str(result)]
    assert main(cmd) == 0 and result.is_file()
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    assert main(cmd) == 2
    assert hashlib.sha256(result.read_bytes()).hexdigest() == digest


def test_cli_fail_returns_one(tmp_path):
    out = tmp_path / "fail.json"
    assert (
        main(
            [
                "score",
                "--gold",
                str(ROOT / "tool_gold_v1.json"),
                "--observed",
                str(ROOT / "tool_observed_fail_v1.json"),
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert json.loads(out.read_text())["gate_passed"] is False


def test_cli_compare_returns_one_on_regression(tmp_path):
    a = score()
    _, _, obs, _ = files()
    obs["cases"][0]["calls"][0]["args"]["query"] = "wrong"
    b = score(obs)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert main(["compare", "--baseline", str(pa), "--candidate", str(pb)]) == 1


def test_inputs_unmodified():
    paths = [ROOT / x for x in ("tool_gold_v1.json", "tool_observed_pass_v1.json")]
    before = [hashlib.sha256(x.read_bytes()).digest() for x in paths]
    score()
    after = [hashlib.sha256(x.read_bytes()).digest() for x in paths]
    assert before == after


def test_ci_workflow_has_real_graph_gate_and_no_paid_runtime():
    repo = Path(__file__).resolve().parents[3]
    workflow = repo / ".github" / "workflows" / "agent-evaluation.yml"
    content = workflow.read_text(encoding="utf-8")
    assert "--network none" in content
    assert "-e LLM_MODE=fake" in content
    assert "python -m evals.agent_eval_ext.unified_cli run" in content
    assert "if: always()" in content and "upload-artifact" in content
    assert 'exit "$rc"' in content


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_boolean_arguments_are_not_python_numeric_aliases(value):
    gold, digest, obs, observed_digest = files()
    gold["cases"][0]["expected_calls"][0]["args"] = {"flag": True}
    obs["cases"][0]["calls"][0]["args"] = {"flag": value}
    result = ts.evaluate(gold, obs, gold_sha256=digest, observed_sha256=observed_digest)
    assert result["cases"][0]["metrics"]["arguments_correct"] is False


def test_nested_boolean_type_is_checked():
    assert not ts._args_match({"a": [{"flag": True}]}, {"a": [{"flag": 1}]})


def test_required_args_subset_allows_extra_keys():
    _, _, obs, _ = files()
    obs["cases"][0]["calls"][0]["args"]["extra"] = "private-canary-DoNotPersist"
    result = score(obs)
    assert result["cases"][0]["pass"] is True
    assert "private-canary" not in json.dumps(result)


def test_direct_api_rejects_duplicate_observations():
    _, _, obs, _ = files()
    obs["cases"].append(copy.deepcopy(obs["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        score(obs)


@pytest.mark.parametrize("field", ["gate_passed", "cases_measured", "metrics", "cases"])
def test_compare_rejects_inconsistent_saved_results(field):
    result = score()
    changed = copy.deepcopy(result)
    changed[field] = {"gate_passed": 1, "cases_measured": 0, "metrics": {}, "cases": []}[field]
    with pytest.raises(ValueError):
        ts.compare(result, changed)


@pytest.mark.parametrize("value", [1, "true", None])
def test_compare_rejects_nonboolean_measured_case(value):
    result = score()
    changed = copy.deepcopy(result)
    changed["cases"][0]["pass"] = value
    with pytest.raises(ValueError):
        ts.compare(result, changed)


def test_compare_rejects_duplicate_case_ids():
    result = score()
    changed = copy.deepcopy(result)
    changed["cases"][1]["case_id"] = changed["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="duplicate"):
        ts.compare(result, changed)


def test_compare_stage_changes_are_not_like_for_like():
    _, _, obs, _ = files()
    before = score(obs)
    obs["cases"][0]["stage"] = "executed"
    after = score(obs)
    assert after["actual_agent_executed"] is False
    assert after["stage_counts"] == {"executed": 1, "proposed": 3}
    assert ts.compare(before, after)["reason"] == "observation_stage_changed"


def test_changed_dataset_id_blocks_comparison():
    result = score()
    changed = copy.deepcopy(result)
    changed["dataset_id"] = "different"
    assert ts.compare(result, changed)["comparable"] is False


def test_unchanged_failing_candidate_does_not_pass_gate():
    _, _, obs, _ = files()
    obs["cases"].pop()
    result = score(obs)
    comparison = ts.compare(result, result)
    assert comparison["regressions"] == 0
    assert comparison["gate_passed"] is False


def test_nonfinite_output_does_not_leave_partial_file(tmp_path):
    target = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        ts.save_new(target, {"value": float("nan")})
    assert not target.exists()


def test_trial_write_failure_precedes_scoring(tmp_path, monkeypatch):
    from evals.agent_eval_ext import tool_safety_cli

    def denied(path):
        raise PermissionError("controlled output denial")

    def unexpected(*args, **kwargs):
        pytest.fail("scoring started before output preflight")

    monkeypatch.setattr(tool_safety_cli, "check_output_path", denied)
    monkeypatch.setattr(tool_safety_cli, "evaluate", unexpected)
    assert (
        main(
            [
                "score",
                "--gold",
                str(ROOT / "tool_gold_v1.json"),
                "--observed",
                str(ROOT / "tool_observed_pass_v1.json"),
                "--out",
                str(tmp_path / "new.json"),
            ]
        )
        == 2
    )


def test_bom_fingerprints_cover_exact_loaded_bytes(tmp_path):
    source = ROOT / "tool_gold_v1.json"
    target = tmp_path / "bom.json"
    target.write_bytes(b"\xef\xbb\xbf" + source.read_bytes())
    _, digest = ts.load_gold(target)
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest != ts.load_gold(source)[1]


def test_invalid_direct_api_cannot_claim_live_provenance():
    _, _, obs, _ = files()
    obs["evidence_kind"] = "actual_agent"
    with pytest.raises(ValueError, match="controlled_fixture"):
        score(obs)
