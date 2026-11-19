"""Case validation, evidence protection and actual compiled-graph replay regressions."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.agent_eval_ext.replay import (
    ReplayCase,
    load_case,
    observed_status,
    score_case,
    write_result_exclusive,
)

GOOD = {
    "schema_version": 1,
    "case_id": "planner-invalid",
    "query": "research task",
    "fault": {"kind": "planner_invalid_schema"},
    "expected": {"status": "failed", "planner_calls": 2, "error_contains": "valid task list"},
}


def test_case_parses():
    assert ReplayCase.from_dict(GOOD).expected_planner_calls == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_id", "../escape"),
        ("case_id", "bad id"),
        ("query", ""),
        ("query", "x" * 2001),
    ],
)
def test_case_rejects_bad_ids_and_queries(field, value):
    with pytest.raises(ValueError):
        ReplayCase.from_dict({**GOOD, field: value})


def test_case_rejects_fault_kind():
    with pytest.raises(ValueError):
        ReplayCase.from_dict({**GOOD, "fault": {"kind": "execute_paid_api"}})


def test_case_rejects_bool_as_attempt_count():
    with pytest.raises(ValueError):
        ReplayCase.from_dict({**GOOD, "expected": {"status": "failed", "planner_calls": True}})


def test_case_rejects_unknown_status():
    with pytest.raises(ValueError):
        ReplayCase.from_dict({**GOOD, "expected": {"status": "unknown"}})


def test_case_rejects_wrong_schema_version():
    with pytest.raises(ValueError):
        ReplayCase.from_dict({**GOOD, "schema_version": 2})


def test_status_distinguishes_failed_and_paused():
    class Interruption:
        value = {"type": "HITL_READY"}

    assert observed_status({"error": "bad"}) == "failed"
    assert (
        observed_status({"error": "old", "__interrupt__": [Interruption()]}) == "awaiting_approval"
    )
    assert observed_status({"__interrupt__": [Interruption()]}) == "awaiting_approval"
    assert observed_status({"final_report": "ready"}) == "completed"
    assert observed_status({}) == "unknown"


def test_status_distinguishes_plan_pause():
    class Interruption:
        value = {"type": "PLAN_READY"}

    assert observed_status({"__interrupt__": [Interruption()]}) == "awaiting_plan"


def test_score_does_not_hide_wrong_attempts():
    spec = ReplayCase.from_dict(GOOD)
    result = score_case(spec, status="failed", planner_calls=1, error="valid task list")
    assert not result["passed"]
    assert not result["assertions"]["planner_calls"]


def test_score_valid_case():
    spec = ReplayCase.from_dict(GOOD)
    assert score_case(spec, status="failed", planner_calls=2, error="valid task list")["passed"]


def test_exclusive_write(tmp_path):
    destination = tmp_path / "report.json"
    write_result_exclusive(destination, {"passed": True})
    with pytest.raises(FileExistsError):
        write_result_exclusive(destination, {"passed": False})
    assert json.loads(destination.read_text())["passed"] is True


def test_case_hash(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(GOOD), encoding="utf-8")
    spec, digest = load_case(path)
    assert spec.case_id == "planner-invalid"
    assert len(digest) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fixture",
    [
        "replay_planner_invalid.json",
        "replay_planner_429.json",
        "replay_planner_522.json",
        "replay_healthy.json",
    ],
)
async def test_real_compiled_graph_offline(fixture, tmp_path):
    from evals.agent_eval_ext.replay import run_replay

    directory = Path(__file__).parents[2] / "evals" / "agent_eval_ext" / "fixtures"
    spec, _ = load_case(directory / fixture)
    original = (directory / fixture).read_bytes()
    result = await run_replay(spec, trace_dir=tmp_path)
    assert result["passed"], result
    assert result["trace_path"]
    assert result["trace_status"] == "recorded"
    assert result["trace_inspection"]["trace_complete"]
    assert result["planner_calls"] == (
        1 if fixture in ("replay_planner_429.json", "replay_healthy.json") else 2
    )
    assert (directory / fixture).read_bytes() == original


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": True},
        {"depth": "deep"},
        {"fault": {"kind": []}},
        {"expected": {"status": []}},
        {"expected": {"status": "failed", "planner_calls": -1}},
        {"expected": {"status": "failed", "planner_calls": 2.0}},
        {"expected": {"status": "failed", "planner_call": 2}},
        {"expected": {"status": "failed", "error_contains": ""}},
        {"fault": {"kind": "none", "script": "ignored"}},
        {"run_script": "ignored"},
    ],
)
def test_case_rejects_ambiguous_or_misspelled_fields(update):
    with pytest.raises(ValueError):
        ReplayCase.from_dict({**GOOD, **update})


@pytest.mark.parametrize("value", [{}, {"type": "OTHER"}, {"type": []}, None])
def test_unknown_interrupt_is_not_approval(value):
    assert observed_status({"__interrupt__": [SimpleNamespace(value=value)]}) == "unknown"


def test_unrequested_assertions_stay_explicitly_unchecked():
    data = deepcopy(GOOD)
    data["expected"] = {"status": "failed"}
    result = score_case(ReplayCase.from_dict(data), status="failed", planner_calls=None, error=None)
    assert result["assertions"]["planner_calls"] is None
    assert result["assertions"]["error_contains"] is None
    assert not score_case(
        ReplayCase.from_dict(GOOD), status="failed", planner_calls=None, error="valid task list"
    )["passed"]


@pytest.mark.parametrize("attempts", [True, -1, 1.5])
def test_score_rejects_invalid_observed_counts(attempts):
    with pytest.raises(ValueError):
        score_case(ReplayCase.from_dict(GOOD), status="failed", planner_calls=attempts, error=None)


def test_case_hash_uses_original_bom_bytes(tmp_path):
    import hashlib

    path = tmp_path / "case.json"
    path.write_text(json.dumps(GOOD), encoding="utf-8-sig")
    before = path.read_bytes()
    _, digest = load_case(path)
    assert digest == hashlib.sha256(before).hexdigest() and path.read_bytes() == before


async def test_missing_trace_cannot_pass_replay(tmp_path):
    from evals.agent_eval_ext.replay import run_replay

    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    result = await run_replay(ReplayCase.from_dict(GOOD), trace_dir=blocker / "child")
    assert result["status"] == "failed" and result["planner_calls"] == 2
    assert result["passed"] is False and result["assertions"]["trace_available"] is False
    assert result["trace_status"] == "unavailable"


async def test_fake_run_isolates_and_restores_real_host_ports(monkeypatch, tmp_path):
    from contextlib import ExitStack

    from evals.agent_eval_ext.replay import run_replay
    from research_engine import cache, corpus, events, graph, llm_factory
    from research_engine.runconfig import get_run_config, reset_run_config, set_run_config

    touched = []

    class HostPort:
        async def get(self, *args):
            touched.append("cache")
            raise AssertionError("host cache reached")

        async def set(self, *args):
            touched.append("cache")
            raise AssertionError("host cache reached")

        async def search(self, *args):
            touched.append("corpus")
            raise AssertionError("host corpus reached")

        async def read(self, *args):
            touched.append("corpus")
            raise AssertionError("host corpus reached")

    async def host_event(*args):
        touched.append("event")
        raise AssertionError("host event sink reached")

    def forbid_provider(*args, **kwargs):
        touched.append("provider")
        raise AssertionError("real provider construction reached")

    monkeypatch.setattr(llm_factory, "_build", forbid_provider)
    outer = replace(get_run_config(), llm_mode="real", corpus_mode=True)
    port = HostPort()
    before_factory = graph.get_llm
    with ExitStack() as scope:
        scope.callback(reset_run_config, set_run_config(outer))
        scope.callback(cache.reset_cache, cache.set_cache(port))
        scope.callback(corpus.reset_corpus, corpus.set_corpus(port))
        scope.callback(events.reset_emitter, events.set_emitter(host_event))
        scope.callback(
            llm_factory.reset_user_keys, llm_factory.set_user_keys({"custom": "outer-test-key"})
        )
        spec = ReplayCase.from_dict(
            {
                **GOOD,
                "fault": {"kind": "none"},
                "expected": {"status": "awaiting_approval", "planner_calls": 1},
            }
        )
        result = await run_replay(spec, trace_dir=tmp_path)
        assert result["passed"] and result["planner_calls"] == 1
        assert get_run_config() is outer
        assert cache.get_cache() is corpus.get_corpus() is port
        assert events._emitter.get() is host_event
        assert llm_factory.api_key_for("custom") == "outer-test-key"
    assert graph.get_llm is before_factory and touched == []


@pytest.mark.parametrize("cancel", [False, True])
async def test_overlapping_replays_rejected_and_patch_restored(monkeypatch, tmp_path, cancel):
    from evals.agent_eval_ext import replay
    from research_engine import graph
    from research_engine.runconfig import get_run_config

    started, release = asyncio.Event(), asyncio.Event()
    invoke = replay._FaultPlanner.ainvoke
    factory = graph.get_llm
    outer = get_run_config()
    restored = []

    async def blocked(self, messages, **kwargs):
        started.set()
        await release.wait()
        return await invoke(self, messages, **kwargs)

    monkeypatch.setattr(replay._FaultPlanner, "ainvoke", blocked)

    async def first():
        try:
            return await replay.run_replay(ReplayCase.from_dict(GOOD), trace_dir=tmp_path)
        finally:
            restored.append(get_run_config() is outer)

    task = asyncio.create_task(first())
    await started.wait()
    with pytest.raises(RuntimeError, match="Another replay"):
        await replay.run_replay(ReplayCase.from_dict(GOOD), trace_dir=tmp_path)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        release.set()
        assert (await task)["passed"]
    assert restored == [True] and graph.get_llm is factory and not replay._REPLAY_LOCK.locked()
    monkeypatch.setattr(replay._FaultPlanner, "ainvoke", invoke)
    assert (await replay.run_replay(ReplayCase.from_dict(GOOD), trace_dir=tmp_path))["passed"]


async def test_graph_construction_failure_restores_factory(monkeypatch, tmp_path):
    from evals.agent_eval_ext import replay
    from research_engine import graph
    from research_engine.runconfig import get_run_config

    original, cfg = graph.get_llm, get_run_config()

    def broken(*args):
        raise RuntimeError("construction failure")

    monkeypatch.setattr(graph, "build_graph", broken)
    with pytest.raises(RuntimeError, match="construction failure"):
        await replay.run_replay(ReplayCase.from_dict(GOOD), trace_dir=tmp_path)
    assert graph.get_llm is original and get_run_config() is cfg
    assert not replay._REPLAY_LOCK.locked()


def test_cli_preflight_blocks_existing_output_before_graph(monkeypatch, tmp_path):
    from evals.agent_eval_ext import replay_cli

    source, output = tmp_path / "case.json", tmp_path / "result.json"
    source.write_text(json.dumps(GOOD))
    output.write_text("original evidence")
    trace = tmp_path / "traces"
    monkeypatch.setattr(
        "sys.argv",
        ["replay", "--case", str(source), "--out", str(output), "--trace-dir", str(trace)],
    )
    monkeypatch.setattr(replay_cli, "run_replay", lambda *a, **k: pytest.fail("graph reached"))
    with pytest.raises(SystemExit) as error:
        replay_cli.main()
    assert error.value.code == 2 and output.read_text() == "original evidence"
    assert not trace.exists()


def test_cli_failed_assertions_save_new_result_and_exit_one(monkeypatch, tmp_path):
    from evals.agent_eval_ext import replay_cli

    raw = deepcopy(GOOD)
    raw["expected"]["planner_calls"] = 99
    source, output = tmp_path / "case.json", tmp_path / "result.json"
    source.write_text(json.dumps(raw))
    original = source.read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--case",
            str(source),
            "--out",
            str(output),
            "--trace-dir",
            str(tmp_path / "traces"),
        ],
    )
    assert replay_cli.main() == 1
    result = json.loads(output.read_text())
    assert result["status"] == "failed" and result["passed"] is False
    assert result["assertions"]["planner_calls"] is False
    assert (output.parent / result["trace_path"]).is_file()
    assert source.read_bytes() == original


def test_cli_preflight_actually_writes_parent_before_graph(monkeypatch, tmp_path):
    from evals.agent_eval_ext import replay_cli

    source, blocker = tmp_path / "case.json", tmp_path / "blocked"
    source.write_text(json.dumps(GOOD))
    blocker.write_text("not a directory")
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--case",
            str(source),
            "--out",
            str(blocker / "result.json"),
            "--trace-dir",
            str(tmp_path / "traces"),
        ],
    )
    monkeypatch.setattr(replay_cli, "run_replay", lambda *a, **k: pytest.fail("graph reached"))
    with pytest.raises(SystemExit) as error:
        replay_cli.main()
    assert error.value.code == 2 and not (tmp_path / "traces").exists()


def test_cli_final_exclusive_write_closes_preflight_race(monkeypatch, tmp_path):
    from evals.agent_eval_ext import replay_cli

    source, output = tmp_path / "case.json", tmp_path / "result.json"
    source.write_text(json.dumps(GOOD))

    async def competing_writer(*args, **kwargs):
        output.write_text("winner evidence")
        return {"passed": True}

    monkeypatch.setattr(replay_cli, "run_replay", competing_writer)
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--case",
            str(source),
            "--out",
            str(output),
            "--trace-dir",
            str(tmp_path / "traces"),
        ],
    )
    with pytest.raises(FileExistsError):
        replay_cli.main()
    assert output.read_text() == "winner evidence"


async def test_unhandled_graph_crash_cannot_match_expected_failed_route(monkeypatch, tmp_path):
    from evals.agent_eval_ext import replay
    from research_engine import graph

    class BrokenGraph:
        async def ainvoke(self, *args):
            raise RuntimeError("valid task list")

    monkeypatch.setattr(graph, "build_graph", lambda *_: BrokenGraph())
    result = await replay.run_replay(ReplayCase.from_dict(GOOD), trace_dir=tmp_path)
    assert result["status"] == "execution_error" and result["passed"] is False
    assert result["execution_error_type"] == "RuntimeError"
    assert result["assertions"]["status"] is False


@pytest.mark.parametrize("run_id", ["../escape", True, "x" * 65])
async def test_bad_run_identity_rejected_without_trace(tmp_path, run_id):
    from evals.agent_eval_ext.replay import run_replay

    with pytest.raises(ValueError, match="run_id"):
        await run_replay(ReplayCase.from_dict(GOOD), trace_dir=tmp_path, run_id=run_id)
    assert not list(tmp_path.iterdir())
