"""Fake-model integration test: the budget gate fires at real GEPA call sites.

No paid model call, no GEPA search, no holdout scoring.
"""

from __future__ import annotations

from pathlib import Path

from evals.agent_eval_ext.planner_evolution.runtime import LiveRuntime
from evals.paid_call_budget import (
    ROLE_JUDGE,
    ROLE_PLANNER,
    ROLE_REFLECTION,
    BudgetConfig,
    PaidCallBudget,
)


def make_runtime(budget, tmp_path: Path):
    from research_engine.runconfig import RunConfig

    cfg = RunConfig()
    runtime = LiveRuntime(
        cfg,
        "fake:judge",
        "fake:reflection",
        30,
        (),
        metric_version="v3",
        budget=budget,
    )
    return runtime


def test_budget_is_wired_into_runtime_and_defaults_off(tmp_path: Path):
    """Runtime accepts a gate and preserves legacy behavior with None."""
    cfg = BudgetConfig(
        run_id="integration",
        quotas={ROLE_PLANNER: 1, ROLE_JUDGE: 1, ROLE_REFLECTION: 1},
        total_cap=3,
        planned_evaluations=1,
    )
    b = PaidCallBudget(tmp_path / "ledger", cfg)
    b.preflight()

    rt_on = make_runtime(b, tmp_path)
    assert rt_on.budget is b

    # Direct constructor with default budget verifies backward compatibility
    from research_engine.runconfig import RunConfig

    rt_default = LiveRuntime(
        RunConfig(), "fake:judge", "fake:reflection", 30, (), metric_version="v3"
    )
    assert rt_default.budget is None

    # _paid executes a fake call directly when no budget is configured
    assert rt_default._paid.__name__ == "_paid"


def test_paid_helper_reserves_for_every_role(tmp_path: Path):
    """The _paid wrapper consumes one slot per call, for all three roles."""
    cfg = BudgetConfig(
        run_id="roles",
        quotas={ROLE_PLANNER: 4, ROLE_JUDGE: 2, ROLE_REFLECTION: 1},
        total_cap=7,
        planned_evaluations=2,
    )
    b = PaidCallBudget(tmp_path / "ledger", cfg)
    b.preflight()
    rt = make_runtime(b, tmp_path)

    import asyncio

    async def fake(*args, **kwargs):
        return {"ok": True}

    # Planner is charged per real request via token_usage.gate_scope, not here.
    asyncio.run(rt._paid(ROLE_JUDGE, "j1", fake, input_data={"x": 2}))
    asyncio.run(rt._paid(ROLE_JUDGE, "j2", fake, input_data={"x": 3}))
    asyncio.run(rt._paid(ROLE_REFLECTION, "r", fake, input_data={"x": 4}))

    state = b.state()
    assert state["used_by_role"] == {ROLE_PLANNER: 0, ROLE_JUDGE: 2, ROLE_REFLECTION: 1}
    assert state["used_total"] == 3
    assert state["remaining_total"] == 4  # planner quota untouched in this test
    assert state["planned_evaluations"] == 2
    assert all(v == "COMPLETED" for v in state["statuses"].values())
    # evaluation count is not confused with paid-call count


def test_non_serializable_result_never_strands_slot_at_sent(tmp_path: Path):
    """A paid call whose SDK result cannot be hashed must still reach COMPLETED.

    Regression: the Judge returned a Pydantic CoverageV2; _sha(result) raised
    TypeError *after* mark_sent but outside the try/except, leaving the slot at
    SENT. state() treats SENT as UNKNOWN, so every later call was denied and the
    whole live run was scored zero -- yet still exited 0.
    """
    import asyncio

    cfg_b = BudgetConfig(
        run_id="nonserial",
        quotas={ROLE_PLANNER: 1, ROLE_JUDGE: 2, ROLE_REFLECTION: 1},
        total_cap=4,
        planned_evaluations=1,
    )
    b = PaidCallBudget(tmp_path / "ledger", cfg_b)
    b.preflight()

    class PydanticLike:
        pass  # not JSON-serializable, like CoverageV2 / AIMessage

    async def fake_judge(*_a, **_k):
        return {"raw": PydanticLike(), "parsed": PydanticLike()}

    asyncio.run(b.paid_call_async(ROLE_JUDGE, "judge:case1", fake_judge, input_data={"x": 1}))
    state = b.state()
    assert state["used_by_role"][ROLE_JUDGE] == 1
    assert state["unknown"] == [], state["statuses"]
    assert all(v == "COMPLETED" for v in state["statuses"].values())
    # A subsequent call is allowed: no phantom UNKNOWN blocking the run.
    asyncio.run(b.paid_call_async(ROLE_JUDGE, "judge:case2", lambda: _coro({"ok": True})))
    assert b.state()["used_by_role"][ROLE_JUDGE] == 2


async def _coro(value):
    return value


def test_cli_budget_preflight_does_not_collide_with_run_dir(monkeypatch, tmp_path: Path):
    """The strict-budget CLI setup must create the exclusive run dir first.

    Regression: previously PaidCallBudget.preflight() (root = out/paid-budget)
    created `out` via parents=True, so _new_dir(out)'s exist_ok=False rejected a
    fresh run with FileExistsError BEFORE any model call. Setup ran at CLI main,
    so this drives main() with Fake routes/runtime -- zero paid calls.
    """
    import json as _json

    from evals.agent_eval_ext.planner_evolution import __main__ as cli
    from evals.agent_eval_ext.planner_evolution import runtime as runtime_mod
    from evals.harness import RUN_CONFIG

    ds = cli.read(cli.DEMO_DATASET)
    ds["provenance"] = "reviewed"
    src = tmp_path / "dataset.json"
    src.write_text(_json.dumps(ds), encoding="utf-8")

    cfg_path = tmp_path / "budget.json"
    cfg_path.write_text(
        _json.dumps(
            {
                "run_id": "cli-ordering-regression",
                "quotas": {ROLE_PLANNER: 1, ROLE_JUDGE: 1, ROLE_REFLECTION: 1},
                "total_cap": 3,
                "planned_evaluations": 1,
                "source": "test",
            }
        ),
        encoding="utf-8",
    )

    class FakeRuntime:
        def __init__(self, *_, **__):
            pass

        def checked_text(self, text):
            return text

    routes = {
        "pipeline": dict(RUN_CONFIG.models),
        "judge": "custom:fake-judge",
        "reflection": "custom:fake-planner",
    }
    monkeypatch.setattr(cli, "version", lambda _: "0.0-test")
    monkeypatch.setattr(cli, "_live_routes", lambda *_: (RUN_CONFIG, routes))
    monkeypatch.setattr(runtime_mod, "LiveRuntime", FakeRuntime)

    class ReachedOptimizer(Exception):
        pass

    def forbidden(*_a, **_k):
        raise ReachedOptimizer()

    monkeypatch.setattr(cli, "optimize_planner", forbidden)

    out = tmp_path / "run1"
    # The old ordering bug raised FileExistsError during setup; the fixed
    # ordering proceeds all the way to the optimizer stub instead.
    import pytest as _pytest

    with _pytest.raises(ReachedOptimizer):
        cli.main(
            [
                "optimize",
                "--live",
                "--dataset",
                str(src),
                "--out-dir",
                str(out),
                "--budget-config",
                str(cfg_path),
                "--strict-budget",
            ]
        )
    # The exclusive run dir and its nested ledger coexist; zero slots consumed.
    assert (out / "write-check.json").is_file()
    ledger = out / "paid-budget"
    assert (ledger / "budget_config.json").is_file()
    slots = ledger / "slots"
    assert slots.is_dir() and list(slots.glob("*.json")) == []


def test_live_unmeasured_search_refuses_candidate_and_exit_zero(monkeypatch, tmp_path: Path):
    """A live run with zero scored observations must fail closed, not exit 0
    with a baseline candidate. Regression from the stranded-SENT incident."""
    import json as _json

    from evals.agent_eval_ext.planner_evolution import __main__ as cli
    from evals.agent_eval_ext.planner_evolution import runtime as runtime_mod
    from evals.harness import RUN_CONFIG

    ds = cli.read(cli.DEMO_DATASET)
    ds["provenance"] = "reviewed"
    src = tmp_path / "dataset.json"
    src.write_text(_json.dumps(ds), encoding="utf-8")

    cfg_path = tmp_path / "budget.json"
    cfg_path.write_text(
        _json.dumps(
            {
                "run_id": "unmeasured-regression",
                "quotas": {ROLE_PLANNER: 1, ROLE_JUDGE: 1, ROLE_REFLECTION: 1},
                "total_cap": 3,
                "planned_evaluations": 1,
                "source": "test",
            }
        ),
        encoding="utf-8",
    )

    class FakeRuntime:
        def __init__(self, *_, **__):
            pass

        def checked_text(self, text):
            return text

    routes = {
        "pipeline": dict(RUN_CONFIG.models),
        "judge": "custom:fake-judge",
        "reflection": "custom:fake-planner",
    }
    monkeypatch.setattr(cli, "version", lambda _: "0.0-test")
    monkeypatch.setattr(cli, "_live_routes", lambda *_: (RUN_CONFIG, routes))
    monkeypatch.setattr(runtime_mod, "LiveRuntime", FakeRuntime)

    def unmeasured_result(*_a, **_k):
        return {
            "candidate": {"override": "baseline prompt", "sha256": "abc"},
            "search": {"validation_gain": None, "evaluations": 0},
            "holdout": {"gate": {"passed": None, "reasons": ["preexperiment_no_holdout"]}},
        }

    monkeypatch.setattr(cli, "optimize_planner", unmeasured_result)

    out = tmp_path / "run1"
    rc = cli.main(
        [
            "optimize",
            "--live",
            "--dataset",
            str(src),
            "--out-dir",
            str(out),
            "--budget-config",
            str(cfg_path),
            "--strict-budget",
        ]
    )
    assert rc == 2
    assert not (out / "candidate.json").exists()
    assert not (out / "experiment.json").exists()
    journal = [_json.loads(p.read_text()) for p in sorted((out / "journal").glob("*.json"))]
    assert journal[-1]["kind"] == "unmeasured_live_run"
