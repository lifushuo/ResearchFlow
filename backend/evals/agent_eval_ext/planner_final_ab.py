"""Execute one frozen Planner final A/B job. No GEPA search, reflection or publish."""

import argparse
import asyncio
import json
from pathlib import Path

from app.services import planner_final_jobs as final
from app.services.evaluation_jobs import _read
from evals import holdout_usage_guard as guard
from evals.agent_eval_ext.long_run import atomic_create
from evals.agent_eval_ext.planner_evolution.__main__ import _live_routes, _new_dir
from evals.agent_eval_ext.planner_evolution.fake_runtime import FakeRuntime
from evals.agent_eval_ext.planner_evolution.journal import ExperimentJournal
from evals.agent_eval_ext.planner_evolution.metrics_v3 import case_score_v3, gate_v3
from evals.agent_eval_ext.planner_evolution.runtime import LiveRuntime
from evals.paid_call_budget import BudgetConfig, PaidCallBudget


async def execute(req, source, out, budget, journal):
    frozen = req["frozen"]
    if req["mode"] == "demo":
        data = final.fake_dataset()
        runtime = FakeRuntime(budget)
        runtime.planner.candidate_prompt = source["candidate"]["override"]
    else:
        # Explicit purpose + permanent admission reservation precede the first read.
        guard.assert_holdout_usage(
            "planner_final_ab",
            dataset_id=guard.HOLDOUT_ID,
            candidate_frozen=True,
            allow_single_final_ab=True,
            audit_log=out / "holdout-usage.jsonl",
        )
        data = json.loads(guard.HOLDOUT_PATH.read_text("utf-8"))
        config, routes = _live_routes(frozen["routes"]["judge"], frozen["routes"]["reflection"])
        if routes != frozen["routes"]:
            raise ValueError("final_model_routes_changed")
        from evals.agent_eval_ext.execution_observer import environment_secrets

        runtime = LiveRuntime(
            config,
            routes["judge"],
            routes["reflection"],
            frozen["request_timeouts"]["planner"],
            environment_secrets(),
            metric_version="v3",
            budget=budget,
            judge_timeout=frozen["request_timeouts"]["judge"],
        )
    if final.content_hash(data) != req["dataset_sha256"]:
        raise ValueError("final_dataset_changed")
    cases = final.as_cases(data)
    journal(
        "final_cases_loaded",
        {
            "simulated": req["mode"] == "demo",
            "case_count": len(cases),
            "holdout_sha256": req["holdout_sha256"],
        },
    )
    arms = {"baseline": [], "candidate": []}
    # Paired order reduces temporal drift; every successful observation is saved
    # before another request, without pretending Journal is a resume checkpoint.
    for case in cases:
        for arm in ("baseline", "candidate"):
            prompt = source[arm]["override"]
            row = await runtime.observe(case, prompt)
            journal(
                "observation",
                {
                    "arm": arm,
                    "prompt_override": prompt,
                    "prompt_sha256": frozen[arm + "_sha256"],
                    "observation": row.model_dump(),
                },
            )
            if case_score_v3(row) is None:
                raise ValueError("final_observation_unmeasured")
            arms[arm].append(row)
    gate = gate_v3(
        cases,
        arms["baseline"],
        arms["candidate"],
        0.05,
        require_live_evidence=req["mode"] == "live",
    )
    result = {
        "schema_version": 1,
        "task_kind": final.KIND,
        "source_job_id": req["source_job_id"],
        "source_sha256": req["source_sha256"],
        "evidence_kind": "live_planner" if req["mode"] == "live" else "fake_planner",
        "metric_version": "v3",
        "metric_rules_sha256": frozen["metric_rules_sha256"],
        "routes": frozen["routes"],
        "limits": frozen["limits"],
        "judge_settings": frozen["judge_settings"],
        "request_timeouts": frozen["request_timeouts"],
        "dataset": data,
        "dataset_sha256": final.content_hash(data),
        "baseline": source["baseline"],
        "candidate": source["candidate"],
        "holdout": {
            **{arm: [r.model_dump() for r in rows] for arm, rows in arms.items()},
            "gate": gate,
        },
        "release_ready": False,
        "full_report_ab": "separate_manual_gate",
    }
    final.verify_result(req, result)
    atomic_create(out / "experiment.json", result)
    journal("completed", {"holdout_gate": gate})
    return 0 if gate["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strict-budget", action="store_true", required=True)
    parser.add_argument("--budget-config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    folder = args.job_dir.resolve()
    req = _read(folder / "request.json")
    root = folder.parents[2]
    if (
        args.budget_config.resolve() != folder / "budget.json"
        or args.out_dir.resolve() != folder.parent.parent / "experiments" / req["job_id"]
        or args.live != (req["mode"] == "live")
    ):
        raise ValueError("final_cli_job_binding_mismatch")
    source = final.validate_inputs(root, folder, req)
    out = _new_dir(args.out_dir)
    journal = ExperimentJournal(out / "journal")
    budget = PaidCallBudget(out / "paid-budget", BudgetConfig(**req["budget"]))
    budget.preflight()
    with budget.execution_lock():
        try:
            journal(
                "started",
                {
                    "task_kind": final.KIND,
                    "source_job_id": req["source_job_id"],
                    "routes": req["frozen"]["routes"],
                    "limits": req["frozen"]["limits"],
                    "dataset": {"dataset_id": req["dataset_preset"]},
                    "simulated": req["mode"] == "demo",
                },
            )
            return asyncio.run(execute(req, source, out, budget, journal))
        except BaseException as exc:
            journal(
                "interrupted",
                {
                    "error_type": type(exc).__name__,
                    "slot_id": getattr(exc, "slot_id", None),
                    "role": getattr(exc, "role", None),
                    "cause_type": getattr(exc, "cause_type", None),
                },
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
