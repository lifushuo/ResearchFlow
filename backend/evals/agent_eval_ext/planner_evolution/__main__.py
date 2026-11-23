"""Planner Prompt 优化 CLI。真实评测显式启用，发布配置交由人工应用。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from difflib import unified_diff
from importlib.metadata import version
from pathlib import Path

from app.services.evaluation_closure import ClosureError, input_hash
from app.services.evaluation_closure.storage import store_at
from app.services.run_config import prompt_settings_payload
from evals.agent_eval_ext.closure_observer import read, sources
from evals.agent_eval_ext.execution_observer import environment_secrets
from evals.agent_eval_ext.long_run import atomic_create, sha256_text
from evals.paid_call_budget import BudgetError
from evals.report_ab_dataset import (
    INDEPENDENT_REPORT_SET,
    fixed_report_queries,
    selected_report_dataset_for_experiment,
)
from research_engine.local import load_env_file, run_config_from_env, usable_providers
from research_engine.prompts import PLANNER_PROMPT_V2
from research_engine.runconfig import ROLES

from . import metrics_v2, metrics_v3
from .experiment import (
    METRIC_VERSION,
    Case,
    Dataset,
    SearchLimits,
    checked_prompt,
    demo_reflection,
    demo_rollout,
    optimize_planner,
)
from .journal import ExperimentJournal
from .judge_prompt import PLANNER_JUDGE_PROMPT_V2
from .release import release_candidate, report_gate

BACKEND = Path(__file__).resolve().parents[3]
DEMO_DATASET = BACKEND / "evals/agent_eval_ext/fixtures/planner_evolution_demo.json"
REPORT_DATASET = BACKEND / "evals/queries.json"


def _new_dir(path: Path) -> Path:
    root = path.resolve()
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    # docs/08：创建目录不证明能够写文件；使用与最终产物相同的独占、flush/fsync 路径。
    atomic_create(root / "write-check.json", {"schema_version": 1, "ok": True})
    return root


def _baseline(path: Path | None) -> str | None:
    if path is None:
        return None
    raw = read(path)
    if not isinstance(raw, dict) or set(raw) != {"planner"} or not isinstance(raw["planner"], str):
        raise ValueError("baseline must be a JSON object with one planner string")
    return checked_prompt(raw["planner"])


def _live_routes(judge: str | None, reflector: str | None):
    from evals.harness import validate_judge
    from evals.judge_config import judge_max_output_tokens, judge_provider_keys

    load_env_file(BACKEND.parent / ".env")
    config = run_config_from_env(fake=False)
    routes = {role: config.model_for(role) for role in ROLES}
    judge = judge or os.getenv("EVAL_JUDGE_MODEL", "")
    reflector = reflector or routes["planner"]
    validate_judge(judge, {**routes, "reflection": reflector})
    judge_max_output_tokens()
    for route in [*routes.values(), reflector]:
        provider, _, model = route.partition(":")
        if not model or provider not in usable_providers(dict(config.provider_keys)):
            raise ValueError(
                f"no configured endpoint or credential for model provider {provider!r}"
            )
    judge_keys = {**config.provider_keys, **judge_provider_keys(judge)}
    if judge.partition(":")[0] not in usable_providers(judge_keys):
        raise ValueError("the configured independent Judge has no usable endpoint or credential")
    return config, {"pipeline": routes, "judge": judge, "reflection": reflector}


def _optimize(args: argparse.Namespace) -> int:
    # Fail closed BEFORE any model call: search / candidate selection must never
    # read the frozen final-acceptance holdout (evals/holdout_usage_guard.py).
    if args.dataset is not None:
        from evals.holdout_usage_guard import assert_holdout_usage

        assert_holdout_usage("search", path=args.dataset)
    dataset = Dataset.model_validate(read(args.dataset or DEMO_DATASET))
    if args.live and args.dataset is None:
        raise ValueError("live optimization requires an explicit dataset file")
    if not args.live and not args.fake and dataset.provenance != "demo":
        raise ValueError("demo mode requires a dataset labeled demo")
    baseline = _baseline(args.baseline)
    if not (1 <= args.proposals <= 20 and 1 <= args.max_evals <= 2000):
        raise ValueError("use 1–20 proposals and 1–2000 evaluations")
    if not (0 < args.min_gain <= 1 and 1 <= args.timeout <= 600):
        raise ValueError("min-gain must be in (0, 1] and request timeout in 1–600 seconds")
    if args.judge_timeout is not None and not 1 <= args.judge_timeout <= 600:
        raise ValueError("Judge timeout must be in 1..600 seconds")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    # Demo rollout 是脚本模拟，无法产生 v2 质量子分——demo 一律走 v1 覆盖率指标。
    metric_version = args.metric_version if (args.live or args.fake) else "v1"
    limits = SearchLimits(args.max_evals, args.proposals, args.seed, args.min_gain)
    # Only new LIVE optimization uses the independent 9-case report A/B set.
    # Demo and historical experiments retain their original fixed 10-case set.
    report_set = INDEPENDENT_REPORT_SET if args.live else REPORT_DATASET
    # Search-only jobs must not even read report/holdout questions for metadata.
    report_hash = None if args.no_holdout else input_hash(fixed_report_queries(report_set))
    optimizer_version = version("gepa")
    # Create the exclusive run directory BEFORE the budget ledger: preflight()
    # creates its own root with parents=True, which would otherwise pre-create
    # out_dir and make _new_dir()'s exist_ok=False check reject a fresh run.
    # Both still happen before any model call, so fail-closed ordering holds.
    out = _new_dir(args.out_dir)
    journal = ExperimentJournal(out / "journal")
    routes = None
    runtime = None
    judge_settings = None
    budget = None
    execution_lock = None
    if (
        getattr(args, "strict_budget", False)
        and (args.live or args.fake)
        and not getattr(args, "budget_config", None)
    ):
        raise ValueError("strict budget requires --budget-config for live runs")
    if getattr(args, "budget_config", None) and (args.live or args.fake):
        from evals.paid_call_budget import BudgetConfig, PaidCallBudget

        raw = json.loads(args.budget_config.read_text(encoding="utf-8"))
        budget = PaidCallBudget(
            out / "paid-budget",
            BudgetConfig(
                run_id=raw["run_id"],
                quotas=raw["quotas"],
                total_cap=raw["total_cap"],
                planned_evaluations=raw["planned_evaluations"],
                source=raw.get("source", "user-authorization"),
            ),
        )
        budget.preflight()
        execution_lock = budget.execution_lock()
        execution_lock.__enter__()
    if args.live:
        from evals.judge_config import judge_max_output_tokens

        from .runtime import PLANNER_JUDGE_PROMPT, LiveRuntime

        config, routes = _live_routes(args.judge, args.reflector)
        runtime = LiveRuntime(
            config,
            routes["judge"],
            routes["reflection"],
            args.timeout,
            environment_secrets(),
            metric_version=metric_version,
            budget=budget,
            judge_timeout=args.judge_timeout,
        )
        judge_settings = {
            "rubric_sha256": input_hash(
                PLANNER_JUDGE_PROMPT if metric_version == "v1" else PLANNER_JUDGE_PROMPT_V2
            ),
            "output_tokens": judge_max_output_tokens(),
        }
        if baseline:
            runtime.checked_text(baseline)
        for case in dataset.cases:
            for text in [case.query, *case.required_aspects]:
                runtime.checked_text(text)

    if args.fake:
        from .fake_runtime import FakeRuntime

        if budget is None or metric_version != "v3" or not args.no_holdout:
            raise ValueError("fake v3 requires strict budget and explicit no-holdout")
        runtime = FakeRuntime(budget)
        routes = {
            "pipeline": {role: "fake:planner" for role in ROLES},
            "judge": "fake:judge",
            "reflection": "fake:reflection",
        }
        judge_settings = {
            "rubric_sha256": input_hash(PLANNER_JUDGE_PROMPT_V2),
            "output_tokens": 8192,
        }
    if args.frozen_config:
        from app.services.planner_gepa_contract import generation_settings

        pinned = read(args.frozen_config)
        if (
            pinned["routes"] != routes
            or pinned["metric_rules_sha256"] != metrics_v3.rules_hash()
            or pinned["baseline_sha256"] != sha256_text(baseline or PLANNER_PROMPT_V2)
            or pinned["judge_settings"] != judge_settings
            or args.min_gain != pinned["min_gain"]
            or metric_version != pinned["metric_version"]
            or not args.no_holdout
            or pinned["dataset_sha256"] != input_hash(dataset.model_dump())
            or pinned["limits"] != asdict(limits)
            or pinned["generation"] != generation_settings(judge_settings["output_tokens"])
            or pinned.get(
                "request_timeouts",
                {"planner": args.timeout, "judge": args.timeout, "reflection": args.timeout},
            )
            != {
                "planner": args.timeout,
                "judge": args.judge_timeout if args.judge_timeout is not None else args.timeout,
                "reflection": args.timeout,
            }
        ):
            raise ValueError("frozen GEPA configuration mismatch")
    # 独占目录与预算账本在任何模型调用前已生效；重复运行必须有新的实验路径。
    journal(
        "started",
        {
            "evidence_kind": "live_planner"
            if args.live
            else "fake_planner"
            if args.fake
            else "scripted_demo",
            "preexperiment_no_holdout": args.no_holdout,
            "metric_version": metric_version,
            "metric_rules_sha256": metrics_v3.rules_hash() if metric_version == "v3" else None,
            "dataset": dataset.model_dump(),
            "baseline_override": baseline,
            "routes": routes,
            "judge_settings": judge_settings,
            "request_timeouts": {
                "planner": args.timeout,
                "judge": args.judge_timeout if args.judge_timeout is not None else args.timeout,
                "reflection": args.timeout,
            },
            "limits": asdict(limits),
        },
    )
    try:
        if runtime is None:
            result = optimize_planner(
                dataset,
                baseline,
                demo_rollout,
                demo_reflection,
                limits,
                record=journal,
                metric_version=metric_version,
                allow_holdout=not args.no_holdout,
            )
        else:
            with asyncio.Runner() as loop:
                result = optimize_planner(
                    dataset,
                    baseline,
                    lambda case, prompt: loop.run(runtime.observe(case, prompt)),
                    lambda prompt: loop.run(runtime.reflect(prompt)),
                    limits,
                    require_live_evidence=True,
                    record=journal,
                    metric_version=metric_version,
                    allow_holdout=not args.no_holdout,
                )
    except BaseException as exc:
        # 失败状态只记异常类型，不把 SDK 原始异常或隐藏推理写入审计文件。
        try:
            journal(
                "interrupted",
                {
                    "error_type": type(exc).__name__,
                    "slot_id": getattr(exc, "slot_id", None),
                    "role": getattr(exc, "role", None),
                    "cause_type": getattr(exc, "cause_type", None),
                },
            )
        except OSError:
            pass  # 磁盘故障时保留原始异常，已经写成的事件仍可独立查看。
        raise
    finally:
        if execution_lock is not None:
            execution_lock.__exit__(None, None, None)

    # Fail closed for live runs: zero scored observations must never be dressed
    # up as a completed search with a candidate and exit code 0. The failed
    # run1 exposed this: a stranded SENT slot denied all observations, yet the
    # CLI wrote a baseline candidate and reported success. A measured gain is
    # required; "no improvement" (gain <= 0) is still a valid, measured result.
    if (args.live or args.fake) and result.get("search", {}).get("validation_gain") is None:
        journal("unmeasured_live_run", {"reason": "validation_gain_unmeasured"})
        raise ValueError(
            "live search produced no measured validation gain; refusing to freeze a candidate"
        )
    candidate = result.get("candidate")
    prompt_diff = None
    if candidate:
        atomic_create(out / "candidate.json", {"planner": candidate["override"]})
        prompt_diff = "\n".join(
            unified_diff(
                (baseline or PLANNER_PROMPT_V2).splitlines(),
                candidate["override"].splitlines(),
                fromfile="baseline",
                tofile="candidate",
                lineterm="",
            )
        )
    experiment = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "evidence_kind": "live_planner"
        if args.live
        else "fake_planner"
        if args.fake
        else "scripted_demo",
        "preexperiment_no_holdout": args.no_holdout,
        # v2/v3 实验记录冻结的评分规则哈希；v1 历史实验保持原样，规则哈希为 null。
        "metric_version": (
            METRIC_VERSION
            if metric_version == "v1"
            else metrics_v3.METRIC_VERSION_V3
            if metric_version == "v3"
            else metrics_v2.METRIC_VERSION_V2
        ),
        "metric_rules_sha256": (
            metrics_v3.rules_hash()
            if metric_version == "v3"
            else metrics_v2.rules_hash()
            if metric_version == "v2"
            else None
        ),
        "optimizer": {"name": "GEPA", "version": optimizer_version},
        "limits": asdict(limits),
        "routes": routes,
        "judge_settings": judge_settings,
        "request_timeouts": {
            "planner": args.timeout,
            "judge": args.judge_timeout if args.judge_timeout is not None else args.timeout,
            "reflection": args.timeout,
        },
        "dataset": dataset.model_dump(),
        "report_dataset_sha256": report_hash,
        "prompt_diff": prompt_diff,
        **result,
    }
    atomic_create(out / "experiment.json", experiment)
    journal("completed", {"holdout_gate": result["holdout"]["gate"]})
    gate = result["holdout"]["gate"]
    mode = "真实 Planner" if args.live else "免费 Fake v3" if args.fake else "脚本演示"
    print(
        f"{mode} · GEPA {optimizer_version} · 留出集门禁 {'待终验' if gate['passed'] is None else '通过' if gate['passed'] else '未通过'}"
    )
    if gate.get("delta") is not None and "baseline_coverage" in gate:
        print(f"留出集覆盖率 {gate['baseline_coverage']:.2%} → {gate['candidate_coverage']:.2%}")
    if gate["reasons"]:
        print(", ".join(gate["reasons"]))
    print(f"实验结果 {out / 'experiment.json'}")
    # 预实验没有正式门禁：完成即成功退出，判定交给人看观测数据。
    return 0 if (args.no_holdout or gate["passed"]) else 1


def _report(args: argparse.Namespace) -> int:
    experiment = read(args.experiment)
    baseline, candidate = release_candidate(experiment)
    routes = experiment["routes"]
    _, current_routes = _live_routes(routes["judge"], routes["reflection"])
    if current_routes != routes:
        raise ValueError("the report must use the same pinned model routes as the experiment")
    report_set, queries = selected_report_dataset_for_experiment(experiment)
    if input_hash(queries) != experiment["report_dataset_sha256"]:
        raise ValueError("the fixed report dataset changed after the experiment")
    if args.case_id is not None and args.case_id not in {case["id"] for case in queries}:
        raise ValueError("the report case is not in the fixed benchmark")

    out = _new_dir(args.out_dir)
    harness_out = out / "report-ab.json"
    candidate_path = out / "candidate.json"
    atomic_create(candidate_path, {"planner": candidate})
    command = [
        sys.executable,
        "-m",
        "evals.harness",
        "--mode",
        "report",
        "--judge",
        routes["judge"],
        "--candidate",
        str(candidate_path),
        "--trace",
        "--trace-dir",
        str(out / "trace"),
        "--progress-dir",
        str(out / "progress"),
        "--out",
        str(harness_out),
    ]
    if report_set != REPORT_DATASET:
        command.extend(["--report-queries", str(report_set)])
    if baseline is not None:
        baseline_path = out / "baseline.json"
        atomic_create(baseline_path, {"planner": baseline})
        command.extend(["--baseline", str(baseline_path)])
    if args.case_id:
        command.extend(["--case-id", args.case_id])
    if args.resume_from:
        command.extend(["--resume-from", str(args.resume_from.resolve())])
    environment = {
        **os.environ,
        "LLM_MODE": "real",
        "RF_JUDGE_BATCH_TRACE_DIR": str(out / "judge-batches"),
    }
    environment.pop("RF_EVAL_RESUME_FROM", None)
    environment.pop("RF_EVAL_EXECUTION_ROOT", None)
    result = subprocess.run(command, cwd=BACKEND, env=environment, check=False)
    if result.returncode:
        return result.returncode
    if args.case_id:
        print(f"单题试运行结果 {harness_out}")
        return 0
    gate = report_gate(experiment, read(harness_out), queries)
    atomic_create(out / "report-gate.json", gate)
    print(f"报告 A/B 门禁通过 {out / 'report-gate.json'}")
    return 0


def _release(args: argparse.Namespace) -> int:
    approver = args.approved_by.strip()
    if not approver or len(approver) > 80:
        raise ValueError("approved-by must name one reviewer (1–80 characters)")
    experiment, report = read(args.experiment), read(args.report)
    baseline, candidate = release_candidate(experiment)
    _, report_queries = selected_report_dataset_for_experiment(experiment)
    gate = report_gate(experiment, report, report_queries)
    out = _new_dir(args.out_dir)
    atomic_create(out / "apply.json", prompt_settings_payload({"planner": candidate}))
    atomic_create(out / "rollback.json", prompt_settings_payload({"planner": baseline}))
    atomic_create(
        out / "release.json",
        {
            "schema_version": 1,
            "version": "planner-" + experiment["candidate"]["sha256"][:12],
            "approved_by": approver,
            "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompt_sha256": experiment["candidate"]["sha256"],
            "parent_prompt_sha256": experiment["baseline"]["sha256"],
            "experiment_sha256": input_hash(experiment),
            "report_sha256": input_hash(report),
            "report_gate": gate,
        },
    )
    print(f"已生成审批记录、发布与回滚配置 {out}")
    return 0


def _dataset(args: argparse.Namespace) -> int:
    from app.services.planner_evolution_center import (
        SEED_DATASET,
        assemble_dataset,
        read_records,
    )

    seed = Dataset.model_validate(read(args.seed or SEED_DATASET))
    records, broken = read_records(args.cases_dir)
    if broken:
        print(f"跳过损坏回流记录: {', '.join(broken)}")
    result = assemble_dataset(seed, records)
    if result["error"]:
        raise ValueError(f"合并后的数据集无效: {result['error']}")
    splits = result["splits"]
    # 活跃数据集直接交给 optimize --dataset，与 UI 看到的合并视图同源。
    atomic_create(args.out, result["dataset"])
    print(
        f"活跃数据集 {args.out}: train={splits.get('train', 0)} "
        f"validation={splits.get('validation', 0)} holdout={splits.get('holdout', 0)}"
        f"（含回流 {result['backflow_case_count']} 题）"
    )
    return 0


def _seed_case(args: argparse.Namespace) -> int:
    if not (args.job_root / "closure-v1.sqlite3").is_file():
        raise ValueError("no evaluation database under job-root")
    detail = store_at(args.job_root).bad_case_detail(args.owner, args.bad_case_id)
    bad_case, latest = detail["bad_case"], detail["last_result"]
    if bad_case["dataset_id"] != "report-benchmark" or not latest:
        raise ValueError("choose a Bad Case from a measured report benchmark run")
    source_cases, _ = sources("report-benchmark", 0)
    source = source_cases.get(bad_case["case_id"])
    if not source or input_hash(source) != latest.get("input_sha256"):
        raise ValueError("latest Bad Case input differs from the report benchmark source")
    case = Case(
        case_id=args.case_id,
        group=args.group,
        split="train",
        query=source["case"]["query"],
        depth=source["case"].get("depth", "balanced"),
        required_aspects=args.aspect,
        source_bad_case_id=args.bad_case_id,
    )
    atomic_create(args.out, case.model_dump())
    print(f"已导出待纳入开发集的规划案例 {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ResearchFlow Planner Prompt optimization and release"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    optimize = commands.add_parser(
        "optimize", help="search with GEPA, then evaluate one winner on holdout"
    )
    mode = optimize.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--demo", action="store_true", help="scripted data and models, no network calls"
    )
    mode.add_argument(
        "--live", action="store_true", help="run configured real models, may incur API costs"
    )
    mode.add_argument(
        "--fake",
        action="store_true",
        help="offline Fake Planner/Judge/Reflection through v3 and request ledger",
    )
    optimize.add_argument("--frozen-config", type=Path, help="immutable web-job protocol snapshot")
    optimize.add_argument("--dataset", type=Path)
    optimize.add_argument("--out-dir", type=Path, required=True)
    optimize.add_argument(
        "--baseline", type=Path, help="single-role JSON for an approved Planner version"
    )
    optimize.add_argument("--judge", help="independent provider:model, default EVAL_JUDGE_MODEL")
    optimize.add_argument("--reflector", help="provider:model, default the configured Planner")
    optimize.add_argument("--proposals", type=int, default=4)
    optimize.add_argument(
        "--max-evals", type=int, default=64, help="Planner/Judge pairs, including holdout"
    )
    optimize.add_argument(
        "--min-gain", type=float, default=0.05, help="minimum absolute holdout gain"
    )
    optimize.add_argument(
        "--judge-timeout",
        type=int,
        help="Judge-only deadline; defaults to --timeout for legacy CLI",
    )
    optimize.add_argument("--seed", type=int, default=42, help="GEPA search seed")
    optimize.add_argument(
        "--budget-config",
        type=Path,
        help="authorized paid-call budget JSON (run_id, quotas{planner,judge,reflection}, total_cap, planned_evaluations)",
    )
    optimize.add_argument(
        "--strict-budget",
        action="store_true",
        help="refuse live runs unless --budget-config is supplied and the execution lock is free",
    )
    optimize.add_argument(
        "--timeout", type=int, default=90, help="deadline per model stage in seconds"
    )
    optimize.add_argument(
        "--metric-version",
        choices=("v1", "v2", "v3"),
        default="v1",
        help=(
            "candidate-selection metric: v1 coverage, v2 composite quality, "
            "or v3 quality-only with coverage as a hard constraint"
        ),
    )
    optimize.add_argument(
        "--no-holdout",
        action="store_true",
        help=(
            "pre-experiment mode: exclude the holdout split from every stage "
            "(no holdout model calls, gate marked not-applicable); for score-"
            "discrimination diagnostics only, never for release decisions"
        ),
    )
    optimize.set_defaults(action=_optimize)

    report = commands.add_parser(
        "report", help="run the real report Harness against both Prompt versions"
    )
    report.add_argument(
        "--live", action="store_true", required=True, help="allow paid report model calls"
    )
    report.add_argument("--experiment", type=Path, required=True)
    report.add_argument("--out-dir", type=Path, required=True)
    report.add_argument(
        "--case-id", help="one case for smoke testing; release uses the full benchmark"
    )
    report.add_argument(
        "--resume-from", type=Path, help="progress directory from a matching report A/B run"
    )
    report.set_defaults(action=_report)

    release = commands.add_parser(
        "release", help="review a passing experiment and export apply/rollback payloads"
    )
    release.add_argument("--experiment", type=Path, required=True)
    release.add_argument("--report", type=Path, required=True)
    release.add_argument("--out-dir", type=Path, required=True)
    release.add_argument("--approved-by", required=True)
    release.set_defaults(action=_release)

    dataset = commands.add_parser(
        "dataset",
        help="assemble the active optimization dataset from the seed and backflow records",
    )
    dataset.add_argument(
        "--cases-dir",
        type=Path,
        required=True,
        help="directory of backflow records (job-root/planner-evolution/cases)",
    )
    dataset.add_argument(
        "--seed", type=Path, help="seed dataset JSON; default the reviewed in-repo set"
    )
    dataset.add_argument("--out", type=Path, required=True)
    dataset.set_defaults(action=_dataset)

    seed = commands.add_parser(
        "seed-case", help="turn an existing Bad Case into a reviewed Planner training example"
    )
    seed.add_argument("--job-root", type=Path, required=True)
    seed.add_argument("--owner", required=True, help="existing evaluation owner's user ID")
    seed.add_argument("--bad-case-id", required=True)
    seed.add_argument("--case-id", required=True)
    seed.add_argument("--group", required=True)
    seed.add_argument(
        "--aspect", action="append", required=True, help="repeat for every required aspect"
    )
    seed.add_argument("--out", type=Path, required=True)
    seed.set_defaults(action=_seed_case)

    args = parser.parse_args(argv)
    try:
        return args.action(args)
    except (
        ClosureError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        BudgetError,
    ) as exc:
        print(f"Planner evolution: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
