"""Planner Prompt 优化 CLI。真实评测显式启用，发布配置交由人工应用。"""

from __future__ import annotations

import argparse
import asyncio
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
from evals.agent_eval_ext.long_run import atomic_create
from research_engine.local import load_env_file, run_config_from_env, usable_providers
from research_engine.prompts import PLANNER_PROMPT_V2
from research_engine.runconfig import ROLES

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
    dataset = Dataset.model_validate(read(args.dataset or DEMO_DATASET))
    if args.live and args.dataset is None:
        raise ValueError("live optimization requires an explicit dataset file")
    if not args.live and dataset.provenance != "demo":
        raise ValueError("demo mode requires a dataset labeled demo")
    baseline = _baseline(args.baseline)
    if not (1 <= args.proposals <= 20 and 1 <= args.max_evals <= 2000):
        raise ValueError("use 1–20 proposals and 1–2000 evaluations")
    if not (0 < args.min_gain <= 1 and 1 <= args.timeout <= 600):
        raise ValueError("min-gain must be in (0, 1] and request timeout in 1–600 seconds")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    limits = SearchLimits(args.max_evals, args.proposals, args.seed, args.min_gain)
    report_hash = input_hash(read(REPORT_DATASET)["queries"])
    optimizer_version = version("gepa")
    routes = None
    runtime = None
    judge_settings = None
    if args.live:
        from evals.judge_config import judge_max_output_tokens

        from .runtime import PLANNER_JUDGE_PROMPT, LiveRuntime

        config, routes = _live_routes(args.judge, args.reflector)
        runtime = LiveRuntime(
            config, routes["judge"], routes["reflection"], args.timeout, environment_secrets()
        )
        judge_settings = {
            "rubric_sha256": input_hash(PLANNER_JUDGE_PROMPT),
            "output_tokens": judge_max_output_tokens(),
        }
        if baseline:
            runtime.checked_text(baseline)
        for case in dataset.cases:
            for text in [case.query, *case.required_aspects]:
                runtime.checked_text(text)

    # 独占目录在任何模型调用前生效，重复运行必须有新的实验路径。
    out = _new_dir(args.out_dir)
    journal = ExperimentJournal(out / "journal")
    journal(
        "started",
        {
            "evidence_kind": "live_planner" if args.live else "scripted_demo",
            "dataset": dataset.model_dump(),
            "baseline_override": baseline,
            "routes": routes,
            "judge_settings": judge_settings,
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
                )
    except BaseException as exc:
        # 失败状态只记异常类型，不把 SDK 原始异常或隐藏推理写入审计文件。
        try:
            journal("interrupted", {"error_type": type(exc).__name__})
        except OSError:
            pass  # 磁盘故障时保留原始异常，已经写成的事件仍可独立查看。
        raise

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
        "evidence_kind": "live_planner" if args.live else "scripted_demo",
        "metric_version": METRIC_VERSION,
        "optimizer": {"name": "GEPA", "version": optimizer_version},
        "limits": asdict(limits),
        "routes": routes,
        "judge_settings": judge_settings,
        "dataset": dataset.model_dump(),
        "report_dataset_sha256": report_hash,
        "prompt_diff": prompt_diff,
        **result,
    }
    atomic_create(out / "experiment.json", experiment)
    journal("completed", {"holdout_gate": result["holdout"]["gate"]})
    gate = result["holdout"]["gate"]
    mode = "真实 Planner" if args.live else "脚本演示"
    print(
        f"{mode} · GEPA {optimizer_version} · 留出集门禁 {'通过' if gate['passed'] else '未通过'}"
    )
    if gate["delta"] is not None:
        print(f"留出集覆盖率 {gate['baseline_coverage']:.2%} → {gate['candidate_coverage']:.2%}")
    if gate["reasons"]:
        print(", ".join(gate["reasons"]))
    print(f"实验结果 {out / 'experiment.json'}")
    return 0 if gate["passed"] else 1


def _report(args: argparse.Namespace) -> int:
    experiment = read(args.experiment)
    baseline, candidate = release_candidate(experiment)
    routes = experiment["routes"]
    _, current_routes = _live_routes(routes["judge"], routes["reflection"])
    if current_routes != routes:
        raise ValueError("the report must use the same pinned model routes as the experiment")
    queries = read(REPORT_DATASET)["queries"]
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
    gate = report_gate(experiment, report, read(REPORT_DATASET)["queries"])
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
    optimize.add_argument("--seed", type=int, default=42, help="GEPA search seed")
    optimize.add_argument(
        "--timeout", type=int, default=90, help="deadline per model stage in seconds"
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
    except (ClosureError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Planner evolution: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
