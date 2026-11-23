"""Planner Judge v3 校准：在构造刺激上测量 Judge 判别力与重复性，为 v3 门禁供证。

背景（用户验收要求的修正）：此前的"Coverage 区分能力测试"只验证了 case_score_v3
的算术，没有验证独立 Judge 能否在真实输出上识别缺漏维度。本脚本补的正是后者——

- 刺激：完整/缺漏/含混三类构造计划（fixtures/planner_judge_v3_stimuli.json），
  预期结果来自构造本身（哪条任务存在），与留出集无关
- 判别力：缺漏计划的 coverage 必须低于完整计划；含混计划的 specificity 必须低于
  完整计划；并对照构造预期逐项核对符合率
- 重复性：完整计划的重复评分给出逐轴 MAE；输出 gate_reference 噪声下界
  （2×质量MAE），仅作参考——评分误差不是效应量，门禁由人决定
- 调用：复用 suggest_planner_coverage_v2（与评测中心同一提示词、同一契约、
  同一独立性检查），默认 --plan 只打印调用计划，零网络零费用；
  真实调用必须 --execute --consent I_CONFIRM_JUDGE_API_COST
- 可靠性：全部调用在单个事件循环内顺序执行。langchain-openai 按
  (base_url, timeout) lru_cache 共享 httpx 异步客户端，每调用一次 asyncio.run
  会让共享池里的旧循环连接抛 RuntimeError("Event loop is closed")——
  首轮 8/15 的根因，已由本地故障注入复现确诊
- 补测：--resume-from 复用首轮已测量判定，只重测未测量调用；
  新证据文件逐条标注来源，合并总数不伪装成单轮结果
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUIRED_CONSENT = "I_CONFIRM_JUDGE_API_COST"
DEFAULT_STIMULI = (
    Path(__file__).resolve().parents[1] / "agent_eval_ext/fixtures/planner_judge_v3_stimuli.json"
)


def load_stimuli(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if raw.get("purpose") != "planner_judge_v3_calibration_stimuli":
        raise ValueError("wrong_stimuli_purpose")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("no_stimulus_cases")
    for case in cases:
        variants = case.get("variants")
        if not isinstance(variants, dict) or set(variants) != {"complete", "missing", "vague"}:
            raise ValueError(f"stimulus {case.get('case_id')} must have exactly three variants")
        aspects = case.get("required_aspects")
        if not isinstance(aspects, list) or not 2 <= len(aspects) <= 12:
            raise ValueError(f"stimulus {case.get('case_id')} needs 2–12 aspects")
        for name, variant in variants.items():
            tasks = variant.get("task_queries")
            if not isinstance(tasks, list) or not 1 <= len(tasks) <= 30:
                raise ValueError(f"stimulus {case.get('case_id')}/{name} needs 1–30 tasks")
            if not isinstance(variant.get("expected"), dict):
                raise ValueError(f"stimulus {case.get('case_id')}/{name} declares no expectation")
    return raw


def call_plan(stimuli: dict, repeats: int) -> list[dict[str, Any]]:
    """每题：完整计划按 repeats 重复评分 + 缺漏、含混各一次。"""
    plan: list[dict[str, Any]] = []
    for case in stimuli["cases"]:
        for repeat in range(1, repeats + 1):
            plan.append({"case_id": case["case_id"], "variant": "complete", "repeat": repeat})
        plan.append({"case_id": case["case_id"], "variant": "missing", "repeat": None})
        plan.append({"case_id": case["case_id"], "variant": "vague", "repeat": None})
    return plan


def _observation_fields(
    *,
    case: dict,
    variant: str,
    repeat: int | None,
    task_queries: list[str],
    suggestion: dict[str, Any],
) -> dict[str, Any]:
    """把 suggest_planner_coverage_v2 的建议投影为与实验同构的评分输入。

    covered 的语义与实验一致：维度匹配到非空任务集才算覆盖。
    """
    covered = [a["aspect_index"] for a in suggestion["aspects"] if a["covered"]]
    return {
        "case_id": case["case_id"],
        # 开发集刺激按 train 标记参与打分；校准用途由报告的 purpose 字段承载。
        "split": "train",
        "required_count": len(case["required_aspects"]),
        "task_queries": task_queries,
        "covered_indices": covered,
        "task_specificity": [t["specificity"] for t in suggestion["task_specificity"]],
        "plan_quality": [a["plan_quality"] for a in suggestion["aspects"]],
        "served_judge": suggestion.get("served_judge"),
        # 每次调用的真实用量（输入/输出 token）：校准证据契约要求逐调用记账。
        "judge_tokens": suggestion.get("judge_tokens"),
        "variant": variant,
        "repeat": repeat,
        "error": None,
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


# Observation 模型是 extra="forbid"：行内的记账字段（variant/repeat/run_source/
# 错误分类等）必须在这里白名单过滤，不能简单减去几个键。
_OBSERVATION_FIELDS = frozenset(
    {
        "case_id",
        "split",
        "required_count",
        "task_queries",
        "covered_indices",
        "task_matches",
        "error",
        "planner_tokens",
        "judge_tokens",
        "served_judge",
        "task_specificity",
        "plan_quality",
        "latency_seconds",
    }
)


def summarize(stimuli: dict, observations: list[dict[str, Any]]) -> dict[str, Any]:
    """离线聚合：判别力、构造预期符合率、重复性、建议门禁。全 fail-closed。"""
    from evals.agent_eval_ext.planner_evolution.experiment import Observation
    from evals.agent_eval_ext.planner_evolution.metrics_v3 import case_score_v3

    by_case: dict[str, dict[str, list[Observation]]] = {}
    unmeasured = {"observations": 0, "case_comparisons": 0}
    for row in observations:
        if row.get("error") is not None:
            unmeasured["observations"] += 1
        scores = by_case.setdefault(row["case_id"], {"complete": [], "missing": [], "vague": []})
        obs = Observation.model_validate({k: v for k, v in row.items() if k in _OBSERVATION_FIELDS})
        scores[row["variant"]].append(obs)

    def scored(obs: Observation) -> dict | None:
        return case_score_v3(obs)

    def axis_mean(obs_list: list[Observation], axis: str) -> float | None:
        values = [s[axis] for o in obs_list if (s := scored(o)) is not None]
        return _mean(values)

    discrimination: dict[str, Any] = {"coverage": {}, "specificity": {}}
    coverage_wins = coverage_total = 0
    specificity_wins = specificity_total = 0
    # 构造预期符合率区分「未测量」与「不满足」：未测量绝不能被计为失败。
    conformance = {
        name: {"checked": 0, "passed": 0, "unmeasured": 0}
        for name in ("complete", "missing", "vague")
    }
    quality_wins_vs_missing = quality_total_vs_missing = 0
    quality_wins_vs_vague = quality_total_vs_vague = 0
    case_records: list[dict[str, Any]] = []

    for case in stimuli["cases"]:
        case_id = case["case_id"]
        arms = by_case.get(case_id, {"complete": [], "missing": [], "vague": []})
        record: dict[str, Any] = {"case_id": case_id}
        complete_cov = axis_mean(arms["complete"], "coverage")
        missing_cov = axis_mean(arms["missing"], "coverage")
        complete_spec = axis_mean(arms["complete"], "specificity")
        vague_spec = axis_mean(arms["vague"], "specificity")
        complete_quality = axis_mean(arms["complete"], "quality")
        missing_quality = axis_mean(arms["missing"], "quality")
        vague_quality = axis_mean(arms["vague"], "quality")
        record["coverage_complete"], record["coverage_missing"] = complete_cov, missing_cov
        record["specificity_complete"], record["specificity_vague"] = complete_spec, vague_spec
        record["quality_complete"], record["quality_missing"] = complete_quality, missing_quality
        record["quality_vague"] = vague_quality
        if complete_cov is None or missing_cov is None:
            unmeasured["case_comparisons"] += 1
        else:
            coverage_total += 1
            if missing_cov < complete_cov:
                coverage_wins += 1
        if complete_spec is None or vague_spec is None:
            unmeasured["case_comparisons"] += 1
        else:
            specificity_total += 1
            if vague_spec < complete_spec:
                specificity_wins += 1
        # 质量分排序：完整计划的质量分必须高于缺漏与含混计划——
        # 这是"Judge 能否稳定排出好坏规划顺序"的直接证据，供门禁决策使用。
        if complete_quality is None or missing_quality is None:
            unmeasured["case_comparisons"] += 1
        else:
            quality_total_vs_missing += 1
            if missing_quality < complete_quality:
                quality_wins_vs_missing += 1
        if complete_quality is None or vague_quality is None:
            unmeasured["case_comparisons"] += 1
        else:
            quality_total_vs_vague += 1
            if vague_quality < complete_quality:
                quality_wins_vs_vague += 1

        # 构造预期逐项核对：这是"Judge 能识别构造差异"的直接证据。
        stimulus = next(c for c in stimuli["cases"] if c["case_id"] == case_id)
        variant_checks = {
            "complete": lambda o, s, expected=stimulus["variants"]["complete"]["expected"]: (
                s["coverage"] == expected["coverage"]
                and s["specificity"] >= expected["specificity_mean_min"]
                and s["plan_quality"] >= expected["plan_quality_mean_min"]
            ),
            "missing": lambda o, s, expected=stimulus["variants"]["missing"]["expected"]: all(
                aspect not in o.covered_indices for aspect in expected["uncovered_aspects"]
            ),
            "vague": lambda o, s, expected=stimulus["variants"]["vague"]["expected"]: (
                s["specificity"] <= expected["specificity_mean_max"]
            ),
        }
        for name, check in variant_checks.items():
            rows = arms[name]
            if not rows:
                continue
            measured = [o for o in rows if scored(o) is not None]
            if not measured:
                conformance[name]["unmeasured"] += 1
                continue
            conformance[name]["checked"] += 1
            conformance[name]["passed"] += int(all(check(o, scored(o)) for o in measured))
            conformance[name]["unmeasured"] += len(rows) - len(measured)
        case_records.append(record)

    # 重复性：完整计划逐轴 MAE（重复评分相对各自均值的平均绝对偏差）。
    per_axis_mae: dict[str, float | None] = {}
    per_case_quality_mae: dict[str, float | None] = {}
    for axis in ("coverage", "specificity", "plan_quality", "quality"):
        maes: list[float] = []
        for case_id, arms in by_case.items():
            scores = [scored(o) for o in arms["complete"]]
            values = [s[axis] for s in scores if s is not None]
            if len(values) < 2:
                continue
            center = sum(values) / len(values)
            mae = sum(abs(v - center) for v in values) / len(values)
            maes.append(mae)
            if axis == "quality":
                per_case_quality_mae[case_id] = round(mae, 6)
        per_axis_mae[axis] = _mean([round(m, 6) for m in maes])

    quality_mae = per_axis_mae["quality"]
    # min_gain 决策参考（只读证据，不自动生效）：Judge 重复评分波动 × 2 只是
    # 门禁的噪声下界。评分误差大小 ≠ 两个 Prompt 的真实质量差异大小——
    # 正式门禁必须同时参考质量排序胜率与开发集效应量，由人决定；当前 min_gain
    # 保持 0.05 不变。
    gate_reference = (
        {
            "noise_lower_bound": round(max(2 * quality_mae, 0.01), 4),
            "basis": "2 × mean quality MAE across complete-plan repeats (n=3)",
            "caveat": (
                "judge repeatability is not effect size; the formal min_gain decision "
                "also needs quality ordering win rates and dev-set deltas, decided by a person"
            ),
        }
        if quality_mae is not None
        else None
    )

    discrimination["coverage"] = {
        "complete_mean": _mean(
            [r["coverage_complete"] for r in case_records if r["coverage_complete"] is not None]
        ),
        "missing_mean": _mean(
            [r["coverage_missing"] for r in case_records if r["coverage_missing"] is not None]
        ),
        "directional_win_rate": round(coverage_wins / coverage_total, 4)
        if coverage_total
        else None,
        "comparisons_measured": coverage_total,
    }
    discrimination["specificity"] = {
        "complete_mean": _mean(
            [
                r["specificity_complete"]
                for r in case_records
                if r["specificity_complete"] is not None
            ]
        ),
        "vague_mean": _mean(
            [r["specificity_vague"] for r in case_records if r["specificity_vague"] is not None]
        ),
        "directional_win_rate": (
            round(specificity_wins / specificity_total, 4) if specificity_total else None
        ),
        "comparisons_measured": specificity_total,
    }
    discrimination["quality"] = {
        "complete_gt_missing_win_rate": (
            round(quality_wins_vs_missing / quality_total_vs_missing, 4)
            if quality_total_vs_missing
            else None
        ),
        "complete_gt_vague_win_rate": (
            round(quality_wins_vs_vague / quality_total_vs_vague, 4)
            if quality_total_vs_vague
            else None
        ),
        "comparisons_measured": {
            "vs_missing": quality_total_vs_missing,
            "vs_vague": quality_total_vs_vague,
        },
    }
    return {
        "discrimination": discrimination,
        "expected_outcome_conformance": conformance,
        "repeatability": {
            "per_axis_mae": per_axis_mae,
            "per_case_quality_mae": per_case_quality_mae,
            "complete_repeats_used": True,
        },
        "gate_reference_min_gain": gate_reference,
        "unmeasured": unmeasured,
        "per_case": case_records,
    }


def _classify_error(exc: BaseException) -> dict[str, Any]:
    """失败证据的脱敏提取：类型、消息、cause 链、HTTP 状态与 request_id（若存在）。

    首轮校准只记录了异常类型，7 次 RuntimeError 无法归因；本函数补齐可诊断性，
    同时保持同一安全边界——消息经 content_snapshot 脱敏截断，原始凭据不落盘。
    """
    from evals.agent_eval_ext.execution_observer import environment_secrets
    from research_engine.trace_content import content_snapshot

    secrets = environment_secrets()

    def snap(value: object, limit: int = 320) -> str:
        return content_snapshot(str(value), secrets=secrets, max_chars=limit)["value"]

    causes: list[str] = []
    cause = exc.__cause__ or exc.__context__
    depth = 0
    while cause is not None and depth < 4:
        causes.append(f"{type(cause).__name__}: {snap(cause, 160)}")
        cause = cause.__cause__ or cause.__context__
        depth += 1
    status = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if isinstance(exc, TimeoutError):
        stage = "http_timeout"
    elif status is not None:
        stage = "http_status"
    elif type(exc).__name__ == "APIConnectionError":
        stage = "connection"
    else:
        stage = "runtime_or_unknown"
    return {
        "type": type(exc).__name__,
        "message": snap(exc),
        "stage": stage,
        "status_code": status if isinstance(status, int) else None,
        "request_id": request_id if isinstance(request_id, str) else None,
        "causes": causes,
    }


def _execute(stimuli: dict, plan: list[dict[str, Any]], timeout: int) -> list[dict[str, Any]]:
    from app.services import planner_review_judge

    observations: list[dict[str, Any]] = []
    stimuli_sha = hashlib.sha256(
        json.dumps(stimuli, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_id = f"planner-judge-v3-calibration-{stimuli_sha[:12]}"

    async def _run_all() -> None:
        # 全部调用共享一个事件循环——这是首轮 8/15 的教训：langchain-openai 按
        # (base_url, timeout) 用 lru_cache 进程级共享 httpx 异步客户端，每次
        # asyncio.run 新建事件循环会让连接池里的旧循环 keep-alive 连接在被复用/
        # 拆卸时抛 RuntimeError("Event loop is closed")（快速失败，与 GEPA 的
        # 单 asyncio.Runner 形态相对照，64 次调用无一失败）。单循环下共享客户端
        # 只在创建它的循环上使用，缺陷不触发。
        for item in plan:
            case = next(c for c in stimuli["cases"] if c["case_id"] == item["case_id"])
            tasks = case["variants"][item["variant"]]["task_queries"]
            evidence = {
                "run_id": run_id,
                "case_id": case["case_id"],
                "evidence_reference": "planner_judge_v3_stimuli",
                "tasks": [
                    {"id": str(index), "query": query} for index, query in enumerate(tasks, 1)
                ],
            }
            started = time.monotonic()
            try:
                suggestion = await planner_review_judge.suggest_planner_coverage_v2(
                    evidence=evidence,
                    question=case["query"],
                    aspects=case["required_aspects"],
                    timeout_seconds=timeout,
                )
                observations.append(
                    {
                        **_observation_fields(
                            case=case,
                            variant=item["variant"],
                            repeat=item["repeat"],
                            task_queries=tasks,
                            suggestion=suggestion,
                        ),
                        "latency_seconds": round(time.monotonic() - started, 2),
                        "run_source": run_id,
                    }
                )
            except Exception as exc:  # noqa: BLE001 — provider errors may embed credentials
                # 未测量保持未测量，绝不按 0 分计，也绝不重试计费。
                # 错误按阶段分类（超时/HTTP 状态/连接/运行时），详情脱敏落盘。
                classified = _classify_error(exc)
                observations.append(
                    {
                        "case_id": case["case_id"],
                        "split": "train",
                        "required_count": len(case["required_aspects"]),
                        "task_queries": tasks,
                        "covered_indices": [],
                        "task_specificity": None,
                        "plan_quality": None,
                        "served_judge": None,
                        "judge_tokens": None,
                        "latency_seconds": round(time.monotonic() - started, 2),
                        "variant": item["variant"],
                        "repeat": item["repeat"],
                        "run_source": run_id,
                        "error_stage": classified["stage"],
                        "error_status": classified["status_code"],
                        "error_request_id": classified["request_id"],
                        "error_causes": classified["causes"],
                        "error": f"judge: {classified['type']}: {classified['message']}",
                    }
                )

    asyncio.run(_run_all())
    return observations


def _load_prior_evidence(
    path: Path, stimuli_sha: str
) -> tuple[dict, set[tuple[str, str, int | None]], str]:
    """读取首轮证据，校验同源后返回（文件解析对象, 已测量调用键集, 文件字节哈希）。

    补测的前提：同一刺激集（哈希一致）、同一证据契约、且首轮文件不被修改。
    已测量的判定复用，未测量的（调用异常）重测。
    """
    raw = path.read_bytes()
    prior = json.loads(raw.decode("utf-8"))
    if prior.get("purpose") != "planner_judge_v3_calibration_evidence":
        raise ValueError("wrong_prior_evidence_purpose")
    if prior.get("stimuli_sha256") != stimuli_sha:
        raise ValueError("prior_evidence_stimuli_mismatch")
    done = {
        (row["case_id"], row["variant"], row["repeat"])
        for row in prior.get("observations", [])
        if row.get("error") is None
    }
    binding = hashlib.sha256(raw).hexdigest()
    return prior, done, binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimuli", type=Path, default=DEFAULT_STIMULI)
    parser.add_argument(
        "--repeats", type=int, default=0, help="complete-plan repeats; 0 uses the fixture value"
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true", help="make real Judge calls (paid)")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "patch run: reuse the prior evidence's measured verdicts and only re-execute "
            "its unmeasured calls; output is a separate merged file, never a rewrite"
        ),
    )
    parser.add_argument("--consent", default=None, help=f"must equal {REQUIRED_CONSENT}")
    args = parser.parse_args(argv)

    stimuli = load_stimuli(args.stimuli)
    repeats = args.repeats or stimuli.get("repeats_on_complete", 3)
    if not 1 <= repeats <= 5:
        raise ValueError("repeats must be 1–5")
    if not 1 <= args.timeout <= 600:
        raise ValueError("timeout must be 1–600")
    plan = call_plan(stimuli, repeats)

    if not args.execute:
        # 计划模式必须完整披露调用构成与每类刺激的人工预期，零网络零费用。
        repeat_calls = sum(1 for p in plan if p["variant"] == "complete" and p["repeat"] > 1)
        print(
            f"校准调用计划（未执行，零费用）: {len(plan)} 次 Judge 调用 = "
            f"{len(plan) - repeat_calls} 个基础刺激 + {repeat_calls} 次完整计划重复评分"
            f"（每题完整计划重复 {repeats} 次，用于波动测量）"
        )
        for case in stimuli["cases"]:
            counts = {
                variant: sum(
                    1 for p in plan if p["case_id"] == case["case_id"] and p["variant"] == variant
                )
                for variant in ("complete", "missing", "vague")
            }
            expected = {
                name: case["variants"][name]["expected"]
                for name in ("complete", "missing", "vague")
            }
            print(f"  {case['case_id']}（{len(case['required_aspects'])} 个必答维度）:")
            print(
                f"    完整×{counts['complete']} 预期: coverage={expected['complete']['coverage']}, "
                f"specificity均值≥{expected['complete']['specificity_mean_min']}, "
                f"plan_quality均值≥{expected['complete']['plan_quality_mean_min']}"
            )
            print(
                f"    缺漏×{counts['missing']} 预期: 维度 "
                f"{expected['missing']['uncovered_aspects']} 未被覆盖"
                f"（coverage≤{expected['missing']['coverage_max']}）"
            )
            print(
                f"    含混×{counts['vague']} 预期: specificity均值≤"
                f"{expected['vague']['specificity_mean_max']}（coverage 不设预期）"
            )
        print(
            "聚合口径: 解析失败/调用异常→未测量（不按 0 分计）；重复评分波动→逐轴 MAE；"
            "好坏规划排序→方向胜率（coverage/specificity/quality 三组）"
        )
        print("真实执行需 --execute --consent I_CONFIRM_JUDGE_API_COST；结果只作证据，门禁由人决定")
        return 0

    if args.consent != REQUIRED_CONSENT:
        print(f"refusing: a real Judge call requires --consent {REQUIRED_CONSENT}", file=sys.stderr)
        return 2
    if args.out is None:
        print("refusing: --out is required with --execute", file=sys.stderr)
        return 2
    if args.out.exists():
        print(
            "refusing: --out already exists (calibration evidence is write-once)", file=sys.stderr
        )
        return 2
    stimuli_sha = hashlib.sha256(
        json.dumps(stimuli, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    prior = None
    prior_binding = None
    if args.resume_from is not None:
        prior, measured_keys, prior_binding = _load_prior_evidence(args.resume_from, stimuli_sha)
        # 只补测首轮未测量的逻辑调用；已测量的判定原样复用，绝不重复计费。
        plan = [
            item
            for item in plan
            if (item["case_id"], item["variant"], item["repeat"]) not in measured_keys
        ]
        if not plan:
            print("refusing: prior evidence has no unmeasured calls to patch", file=sys.stderr)
            return 2

    from app.services.planner_review_judge import configured_judge_route
    from research_engine.local import load_env_file, run_config_from_env
    from research_engine.runconfig import reset_run_config, set_run_config

    load_env_file(Path(__file__).resolve().parents[2].parent / ".env")
    config = run_config_from_env(fake=False)  # 缺密钥时在此退出并给出供应商指引，不发起半途调用
    # configured_judge_route 从进程配置读取 real 模式与研究路由做独立性预检；
    # 只构建不安装会让全部调用 fail-closed（fake 模式拒绝），先于任何付费请求。
    config_token = set_run_config(config)
    try:
        judge_route = configured_judge_route()
        observations = _execute(stimuli, plan, args.timeout)
    finally:
        reset_run_config(config_token)

    unmeasured_count = sum(1 for row in observations if row["error"] is not None)
    stimuli_path_str = str(args.stimuli)
    report = {
        "schema_version": 1,
        "purpose": "planner_judge_v3_calibration_evidence",
        "created_at": datetime.now(UTC).isoformat(),
        "stimuli_path": stimuli_path_str,
        "stimuli_sha256": stimuli_sha,
        "judge_route_env": "EVAL_JUDGE_MODEL",
        "judge_route_resolved": judge_route,
        "timeout_seconds": args.timeout,
        "repeats": repeats,
        "calls_planned": len(plan),
        "calls_executed": len(observations),
        "calls_unmeasured": unmeasured_count,
        "observations": observations,
    }
    if prior is not None:
        # 跨运行补测：首轮判定原样保留并标注来源，绝不伪装成单轮 15/15。
        prior_rows = [
            {**row, "run_source": f"prior:{prior.get('created_at')}"}
            for row in prior.get("observations", [])
        ]
        prior_measured = sum(1 for row in prior_rows if row.get("error") is None)
        # 合并摘要描述"最终证据状态"：同一逻辑调用取最新一轮判定；
        # 首轮失败行保留在 prior_observations 供审计，不参与聚合。
        final_by_key: dict[tuple[str, str, int | None], dict[str, Any]] = {
            (row["case_id"], row["variant"], row["repeat"]): row for row in prior_rows
        }
        for row in observations:
            final_by_key[(row["case_id"], row["variant"], row["repeat"])] = row
        final_rows = list(final_by_key.values())
        report.update(
            {
                "patched_from": {
                    "path": str(args.resume_from),
                    "sha256": prior_binding,
                    "created_at": prior.get("created_at"),
                },
                "prior_observations": prior_rows,
                "merged_final_state_rows": len(final_rows),
                "merged_totals": {
                    "prior_measured": prior_measured,
                    "new_executed": len(observations),
                    "new_measured": len(observations) - unmeasured_count,
                    "combined_measured": prior_measured + len(observations) - unmeasured_count,
                },
                "warning": (
                    "Evidence only. gate_reference_min_gain is a noise lower bound from judge "
                    "repeatability on development stimuli, not an effect-size measurement; the "
                    "formal min_gain decision belongs to a person, using the ordering win rates "
                    "and dev-set deltas, and must not be derived from holdout results. "
                    "This file PATCHES a prior run: combined totals span two runs and must "
                    "never be presented as one continuous 15-call session."
                ),
            }
        )
        report["summary"] = summarize(stimuli, final_rows)
    else:
        report["warning"] = (
            "Evidence only. gate_reference_min_gain is a noise lower bound from judge "
            "repeatability on development stimuli, not an effect-size measurement; the "
            "formal min_gain decision belongs to a person, using the ordering win rates "
            "and dev-set deltas, and must not be derived from holdout results."
        )
        report["summary"] = summarize(stimuli, observations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    measured = len(observations) - unmeasured_count
    if prior is not None:
        print(
            f"补测证据已落盘: {args.out} "
            f"(本轮 {measured}/{len(observations)} 可测量; 合并 {prior_measured}+{measured} "
            f"来自两次运行, 来源已逐条标注)"
        )
    else:
        print(
            f"校准证据已落盘: {args.out} "
            f"({measured}/{len(observations)} 次调用可测量, 未测量 {unmeasured_count})"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"planner judge calibration: {exc}", file=sys.stderr)
        sys.exit(2)
