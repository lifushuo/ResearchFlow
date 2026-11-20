"""审阅端复核 Planner 留出集和现有报告 Harness 产物，生成发布与回滚配置。"""

from __future__ import annotations

import math
from typing import Any

from app.services.evaluation_closure import input_hash
from evals.agent_eval_ext.long_run import JUDGE_COMPLETED, sha256_text
from research_engine.prompts import PLANNER_PROMPT_V2

from .experiment import METRIC_VERSION, Dataset, Observation, checked_prompt, make_gate


def _independent_served_model(model: object, experiment: dict) -> str:
    """按实际响应身份校验；没有身份时拒绝发布，不替供应商补造元数据。"""
    from evals.harness import validate_judge

    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise ValueError("actual Judge identity is unverified")
    routes = experiment["routes"]
    pipeline = dict(routes["pipeline"])
    if routes.get("reflection"):
        pipeline["reflection"] = routes["reflection"]
    validate_judge(f"{routes['judge'].partition(':')[0]}:{model}", pipeline)
    return model


def release_candidate(experiment: dict) -> tuple[str | None, str]:
    """重新计算门禁，拒绝演示、不同数据集以及不完整的逐题评分。"""
    if (
        experiment.get("schema_version") != 1
        or experiment.get("evidence_kind") != "live_planner"
        or experiment.get("metric_version") != METRIC_VERSION
    ):
        raise ValueError("release requires a real Planner experiment with the current metric")
    dataset = Dataset.model_validate(experiment["dataset"])
    if (
        dataset.provenance != "reviewed"
        or input_hash(dataset.model_dump()) != experiment["dataset_sha256"]
    ):
        raise ValueError("release requires the unchanged reviewed dataset")
    baseline = experiment["baseline"]["override"]
    candidate = (experiment.get("candidate") or {}).get("override")
    if not candidate:
        raise ValueError("no candidate passed validation selection")
    checked_prompt(candidate)
    if baseline is not None:
        checked_prompt(baseline)
    baseline_hash = sha256_text(PLANNER_PROMPT_V2 if baseline is None else baseline)
    if (
        baseline_hash != experiment["baseline"]["sha256"]
        or sha256_text(candidate) != experiment["candidate"]["sha256"]
        or sha256_text(candidate) == baseline_hash
    ):
        raise ValueError("Prompt versions do not match the experiment")
    minimum_gain = float(experiment["limits"]["minimum_gain"])
    if not 0 < minimum_gain <= 1:
        raise ValueError("invalid positive gain threshold")
    holdout = experiment["holdout"]
    before, after = (
        [Observation.model_validate(row) for row in holdout[arm]]
        for arm in ("baseline", "candidate")
    )
    gate = make_gate(
        [case for case in dataset.cases if case.split == "holdout"],
        before,
        after,
        minimum_gain,
        require_live_evidence=True,
    )
    if not gate["passed"]:
        raise ValueError(f"Planner holdout gate failed: {', '.join(gate['reasons'])}")
    for row in before + after:
        _independent_served_model(row.served_judge, experiment)
    return baseline, candidate


def report_gate(experiment: dict, report: dict, queries: list[dict]) -> dict[str, Any]:
    """完整固定报告集 A/B，校验被测 Prompt、独立 Judge、逐题和全量 Claim。"""
    from evals.harness import aggregate, check_release_criteria
    from evals.metrics import CITE_RE, METRICS_VERSION, claim_lines

    method = report.get("method") or {}
    challenger = report.get("candidate") or {}
    judge = report.get("judge") or {}
    expected = [case["id"] for case in queries]
    sample_hash = input_hash(queries)
    provenance_ok = (
        report.get("llm_mode") == "real"
        and report.get("mode") == "report"
        and report.get("models") == experiment["routes"]["pipeline"]
        and method.get("metrics_version") == METRICS_VERSION
        and method.get("case_set_sha256") == sample_hash == experiment["report_dataset_sha256"]
        and method.get("baseline_prompt_sha256") == {"planner": experiment["baseline"]["sha256"]}
        and challenger.get("prompt_sha256") == {"planner": experiment["candidate"]["sha256"]}
        and challenger.get("prompt_overrides") == {"planner": experiment["candidate"]["override"]}
        and judge.get("route") == experiment["routes"]["judge"]
    )
    if not provenance_ok:
        raise ValueError(
            "report was not measured on the exact same models, prompt versions and full dataset"
        )

    rows_by_arm = [report.get("results") or [], challenger.get("results") or []]
    if any([row.get("id") for row in arm] != expected for arm in rows_by_arm):
        raise ValueError("report evaluation does not contain both complete case sets")
    reliability = report.get("reliability") or {}
    if reliability.get("status") != JUDGE_COMPLETED:
        raise ValueError("report evaluation has no durable completed Judge run")
    served = []
    for arm in rows_by_arm:
        actual_models = set()
        for row in arm:
            # docs/08：门禁与聚合使用同一完成状态；1 / "true" 都不是真正的 bool。
            if type(row.get("completed")) is not bool:
                raise ValueError("completed must be a strict boolean")
            if not row["completed"]:
                continue
            text, verdicts = row.get("report"), row.get("claim_verdicts")
            if not isinstance(text, str) or not isinstance(verdicts, list):
                raise ValueError("a completed report requires text and Claim judgments")
            claims = [line for line in claim_lines(text) if CITE_RE.search(line)]
            if (
                row.get("judge_status") != JUDGE_COMPLETED
                or not claims
                or any(not isinstance(v, dict) for v in verdicts)
                or [verdict.get("claim") for verdict in verdicts] != claims
                or any(
                    v.get("judged") is not True or type(v.get("supported")) is not bool
                    for v in verdicts
                )
            ):
                raise ValueError("a completed report has missing or uncommitted Claim judgments")
            support = sum(verdict["supported"] for verdict in verdicts) / len(verdicts)
            recorded_support = row.get("citation_support_rate")
            if (
                type(recorded_support) not in (int, float)
                or not math.isfinite(recorded_support)
                or recorded_support != round(support, 4)
            ):
                raise ValueError("citation support disagrees with the underlying Claim judgments")
            actual_models.update(
                _independent_served_model(v.get("judged_by"), experiment) for v in verdicts
            )
        served.append(actual_models)
    for arm, actual in zip(("baseline", "candidate"), served, strict=True):
        summary = (judge.get(arm) or {}).get("served_models")
        if (
            not isinstance(summary, list)
            or any(not isinstance(model, str) for model in summary)
            or len(summary) != len(set(summary))
            or set(summary) != actual
        ):
            raise ValueError("reported Judge identity disagrees with individual Claim judgments")
    if served[0] != served[1] or len(served[0]) != 1:
        raise ValueError(
            "the reported Judge identity differs between arms or changed within an arm"
        )

    before, after = [aggregate(arm) for arm in rows_by_arm]
    if (
        before["full_judge_coverage"] != 1.0
        or check_release_criteria(after)["gate_passed"] is not True
    ):
        raise ValueError("report gate failed on coverage, completion or citation support")
    if before["citation_support_rate"] is None or any(
        after[metric] < before[metric] for metric in ("completion_rate", "citation_support_rate")
    ):
        raise ValueError("report completion or citation support regressed")
    new_failures = [
        old["id"]
        for old, new in zip(*rows_by_arm, strict=True)
        if old.get("completed") is True and new.get("completed") is not True
    ]
    if new_failures:
        raise ValueError(f"report completion regressed on cases: {', '.join(new_failures)}")
    return {
        "passed": True,
        "baseline": before,
        "candidate": after,
        "cases": len(expected),
        "judge_identity_status": "verified",
        "served_judge": next(iter(served[0])),
    }
