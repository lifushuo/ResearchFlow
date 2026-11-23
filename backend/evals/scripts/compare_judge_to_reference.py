"""把一次真实 Judge 建议与参考标注草案逐维度对比。

方向定义以参考标签为基准：参考判「缺失」而 Judge 判「已覆盖」= false_covered
（对 Planner 优化最危险的误差方向）。任一方未测量（null）的维度保持未测量，
不计入一致率——与 score_planner_judge_calibration 的口径一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def compare(reference: dict, suggestion: dict) -> dict:
    if reference.get("purpose") != "glm_model_assisted_reference":
        raise ValueError("wrong_reference_purpose")
    ref_rows = {row["aspect_index"]: row for row in reference.get("aspects", [])}
    suggestion_aspects = suggestion.get("aspects") or []
    if len(suggestion_aspects) != len(ref_rows):
        raise ValueError("judge_must_return_every_aspect_exactly_once")
    rows, measured, agree, false_covered, false_missing = [], 0, 0, 0, 0
    for item in suggestion_aspects:
        ref = ref_rows.get(item.get("aspect_index"))
        if ref is None:
            raise ValueError("judge_returned_an_unknown_aspect_index")
        row = {
            "aspect_index": item["aspect_index"],
            "aspect": ref.get("aspect"),
            "reference_covered": ref.get("reference_covered"),
            "judge_covered": item.get("covered"),
            "judge_task_ids": item.get("task_ids"),
            "judge_reason": item.get("reason"),
            "reference_reason": ref.get("reason"),
        }
        ref_covered, judge_covered = ref.get("reference_covered"), item.get("covered")
        if ref_covered is None or judge_covered is None:
            row["status"] = "unmeasured"
            rows.append(row)
            continue
        judge_covered = bool(judge_covered)
        ref_covered = bool(ref_covered)
        measured += 1
        if ref_covered == judge_covered:
            agree += 1
            row["status"] = "agree"
        else:
            row["status"] = (
                "false_covered" if (not ref_covered and judge_covered) else "false_missing"
            )
            if row["status"] == "false_covered":
                false_covered += 1
            else:
                false_missing += 1
        rows.append(row)
    if measured == 0:
        raise ValueError("no_reference_and_judge_pairs_measured")
    count = measured
    return {
        "schema_version": 1,
        "purpose": "judge_vs_glm_reference_comparison",
        "case_id": reference.get("case_id"),
        "label_source": reference.get("label_source"),
        "human_confirmed": reference.get("human_confirmed") is True,
        "caveat": (
            "参考标签由 GLM 起草；除非 human_confirmed=true 并经用户本人复核，"
            "本对比不构成 Judge 对人工真值的一致率。"
        ),
        "measured_aspects": count,
        "agreement": round(agree / count, 4),
        "false_covered": false_covered,
        "false_missing": false_missing,
        "rows": rows,
    }


def compare_v2(reference: dict, suggestion: dict) -> dict:
    """v2 对比：任务 specificity 与维度 plan_quality 的逐项一致率与平均绝对误差。

    判分方向约定：|Judge − 参考| = 0 为一致；差异 ≥1 分进入争议清单。
    参考的 plan_quality=0（未覆盖）而 Judge 判 covered 时，同时计入 false_covered。
    """
    ref_spec = {
        row["task_id"]: row.get("specificity") for row in reference.get("task_specificity", [])
    }
    judge_spec = {
        row["task_id"]: row.get("specificity") for row in suggestion.get("task_specificity", [])
    }
    if set(ref_spec) != set(judge_spec):
        raise ValueError("judge must score exactly the reference task set")
    ref_plan = {
        row["aspect_index"]: row.get("plan_quality")
        for row in reference.get("aspect_plan_quality", [])
    }
    judge_aspects = {row["aspect_index"]: row for row in suggestion.get("aspects", [])}
    if set(ref_plan) != set(judge_aspects):
        raise ValueError("judge must return every reference aspect exactly once")

    spec_rows, spec_exact, spec_mae = [], 0, 0.0
    for task_id in sorted(ref_spec):
        ref_value, judge_value = ref_spec[task_id], judge_spec[task_id]
        delta = None if (ref_value is None or judge_value is None) else abs(judge_value - ref_value)
        status = "unmeasured" if delta is None else ("agree" if delta == 0 else "disputed")
        if delta is not None:
            spec_exact += 1 if delta == 0 else 0
            spec_mae += delta
        spec_rows.append(
            {
                "task_id": task_id,
                "reference": ref_value,
                "judge": judge_value,
                "delta": delta,
                "status": status,
            }
        )
    spec_measured = sum(1 for row in spec_rows if row["status"] != "unmeasured")

    plan_rows, plan_exact, plan_mae = [], 0, 0.0
    false_covered = 0
    for aspect_index in sorted(ref_plan):
        ref_value = ref_plan[aspect_index]
        judge_value = judge_aspects[aspect_index].get("plan_quality")
        judge_covered = bool(judge_aspects[aspect_index].get("covered"))
        delta = None if (ref_value is None or judge_value is None) else abs(judge_value - ref_value)
        status = "unmeasured" if delta is None else ("agree" if delta == 0 else "disputed")
        if delta is not None:
            plan_exact += 1 if delta == 0 else 0
            plan_mae += delta
        if ref_value == 0 and judge_covered:
            false_covered += 1
        plan_rows.append(
            {
                "aspect_index": aspect_index,
                "aspect": judge_aspects[aspect_index].get("aspect"),
                "reference_plan_quality": ref_value,
                "judge_plan_quality": judge_value,
                "judge_covered": judge_covered,
                "delta": delta,
                "status": status,
            }
        )
    plan_measured = sum(1 for row in plan_rows if row["status"] != "unmeasured")
    if spec_measured == 0 and plan_measured == 0:
        raise ValueError("no_reference_and_judge_pairs_measured")
    covered_ref = sum(1 for value in ref_plan.values() if value and value > 0)
    covered_judge = sum(1 for row in suggestion.get("aspects", []) if row.get("covered"))
    return {
        "purpose": "judge_vs_glm_reference_comparison_v2",
        "case_id": reference.get("case_id"),
        "label_source": reference.get("label_source"),
        "human_confirmed": reference.get("human_confirmed") is True,
        "metric_version": suggestion.get("metric_version"),
        "caveat": (
            "参考标签由 GLM 起草；除非 human_confirmed=true 并经用户本人复核，"
            "本对比不构成 Judge 对人工真值的一致率。"
        ),
        "specificity": {
            "measured_tasks": spec_measured,
            "exact_match": spec_exact,
            "exact_match_rate": round(spec_exact / spec_measured, 4) if spec_measured else None,
            "mean_absolute_error": round(spec_mae / spec_measured, 4) if spec_measured else None,
            "rows": spec_rows,
        },
        "plan_quality": {
            "measured_aspects": plan_measured,
            "exact_match": plan_exact,
            "exact_match_rate": round(plan_exact / plan_measured, 4) if plan_measured else None,
            "mean_absolute_error": round(plan_mae / plan_measured, 4) if plan_measured else None,
            "false_covered": false_covered,
            "rows": plan_rows,
        },
        "coverage_reference": covered_ref,
        "coverage_judge": covered_judge,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--suggestion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=("v1", "v2"), default="v1")
    args = parser.parse_args(argv)

    reference = json.loads(args.reference.read_text(encoding="utf-8-sig"))
    suggestion = json.loads(args.suggestion.read_text(encoding="utf-8-sig")).get("suggestion")
    result = (
        compare_v2(reference, suggestion) if args.kind == "v2" else compare(reference, suggestion)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.kind == "v2":
        spec, plan = result["specificity"], result["plan_quality"]
        print(
            f"comparison saved: {args.out} — specificity exact {spec['exact_match']}/"
            f"{spec['measured_tasks']} (MAE {spec['mean_absolute_error']}), plan_quality exact "
            f"{plan['exact_match']}/{plan['measured_aspects']} (MAE {plan['mean_absolute_error']}, "
            f"false_covered={plan['false_covered']}; human_confirmed={result['human_confirmed']})"
        )
    else:
        print(
            f"comparison saved: {args.out} — agreement {result['agreement']:.0%} on "
            f"{result['measured_aspects']} measured aspects "
            f"(false_covered={result['false_covered']}, false_missing={result['false_missing']}; "
            f"human_confirmed={result['human_confirmed']})"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f"comparison refused: {exc}", file=sys.stderr)
        sys.exit(2)
