"""v2 Judge 建议与对比工具的离线回归（Fake LLM，零模型调用）。"""

from __future__ import annotations

import json

import pytest

from evals.scripts.compare_judge_to_reference import compare_v2
from evals.scripts.run_planner_judge_suggestion import main as suggestion_main

CASE_ID = "planner-review-llm-memory-ca4295f7"
RUN_ID = "planner-probe:case-1:ff00"


def _reference_v2() -> dict:
    return {
        "purpose": "glm_model_assisted_reference_v2",
        "label_source": "glm_model_assisted_draft",
        "human_confirmed": False,
        "case_id": "case-1",
        "task_specificity": [
            {"task_id": 1, "specificity": 2},
            {"task_id": 2, "specificity": 1},
        ],
        "aspect_plan_quality": [
            {"aspect_index": 1, "aspect": "机制", "plan_quality": 2},
            {"aspect_index": 2, "aspect": "开销", "plan_quality": 0},
        ],
    }


def _suggestion_v2(spec: dict[int, int], plan: dict[int, int], covered: dict[int, bool]) -> dict:
    return {
        "metric_version": "planner-coverage-quality/v2",
        "aspects": [
            {
                "aspect_index": index,
                "aspect": "机制" if index == 1 else "开销",
                "covered": covered[index],
                "task_ids": [1] if covered[index] else [],
                "plan_quality": plan[index],
            }
            for index in (1, 2)
        ],
        "task_specificity": [
            {"task_id": task_id, "task_query": f"任务 {task_id}", "specificity": value}
            for task_id, value in spec.items()
        ],
    }


def test_compare_v2_counts_exact_match_mae_and_disputes():
    result = compare_v2(
        _reference_v2(),
        _suggestion_v2({1: 2, 2: 2}, {1: 2, 2: 0}, {1: True, 2: True}),
    )
    assert result["specificity"]["exact_match"] == 1
    assert result["specificity"]["mean_absolute_error"] == 0.5
    assert result["plan_quality"]["exact_match"] == 2
    assert result["plan_quality"]["mean_absolute_error"] == 0.0
    # 参考 plan_quality=0（未覆盖）而 Judge 判 covered → false_covered 计数。
    assert result["plan_quality"]["false_covered"] == 1
    assert result["coverage_reference"] == 1
    assert result["coverage_judge"] == 2
    disputes = [row for row in result["specificity"]["rows"] if row["status"] == "disputed"]
    assert [row["task_id"] for row in disputes] == [2]


def test_compare_v2_fail_closed_on_task_set_mismatch():
    with pytest.raises(ValueError, match="exactly the reference task set"):
        compare_v2(
            _reference_v2(),
            _suggestion_v2({1: 2, 3: 2}, {1: 2, 2: 0}, {1: True, 2: True}),
        )


def test_suggestion_v2_cli_writes_side_file_without_touching_sheet(tmp_path, monkeypatch):
    import app.services.planner_review_judge as service

    case = {
        "case_id": CASE_ID,
        "original_query": "研究问题。",
        "planner_trace_reference": "planner-probe:case-1:ff00",
        "planner_task_queries": ["检索记忆写入路径", "核对延迟测量"],
        "aspects": [
            {"aspect_index": 1, "required_aspect": "机制"},
            {"aspect_index": 2, "required_aspect": "开销"},
        ],
    }
    sheet_path = tmp_path / "sheet.json"
    sheet_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "manual_planner_judge_calibration_only",
                "cases": [case],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captured = {}

    async def fake_suggest_v2(**kwargs):
        captured.update(kwargs)
        return {
            "status": "suggested",
            "metric_version": "planner-coverage-quality/v2",
            "judge_route": "custom:qwen4-max",
            "served_judge": "qwen4-max",
            "judge_tokens": {"input": 500, "output": 400},
            "coverage": 0.5,
            "automatic_label": False,
            "aspects": [
                {
                    "aspect_index": 1,
                    "aspect": "机制",
                    "covered": True,
                    "task_ids": [1],
                    "plan_quality": 2,
                }
            ],
            "task_specificity": [{"task_id": 1, "task_query": "q1", "specificity": 2}],
        }

    monkeypatch.setattr(service, "suggest_planner_coverage_v2", fake_suggest_v2)
    out = tmp_path / "suggestion-v2.json"
    assert (
        suggestion_main(
            [
                "--sheet",
                str(sheet_path),
                "--case-id",
                CASE_ID,
                "--run-id",
                RUN_ID,
                "--out",
                str(out),
                "--consent",
                "I_CONFIRM_JUDGE_API_COST",
                "--metric-version",
                "v2",
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["purpose"] == "manual_planner_judge_calibration_suggestion_v2"
    assert payload["suggestion"]["metric_version"] == "planner-coverage-quality/v2"
    assert captured["evidence"]["tasks"] == [
        {"id": "1", "query": "检索记忆写入路径"},
        {"id": "2", "query": "核对延迟测量"},
    ]
    # 校准表不被修改：judge_covered 仍全部为 null。
    sheet_after = json.loads(sheet_path.read_text(encoding="utf-8"))
    assert all(a.get("judge_covered") is None for a in sheet_after["cases"][0]["aspects"])


def test_suggestion_v2_cli_requires_consent(tmp_path):
    sheet_path = tmp_path / "sheet.json"
    sheet_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "manual_planner_judge_calibration_only",
                "cases": [
                    {
                        "case_id": CASE_ID,
                        "original_query": "研究问题。",
                        "planner_trace_reference": None,
                        "planner_task_queries": ["任务"],
                        "aspects": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "suggestion.json"
    assert (
        suggestion_main(
            ["--sheet", str(sheet_path), "--case-id", CASE_ID, "--run-id", "r", "--out", str(out)]
        )
        == 2
    )
    assert not out.exists()
