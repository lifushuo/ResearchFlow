"""Judge 建议与 GLM 参考标注草案的逐维度对比回归（离线）。"""

from __future__ import annotations

import pytest

from evals.scripts.compare_judge_to_reference import compare


def _reference() -> dict:
    return {
        "purpose": "glm_model_assisted_reference",
        "label_source": "glm_model_assisted_draft",
        "human_confirmed": False,
        "case_id": "case-1",
        "aspects": [
            {"aspect_index": 1, "aspect": "机制", "reference_covered": True},
            {"aspect_index": 2, "aspect": "开销", "reference_covered": False},
            {"aspect_index": 3, "aspect": "失效", "reference_covered": None},
        ],
    }


def _suggestion(covered: dict[int, bool | None]) -> dict:
    return {
        "aspects": [
            {"aspect_index": index, "covered": value, "task_ids": [], "reason": ""}
            for index, value in covered.items()
        ]
    }


def test_counts_both_error_directions_and_keeps_unmeasured():
    result = compare(_reference(), _suggestion({1: True, 2: True, 3: None}))
    assert result["measured_aspects"] == 2
    assert result["agreement"] == 0.5
    assert result["false_covered"] == 1
    assert result["false_missing"] == 0
    assert [row["status"] for row in result["rows"]] == ["agree", "false_covered", "unmeasured"]
    assert result["human_confirmed"] is False


def test_false_missing_is_counted_separately():
    result = compare(_reference(), _suggestion({1: False, 2: False, 3: False}))
    assert result["false_missing"] == 1
    assert result["false_covered"] == 0
    assert result["agreement"] == 0.5


def test_structural_violations_fail_closed():
    with pytest.raises(ValueError, match="wrong_reference_purpose"):
        compare({"purpose": "other"}, {"aspects": []})
    with pytest.raises(ValueError, match="exactly_once"):
        compare(_reference(), _suggestion({1: True, 2: True}))
    with pytest.raises(ValueError, match="unknown_aspect_index"):
        compare(_reference(), _suggestion({1: True, 2: True, 9: True}))
    with pytest.raises(ValueError, match="no_reference_and_judge_pairs_measured"):
        compare(_reference(), _suggestion({1: None, 2: None, 3: None}))
