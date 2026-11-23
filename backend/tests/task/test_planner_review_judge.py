"""Planner 逐题复核的独立 Judge：所有模型均为 Fake，不产生 API 费用。"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from app.services.planner_review_judge import (
    ReviewCoverage,
    build_review_advice,
    suggest_planner_coverage,
)


def _evidence():
    return {
        "run_id": "pipe-aaaaaaaaaaaaaaaaaaaaaaaa--report-benchmark",
        "case_id": "llm-memory",
        "evidence_reference": "auto:pipe-aaaaaaaaaaaaaaaaaaaaaaaa--report-benchmark:" + "a" * 64,
        "tasks": [
            {"id": "1", "query": "检索记忆写入与检索实现"},
            {"id": "2", "query": "调查长程记忆 Benchmark"},
        ],
    }


def _verdict():
    return ReviewCoverage.model_validate(
        {
            "matches": [
                {"aspect_index": 1, "task_ids": [1], "reason": "任务一直接调查写入和检索"},
                {"aspect_index": 2, "task_ids": [], "reason": "任务列表没有延迟或成本测量"},
            ],
        }
    )


def test_advice_is_not_an_automatic_training_label():
    result = build_review_advice(
        evidence=_evidence(),
        question="长程记忆优缺点？",
        aspects=["memory mechanism", "latency or cost"],
        verdict=_verdict(),
        judge_route="custom:qwen4-max",
        served_model="qwen4-max",
        tokens=(140, 80),
    )
    assert result["automatic_label"] is False
    assert result["coverage"] == 0.5
    assert result["aspects"][0]["task_ids"] == [1]
    assert result["aspects"][1]["covered"] is False
    assert result["evidence_reference"] == _evidence()["evidence_reference"]


@pytest.mark.parametrize(
    "matches",
    [
        [{"aspect_index": 1, "task_ids": [], "reason": "缺少第二个维度"}],
        [
            {"aspect_index": 1, "task_ids": [3], "reason": "不存在的任务"},
            {"aspect_index": 2, "task_ids": [], "reason": "空列表"},
        ],
        [
            {"aspect_index": 1, "task_ids": [], "reason": "重复维度"},
            {"aspect_index": 1, "task_ids": [1], "reason": "重复维度"},
        ],
    ],
)
def test_invalid_structured_judge_feedback_is_rejected(matches):
    with pytest.raises(ValueError):
        build_review_advice(
            evidence=_evidence(),
            question="题目",
            aspects=["机制", "开销"],
            verdict=ReviewCoverage.model_validate({"matches": matches}),
            judge_route="custom:qwen4-max",
            served_model=None,
            tokens=None,
        )


@pytest.mark.asyncio
async def test_one_judge_call_uses_saved_tasks_and_does_not_rerun_planner(monkeypatch):
    import app.services.planner_review_judge as service

    calls = []

    class FakeStructuredJudge:
        def with_structured_output(self, schema, **kwargs):
            assert schema is ReviewCoverage
            return self

        async def ainvoke(self, messages):
            calls.append(messages)
            assert "检索记忆写入与检索实现" in messages[1].content
            assert "latency or cost" in messages[1].content
            return {"parsed": _verdict(), "raw": None, "parsing_error": None}

    @contextmanager
    def fake_config(route):
        assert route == "custom:qwen4-max"
        yield

    monkeypatch.setattr(service, "configured_judge_route", lambda: "custom:qwen4-max")
    monkeypatch.setattr(service, "judge_client_config", fake_config)
    monkeypatch.setattr(service.llm_factory, "get_llm", lambda role: FakeStructuredJudge())
    monkeypatch.setattr(service.llm_factory, "served_model_id", lambda raw: "qwen4-max")
    monkeypatch.setattr(service.llm_factory, "token_counts", lambda raw: (12, 8))

    result = await suggest_planner_coverage(
        evidence=_evidence(),
        question="研究长程记忆",
        aspects=["memory mechanism", "latency or cost"],
    )
    assert len(calls) == 1
    assert result["aspects"][1]["covered"] is False
    assert result["automatic_label"] is False
