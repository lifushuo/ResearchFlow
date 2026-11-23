"""Fake harness 9 题 report A/B 与 report_gate 绑定集成测试（零模型调用）。

run_one 被替换为脚本行，因此没有任何 Planner/Judge/模型调用；
被测的是 harness 的 9 题固定集接线、哈希三方一致与 report_gate 的 fail-closed 行为。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

# 见 test_evals_result_paths.py——导入 evals.harness 会执行 load_env_file()，
# 必须先快照 os.environ 再恢复，避免 .env 泄漏进整个测试会话。
_ENV_BEFORE = dict(os.environ)
from evals import harness  # noqa: E402 — 必须跟随上面的快照

os.environ.clear()
os.environ.update(_ENV_BEFORE)

from app.services.evaluation_closure import input_hash  # noqa: E402
from evals.agent_eval_ext.long_run import canonical_hash, sha256_text  # noqa: E402
from evals.agent_eval_ext.planner_evolution.release import report_gate  # noqa: E402
from evals.report_ab_dataset import (  # noqa: E402
    INDEPENDENT_REPORT_SET,
    ORIGINAL_REPORT_SET,
    fixed_report_queries,
    fixed_set_hash,
    selected_report_dataset_for_experiment,
)
from research_engine.prompts import PLANNER_PROMPT_V2  # noqa: E402


def _fake_row(query: dict, arm: str) -> dict:
    """聚合所需的完整行；completed 严格 bool，judge 字段由本测试的替换臂提供。"""
    return {
        "id": query["id"],
        "domain": query.get("domain"),
        "completed": True,
        "error": None,
        "source_count": 5,
        "uncited_claim_count": 0,
        "resolution_rate": 1.0,
        "citation_support_rate": 1.0,
        "claim_verdicts": [],
        "latency_s": 0.1,
        "cost_usd": 0.0,
        "evaluation_arm": arm,
    }


async def test_fake_harness_nine_case_ab_binds_independent_report_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows9 = fixed_report_queries(INDEPENDENT_REPORT_SET)
    assert len(rows9) == 9
    assert "llm-memory" not in {row["id"] for row in rows9}

    # 固定路由：报告文档的 llm_mode/models 必须与实验绑定完全一致才能推进门禁。
    models = {role: f"custom:calib-{role}" for role in harness._ROLES}
    monkeypatch.setattr(
        harness, "RUN_CONFIG", replace(harness.RUN_CONFIG, llm_mode="real", models=models)
    )
    # judge_section 的计票细节不是本测试对象；这里只需要文档如实记录所用路由。
    monkeypatch.setattr(
        harness, "judge_section", lambda route, rows, candidate_rows: {"route": route}
    )
    calls: list[tuple[str, str | None, dict | None]] = []

    async def fake_run_one(query: dict, **kwargs) -> dict:
        # 基线臂不带 prompt_overrides（--baseline 未传时 evaluation_arm 也不注入）；
        # 候选臂总是携带 {"planner": ...} 覆盖。
        overrides = kwargs.get("prompt_overrides")
        arm = "candidate" if overrides else "baseline"
        calls.append((query["id"], arm, overrides))
        return _fake_row(query, arm)

    monkeypatch.setattr(harness, "run_one", fake_run_one)

    candidate_text = "Calibrated candidate planner prompt for the nine-case A/B."
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps({"planner": candidate_text}), encoding="utf-8")
    out = tmp_path / "report-ab.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--mode",
            "report",
            "--judge",
            "custom:calib-judge",
            "--report-queries",
            str(INDEPENDENT_REPORT_SET),
            "--candidate",
            str(candidate_path),
            "--out",
            str(out),
        ],
    )
    await harness.main()

    doc = json.loads(out.read_text(encoding="utf-8"))
    expected_ids = [row["id"] for row in rows9]
    # 两臂各 9 题，顺序与固定集一致，且不含 llm-memory。
    assert [row["id"] for row in doc["results"]] == expected_ids
    assert [row["id"] for row in doc["candidate"]["results"]] == expected_ids
    assert len(calls) == 18
    assert [arm for _, arm, _ in calls[:9]] == ["baseline"] * 9
    assert all(overrides == {"planner": candidate_text} for _, _, overrides in calls[9:])

    # 9 题集哈希：experiment 绑定（input_hash）== harness 落盘（canonical_hash）
    # == fixed_set_hash，三套实现必须对同一固定集给出同一指纹。
    bound = fixed_set_hash(rows9)
    assert bound == input_hash(rows9) == canonical_hash(rows9)
    assert doc["method"]["case_set_sha256"] == bound
    assert doc["method"]["query_set"] == "evals/queries-planner-gepa-ab-v1.json"
    assert doc["llm_mode"] == "real"
    assert doc["models"] == models

    # 实验以该哈希唯一恢复 9 题集；未知哈希 fail-closed。
    baseline_sha = sha256_text(PLANNER_PROMPT_V2)
    candidate_sha = sha256_text(candidate_text)
    experiment = {
        "report_dataset_sha256": bound,
        "routes": {"pipeline": models, "judge": "custom:calib-judge"},
        "baseline": {"override": None, "sha256": baseline_sha},
        "candidate": {"override": candidate_text, "sha256": candidate_sha},
    }
    restored_path, restored_rows = selected_report_dataset_for_experiment(experiment)
    assert restored_path == INDEPENDENT_REPORT_SET
    assert restored_rows == rows9
    with pytest.raises(ValueError, match="not_recognized"):
        selected_report_dataset_for_experiment({"report_dataset_sha256": "f" * 64})

    # report_gate：实验绑定 9 题时，文档的 provenance（模型/哈希/Prompt 版本/数据集 SHA）
    # 全部通过、两臂行集完整，然后停在可靠性证据缺失——证明绑定检查已全部走通。
    with pytest.raises(ValueError, match="no durable completed Judge run"):
        report_gate(experiment, doc, rows9)

    # 错误切换：同一个实验、同一份文档，换回原始 10 题集 → 数据集哈希不再匹配，直接拒绝。
    with pytest.raises(ValueError, match="not measured on the exact same models"):
        report_gate(experiment, doc, fixed_report_queries(ORIGINAL_REPORT_SET))

    # 基线 Prompt 指纹：baseline 无 override 时指纹必须来自出厂 Prompt。
    assert doc["method"]["baseline_prompt_sha256"] == {"planner": baseline_sha}
    assert doc["candidate"]["prompt_sha256"] == {"planner": candidate_sha}
