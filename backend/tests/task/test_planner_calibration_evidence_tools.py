"""校准证据工具：探针/回填/Judge 建议三件套的离线回归（零模型调用）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.scripts.fill_planner_judge_calibration_evidence import (
    fill_case,
    normalized_tasks,
    verified_tasks_from_probe,
)
from evals.scripts.run_planner_judge_suggestion import main as suggestion_main
from evals.scripts.run_planner_only_probe import (
    build_probe_payload,
    fingerprint,
)
from evals.scripts.run_planner_only_probe import (
    main as probe_main,
)

QUERY = "What are the leading approaches to long-term memory in LLM agents, and their trade-offs?"
CASE_ID = "planner-review-llm-memory-ca4295f7"
RUN_ID = "pipe-ca4295f7ee61f0b9d2896e52--report-benchmark"
TASKS = [
    {"id": "1", "query": "survey long-term memory mechanisms for LLM agents"},
    {"id": "2", "query": "MemGPT virtual context management architectures"},
]


def _canonical(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _evidence(query_sha256: str | None = None) -> dict:
    evidence = {
        "schema_version": 1,
        "case_id": "llm-memory",
        "query_sha256": query_sha256 or hashlib.sha256(QUERY.encode("utf-8")).hexdigest(),
        "tasks": TASKS,
        "prompt_sha256": "b" * 64,
        "trace_id": "trace-01",
    }
    evidence["evidence_sha256"] = hashlib.sha256(_canonical(evidence).encode("utf-8")).hexdigest()
    return evidence


def _sheet() -> dict:
    return {
        "schema_version": 1,
        "purpose": "manual_planner_judge_calibration_only",
        "cases": [
            {
                "case_id": CASE_ID,
                "original_query": QUERY,
                "planner_trace_reference": None,
                "planner_task_queries": [],
                "aspects": [
                    {
                        "aspect_index": index,
                        "required_aspect": aspect,
                        "human_covered": None,
                        "judge_covered": None,
                        "agreement": None,
                    }
                    for index, aspect in enumerate(
                        ["memory mechanism", "measured effectiveness"], 1
                    )
                ],
            }
        ],
    }


# ---------- 回填（archive） ----------
def test_archive_fill_restores_real_tasks_and_keeps_aspects(tmp_path: Path) -> None:
    sheet_path = tmp_path / "sheet.json"
    sheet_path.write_text(json.dumps(_sheet(), ensure_ascii=False), encoding="utf-8")
    raw = tmp_path / "raw_result.json"
    raw.write_text(
        json.dumps({"results": [{"id": "llm-memory", "planner_evidence": _evidence()}]}),
        encoding="utf-8",
    )

    assert (fill_planner_main(tmp_path, sheet_path, raw)) == 0
    filled = json.loads(sheet_path.read_text(encoding="utf-8"))
    case = filled["cases"][0]
    assert case["planner_task_queries"] == [t["query"] for t in TASKS]
    assert case["planner_trace_reference"].startswith(f"auto:{RUN_ID}:")
    # 17 个必答维度原样保留；标签仍为空，等待人工与 Judge。
    assert [a["required_aspect"] for a in case["aspects"]] == [
        "memory mechanism",
        "measured effectiveness",
    ]
    assert all(a["human_covered"] is None and a["judge_covered"] is None for a in case["aspects"])


def fill_planner_main(
    tmp_path: Path, sheet_path: Path, raw: Path, extra: list[str] | None = None
) -> int:
    from evals.scripts.fill_planner_judge_calibration_evidence import main as fill_main

    return fill_main(
        [
            "--sheet",
            str(sheet_path),
            "--case-id",
            CASE_ID,
            "--source",
            str(raw),
            "--kind",
            "archive",
            "--match-id",
            "llm-memory",
            "--run-id",
            RUN_ID,
        ]
    )


def test_archive_fill_rejects_question_mismatch_and_tampering(tmp_path: Path) -> None:
    sheet_path = tmp_path / "sheet.json"
    sheet_path.write_text(json.dumps(_sheet(), ensure_ascii=False), encoding="utf-8")
    wrong_question = tmp_path / "wrong-question.json"
    wrong_question.write_text(
        json.dumps({"results": [{"id": "llm-memory", "planner_evidence": _evidence("0" * 64)}]}),
        encoding="utf-8",
    )
    tampered = _evidence()
    tampered["tasks"][0]["query"] = "被篡改的任务内容"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(
        json.dumps({"results": [{"id": "llm-memory", "planner_evidence": tampered}]}),
        encoding="utf-8",
    )
    # 直接调用 main() 时 ValueError 透传（CLI 入口统一捕获后以退出码 2 拒绝）。
    with pytest.raises(ValueError, match="question fingerprint"):
        fill_planner_main(tmp_path, sheet_path, wrong_question)
    with pytest.raises(ValueError, match="content fingerprint"):
        fill_planner_main(tmp_path, sheet_path, tampered_path)
    # 两次拒绝都不允许改动校准表。
    assert json.loads(sheet_path.read_text(encoding="utf-8")) == _sheet()


def test_fill_refuses_overwrite_and_labeled_cases(tmp_path: Path) -> None:
    sheet = _sheet()
    sheet["cases"][0]["planner_task_queries"] = ["已有任务"]
    assert "refusing to overwrite" in _fill_error(sheet, CASE_ID)
    labeled = _sheet()
    labeled["cases"][0]["aspects"][0]["human_covered"] = True
    assert "human or judge labels" in _fill_error(labeled, CASE_ID)
    assert "no case" in _fill_error(sheet, "unknown-case")


def _fill_error(sheet: dict, case_id: str) -> str:
    with pytest.raises(ValueError) as excinfo:
        fill_case(sheet, case_id=case_id, tasks=["x" * 10], trace_reference="auto:x:1")
    return str(excinfo.value)


# ---------- 回填（probe） ----------
def test_probe_fill_happy_path_and_fake_refusal(tmp_path: Path) -> None:
    probe = build_probe_payload(
        case_id="tsp-baselines",
        query="比较学习型 TSP 求解器与 LKH。",
        depth="balanced",
        llm_mode="real",
        prompt_sha256="c" * 64,
        planner_tokens=(10, 4),
        tasks=TASKS,
    )
    evidence = verified_tasks_from_probe(probe, "tsp-baselines")
    assert normalized_tasks(evidence) == [t["query"] for t in TASKS]
    fake = {**probe, "llm_mode": "fake"}
    with pytest.raises(ValueError, match="fake mode"):
        verified_tasks_from_probe(fake, "tsp-baselines")


# ---------- 探针（fake 接线测试；real 调用由用户触发） ----------
def test_probe_script_records_scripted_tasks_only_with_explicit_fake_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "probe.json"
    # 测试隔离：真实模式下探针必须看到无密钥环境才走 SystemExit 拒绝路径。
    # 挂载仓库的容器可能带有 .env（CUSTOM_API_KEY 等）或 compose 注入的
    # MODEL_* 路由——屏蔽 .env 加载并清空相关变量，恢复测试的无密钥前提。
    # 安全行为本身（缺密钥拒绝、不落盘、不给半途调用）保持原断言，不放宽。
    import research_engine.local as engine_local

    for name in (
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "CUSTOM_API_KEY",
        "CUSTOM_BASE_URL",
        "MODEL_PLANNER",
        "MODEL_EXECUTOR",
        "MODEL_CRITIC",
        "MODEL_SYNTHESIZER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(engine_local, "load_env_file", lambda *args, **kwargs: False)
    with pytest.raises(SystemExit):
        probe_main(
            [
                "--case-id",
                "tsp-baselines",
                "--query",
                "比较学习型 TSP 求解器与 LKH。",
                "--out",
                str(out),
            ]
        )
    assert not out.exists()
    assert (
        probe_main(
            [
                "--case-id",
                "tsp-baselines",
                "--query",
                "比较学习型 TSP 求解器与 LKH。",
                "--out",
                str(out),
                "--allow-fake",
            ]
        )
        == 0
    )
    probe = json.loads(out.read_text(encoding="utf-8"))
    assert probe["llm_mode"] == "fake"
    assert probe["tasks"] and all(5 <= len(t["query"]) for t in probe["tasks"])
    assert probe["probe_evidence_sha256"] == fingerprint(probe)
    # fake 探针不能通过回填校验——它不可能给真实校准提供证据。
    with pytest.raises(ValueError, match="fake mode"):
        verified_tasks_from_probe(probe, "tsp-baselines")


# ---------- Judge 建议 CLI ----------
def test_judge_suggestion_writes_side_file_without_touching_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sheet_path = tmp_path / "sheet.json"
    case = _sheet()["cases"][0]
    case["planner_task_queries"] = [t["query"] for t in TASKS]
    case["planner_trace_reference"] = f"auto:{RUN_ID}:{'a' * 64}"
    sheet_path.write_text(
        json.dumps({**_sheet(), "cases": [case]}, ensure_ascii=False), encoding="utf-8"
    )

    import app.services.planner_review_judge as service

    captured = {}

    async def fake_suggest(**kwargs):
        captured.update(kwargs)
        return {
            "status": "suggested",
            "aspects": [
                {"aspect": "memory mechanism", "covered": True, "task_ids": [1], "reason": "任务 1"}
            ],
            "judge_route": "custom:calib-judge",
            "served_judge": None,
            "automatic_label": False,
        }

    monkeypatch.setattr(service, "suggest_planner_coverage", fake_suggest)
    out = tmp_path / "suggestion.json"
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
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["suggestion"]["automatic_label"] is False
    assert captured["evidence"]["tasks"] == [
        {"id": "1", "query": TASKS[0]["query"]},
        {"id": "2", "query": TASKS[1]["query"]},
    ]
    # 校准表未被修改：judge_covered 仍全部为 null。
    sheet_after = json.loads(sheet_path.read_text(encoding="utf-8"))
    assert all(
        a["judge_covered"] is None and a["human_covered"] is None
        for a in sheet_after["cases"][0]["aspects"]
    )


def test_judge_suggestion_requires_consent_and_filled_tasks(tmp_path: Path) -> None:
    sheet_path = tmp_path / "sheet.json"
    sheet_path.write_text(json.dumps(_sheet(), ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "suggestion.json"
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
            ]
        )
        == 2
    )
    assert not out.exists()
