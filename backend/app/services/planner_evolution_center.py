"""评测中心与 Planner 自进化的回流桥梁。

回流记录存放在评测卷 ``job_root()/planner-evolution/`` 下，与闭环数据库同源同权限：

- ``cases/``       复核归因为 Planner 漏项的训练例记录，参与优化数据集组装
- ``reviews/``     归因为检索或报告环节的复核记录，仅作证据保留，不进数据集
- ``experiments/`` 约定的 CLI 实验输出目录（``planner_evolution optimize --out-dir``）

种子数据集与基准标注来自仓库内 ``evals/datasets/planner/``，只读；回流记录只追加，
不修改既有记录。评测轨迹默认只含元数据，Planner 实际任务列表不被补造：
回流记录显式标记任务证据状态，复核结论由复核人对照评测中心轨迹填写。
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.evaluation_closure import EvaluationClosure, input_hash
from evals.agent_eval_ext.closure_observer import BACKEND, read, sources
from evals.agent_eval_ext.long_run import atomic_create
from evals.agent_eval_ext.planner_evolution.experiment import Case, Dataset, normalized

PLANNER_DIR = "planner-evolution"
RECORD_KIND = "planner_backflow_v1"
STAGES = ("planner", "retrieval", "report")
MAX_RECORD_BYTES = 256 * 1024
MAX_EXPERIMENT_DIRS = 50
SEED_DATASET = BACKEND / "evals/datasets/planner/planner-reviewed-v1.json"
BENCHMARK_ANNOTATIONS = BACKEND / "evals/datasets/planner/benchmark-aspects-v1.json"
_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$")


class BackflowError(ValueError):
    """回流输入与既有证据冲突，消息面向复核人，不泄露其他用户资源。"""


class BackflowSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["bad_case", "manual"]
    bad_case_id: str | None = None
    benchmark_case_id: str | None = None
    run_id: str | None = None
    error_key: str | None = None


class BackflowReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["planner", "retrieval", "report"]
    review_note: str = Field(min_length=2, max_length=2000)
    reviewed_at: str
    source: BackflowSource
    benchmark_aspects: list[str] | None = None
    # 评测轨迹默认只含元数据；未采集的任务列表不补造，只能显式披露缺失。
    task_evidence_status: Literal["not_collected", "unavailable"] = "not_collected"


class BackflowRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    kind: Literal["planner_backflow_v1"]
    case: Case
    review: BackflowReview


def evolution_root(job_root: Path) -> Path:
    return job_root / PLANNER_DIR


def _records_dir(root: Path, subdir: str) -> Path:
    directory = evolution_root(root) / subdir
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise BackflowError("回流目录状态异常，请检查评测卷")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _read_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
        raise BackflowError(f"回流记录不可读: {path.name}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise BackflowError(f"回流记录损坏: {path.name}") from exc
    if not isinstance(doc, dict) or doc.get("kind") != RECORD_KIND:
        raise BackflowError(f"回流记录类型不符: {path.name}")
    return doc


def load_benchmark_annotations() -> dict[str, Any]:
    """读取并校验 report-benchmark 的必答维度标注；标注键必须落在基准题集内。"""
    doc = read(BENCHMARK_ANNOTATIONS)
    if not isinstance(doc, dict) or doc.get("schema_version") != 1:
        raise ValueError("基准维度标注结构无效")
    if doc.get("dataset_id") != "report-benchmark":
        raise ValueError("基准维度标注只服务 report-benchmark")
    benchmark_cases, _ = sources("report-benchmark", 0)
    aspects = doc.get("aspects")
    if not isinstance(aspects, dict) or not aspects:
        raise ValueError("基准维度标注缺少 aspects")
    for case_id, row in aspects.items():
        if case_id not in benchmark_cases:
            raise ValueError(f"维度标注引用了不存在的基准题: {case_id}")
        if not isinstance(row, dict) or not isinstance(row.get("required_aspects"), list):
            raise ValueError(f"维度标注缺少必答维度: {case_id}")
        labels = row["required_aspects"]
        if not 2 <= len(labels) <= 12 or len({normalized(x) for x in labels}) != len(labels):
            raise ValueError(f"必答维度数量或去重无效: {case_id}")
    return doc


def load_seed_dataset() -> Dataset:
    return Dataset.model_validate(read(SEED_DATASET))


def read_records(directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """读取一个回流记录目录；损坏文件跳过并披露名称，不阻塞其余证据。"""
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise BackflowError("回流目录状态异常，请检查评测卷")
    records: list[dict[str, Any]] = []
    broken: list[str] = []
    if not directory.is_dir():
        return records, broken
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(_read_record(path))
        except BackflowError:
            broken.append(path.name)
    return records, broken


def _existing_records(root: Path) -> list[dict[str, Any]]:
    return read_records(evolution_root(root) / "cases")[0]


def _check_group_split(seed: Dataset, group: str, split: str) -> None:
    for case in seed.cases:
        if case.group.casefold() == group.casefold() and case.split != split:
            raise BackflowError(
                f"分组 {group} 在种子数据集中属于 {case.split}，回流题目不能跨 split"
            )


def _check_dupes(
    seed: Dataset,
    records: list[dict[str, Any]],
    *,
    case_id: str,
    query: str,
    bad_case_id: str | None,
) -> None:
    """去重顺序：案例 ID → 坏例来源 → 问题文本；reviews/ 记录同样参与坏例去重。"""
    question = normalized(query)
    for case in seed.cases:
        if case.case_id == case_id:
            raise BackflowError(f"案例 ID 已存在于种子数据集: {case_id}")
        if normalized(case.query) == question:
            raise BackflowError("相同问题已存在于种子数据集")
    for record in records:
        case = Case.model_validate(record["case"])
        if case.case_id == case_id:
            raise BackflowError(f"案例 ID 已有回流记录: {case_id}")
        if bad_case_id and case.source_bad_case_id == bad_case_id:
            raise BackflowError(f"该坏例已回流: {bad_case_id}")
        if normalized(case.query) == question:
            raise BackflowError("相同问题已有回流记录")


def backflow(
    store: EvaluationClosure,
    owner: str,
    root: Path,
    *,
    case_id: str,
    group: str,
    aspects: list[str],
    review_note: str,
    stage: str,
    bad_case_id: str | None = None,
    manual_query: str | None = None,
    manual_depth: str = "balanced",
) -> dict[str, Any]:
    """把一条复核结论落成结构化回流记录；归因决定进训练集还是仅存证。

    坏例回流沿用 ``seed-case`` 的同一套校验：坏例必须属于 report-benchmark、
    已有最近一次测量，且原始输入指纹与基准题源一致。归因不是 planner 的记录
    写入 ``reviews/``，只作证据保留。
    """
    if stage not in STAGES:
        raise BackflowError(f"归因阶段无效: {stage}")
    if not _ID.fullmatch(case_id or ""):
        raise BackflowError("案例 ID 必须是 1–80 位字母、数字、点、下划线或连字符")
    if not _ID.fullmatch(group or ""):
        raise BackflowError("分组必须是 1–80 位字母、数字、点、下划线或连字符")
    if not isinstance(aspects, list) or not 2 <= len(aspects) <= 12:
        raise BackflowError("必答维度需要 2–12 条")
    if any(not isinstance(x, str) or not 2 <= len(x.strip()) <= 180 for x in aspects):
        raise BackflowError("每条必答维度需要 2–180 个字符")
    if len({normalized(x) for x in aspects}) != len(aspects):
        raise BackflowError("必答维度重复")
    if not isinstance(review_note, str) or not 2 <= len(review_note.strip()) <= 2000:
        raise BackflowError("复核结论必填（2–2000 字符）")
    if (bad_case_id is None) == (manual_query is None):
        raise BackflowError("必须且只能选择坏例回流或手动回流之一")

    seed = load_seed_dataset()
    annotations = load_benchmark_annotations()
    benchmark_aspects = None
    query, depth, source = None, manual_depth, None
    if bad_case_id is not None:
        detail = store.bad_case_detail(owner, bad_case_id)
        bad_case, last_result = detail["bad_case"], detail["last_result"]
        if bad_case["dataset_id"] != "report-benchmark" or not last_result:
            raise BackflowError("坏例必须来自已测量的 report-benchmark 运行")
        benchmark_cases, _ = sources("report-benchmark", 0)
        source_case = benchmark_cases.get(bad_case["case_id"])
        if not source_case or input_hash(source_case) != last_result.get("input_sha256"):
            raise BackflowError("坏例最近一次输入与基准题源不一致，请先重新运行评测")
        query = source_case["case"]["query"]
        depth = source_case["case"].get("depth", "balanced")
        benchmark_aspects = (
            annotations["aspects"].get(bad_case["case_id"], {}).get("required_aspects")
        )
        source = BackflowSource(
            type="bad_case",
            bad_case_id=bad_case_id,
            benchmark_case_id=bad_case["case_id"],
            run_id=bad_case["last_run_id"],
            error_key=bad_case["error_key"],
        )
    else:
        assert manual_query is not None
        if not 10 <= len(manual_query.strip()) <= 2000:
            raise BackflowError("手动回流的研究问题需要 10–2000 个字符")
        if manual_depth not in ("fast", "balanced", "comprehensive"):
            raise BackflowError("研究深度无效")
        query = manual_query.strip()
        source = BackflowSource(type="manual")

    _check_group_split(seed, group, "train")
    records = _existing_records(root) + read_records(evolution_root(root) / "reviews")[0]
    _check_dupes(seed, records, case_id=case_id, query=query, bad_case_id=bad_case_id)

    record = BackflowRecord(
        schema_version=1,
        kind=RECORD_KIND,
        case=Case(
            case_id=case_id,
            group=group,
            split="train",
            query=query,
            depth=depth,
            required_aspects=aspects,
            source_bad_case_id=bad_case_id,
        ),
        review=BackflowReview(
            stage=stage,
            review_note=review_note.strip(),
            reviewed_at=datetime.now(UTC).isoformat(timespec="seconds"),
            source=source,
            benchmark_aspects=benchmark_aspects,
        ),
    )
    # 归因不是 planner 的复核写入 reviews/，只作证据保留，不进优化数据集。
    directory = _records_dir(root, "cases" if stage == "planner" else "reviews")
    target = directory / f"{case_id}.json"
    if target.exists():
        raise BackflowError(f"记录已存在: {target.name}")
    try:
        atomic_create(target, record.model_dump())
    except OSError as exc:
        raise BackflowError(f"回流记录写入失败: {type(exc).__name__}") from exc
    store.record_audit(
        owner,
        "planner_backflow",
        case_id,
        {
            "stage": stage,
            "bad_case_id": bad_case_id,
            "benchmark_case_id": source.benchmark_case_id,
            "run_id": source.run_id,
            "aspect_count": len(aspects),
        },
    )
    return record.model_dump()


def assemble_dataset(seed: Dataset, records: list[dict[str, Any]]) -> dict[str, Any]:
    """种子数据集加上 planner 归因回流记录；冲突在这里统一披露。"""
    backflow_cases: list[Case] = []
    for record in records:
        if record.get("review", {}).get("stage") != "planner":
            continue
        backflow_cases.append(Case.model_validate(record["case"]))
    error = None
    merged = None
    try:
        merged = Dataset(
            schema_version=1,
            dataset_id=seed.dataset_id,
            provenance="reviewed",
            cases=[*seed.cases, *backflow_cases],
        )
    except ValueError as exc:
        error = str(exc)
    payload: dict[str, Any] = {
        "seed_case_count": len(seed.cases),
        "backflow_case_count": len(backflow_cases),
        "dataset_sha256": input_hash(merged.model_dump()) if merged else None,
        "error": error,
    }
    if merged is not None:
        splits: dict[str, int] = {}
        for case in merged.cases:
            splits[case.split] = splits.get(case.split, 0) + 1
        payload["dataset"] = merged.model_dump()
        payload["splits"] = splits
        payload["group_count"] = len({case.group for case in merged.cases})
    return payload


def merged_dataset(root: Path) -> dict[str, Any]:
    """种子数据集加回流记录的合并视图；回写时已预检，这里兜底披露。"""
    seed = load_seed_dataset()
    records, broken = read_records(evolution_root(root) / "cases")
    reviews, _ = read_records(evolution_root(root) / "reviews")
    payload = assemble_dataset(seed, records)
    payload["review_record_count"] = len(reviews)
    payload["unreadable_record_count"] = len(broken)
    return payload


def _project_experiment(doc: dict[str, Any]) -> dict[str, Any]:
    holdout = doc.get("holdout") or {}
    gate = holdout.get("gate") or {}
    search = doc.get("search") or {}
    routes = doc.get("routes") or {}
    return {
        "created_at": doc.get("created_at"),
        "evidence_kind": doc.get("evidence_kind"),
        "metric_version": doc.get("metric_version"),
        "optimizer": doc.get("optimizer"),
        "limits": doc.get("limits"),
        "dataset_sha256": doc.get("dataset_sha256"),
        "case_count": len((doc.get("dataset") or {}).get("cases", [])),
        "routes": routes,
        "judge_settings": doc.get("judge_settings"),
        "validation_gain": search.get("validation_gain"),
        "holdout_gate": gate,
        "candidate": (doc.get("candidate") or {}).get("sha256"),
        "has_prompt_diff": bool(doc.get("prompt_diff")),
    }


def _project_report_gate(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": doc.get("passed"),
        "cases": doc.get("cases"),
        "served_judge": doc.get("served_judge"),
        "judge_identity_status": doc.get("judge_identity_status"),
        "baseline": doc.get("baseline"),
        "candidate": doc.get("candidate"),
    }


def _project_release(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": doc.get("version"),
        "approved_by": doc.get("approved_by"),
        "approved_at": doc.get("approved_at"),
        "prompt_sha256": doc.get("prompt_sha256"),
        "parent_prompt_sha256": doc.get("parent_prompt_sha256"),
        "experiment_sha256": doc.get("experiment_sha256"),
        "report_sha256": doc.get("report_sha256"),
    }


def _read_projected(path: Path, project) -> dict[str, Any] | None:
    try:
        return project(read(path))
    except OSError, ValueError:
        return None


def list_experiments(root: Path) -> list[dict[str, Any]]:
    """扫描约定目录中的实验证据；只投影摘要，不返回逐题观测正文。"""
    base = evolution_root(root) / "experiments"
    if not base.is_dir() or base.is_symlink():
        return []
    rows: list[dict[str, Any]] = []
    for directory in sorted(base.iterdir())[:MAX_EXPERIMENT_DIRS]:
        if not directory.is_dir() or directory.is_symlink():
            continue
        rows.append(
            {
                "name": directory.name,
                "experiment": _read_projected(directory / "experiment.json", _project_experiment),
                "report_gate": _read_projected(
                    directory / "report-gate.json", _project_report_gate
                ),
                "release": _read_projected(directory / "release.json", _project_release),
            }
        )
    return rows


def benchmark_bad_cases(
    store: EvaluationClosure, owner: str, *, review_status: str | None = None
) -> dict[str, Any]:
    """report-benchmark 坏例清单，附基准题原文与维度标注预填。"""
    if review_status is not None and review_status not in ("new", "confirmed", "ignored", "fixed"):
        raise BackflowError("复核状态无效")
    benchmark_cases, _ = sources("report-benchmark", 0)
    annotations = load_benchmark_annotations()["aspects"]
    rows = []
    for bad_case in store.list_bad_cases(owner, review_status=review_status):
        if bad_case["dataset_id"] != "report-benchmark":
            continue
        source_case = benchmark_cases.get(bad_case["case_id"], {}).get("case", {})
        rows.append(
            {
                "bad_case_id": bad_case["bad_case_id"],
                "case_id": bad_case["case_id"],
                "error_key": bad_case["error_key"],
                "review_status": bad_case["review_status"],
                "first_run_id": bad_case["first_run_id"],
                "last_run_id": bad_case["last_run_id"],
                "last_seen_ms": bad_case["last_seen_ms"],
                "query": source_case.get("query"),
                "depth": source_case.get("depth", "balanced"),
                "suggested_aspects": annotations.get(bad_case["case_id"], {}).get(
                    "required_aspects"
                ),
            }
        )
    return {"bad_cases": rows, "annotated_cases": len(annotations)}


def overview(store: EvaluationClosure, owner: str, root: Path) -> dict[str, Any]:
    """自进化闭环的分阶段状态；只汇总，不代答门禁结论。"""
    dataset_state = merged_dataset(root)
    experiments = list_experiments(root)
    gates = [
        row["experiment"]["holdout_gate"]
        for row in experiments
        if row["experiment"] and row["experiment"].get("holdout_gate")
    ]
    report_gates = [row["report_gate"] for row in experiments if row["report_gate"]]
    releases = [row["release"] for row in experiments if row["release"]]
    bad_cases = store.list_bad_cases(owner)
    benchmark_open = sum(
        1
        for row in bad_cases
        if row["dataset_id"] == "report-benchmark" and row["review_status"] in ("new", "confirmed")
    )
    flowed = {
        Case.model_validate(record["case"]).source_bad_case_id
        for record in read_records(evolution_root(root) / "cases")[0]
        + read_records(evolution_root(root) / "reviews")[0]
    }
    annotations = load_benchmark_annotations()["aspects"]
    return {
        "dataset": {
            "seed_case_count": dataset_state["seed_case_count"],
            "backflow_case_count": dataset_state["backflow_case_count"],
            "review_record_count": dataset_state["review_record_count"],
            "group_count": dataset_state.get("group_count"),
            "splits": dataset_state.get("splits"),
            "dataset_sha256": dataset_state["dataset_sha256"],
            "error": dataset_state["error"],
        },
        "benchmark": {
            "case_count": len(sources("report-benchmark", 0)[0]),
            "annotated_case_count": len(annotations),
        },
        "bad_cases": {
            "open_benchmark_count": benchmark_open,
            "flowed_count": len({x for x in flowed if x}),
        },
        "experiments": {
            "directory_count": len(experiments),
            "last_holdout_gate_passed": gates[-1]["passed"] if gates else None,
            "last_report_gate_passed": (report_gates[-1].get("passed") if report_gates else None),
            "release_count": len(releases),
        },
    }
