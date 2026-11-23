"""评测中心与 Planner 自进化的回流桥梁。

回流记录存放在评测卷 ``job_root()/planner-evolution/`` 下，与闭环数据库同源同权限：

- ``cases/``       复核归因为 Planner 漏项的训练例记录，参与优化数据集组装
- ``reviews/``     检索/报告问题以及规划完整的复核记录，仅作证据保留，不进数据集
- ``experiments/`` 约定的 CLI 实验输出目录（``planner_evolution optimize --out-dir``）

种子数据集与基准标注来自仓库内 ``evals/datasets/planner/``，只读；回流记录只追加，
不修改既有记录。评测轨迹默认只含元数据，Planner 实际任务列表不被补造：
回流记录显式标记任务证据状态，复核结论由复核人对照评测中心轨迹填写。
"""

from __future__ import annotations

import hashlib
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
STAGES = ("planner", "planner_complete", "retrieval", "report")
MAX_RECORD_BYTES = 256 * 1024
MAX_EXPERIMENT_DIRS = 50
SEED_DATASET = BACKEND / "evals/datasets/planner/planner-reviewed-v1.json"
BENCHMARK_ANNOTATIONS = BACKEND / "evals/datasets/planner/benchmark-aspects-v1.json"
_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$")


class BackflowError(ValueError):
    """回流输入与既有证据冲突，消息面向复核人，不泄露其他用户资源。"""


class BackflowSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["bad_case", "manual", "evaluation_case"]
    bad_case_id: str | None = None
    benchmark_case_id: str | None = None
    run_id: str | None = None
    error_key: str | None = None


class BackflowReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["planner", "planner_complete", "retrieval", "report"]
    review_note: str = Field(min_length=2, max_length=2000)
    reviewed_at: str
    source: BackflowSource
    benchmark_aspects: list[str] | None = None
    # reviewer_supplied 为人工复制的任务，trace_verified 为自动归档且提交时
    # 已重新核对原文；两者都是显式来源标记，缺失证据只能披露，不能补造。
    task_evidence_status: Literal[
        "not_collected", "unavailable", "reviewer_supplied", "trace_verified"
    ] = "not_collected"
    planner_tasks: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    evidence_reference: str | None = None
    evidence_sha256: str | None = None
    review_confirmed: bool = False


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


def _planner_evidence_for_run(
    store: EvaluationClosure, owner: str, root: Path, run_id: str, case_id: str
) -> dict[str, Any]:
    """已测报告案例（包括 passed）均可取证，但必须同时验证用户、题源和归档。

    四重归属校验：SQLite 逐题记录所有权 → Run 与 Pipeline 一致性 → Pipeline
    所有权 → 基准题源与归档指纹一致。SHA-256 只证明记录内容自写入后未被
    改动，不是防恶意篡改磁盘的签名。"""
    from app.services.evaluation_pipeline_v13 import PIPE_RE, get_pipeline

    result = {"status": "unavailable", "run_id": run_id, "tasks": [], "reason": "证据未归档"}
    if not run_id.endswith("--report-benchmark"):
        return result
    pipeline_id = run_id.removesuffix("--report-benchmark")
    if not PIPE_RE.fullmatch(pipeline_id):
        return result
    # Case 必须实际完成测量；预期题、页面路径或伪造的 Run ID 均无效。
    detail = store.case_result(owner, run_id, case_id)
    if (
        detail["run"]["dataset_id"] != "report-benchmark"
        or detail["run"]["execution_mode"] != "real"
    ):
        return result
    if detail["run"].get("source_pipeline_id") not in (None, pipeline_id):
        return result
    try:
        pipeline = get_pipeline(root, owner, pipeline_id)
    except KeyError, ValueError:
        return result
    if "report-benchmark" not in pipeline.get("datasets", []):
        return result
    path = root / "pipelines" / pipeline_id / "artifacts" / "report-benchmark" / "raw_result.json"
    if any(
        part.is_symlink()
        for part in (path, path.parent, path.parent.parent, path.parent.parent.parent)
    ):
        return result
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        return result
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError, UnicodeError:
        return result
    rows = doc.get("results") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict) or row.get("id") != case_id:
            continue
        ev = row.get("planner_evidence")
        if (
            not isinstance(ev, dict)
            or ev.get("schema_version") != 1
            or ev.get("case_id") != case_id
        ):
            return result
        signature = ev.get("evidence_sha256")
        unsigned = {k: v for k, v in ev.items() if k != "evidence_sha256"}
        expected = hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if not isinstance(signature, str) or signature != expected:
            return result
        benchmark, _ = sources("report-benchmark", 0)
        fixture = benchmark.get(case_id)
        if not fixture or input_hash(fixture) != detail["result"]["input_sha256"]:
            return result
        question = fixture.get("case", {}).get("query")
        if not isinstance(question, str) or hashlib.sha256(question.encode()).hexdigest() != ev.get(
            "query_sha256"
        ):
            return result
        if not isinstance(ev.get("prompt_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", ev["prompt_sha256"]
        ):
            return result
        if not isinstance(ev.get("trace_id"), str) or not re.fullmatch(
            r"[0-9a-zA-Z_-]{1,64}", ev["trace_id"]
        ):
            return result
        tasks = ev.get("tasks")
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= 30:
            return result
        if any(
            not isinstance(t, dict)
            or not isinstance(t.get("query"), str)
            or not 5 <= len(t["query"]) <= 500
            or not isinstance(t.get("id"), str)
            for t in tasks
        ):
            return result
        return {
            "status": "recorded",
            "run_id": run_id,
            "case_id": case_id,
            "result_status": detail["result"]["status"],
            "tasks": tasks,
            "prompt_sha256": ev["prompt_sha256"],
            "trace_id": ev["trace_id"],
            "evidence_reference": f"auto:{run_id}:{signature}",
        }
    return result


def _planner_evidence_for_bad_case(
    store: EvaluationClosure, owner: str, root: Path, bad_case_id: str
) -> dict[str, Any]:
    """原坏例接口保持兼容：坏例身份验证后共享可信归档解析器。"""
    detail = store.bad_case_detail(owner, bad_case_id)
    bc = detail["bad_case"]
    return _planner_evidence_for_run(store, owner, root, bc["last_run_id"], bc["case_id"])


def planner_evidence_for_bad_case(
    store: EvaluationClosure, owner: str, root: Path, bad_case_id: str
) -> dict[str, Any]:
    """对外入口：API 与回流提交共用同一取证路径，保证展示与校验看到同一份归档。"""
    return _planner_evidence_for_bad_case(store, owner, root, bad_case_id)


def planner_evidence_for_case(
    store: EvaluationClosure, owner: str, root: Path, run_id: str, case_id: str
) -> dict[str, Any]:
    """普通已测案例（含 passed）的取证入口，与坏例路径共享同一解析器。"""
    return _planner_evidence_for_run(store, owner, root, run_id, case_id)


def reviewable_evaluation_cases(store: EvaluationClosure, owner: str) -> dict[str, Any]:
    """普通通过题与失败题都可进入人工复核；只列真实、已测、有基准标注的题。"""
    benchmark, _ = sources("report-benchmark", 0)
    annotated = load_benchmark_annotations()["aspects"]
    rows = []
    for run in store.runs(owner):
        if run["dataset_id"] != "report-benchmark" or run["execution_mode"] != "real":
            continue
        for row in store.case_results(owner, run["run_id"]):
            if row["case_id"] not in annotated or row["case_id"] not in benchmark:
                continue
            if row["input_sha256"] != input_hash(benchmark[row["case_id"]]):
                continue
            source = benchmark[row["case_id"]]["case"]
            rows.append(
                {
                    "run_id": run["run_id"],
                    "case_id": row["case_id"],
                    "result_status": row["status"],
                    "query": source["query"],
                    "depth": source.get("depth", "balanced"),
                    "required_aspects": annotated[row["case_id"]]["required_aspects"],
                }
            )
            if len(rows) >= 30:
                return {"cases": rows}
    return {"cases": rows}


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
    evaluation_run_id: str | None = None,
    evaluation_case_id: str | None = None,
    manual_query: str | None = None,
    manual_depth: str = "balanced",
    planner_tasks: list[str] | None = None,
    missing_aspects: list[str] | None = None,
    evidence_reference: str | None = None,
    review_confirmed: bool = False,
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
    if sum(x is not None for x in (bad_case_id, manual_query, evaluation_run_id)) != 1:
        raise BackflowError("必须且只能选择坏例、已测评测案例或手动回流之一")
    if (evaluation_run_id is None) != (evaluation_case_id is None):
        raise BackflowError("普通评测案例必须同时提供 Run ID 与 Case ID")

    # 严格准入：报告未完成 ≠ Planner 漏项。只有人工审阅任务清单并明确指出
    # 哪些必答维度未被任何任务覆盖，才能写入 GEPA 的 train split。
    planner_tasks = planner_tasks or []
    missing_aspects = missing_aspects or []
    if stage in ("planner", "planner_complete"):
        if not review_confirmed:
            raise BackflowError("Planner 复核需要人工确认任务证据，不能仅凭报告结果判定")
        if not isinstance(planner_tasks, list) or not 1 <= len(planner_tasks) <= 30:
            raise BackflowError("请提供实际 Planner 任务列表（1–30 条）")
        if any(not isinstance(t, str) or not 5 <= len(t.strip()) <= 500 for t in planner_tasks):
            raise BackflowError("实际 Planner 任务每条需 5–500 字符")
        if stage == "planner" and (not isinstance(missing_aspects, list) or not missing_aspects):
            raise BackflowError("必须指出至少一条未被 Planner 任务覆盖的必答维度")
        if stage == "planner_complete" and missing_aspects:
            raise BackflowError("规划完整不能同时标记遗漏维度")
        labels = {normalized(x): x for x in aspects}
        if any(not isinstance(x, str) or normalized(x) not in labels for x in missing_aspects):
            raise BackflowError("漏项维度必须来自当前必答维度列表")
        if len({normalized(x) for x in missing_aspects}) != len(missing_aspects):
            raise BackflowError("漏项维度重复")
        if (
            not isinstance(evidence_reference, str)
            or not 4 <= len(evidence_reference.strip()) <= 200
        ):
            raise BackflowError("请填写 Planner Trace 或任务列表的可追溯证据编号")
    elif planner_tasks or missing_aspects or evidence_reference or review_confirmed:
        raise BackflowError("检索/报告归因不能携带 Planner 漏项确认字段")

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
    elif evaluation_run_id is not None:
        if stage not in ("planner", "planner_complete"):
            raise BackflowError("普通评测案例只允许 Planner 维度复核")
        assert evaluation_case_id is not None
        archived = _planner_evidence_for_run(
            store, owner, root, evaluation_run_id, evaluation_case_id
        )
        if archived["status"] != "recorded":
            raise BackflowError("该评测案例缺少可验证的 Planner 任务证据")
        fixture, _ = sources("report-benchmark", 0)
        row = fixture[evaluation_case_id]["case"]
        query = row["query"]
        depth = row.get("depth", "balanced")
        benchmark_aspects = annotations["aspects"][evaluation_case_id]["required_aspects"]
        source = BackflowSource(
            type="evaluation_case", benchmark_case_id=evaluation_case_id, run_id=evaluation_run_id
        )
        if aspects != benchmark_aspects:
            raise BackflowError("普通评测案例必须使用冻结的必答维度，不能在复核时改写标注")
    else:
        assert manual_query is not None
        if not 10 <= len(manual_query.strip()) <= 2000:
            raise BackflowError("手动回流的研究问题需要 10–2000 个字符")
        if manual_depth not in ("fast", "balanced", "comprehensive"):
            raise BackflowError("研究深度无效")
        query = manual_query.strip()
        source = BackflowSource(type="manual")

    if stage == "planner":
        _check_group_split(seed, group, "train")
        if evaluation_run_id is not None and group not in {
            case.group for case in seed.cases if case.split == "train"
        }:
            raise BackflowError("普通评测漏项必须归入已有开发集主题组，防止绕过冻结验证/留出分组")
    records = _existing_records(root) + read_records(evolution_root(root) / "reviews")[0]
    if stage == "planner_complete":
        if any(
            rec.get("review", {}).get("source", {}).get("run_id") == source.run_id
            and rec.get("review", {}).get("source", {}).get("benchmark_case_id")
            == source.benchmark_case_id
            for rec in records
        ):
            raise BackflowError("本次评测案例已经人工复核")
        if any(rec.get("case", {}).get("case_id") == case_id for rec in records):
            raise BackflowError("案例 ID 已有回流记录")
    else:
        _check_dupes(seed, records, case_id=case_id, query=query, bad_case_id=bad_case_id)
        if evaluation_run_id and any(
            rec.get("review", {}).get("source", {}).get("run_id") == evaluation_run_id
            and rec.get("review", {}).get("source", {}).get("benchmark_case_id")
            == evaluation_case_id
            for rec in records
        ):
            raise BackflowError("本次评测案例已经人工复核")

    # 保持去重优先级：重复坏例先报「已回流」，证据校验在后。
    # 坏例已有明确 Run ID，证据引用必须指向同一运行；只做关联约束，
    # 无法自动从 Trace 反证 Planner 内容。
    if (
        stage in ("planner", "planner_complete")
        and bad_case_id is not None
        and source.run_id not in evidence_reference
    ):
        raise BackflowError("Planner Trace 证据编号必须包含该坏例的 Run ID")

    # auto: 引用来自自动归档。提交时重新核对归档状态与原文一致，防止
    # 审核到提交之间归档被改动后仍按旧内容回流；人工复制路径不走这里。
    verified_trace = False
    if (
        stage in ("planner", "planner_complete")
        and evidence_reference
        and evidence_reference.startswith("auto:")
    ):
        if bad_case_id is not None:
            archived = _planner_evidence_for_bad_case(store, owner, root, bad_case_id)
        elif evaluation_run_id is not None:
            archived = _planner_evidence_for_run(
                store, owner, root, evaluation_run_id, evaluation_case_id
            )
        else:
            raise BackflowError("自动取证仅支持真实评测运行")
        if (
            archived.get("status") != "recorded"
            or archived.get("evidence_reference") != evidence_reference
        ):
            raise BackflowError("自动 Planner 证据不存在或已变化，请刷新并重新审核")
        if planner_tasks != [row["query"] for row in archived["tasks"]]:
            raise BackflowError("自动 Planner 任务内容与归档原文不一致")
        verified_trace = True

    # 普通评测案例没有人工复制路径：必须走自动归档校验，防止伪造任务列表。
    if evaluation_run_id is not None and not verified_trace:
        raise BackflowError("普通评测案例回流必须使用可验证的自动证据")

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
            task_evidence_status=(
                "trace_verified"
                if verified_trace
                else "reviewer_supplied"
                if stage == "planner"
                else "not_collected"
            ),
            planner_tasks=[t.strip() for t in planner_tasks],
            missing_aspects=[labels[normalized(x)] for x in missing_aspects]
            if stage == "planner"
            else [],
            evidence_reference=evidence_reference.strip()
            if stage in ("planner", "planner_complete")
            else None,
            evidence_sha256=input_hash(
                {
                    "tasks": [t.strip() for t in planner_tasks],
                    "missing_aspects": [labels[normalized(x)] for x in missing_aspects],
                    "evidence_reference": evidence_reference.strip(),
                    "run_id": source.run_id,
                }
            )
            if stage in ("planner", "planner_complete")
            else None,
            review_confirmed=(stage in ("planner", "planner_complete") and review_confirmed),
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
        review = record.get("review", {})
        # 旧版只有“人工选 planner”的历史记录不自动获得可信训练资格。
        if (
            review.get("stage") != "planner"
            or review.get("task_evidence_status") not in ("reviewer_supplied", "trace_verified")
            or not review.get("review_confirmed")
            or not review.get("planner_tasks")
            or not review.get("missing_aspects")
            or not review.get("evidence_sha256")
        ):
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


def benchmark_cases() -> dict[str, Any]:
    """report-benchmark 题目与人工维度标注，供自进化页从评测中心选题回流。

    题目原文来自评测中心的同源题集，维度来自人工维护标注；两者都不在这里生成。
    未标注的题保留 ``required_aspects=None``，由复核人在回流表单里自行填写。
    """
    cases, _ = sources("report-benchmark", 0)
    annotations = load_benchmark_annotations()["aspects"]
    rows = []
    for case_id, source in cases.items():
        case = source["case"]
        annotation = annotations.get(case_id, {})
        rows.append(
            {
                "case_id": case_id,
                "query": case.get("query"),
                "depth": case.get("depth", "balanced"),
                "required_aspects": annotation.get("required_aspects"),
                "rationale": annotation.get("rationale"),
            }
        )
    return {
        "dataset_id": "report-benchmark",
        "case_count": len(rows),
        "annotated_case_count": len(annotations),
        "cases": rows,
    }


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
