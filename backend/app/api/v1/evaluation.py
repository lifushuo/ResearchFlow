"""按当前用户隔离的统一评测、证据查询、比较与质量门禁接口。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.evaluation_closure import make_router
from app.dependencies import get_current_user
from app.models.user import User
from app.services.evaluation_closure import ClosureError, EvaluationClosure, NotFound
from app.services.evaluation_closure.storage import store_at
from app.services.evaluation_dataset_registry import (
    DatasetRegistryError,
    get_dataset,
    list_datasets,
)
from app.services.evaluation_jobs import JobError, job_root
from app.services.evaluation_pipeline_v13 import (
    PipelineBusy,
    PipelineError,
    create_pipeline,
    get_pipeline,
    list_pipelines,
)
from app.services.evaluation_pipeline_v13 import (
    capabilities as pipeline_capabilities,
)
from app.services.planner_evolution_center import (
    BackflowError,
    backflow,
    benchmark_bad_cases,
    benchmark_cases,
    list_experiments,
    load_benchmark_annotations,
    merged_dataset,
    overview,
    planner_evidence_for_bad_case,
    planner_evidence_for_case,
    reviewable_evaluation_cases,
)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


def verify_evaluation_action(action: str | None) -> None:
    if action != "run-pipeline-v1.3":
        raise HTTPException(status_code=403, detail="Explicit evaluation action required")


def _closure_action(
    action: str | None = Header(default=None, alias="X-ResearchFlow-Eval-Action"),
) -> None:
    verify_evaluation_action(action)


def _closure_owner(user: User = Depends(get_current_user)) -> str:
    return str(user.id)


def _closure_store() -> EvaluationClosure:
    try:
        return store_at(job_root())
    except OSError, ClosureError, JobError, sqlite3.Error:
        raise HTTPException(503, "Closure storage unavailable") from None


router.include_router(
    make_router(_closure_store, _closure_owner, _closure_action), prefix="/closure"
)


def _diagnostic_root() -> Path:
    try:
        return job_root()
    except JobError:
        raise HTTPException(503, "Evaluation storage unavailable") from None


def _diagnostic_call(function, *args):
    try:
        return function(*args)
    except KeyError:
        raise HTTPException(404, "Evaluation resource not found") from None
    except ValueError:
        raise HTTPException(
            409, "Diagnostic evidence invalid or changed; refresh the run"
        ) from None
    except OSError:
        raise HTTPException(
            503, "diagnostic_storage_unavailable: 诊断记录暂时无法读取，请稍后刷新。"
        ) from None


@router.get("/closure/execution-incidents")
def execution_incidents(
    current_user: User = Depends(get_current_user),
    root: Path = Depends(_diagnostic_root),
    pipeline_id: str | None = None,
    dataset_id: str | None = None,
):
    from app.services.evaluation_diagnostics import incident_queue

    return {
        "incidents": _diagnostic_call(
            incident_queue, root, str(current_user.id), pipeline_id, dataset_id
        )
    }


@router.get("/pipelines/{pipeline_id}/diagnostics/{dataset_id}")
def evaluation_diagnostics(
    pipeline_id: str,
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    root: Path = Depends(_diagnostic_root),
    case_id: str | None = None,
):
    from app.services.evaluation_diagnostics import read_diagnostics

    return _diagnostic_call(
        read_diagnostics, root, str(current_user.id), pipeline_id, dataset_id, case_id
    )


def _evaluation_root(_user: User = Depends(get_current_user)) -> Path:
    return _diagnostic_root()


@router.get("/datasets")
def evaluation_datasets(_root: Path = Depends(_evaluation_root)):
    """需要登录；注册表只读取纳入版本控制的数据集。"""
    try:
        return list_datasets()
    except (DatasetRegistryError, OSError) as exc:
        raise HTTPException(
            status_code=422, detail="Evaluation datasets unavailable or invalid"
        ) from exc


@router.get("/datasets/{dataset_id}")
def evaluation_dataset_detail(
    dataset_id: str,
    offset: int = 0,
    limit: int = 50,
    _root: Path = Depends(_evaluation_root),
):
    try:
        return get_dataset(dataset_id, offset=offset, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Evaluation dataset not found") from exc
    except (DatasetRegistryError, OSError) as exc:
        raise HTTPException(
            status_code=422, detail="Invalid dataset registry or pagination"
        ) from exc


class PipelineRequestV13(BaseModel):
    model_config = ConfigDict(extra="forbid")
    datasets: list[str]
    request_id: str
    max_cases: int = 1
    consent: str | None = None
    case_id: str | None = Field(default=None, min_length=1, max_length=120, strict=True)


@router.get("/pipelines/capabilities")
def evaluation_pipeline_capabilities_v13(_root: Path = Depends(_evaluation_root)):
    return pipeline_capabilities()


@router.post("/pipelines", status_code=202)
def evaluation_create_pipeline_v13(
    payload: PipelineRequestV13,
    _root: Path = Depends(_evaluation_root),
    current_user: User = Depends(get_current_user),
    action: str | None = Header(default=None, alias="X-ResearchFlow-Eval-Action"),
):
    verify_evaluation_action(action)
    try:
        return create_pipeline(
            job_root(),
            str(current_user.id),
            payload.datasets,
            payload.request_id,
            payload.max_cases,
            payload.consent,
            **({"case_id": payload.case_id} if payload.case_id is not None else {}),
        )
    except PipelineBusy as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/pipelines")
def evaluation_list_pipelines_v13(
    _root: Path = Depends(_evaluation_root),
    current_user: User = Depends(get_current_user),
):
    return {"pipelines": list_pipelines(job_root(), str(current_user.id))}


@router.get("/pipelines/{pipeline_id}")
def evaluation_get_pipeline_v13(
    pipeline_id: str,
    _root: Path = Depends(_evaluation_root),
    current_user: User = Depends(get_current_user),
):
    try:
        return get_pipeline(job_root(), str(current_user.id), pipeline_id)
    except (KeyError, PipelineError) as exc:
        raise HTTPException(status_code=404, detail="Pipeline not found") from exc


def verify_backflow_action(action: str | None) -> None:
    # 与评测执行分开的显式动作头：回流写入评测卷，不能被隐性触发。
    if action != "planner-backflow-v1":
        raise HTTPException(status_code=403, detail="Explicit backflow action required")


def _evolution_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except NotFound:
        raise HTTPException(404, "Resource not found") from None
    except (BackflowError, ClosureError) as exc:
        raise HTTPException(409, str(exc)) from None
    except JobError, sqlite3.Error:
        raise HTTPException(503, "Evaluation storage unavailable") from None
    except OSError:
        raise HTTPException(503, "Evaluation storage unavailable") from None
    except ValueError as exc:
        # JobError 是 ValueError 子类，已在上方先行拦截；其余值错误面向复核人展示。
        raise HTTPException(422, str(exc)) from None


class BackflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=80)
    group: str = Field(min_length=1, max_length=80)
    aspects: list[str]
    review_note: str
    stage: Literal["planner", "planner_complete", "retrieval", "report"]
    bad_case_id: str | None = Field(default=None, min_length=1, max_length=120)
    evaluation_run_id: str | None = Field(default=None, min_length=1, max_length=256)
    evaluation_case_id: str | None = Field(default=None, min_length=1, max_length=120)
    manual_query: str | None = None
    manual_depth: Literal["fast", "balanced", "comprehensive"] = "balanced"
    planner_tasks: list[str] = Field(default_factory=list, max_length=30)
    missing_aspects: list[str] = Field(default_factory=list, max_length=12)
    evidence_reference: str | None = Field(default=None, max_length=200)
    review_confirmed: bool = False


@router.get("/evolution/overview")
def evolution_overview(
    store: EvaluationClosure = Depends(_closure_store),
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
):
    return _evolution_call(overview, store, owner, root)


@router.get("/evolution/dataset")
def evolution_dataset(root: Path = Depends(_evaluation_root)):
    return _evolution_call(merged_dataset, root)


@router.get("/evolution/annotations")
def evolution_annotations(root: Path = Depends(_evaluation_root)):
    return _evolution_call(load_benchmark_annotations)


@router.get("/evolution/bad-cases")
def evolution_bad_cases(
    review_status: str | None = None,
    store: EvaluationClosure = Depends(_closure_store),
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
):
    return _evolution_call(benchmark_bad_cases, store, owner, review_status=review_status)


@router.get("/evolution/bad-cases/{bad_case_id}/planner-evidence")
def evolution_planner_evidence(
    bad_case_id: str,
    store: EvaluationClosure = Depends(_closure_store),
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
):
    """仅返回当前账号真实评测 Run 的已归档 Planner 任务，不自动判定漏项。"""
    return _evolution_call(planner_evidence_for_bad_case, store, owner, root, bad_case_id)


@router.get("/evolution/reviewable-cases")
def evolution_reviewable_cases(
    store: EvaluationClosure = Depends(_closure_store),
    owner: str = Depends(_closure_owner),
):
    """将真实已测评测案例（包括 passed）提供人工复核；不自动定义坏例。"""
    return _evolution_call(reviewable_evaluation_cases, store, owner)


@router.get("/evolution/reviewable-cases/{run_id}/{case_id}/planner-evidence")
def evolution_reviewable_case_evidence(
    run_id: str,
    case_id: str,
    store: EvaluationClosure = Depends(_closure_store),
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
):
    return _evolution_call(planner_evidence_for_case, store, owner, root, run_id, case_id)


class PlannerJudgeRequest(BaseModel):
    """只有用户显式允许一次收费 Judge 调用，才能分析冻结的评测证据。"""

    model_config = ConfigDict(extra="forbid")
    evidence_reference: str = Field(min_length=20, max_length=256)
    acknowledge_cost: Literal[True]


@router.post("/evolution/reviewable-cases/{run_id}/{case_id}/judge-suggestion")
async def evolution_planner_judge_suggestion(
    run_id: str,
    case_id: str,
    payload: PlannerJudgeRequest,
    store: EvaluationClosure = Depends(_closure_store),
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
    action: str | None = Header(default=None, alias="X-ResearchFlow-Eval-Action"),
):
    """读取可信 Planner Trace，调用一次独立 Judge；不写训练集，也不替人勾选。"""

    from app.services.planner_review_judge import suggest_planner_coverage
    from evals.agent_eval_ext.closure_observer import sources

    if action != "planner-judge-v1":
        raise HTTPException(403, "Explicit Planner Judge action required")

    # 必须先通过账号、Pipeline、原题、任务指纹校验，再考虑模型调用。
    evidence = _evolution_call(planner_evidence_for_case, store, owner, root, run_id, case_id)
    if (
        evidence.get("status") != "recorded"
        or evidence.get("evidence_reference") != payload.evidence_reference
    ):
        raise HTTPException(409, "Planner 证据不可用或已经改变；请重新选择案例")
    annotations = _evolution_call(load_benchmark_annotations)
    aspects = annotations["aspects"].get(case_id, {}).get("required_aspects")
    benchmark, _ = sources("report-benchmark", 0)
    source = benchmark.get(case_id, {}).get("case", {})
    if not isinstance(aspects, list) or not isinstance(source.get("query"), str):
        raise HTTPException(409, "没有可核验的冻结必答维度")
    try:
        return await suggest_planner_coverage(
            evidence=evidence,
            question=source["query"],
            aspects=aspects,
        )
    except TimeoutError:
        raise HTTPException(504, "Planner Judge 超时；未产生有效评分，请检查调用日志") from None
    except ValueError:
        raise HTTPException(409, "Judge 配置或结构化反馈无效，请检查评分模型设置") from None
    except Exception:
        # 供应商错误及返回内容可能含密钥、敏感端点，不在 HTTP 中回显。
        raise HTTPException(
            503, "Planner Judge 调用失败；未生成覆盖判断，请检查服务端日志"
        ) from None


@router.get("/evolution/benchmark-cases")
def evolution_benchmark_cases(_root: Path = Depends(_evaluation_root)):
    """评测中心基准题目与人工标注，供自进化页选题回流（只读）。"""
    return _evolution_call(benchmark_cases)


@router.get("/evolution/experiments")
def evolution_experiments(root: Path = Depends(_evaluation_root)):
    return _evolution_call(list_experiments, root)


@router.post("/evolution/backflow")
def evolution_backflow(
    payload: BackflowRequest,
    store: EvaluationClosure = Depends(_closure_store),
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
    action: str | None = Header(default=None, alias="X-ResearchFlow-Eval-Action"),
):
    verify_backflow_action(action)
    return _evolution_call(
        backflow,
        store,
        owner,
        root,
        case_id=payload.case_id,
        group=payload.group,
        aspects=payload.aspects,
        review_note=payload.review_note,
        stage=payload.stage,
        bad_case_id=payload.bad_case_id,
        evaluation_run_id=payload.evaluation_run_id,
        evaluation_case_id=payload.evaluation_case_id,
        manual_query=payload.manual_query,
        manual_depth=payload.manual_depth,
        planner_tasks=payload.planner_tasks,
        missing_aspects=payload.missing_aspects,
        evidence_reference=payload.evidence_reference,
        review_confirmed=payload.review_confirmed,
    )


# Planner GEPA 作业生命周期：独立于报告 A/B 与发布审批。
class GEPAJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=36, max_length=36)
    mode: Literal["demo", "live"] = "demo"
    # demo 作业由服务端强制记录为 v1（脚本 rollout 无法产生 v2 质量子分）。
    metric_version: Literal["v3"] = "v3"
    max_evals: int = Field(default=64, ge=4, le=200)
    dataset_preset: Literal["smoke", "full"] = "full"
    request_quotas: dict[str, int] | None = None
    proposals: int = Field(default=4, ge=1, le=8)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    consent: str | None = None


@router.get("/evolution/jobs/capabilities")
def evolution_gepa_capabilities(owner: str = Depends(_closure_owner)):
    from app.services.planner_evolution_jobs import _paid_allowed
    from app.services.planner_gepa_contract import request_timeouts

    return {
        "request_timeouts": request_timeouts(),
        "demo_enabled": True,
        "live_enabled": _paid_allowed(owner),
        "max_evals_limit": 200,
        "proposals_limit": 8,
        "estimated_api_cost_is_not_a_hard_dollar_cap": True,
    }


@router.post("/evolution/jobs", status_code=202)
def evolution_create_gepa_job(
    payload: GEPAJobRequest,
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
    action: str | None = Header(default=None, alias="X-ResearchFlow-Eval-Action"),
):
    from app.services.planner_evolution_jobs import (
        EvolutionJobBusy,
        EvolutionJobError,
        create_job,
    )

    if action != "planner-gepa-optimize-v1":
        raise HTTPException(403, "Explicit GEPA optimization action required")
    try:
        return create_job(
            root,
            owner,
            request_id=payload.request_id,
            mode=payload.mode,
            metric_version=payload.metric_version,
            dataset_preset=payload.dataset_preset,
            request_quotas=payload.request_quotas,
            max_evals=payload.max_evals,
            proposals=payload.proposals,
            seed=payload.seed,
            consent=payload.consent,
        )
    except EvolutionJobBusy as exc:
        raise HTTPException(429, str(exc)) from exc
    except (EvolutionJobError, OSError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/evolution/jobs")
def evolution_list_gepa_jobs(
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
):
    from app.services.planner_evolution_jobs import list_jobs

    return {"jobs": list_jobs(root, owner)}


class PlannerFinalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=36, max_length=36)
    mode: Literal["demo", "live"] = "demo"
    consent: str | None = None
    request_quotas: dict[str, int] | None = None


@router.get("/evolution/jobs/{job_id}/final/preview")
def evolution_final_preview(
    job_id: str,
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
):
    from app.services.planner_final_jobs import preview

    try:
        return preview(root, owner, job_id)
    except KeyError:
        raise HTTPException(404, "Source experiment not found") from None
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/evolution/jobs/{job_id}/final", status_code=202)
def evolution_start_final(
    job_id: str,
    payload: PlannerFinalRequest,
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
    action: str | None = Header(default=None, alias="X-ResearchFlow-Eval-Action"),
):
    from app.services.planner_evolution_jobs import EvolutionJobBusy
    from app.services.planner_final_jobs import create_final_job

    if action != "planner-final-ab-v1":
        raise HTTPException(403, "Explicit Planner final A/B action required")
    try:
        return create_final_job(root, owner, job_id, **payload.model_dump())
    except KeyError:
        raise HTTPException(404, "Source experiment not found") from None
    except EvolutionJobBusy as exc:
        raise HTTPException(429, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/evolution/jobs/{job_id}")
def evolution_get_gepa_job(
    job_id: str,
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
):
    from app.services.planner_evolution_jobs import get_job

    try:
        return get_job(root, owner, job_id)
    except KeyError, ValueError:
        raise HTTPException(404, "GEPA job not found") from None


@router.get("/evolution/jobs/{job_id}/result")
def evolution_gepa_job_result(
    job_id: str,
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
):
    from app.services.planner_evolution_jobs import job_result

    try:
        return job_result(root, owner, job_id)
    except KeyError, ValueError, OSError:
        raise HTTPException(404, "GEPA result not available") from None


@router.get("/evolution/jobs/{job_id}/audit")
def evolution_gepa_job_audit(
    job_id: str,
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
):
    from app.services.planner_evolution_jobs import job_audit

    try:
        return job_audit(root, owner, job_id)
    except KeyError, ValueError, OSError:
        raise HTTPException(404, "GEPA audit unavailable") from None


@router.post("/evolution/jobs/{job_id}/cancel")
def evolution_cancel_gepa_job(
    job_id: str,
    root: Path = Depends(_evaluation_root),
    owner: str = Depends(_closure_owner),
    action: str | None = Header(default=None, alias="X-ResearchFlow-Eval-Action"),
):
    from app.services.planner_evolution_jobs import cancel_job

    if action != "planner-gepa-cancel-v1":
        raise HTTPException(403, "Explicit GEPA cancellation action required")
    try:
        return cancel_job(root, owner, job_id)
    except KeyError, ValueError:
        raise HTTPException(404, "GEPA job not found") from None
