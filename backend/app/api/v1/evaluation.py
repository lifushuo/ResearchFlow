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
    stage: Literal["planner", "retrieval", "report"]
    bad_case_id: str | None = Field(default=None, min_length=1, max_length=120)
    manual_query: str | None = None
    manual_depth: Literal["fast", "balanced", "comprehensive"] = "balanced"


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
        manual_query=payload.manual_query,
        manual_depth=payload.manual_depth,
    )
