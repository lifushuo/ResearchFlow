"""服务端评测结果接口；用户身份来自现有登录会话。"""

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.services.evaluation_closure import ClosureError, EvaluationClosure, NotFound


class ClosureBody(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GateBody(ClosureBody):
    min_measured: int = Field(default=1, ge=1, strict=True)
    min_pass_rate: float = Field(default=0.9, ge=0, le=1, strict=True)
    max_p95_latency_ms: int | None = Field(default=None, ge=0, strict=True)
    fail_on_blocked: bool = Field(default=True, strict=True)
    fail_on_unmeasured: bool = Field(default=True, strict=True)
    required_metric_minimums: dict[str, Annotated[float, Field(strict=True)]] = Field(
        default_factory=dict
    )


def make_router(
    store: EvaluationClosure | Callable[..., EvaluationClosure],
    verified_owner_dependency: Callable[..., str],
    verified_mutation_dependency: Callable[..., Any],
) -> APIRouter:
    router = APIRouter(tags=["Evaluation Closure"])

    def fixed_store() -> EvaluationClosure:
        assert isinstance(store, EvaluationClosure)
        return store

    get_store = fixed_store if isinstance(store, EvaluationClosure) else store

    def call(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotFound:
            raise HTTPException(404, "Resource not found") from None
        except ClosureError as exc:
            raise HTTPException(409, str(exc)) from None

    @router.get("/runs")
    def runs(
        owner: str = Depends(verified_owner_dependency), db: EvaluationClosure = Depends(get_store)
    ):
        return call(db.runs, owner)

    @router.get("/bad-cases")
    def bad_cases(
        review_status: str | None = None,
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0),
        owner: str = Depends(verified_owner_dependency),
        db: EvaluationClosure = Depends(get_store),
    ):
        return call(
            db.list_bad_cases, owner, review_status=review_status, limit=limit, offset=offset
        )

    @router.get("/bad-cases/{bad_case_id}")
    def detail(
        bad_case_id: str,
        owner: str = Depends(verified_owner_dependency),
        db: EvaluationClosure = Depends(get_store),
    ):
        return call(db.bad_case_detail, owner, bad_case_id)

    @router.post("/runs/{run_id}/gate")
    def gate(
        run_id: str,
        body: GateBody,
        owner: str = Depends(verified_owner_dependency),
        _action: Any = Depends(verified_mutation_dependency),
        db: EvaluationClosure = Depends(get_store),
    ):
        return call(db.quality_gate, owner, run_id, **body.model_dump())

    @router.get("/compare")
    def compare(
        baseline_run: str,
        candidate_run: str,
        owner: str = Depends(verified_owner_dependency),
        db: EvaluationClosure = Depends(get_store),
    ):
        return call(db.compare, owner, baseline_run, candidate_run)

    return router
