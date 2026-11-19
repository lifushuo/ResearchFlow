"""研究运行的查询、审核和导出接口。

多个页面需要读取同一次研究的数据，因此用一次聚合查询返回完整视图，减少请求次数，也把权限校验集中在一处。字段和三态值直接沿用领域层，不能把“未检查”和“未证实”合并。

来源的 citation_index 允许为空：这表示找到过该来源，但没有给它分配引用编号。即使有编号，也要看当前修订的 cited_indices，才能知道报告是否实际引用它。

计划审核和报告审核统一经过 app.run_lifecycle 与 app.authorization；只有获批的报告审核能授权生成最终产物，本文件不重复制定授权规则。"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import metrics, run_bundle, run_lifecycle
from app.db.base import AsyncSessionLocal, get_db
from app.db.redis import get_redis
from app.dependencies import get_current_user
from app.errors import Conflict, NotFound, Unprocessable
from app.logconfig import bind_research_run_context
from app.models.agent_log import AgentLog
from app.models.project import Project
from app.models.research import (
    RESEARCH_DEPTHS,
    Contradiction,
    Evidence,
    ResearchPlan,
    ResearchRun,
    Source,
)
from app.models.review import ResearchArtifact, Review
from app.models.revision import Claim, ClaimEvidenceLink, Revision
from app.models.user import User
from app.ports import CorpusLocator
from app.run_dispatch import RunDispatcher

# 请求结构放在 app/schemas/runs.py，桌面端可以复用，而不用引入服务端配置依赖。
# 这里继续导出这些结构，兼容既有路由和测试的导入位置。
from app.schemas.corpus import ExtractedTextResponse
from app.schemas.runs import (  # noqa: F401  （供其他模块导入）
    CreateRunRequest,
    PlanReviewRequest,
    ReportReviewRequest,
)
from app.services import cited_documents, corpus_ingest, model_routing
from app.services.corpus_selection import SelectionError, freeze_selection
from app.services.document_headers import download_headers, media_type_for
from app.services.event_stream import sse_frames
from app.services.host_ports import get_corpus_locator
from app.services.sse import SSE_HEADERS
from app.workers.dispatch import get_run_dispatcher
from research_engine import citation_rate
from research_engine.bundle import render_model_attribution_md, stamp_demo_md

logger = structlog.get_logger()
router = APIRouter(prefix="/runs", tags=["Runs"])

# 实时事件流收到完成、失败或两个人工关卡事件后关闭。图暂停时要等人操作，继续保持连接不会收到新事件。
# 此列表与会话事件流及 sidecar._TERMINAL_EVENTS 保持一致。
_TERMINAL_EVENTS = ("COMPLETED", "FAILED", "HITL_READY", "PLAN_READY")

# 历史事件回放只在真正的终态停止，不在人工关卡处停止。
#:
# 实时监听和历史回放必须使用不同的停止条件。
# 实时监听遇到人工关卡应关闭；历史回放则必须继续读关卡之后的事件。
# 客户端重连且未传 Last-Event-ID 时，会从头回放；若在旧 PLAN_READY 处停止，就会漏掉后续执行，也无法进入实时监听。
# 因此历史回放只在 COMPLETED、FAILED 停止，与会话事件流规则一致。
_REPLAY_STOP_EVENTS = ("COMPLETED", "FAILED")

# 这些状态表示当前正在等人处理；回放完已有事件后直接关闭，不再监听。
#:
# 是否继续监听要看运行当前的状态，而不是历史中是否出现过人工关卡。
# 这样既能完整回放已经越过关卡的任务，也不会让仍停在关卡的任务一直占用连接。
_SUSPENDED_STATUSES = ("COMPLETED", "FAILED", "CANCELLED", "AWAITING_PLAN", "AWAITING_REVIEW")


async def _run_or_404(db: AsyncSession, run_id: uuid.UUID, owner_id: uuid.UUID) -> ResearchRun:
    """检查运行是否属于当前用户，同时绑定请求的日志关联标识。

    owner_id 直接存于运行表，因此不必跨表检查归属。别人的运行也返回 404，避免通过 403 泄露它确实存在。
    绑定放在共享函数里，而不是服务端 Depends 中，因为桌面端会直接调用这些处理函数；否则桌面日志会缺少关联标识。未解析具体运行的请求不绑定运行标识。"""
    run = (
        await db.execute(
            select(ResearchRun).where(ResearchRun.id == run_id, ResearchRun.owner_id == owner_id)
        )
    ).scalar_one_or_none()
    if run is None:
        raise NotFound("Run not found.")
    bind_research_run_context(str(run.id), user_id=str(owner_id))
    return run


async def project_run(db: AsyncSession, run: ResearchRun) -> dict:
    """返回完整研究视图；服务端和桌面端共用同一种返回结构。"""
    plans = (
        (
            await db.execute(
                select(ResearchPlan)
                .where(ResearchPlan.run_id == run.id)
                .order_by(ResearchPlan.version.asc())
            )
        )
        .scalars()
        .all()
    )
    sources = (
        (
            await db.execute(
                select(Source)
                .where(Source.run_id == run.id)
                # 未引用来源排在最后，但仍展示，便于用户区分“检索到”和“实际引用”。
                .order_by(Source.citation_index.is_(None), Source.citation_index.asc())
            )
        )
        .scalars()
        .all()
    )
    evidence = (
        (
            await db.execute(
                select(Evidence)
                .where(Evidence.run_id == run.id)
                .order_by(Evidence.sequence.asc(), Evidence.id.asc())
            )
        )
        .scalars()
        .all()
    )
    revisions = (
        (
            await db.execute(
                select(Revision).where(Revision.run_id == run.id).order_by(Revision.version.asc())
            )
        )
        .scalars()
        .all()
    )
    claims = (
        (
            await db.execute(
                select(Claim).where(Claim.run_id == run.id).order_by(Claim.position.asc())
            )
        )
        .scalars()
        .all()
    )
    links = (
        (await db.execute(select(ClaimEvidenceLink).where(ClaimEvidenceLink.run_id == run.id)))
        .scalars()
        .all()
    )
    contradictions = (
        (await db.execute(select(Contradiction).where(Contradiction.run_id == run.id)))
        .scalars()
        .all()
    )
    reviews = (
        (
            await db.execute(
                select(Review).where(Review.run_id == run.id).order_by(Review.sequence.asc())
            )
        )
        .scalars()
        .all()
    )
    artifact = (
        await db.execute(select(ResearchArtifact).where(ResearchArtifact.run_id == run.id))
    ).scalar_one_or_none()

    return {
        "run": {
            "id": str(run.id),
            "project_id": str(run.project_id),
            "question": run.question,
            "status": run.status,
            "depth": run.depth,
            "corpus_mode": run.corpus_mode,
            # 记录本次允许读取的范围及实际打开的语料库状态。创建和开始执行时分别保存。
            # 网页研究及早期版本的运行可能为空；早期运行默认读取项目全部语料。
            # 参见 internal/rfcs/V3.1-corpus-scope.md §8.6。
            "corpus_selection": run.corpus_selection,
            "corpus_id": run.corpus_id,
            "corpus_version": run.corpus_version,
            "offline": run.offline,
            # 按所选文件研究时，保留逐条结论的证据核验结果：哪些通过、哪些被拒绝或修复、哪些无法测量。
            # 参见 RFC V3.1 §9.6。
            "grounding_report": run.grounding_report,
            "demo": run.demo,
            "skip_plan_gate": run.skip_plan_gate,
            "model_routing": run.model_routing,
            "cost_usd": float(run.cost_usd),
            "tokens_input": run.tokens_input,
            "tokens_output": run.tokens_output,
            "elapsed_seconds": float(run.elapsed_seconds) if run.elapsed_seconds else None,
            # NULL 表示未测量，不能显示成 0；未知和实测为零是两回事。
            "citation_resolution_rate": (
                float(run.citation_resolution_rate)
                if run.citation_resolution_rate is not None
                else None
            ),
            "error_message": run.error_message,
            "cancelled_at": run.cancelled_at.isoformat() if run.cancelled_at else None,
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat(),
        },
        "plans": [
            {
                "id": str(p.id),
                "version": p.version,
                "tasks": p.tasks,
                "outline_sections": p.outline_sections,
                "origin": p.origin,
                "approved_at": p.approved_at.isoformat() if p.approved_at else None,
            }
            for p in plans
        ],
        "sources": [
            {
                "id": str(s.id),
                "url": s.url,
                "title": s.title,
                "kind": s.kind,
                "retrieval_status": s.retrieval_status,
                # None 表示检索过但未引用，前端不能自行补编号。
                "citation_index": s.citation_index,
                "corpus_document_id": s.corpus_document_id,
            }
            for s in sources
        ],
        "evidence": [
            {
                "id": str(e.id),
                "source_id": str(e.source_id),
                "sequence": e.sequence,
                "task_id": e.task_id,
                "snippet": e.snippet,
                "content_hash": e.content_hash,
                "key_fact": e.key_fact,
                # 保留原始三态：UNCHECKED（未检查）不同于 UNATTESTED（未证实）。
                "provenance_state": e.provenance_state,
                "attested_against": e.attested_against,
                "attestation_run_at": (
                    e.attestation_run_at.isoformat() if e.attestation_run_at else None
                ),
            }
            for e in evidence
        ],
        "revisions": [
            {
                "id": str(r.id),
                "version": r.version,
                "report_markdown": r.report_markdown,
                "report_hash": r.report_hash,
                # 结构化视图是补充字段；旧修订可以为空。
                # 既有客户端和前端引用渲染仍读取 report_markdown，不依赖这个新视图。
                "report_document": r.report_document,
                "evidence_watermark": r.evidence_watermark,
                # 这里只统计当前修订正文实际引用的来源，不根据 citation_index 推断。
                # 与引用解析率使用相同边界：参考文献列表提到来源，不等于正文引用了它。
                "cited_indices": sorted(
                    set(
                        citation_rate.cited_indices(
                            citation_rate.body_before_sources(r.report_markdown)
                        )
                    )
                ),
                "created_at": r.created_at.isoformat(),
            }
            for r in revisions
        ],
        "claims": [
            {
                "id": str(c.id),
                "revision_id": str(c.revision_id),
                "position": c.position,
                "text": c.text,
                "extraction_method": c.extraction_method,
                "verification_state": c.verification_state,
                "verification_method": c.verification_method,
                "lineage_id": str(c.lineage_id) if c.lineage_id else None,
            }
            for c in claims
        ],
        "claim_evidence_links": [
            {
                "id": str(link.id),
                "claim_id": str(link.claim_id),
                "evidence_id": str(link.evidence_id),
                "stance": link.stance,
                "origin": link.origin,
            }
            for link in links
        ],
        "contradictions": [
            {
                "id": str(c.id),
                # 来源锚点决定这是不是有效冲突；证据锚点只是进一步定位，允许为空。
                "source_a_id": str(c.source_a_id) if c.source_a_id else None,
                "source_b_id": str(c.source_b_id) if c.source_b_id else None,
                "evidence_a_id": str(c.evidence_a_id) if c.evidence_a_id else None,
                "evidence_b_id": str(c.evidence_b_id) if c.evidence_b_id else None,
                "quote_a": c.quote_a,
                "quote_b": c.quote_b,
                "summary_a": c.summary_a,
                "summary_b": c.summary_b,
                "nature": c.nature,
                "dimension": c.dimension,
                "detection_state": c.detection_state,
                "review_state": c.review_state,
            }
            for c in contradictions
        ],
        "reviews": [
            {
                "id": str(r.id),
                "sequence": r.sequence,
                "gate": r.gate,
                "decision": r.decision,
                "revision_id": str(r.revision_id) if r.revision_id else None,
                "plan_version_id": str(r.plan_version_id) if r.plan_version_id else None,
                "feedback": r.feedback,
                "reviewed_hash": r.reviewed_hash,
                "created_at": r.created_at.isoformat(),
            }
            for r in reviews
        ],
        "artifact": (
            {
                "id": str(artifact.id),
                "artifact_hash": artifact.artifact_hash,
                "format_version": artifact.format_version,
                "review_id": str(artifact.review_id) if artifact.review_id else None,
                "review_gate": artifact.review_gate,
                "review_decision": artifact.review_decision,
                "revision_id": str(artifact.revision_id) if artifact.revision_id else None,
                "demo": artifact.demo,
                "created_at": artifact.created_at.isoformat(),
            }
            if artifact
            else None
        ),
    }


def grounding_refusal(run: ResearchRun, report_text: str) -> str | None:
    """根据已保存记录，判断所选文件报告为何不能获批（RFC V3.1 §9.6）。

    引擎已经阻止未经校验的草稿进入审批；接口也执行同一限制，避免客户端绕过它。"""
    if run.corpus_selection is None:
        return None
    report = run.grounding_report or {}
    if report.get("status") != "complete" or report.get("claims_unmeasured"):
        return "This report's claims were not all checked against your selected files."
    digest = hashlib.sha256((report_text or "").encode("utf-8")).hexdigest()
    if digest != report.get("validated_text_sha256"):
        return "This report differs from the text that was checked against your selected files."
    return None


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_run(
    body: CreateRunRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dispatcher: RunDispatcher = Depends(get_run_dispatcher),
    corpus: CorpusLocator = Depends(get_corpus_locator),
):
    """打开一次研究运行。

    仅创建领域对象行。执行仍沿用现有流水线——这个里程碑修改的是运行结果
    *持久化的位置*，而不是研究本身的执行方式，因此这里故意不额外引入第二个执行器。
    """
    project = (
        await db.execute(
            select(Project).where(Project.id == body.project_id, Project.user_id == current_user.id)
        )
    ).scalar_one_or_none()
    if project is None:
        raise NotFound("Project not found.")
    if body.depth not in RESEARCH_DEPTHS:
        raise Unprocessable("Unknown depth.")

    # 在这里拒绝而不是修复：只有通过 `validate` 校验后才允许启动，
    # 因此一条运行不能在模型根本不可路由的情况下半途中失败。创建时为快照打上
    # 标记是为了让选择持久化——`run_config_for_run` 会将存在的 `model_routing` 视为
    # 权威来源，因此每个角色、每次恢复，以及报告自身的模型归属都会读取该请求命名的模型。
    routing = None
    if body.model_routing is not None:
        try:
            routing = model_routing.validate(body.model_routing)
        except model_routing.InvalidRouting as exc:
            raise Unprocessable(str(exc)) from exc

    # 仅语料库运行会读取一组冻结的文档版本，并在这里解析后永远不再重新解析
    # （internal/rfcs/V3.1-corpus-scope.md §8.2）。在记录行创建之前就拒绝：
    # 空的或无法解析的选择绝不能启动，也绝不能扩大到整个语料库。
    corpus_mode = body.corpus_mode or body.corpus_selection is not None or body.offline
    if body.corpus_selection is not None and "corpus_mode" in body.model_fields_set:
        if not body.corpus_mode:
            raise Unprocessable(
                "A file selection is corpus-only research; corpus_mode cannot be false."
            )
    selection = None
    if corpus_mode:
        requested = body.corpus_selection
        try:
            selection = await freeze_selection(
                await corpus.for_project(project.id),
                folder_ids=requested.folder_ids if requested else None,
                doc_keys=requested.doc_keys if requested else None,
                whole_corpus=requested is None,
            )
        except SelectionError as exc:
            if exc.code == "selection_not_found":
                raise NotFound(str(exc)) from exc
            raise Unprocessable(str(exc)) from exc

    run = await run_lifecycle.create_run(
        db,
        owner_id=current_user.id,
        project_id=project.id,
        question=body.question,
        depth=body.depth,
        corpus_mode=corpus_mode,
        corpus_selection=selection,
        offline=body.offline,
        skip_plan_gate=body.skip_plan_gate,
        topic_seeds=body.topic_seeds,
        outline_template=body.outline_template,
        model_routing=routing,
    )
    # 在分发前提交：如果 worker 最先拿到消息却找不到该行记录，
    # 那么本来即将创建的运行会失败。
    await db.commit()

    # 这是唯一一条不经过 `_run_or_404` 的运行路由——它创建运行而不是解析已有运行——
    # 因此会在这里绑定自己的身份，并且在分发前绑定，确保交接日志记录在该运行自身下。
    bind_research_run_context(str(run.id), user_id=str(current_user.id))

    if body.dispatch:
        await dispatcher.start(str(run.id), str(current_user.id))
    return {"run_id": str(run.id), "status": run.status, "dispatched": body.dispatch}


@router.get("")
async def list_runs(
    project_id: uuid.UUID | None = None,
    archived: bool = False,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """当前用户的运行记录，按创建时间倒序排列，可按项目单独筛选。

    这是一个摘要而非完整图谱：历史页面只展示列表，
    若每条记录都返回完整投影数据，会携带每份报告的 markdown 内容来绘制表格。
    """
    query = select(ResearchRun).where(ResearchRun.owner_id == current_user.id)
    if project_id is not None:
        query = query.where(ResearchRun.project_id == project_id)
    if archived:
        query = query.where(ResearchRun.archived_at.is_not(None))
    else:
        query = query.where(ResearchRun.archived_at.is_(None))
    runs = (
        (await db.execute(query.order_by(ResearchRun.created_at.desc()).limit(min(limit, 200))))
        .scalars()
        .all()
    )
    artifacts = {
        a.run_id
        for a in (
            await db.execute(
                select(ResearchArtifact).where(ResearchArtifact.owner_id == current_user.id)
            )
        )
        .scalars()
        .all()
    }
    return {
        "runs": [
            {
                "id": str(r.id),
                "project_id": str(r.project_id),
                "question": r.question,
                "status": r.status,
                "depth": r.depth,
                "demo": r.demo,
                "cost_usd": float(r.cost_usd),
                "citation_resolution_rate": (
                    float(r.citation_resolution_rate)
                    if r.citation_resolution_rate is not None
                    else None
                ),
                "has_artifact": r.id in artifacts,
                "archived_at": r.archived_at.isoformat() if r.archived_at else None,
                "created_at": r.created_at.isoformat(),
            }
            for r in runs
        ]
    }


@router.get("/{run_id}")
async def get_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """一次返回完整研究视图，避免多个页面分别请求同一批数据。"""
    return await project_run(db, await _run_or_404(db, run_id, current_user.id))


@router.post("/{run_id}/plan-review", status_code=status.HTTP_201_CREATED)
async def submit_plan_review(
    run_id: uuid.UUID,
    body: PlanReviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dispatcher: RunDispatcher = Depends(get_run_dispatcher),
):
    """审核最新计划版本；批准计划不等于批准报告，也不能授权生成最终产物。"""
    run = await _run_or_404(db, run_id, current_user.id)
    plan = (
        (
            await db.execute(
                select(ResearchPlan)
                .where(ResearchPlan.run_id == run.id)
                .order_by(ResearchPlan.version.desc())
            )
        )
        .scalars()
        .first()
    )
    if plan is None:
        raise Conflict("This run has no plan to review.")
    # None 表示未编辑、沿用提案；空列表 [] 则表示审核者排除了全部任务，应返回 422。
    proposed = plan.tasks or []
    tasks = body.tasks if body.tasks is not None else proposed
    kept = [t for t in tasks if t.get("include", True)]

    # 在保存审核决定前拒绝空计划，避免留下“已批准”但根本无法收集证据的记录。
    # 审核记录本身也是需要可信保存的证据。
    #
    # 这里与会话入口 research.submit_plan 共用同一限制和提示。
    # 此前全部任务 include=false 的计划也能获批，导致没有证据却生成报告。
    # 桌面端直接复用本处理函数，所以也受同一规则保护。
    if body.decision == "APPROVED" and not kept:
        raise Unprocessable(
            "Keep at least one task — a plan with nothing in it researches nothing."
        )

    # 人工编辑会新增计划版本，不覆盖模型提案，便于对照原建议与实际执行内容。
    # origin 标记来源；审核绑定实际批准的版本，而不是被替换的旧版本。
    if body.decision == "APPROVED" and body.tasks is not None and kept != proposed:
        plan = await run_lifecycle.record_plan(
            db,
            run,
            tasks=kept,
            outline_sections=plan.outline_sections,
            origin="HUMAN_EDITED",
        )

    review = await run_lifecycle.record_plan_review(
        db, run, plan, reviewer_id=current_user.id, decision=body.decision, feedback=body.feedback
    )
    if body.decision == "APPROVED" and body.dispatch:
        # 在同一事务中同时保存批准决定并将运行改为 RUNNING，然后才入队。
        # 否则客户端提交后立即查询，可能仍看到 AWAITING_PLAN，误以为任务没有继续，进而关闭事件流。
        await run_lifecycle.set_status(db, run, "RUNNING")
    await db.commit()

    # 通过 runner.resume(plan=…) 从计划关卡继续，复用已暂停的图，不重跑用户已付费的研究步骤。
    if body.decision == "APPROVED" and body.dispatch:
        # 交给执行器的是实际获批的 kept，而非未筛选的 plan.tasks。
        # 图内还会过滤 include=false，但这里也必须保证恢复参数与审核记录一致。
        await dispatcher.resume_plan(
            str(run.id),
            str(current_user.id),
            {"tasks": kept, "sections": plan.outline_sections},
        )
    return {"review_id": str(review.id), "gate": "PLAN", "decision": review.decision}


@router.post("/{run_id}/report-review", status_code=status.HTTP_201_CREATED)
async def submit_report_review(
    run_id: uuid.UUID,
    body: ReportReviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dispatcher: RunDispatcher = Depends(get_run_dispatcher),
):
    """审核报告修订。批准和冻结产物放在同一事务；如果无法生成可验证产物，就不能留下批准成功的记录。"""
    run = await _run_or_404(db, run_id, current_user.id)
    # 在事务与尽力执行的步骤之前读取状态，避免后续回滚使 ORM 对象失效。
    # was_completed 用来识别首次完成；重复批准沿用原行为，但不能重复增加完成计数。
    was_completed = run.status == "COMPLETED"
    spend = (run.cost_usd, run.tokens_input, run.tokens_output, run.model_routing)
    query = select(Revision).where(Revision.run_id == run.id)
    if body.revision_version is not None:
        query = query.where(Revision.version == body.revision_version)
    revision = (await db.execute(query.order_by(Revision.version.desc()))).scalars().first()
    if revision is None:
        raise Conflict("This run has no report to review.")
    if body.decision == "APPROVED":
        refused = grounding_refusal(run, revision.report_markdown)
        if refused:
            raise Conflict(refused)

    try:
        review = await run_lifecycle.record_report_review(
            db,
            run,
            revision,
            reviewer_id=current_user.id,
            decision=body.decision,
            feedback=body.feedback,
        )
        artifact_id = None
        if body.decision == "APPROVED":
            run.status = "COMPLETED"
            # 先测量再冻结产物，与批准处于同一事务，确保指标对应审核者批准的那份正文。
            await run_lifecycle.measure_citation_resolution(db, run, revision)
            artifact = await run_lifecycle.create_artifact(db, run)
            artifact_id = str(artifact.id)
        elif body.decision == "REWORK_REQUESTED":
            run.status = "RUNNING"
        await db.commit()
        if body.decision == "REWORK_REQUESTED" and body.dispatch:
            # 拒绝草稿后只回到 SYNTHESIZER，使用已有证据重新写作，不重跑整个检索执行流程。
            await dispatcher.rework(str(run.id), str(current_user.id), body.feedback)
    except run_lifecycle.LifecycleError as exc:
        await db.rollback()
        raise Conflict(str(exc)) from exc

    # 在索引写入之前从 ORM 对象取出响应所需数据。
    # 索引写入失败可能回滚并使全部对象失效；此后才读取 review 会触发异步上下文错误，让已经提交的批准请求返回 500。
    response = {
        "review_id": str(review.id),
        "gate": "REPORT",
        "decision": review.decision,
        "artifact_id": artifact_id,
    }

    if body.decision == "APPROVED" and not was_completed:
        # 运行在这里正式完成。
        # 报告批准后不会再以 approved=True 恢复引擎，所以 persist_outcome 只看到审核暂停，看不到最终完成；完成计数必须在此记录。
        cost_usd, tokens_input, tokens_output, model_routing = spend
        metrics.observe_completed_run(
            cost_usd=cost_usd,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model_routing=model_routing,
        )

    if body.decision == "APPROVED":
        # 两个索引都在批准事务提交后写入，不能让向量化费用或供应商故障影响已经批准的报告。
        # 这两次调用都自行处理异常，不向外抛出。
        await _ingest_report_into_corpus(db, run, revision.report_markdown)
        await _ingest_report_into_memory(db, run, revision.report_markdown)

    return response


async def _ingest_report_into_corpus(db: AsyncSession, run, report_markdown: str) -> None:
    """尽力把已批准报告保存进项目语料库；失败不向外抛出，原因见 report_corpus.ingest_report。"""
    try:
        from app import adapters
        from app.run_execution import provider_keys_for
        from app.services.report_corpus import ingest_report

        store = await adapters.ServerCorpusLocator().ensure(
            run.project_id, keys=await provider_keys_for(db, run.owner_id)
        )
        await ingest_report(store, report_id=str(run.id), report_markdown=report_markdown)
    except Exception as e:  # noqa: BLE001 — 保存失败的处理原则见 report_corpus.ingest_report
        logger.warning("report_corpus_ingest_setup_failed", run_id=str(run.id), error=str(e))


async def _ingest_report_into_memory(db: AsyncSession, run, report_markdown: str) -> None:
    """把获批报告加入项目记忆，供项目聊天检索。

    只有批准时才建立索引，以审核作为内容质量边界。向量化服务故障不能推翻已经成功提交的研究；缺口由 memory/status 披露，并在下次重建索引时修复。"""
    from app.services import memory

    if not memory.is_available(db):
        # 桌面端按设计不支持项目记忆，接口契约和界面都已说明；这里跳过即可，不应把正常差异当成故障报警。
        return
    # 在索引步骤之前读取运行属性。
    # 下面可能回滚，使 run 失效；之后再取属性会触发 MissingGreenlet，误把向量化故障变成批准接口的 500。
    run_id = str(run.id)
    try:
        from app import adapters
        from app.run_execution import provider_keys_for

        embedder = await adapters.embeddings_for(await provider_keys_for(db, run.owner_id))
        result = await memory.ingest_run(db, run, report_markdown, embedder)
        if result.skipped:
            logger.info("memory_ingest_skipped", run_id=run_id, reason=result.reason)
    except Exception as e:  # noqa: BLE001 — 按函数说明，不能让索引故障影响已提交的运行
        await db.rollback()
        logger.warning("memory_ingest_failed", run_id=run_id, error=str(e))


@router.get("/{run_id}/bundle.json")
async def get_bundle(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """返回可验证证据包；存在冻结产物时直接读取它。

    批准前实时组装，让用户检查将被批准的内容；批准后以产物 payload 为准，避免重新组装时混入后来变化的数据。"""
    run = await _run_or_404(db, run_id, current_user.id)
    artifact = (
        await db.execute(select(ResearchArtifact).where(ResearchArtifact.run_id == run.id))
    ).scalar_one_or_none()
    if artifact is not None:
        payload = artifact.payload
    else:
        manifest, reason = await run_bundle.assemble_with_reason(db, run.id)
        if manifest is None:
            raise Conflict(f"No bundle can be assembled for this run ({reason}).")
        payload = manifest.model_dump()

    filename = f"research-{str(run.id)[:8]}.bundle.json"
    return Response(
        content=json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{run_id}/verification")
async def get_verification(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """调用随项目发布的独立校验器，返回每项检查结果。

    不能只给一个布尔值：用户需要知道哪些检查通过。assembled=false 表示没有执行校验，不能当成校验失败。"""
    run = await _run_or_404(db, run_id, current_user.id)
    artifact = (
        await db.execute(select(ResearchArtifact).where(ResearchArtifact.run_id == run.id))
    ).scalar_one_or_none()
    if artifact is not None:
        from research_engine.bundle import BundleManifest

        manifest, reason = BundleManifest.model_validate(artifact.payload), None
    else:
        manifest, reason = await run_bundle.assemble_with_reason(db, run.id)
    if manifest is None:
        return {"assembled": False, "reason": reason, "passed": None, "checks": []}

    result = run_bundle.verify(manifest)
    return {
        "assembled": True,
        "reason": None,
        "passed": result.passed,
        "bundle_hash": manifest.bundle_hash,
        "frozen": artifact is not None,
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in result.checks],
        # 与独立校验器展示同一份来源记录，保证页面和离线校验对运行指令的解释一致。
        # 这里只返回哈希；完整提示词由证据包导出提供。
        "bundle_version": manifest.bundle_version,
        "prompt_overrides_status": result.prompt_overrides_status,
        "prompt_provenance": [
            {
                "purpose": r.purpose,
                "role": r.role,
                "policy": r.policy,
                "overridden": r.overridden,
                "effective_prompt_sha256": r.effective_prompt_sha256,
            }
            for r in result.prompt_provenance
        ],
    }


# 实时事件流（SSE）


async def _cited_store(db, run_id, document_id, current_user, corpus, also_in):
    run = await _run_or_404(db, run_id, current_user.id)
    urls = (
        await db.execute(select(Source.url).where(Source.run_id == run.id, Source.kind == "CORPUS"))
    ).scalars()
    store = await cited_documents.store_holding(
        corpus,
        document_id,
        cited=cited_documents.recorded_ids(urls),
        hashes=cited_documents.recorded_hashes(run.corpus_selection),
        corpus_id=run.corpus_id,
        projects=[run.project_id, *also_in],
    )
    if store is None:
        raise NotFound(corpus_ingest.DOCUMENT_NOT_FOUND)
    return store


@router.get("/{run_id}/cited-documents/{document_id}", response_model=ExtractedTextResponse)
async def get_cited_document_text(
    run_id: uuid.UUID,
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
    also_in: list[uuid.UUID] = Depends(cited_documents.no_other_projects),
):
    """读取本次引用的特定文档版本的提取正文，无论它现在存放在哪里。
    具体规则见 app/services/cited_documents.py 与 RFC V3.1 §7.3。"""
    store = await _cited_store(db, run_id, document_id, current_user, corpus, also_in)
    found = await store.extracted(document_id)
    if found is None:
        raise NotFound(corpus_ingest.DOCUMENT_NOT_FOUND)
    return ExtractedTextResponse(**found)


@router.get("/{run_id}/cited-documents/{document_id}/download")
async def download_cited_document(
    run_id: uuid.UUID,
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
    also_in: list[uuid.UUID] = Depends(cited_documents.no_other_projects),
):
    """按相同的版本与权限规则，返回本次引用文档的原始文件。"""
    store = await _cited_store(db, run_id, document_id, current_user, corpus, also_in)
    found = await store.blob(document_id)
    if found is None:
        raise NotFound(corpus_ingest.DOCUMENT_NOT_FOUND)
    data, filename, kind = found
    return Response(
        content=data, media_type=media_type_for(kind), headers=download_headers(kind, filename)
    )


@router.get("/{run_id}/stream")
async def stream_run(
    run_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """研究运行的实时事件流，复用会话流的通道与回放规则。

    agent_logs 保存历史，Redis 提供实时消息；worker 通过 adapters.agent_log_sink 写入，所以 agent_logs.session_id 允许指向会话或研究运行。重连依靠 Last-Event-ID 回放。
    实时流在 PLAN_READY、HITL_READY 及完成、失败事件处关闭，避免图等待人工操作时一直空等。"""
    run = await _run_or_404(db, run_id, current_user.id)
    last_event_id = request.headers.get("last-event-id")
    after_id = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0

    async def gen() -> AsyncGenerator[str]:
        # 延迟导入 adapters，避免桌面端导入共享路由时，连带加载服务端配置和 Redis。
        # 桌面并不挂载这个实时接口，不能因此被迫安装服务端依赖。
        from app import adapters

        channel = f"session:{run_id}:events"
        pubsub = redis.pubsub()
        # 先订阅再读取历史快照，确保两步之间发生的事件进入订阅队列，不会两边都漏掉。
        await pubsub.subscribe(channel)
        try:
            async with AsyncSessionLocal() as sdb:
                backlog = (
                    (
                        await sdb.execute(
                            select(AgentLog)
                            .where(AgentLog.session_id == run_id, AgentLog.id > after_id)
                            .order_by(AgentLog.id.asc())
                        )
                    )
                    .scalars()
                    .all()
                )

            # 监听循环统一由 app/services/event_stream.py 实现，供两个主机的运行与会话流复用。
            # 这里只负责各入口确实不同的部分：历史从哪里读、实时消息从哪里来。
            async for frame in sse_frames(
                connected={"type": "connected", "run_id": str(run_id)},
                backlog=[(row.id, row.payload) for row in backlog],
                live=adapters.redis_event_stream(pubsub),
                replay_stop=_REPLAY_STOP_EVENTS,
                terminal_stop=_TERMINAL_EVENTS,
                already_done=run.status in _SUSPENDED_STATUSES,
                seen_from=after_id,
            ):
                yield frame
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


# 报告导出


async def _latest_revision(db: AsyncSession, run: ResearchRun, version: int | None):
    query = select(Revision).where(Revision.run_id == run.id)
    if version is not None:
        query = query.where(Revision.version == version)
    revision = (await db.execute(query.order_by(Revision.version.desc()))).scalars().first()
    if revision is None:
        raise NotFound("This run has no report yet.")
    return revision


async def _source_dicts(db: AsyncSession, run_id) -> list[dict]:
    """生成导出报告所需的编号来源列表；未引用来源没有 [n]，必须排除，不能凭空编造编号。"""
    rows = (
        (
            await db.execute(
                select(Source)
                .where(Source.run_id == run_id, Source.citation_index.isnot(None))
                .order_by(Source.citation_index)
            )
        )
        .scalars()
        .all()
    )
    return [{"index": s.citation_index, "url": s.url, "title": s.title or ""} for s in rows]


@router.get("/{run_id}/export.md")
async def export_markdown(
    run_id: uuid.UUID,
    revision_version: int | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """输出已批准的正文，并追加两个主机共用的模型署名页脚。"""
    run = await _run_or_404(db, run_id, current_user.id)
    revision = await _latest_revision(db, run, revision_version)
    body = _demo_stamped(run, revision.report_markdown) + render_model_attribution_md(
        run.model_routing, _checker_accuracy(run)
    )
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="research-{str(run.id)[:8]}.md"'},
    )


@router.get("/{run_id}/export.pdf")
async def export_pdf(
    run_id: uuid.UUID,
    revision_version: int | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """复用 app.services.export.render_pdf，避免维护两套 PDF 生成流程。"""
    run = await _run_or_404(db, run_id, current_user.id)
    revision = await _latest_revision(db, run, revision_version)
    from app.services import export

    pdf = export.render_pdf(
        _demo_stamped(run, revision.report_markdown),
        await _source_dicts(db, run.id),
        title=run.question[:120],
        model_routing=run.model_routing,
        checker_accuracy=_checker_accuracy(run),
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="research-{str(run.id)[:8]}.pdf"'},
    )


def _checker_accuracy(run: ResearchRun) -> dict | None:
    """读取所选文件运行记录的 D-24 标记；其他运行没有该标记。"""
    return (run.grounding_report or {}).get("checker_accuracy")


def _demo_stamped(run: ResearchRun, report: str) -> str:
    """演示运行的文字导出需要演示标记，但证据包正文不能追加该标记。

    正文哈希必须与人工批准时一致，否则会破坏校验。证据包改用受哈希保护的 demo 字段说明演示身份，与会话导出保持一致。"""
    # 直接调用引擎提供的演示标记函数（tests/test_demo_stamp_parity.py）。
    # 此前经 app.api.v1.research 间接读取常量，会连带导入 Celery，使未打包 Celery 的桌面端导出时报错。
    # 延迟导入只能推迟这个错误，不能解决依赖问题；引擎本身不依赖主机包，因此可直接使用。
    return stamp_demo_md(report, demo=bool(run.demo))


# 取消运行


@router.post("/{run_id}/cancel")
async def cancel_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """持久化用户的取消请求，但不承诺执行立即停止。

    当前引擎不会在每个节点之间检查取消，在途运行可能继续到下一个暂停点。这里保证保存 CANCELLED、取消时间和操作者；界面不能声称所有工作已立即终止。引擎抢占式取消属于另一项改造。"""
    run = await _run_or_404(db, run_id, current_user.id)
    if run.status in ("COMPLETED", "FAILED", "CANCELLED"):
        raise Conflict(f"This run is already {run.status}.")
    await run_lifecycle.request_cancel(db, run, by=current_user.id)
    await db.commit()
    return {
        "status": run.status,
        "cancelled_at": run.cancelled_at.isoformat(),
        # 在返回数据中明确说明取消的边界，避免客户端作出超出实际能力的承诺。
        "advisory": True,
        "detail": (
            "Cancellation recorded. Work already in flight runs to its next checkpoint; "
            "no new research will be started for this run."
        ),
    }


# 归档、恢复与删除


@router.post("/{run_id}/archive")
async def archive_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """标记运行已归档；重复请求不会重复产生效果。"""
    run = await _run_or_404(db, run_id, current_user.id)
    await run_lifecycle.archive_run(db, run)
    await db.commit()
    return {
        "status": "ok",
        "archived": True,
        "archived_at": run.archived_at.isoformat() if run.archived_at else None,
    }


@router.post("/{run_id}/unarchive")
async def unarchive_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """恢复已归档运行；重复请求不会重复产生效果。"""
    run = await _run_or_404(db, run_id, current_user.id)
    await run_lifecycle.unarchive_run(db, run)
    await db.commit()
    return {
        "status": "ok",
        "archived": False,
        "archived_at": None,
    }


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """永久删除运行及关联记录。

    非终态运行返回 409，避免删除正在工作的任务。显式清理多态代理日志、产物与审核，再删除运行并级联清理依赖表；LangGraph 检查点通过 checkpoints.delete_thread 清理。"""
    run = await _run_or_404(db, run_id, current_user.id)
    try:
        await run_lifecycle.delete_run(db, run)
    except run_lifecycle.LifecycleError as exc:
        raise Conflict(str(exc)) from exc
    await db.commit()

    from app.services import checkpoints

    try:
        await checkpoints.delete_thread(str(run_id))
    except Exception as e:  # noqa: BLE001 — 用户记录已删除；检查点清理失败记日志，不推翻已完成的删除
        logger.warning("checkpoint_cleanup_failed", run_id=str(run_id), error=str(e))

    logger.info("run_deleted", run_id=str(run_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
