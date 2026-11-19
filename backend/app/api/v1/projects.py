"""项目管理接口（docs/14 §8）。

删除项目会删除其中研究。会话通过外键级联删除，但 LangGraph 检查点不受外键管理，必须显式清理，否则会残留包含网页正文的完整代理状态。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.dependencies import get_current_user
from app.models.agent_log import AgentLog
from app.models.project import Project
from app.models.research import ResearchRun
from app.models.review import ResearchArtifact, Review
from app.models.session import Session, SessionStatus
from app.models.user import User
from app.ports import CheckpointDeleter, CorpusLocator
from app.schemas.project import (
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdateRequest,
)
from app.services.host_ports import get_checkpoint_deleter, get_corpus_locator

logger = structlog.get_logger()
router = APIRouter(prefix="/projects", tags=["Projects"])

DEFAULT_PROJECT_NAME = "General"


async def _counts(db: AsyncSession, project_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    """为本页项目统计研究数量，同时计入会话与运行。"""
    if not project_ids:
        return {}
    norm_ids = [uuid.UUID(str(pid)) for pid in project_ids]
    v1_rows = (
        await db.execute(
            select(Session.project_id, func.count())
            .where(Session.project_id.in_(norm_ids))
            .group_by(Session.project_id)
        )
    ).all()
    run_rows = (
        await db.execute(
            select(ResearchRun.project_id, func.count())
            .where(ResearchRun.project_id.in_(norm_ids))
            .group_by(ResearchRun.project_id)
        )
    ).all()
    counts: dict[uuid.UUID, int] = {pid: 0 for pid in norm_ids}
    for pid, n in v1_rows:
        counts[uuid.UUID(str(pid))] = counts.get(uuid.UUID(str(pid)), 0) + n
    for pid, n in run_rows:
        counts[uuid.UUID(str(pid))] = counts.get(uuid.UUID(str(pid)), 0) + n
    return counts


def _to_response(project: Project, session_count: int = 0) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        archived_at=project.archived_at,
        created_at=project.created_at,
        session_count=session_count,
    )


async def get_or_create_default_project(db: AsyncSession, user_id: uuid.UUID) -> Project:
    """获取用户默认项目，缺少时创建。

    新账号无需先建项目就能研究；行为与历史数据迁移创建的 General 项目一致。"""
    existing = (
        await db.execute(
            select(Project)
            .where(
                Project.user_id == user_id,
                func.lower(Project.name) == DEFAULT_PROJECT_NAME.lower(),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing:
        return existing

    project = Project(user_id=user_id, name=DEFAULT_PROJECT_NAME)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


async def resolve_project(
    db: AsyncSession, user_id: uuid.UUID, project_id: uuid.UUID | None
) -> Project:
    """解析请求指定的项目；没有指定则使用用户默认项目。"""
    if project_id is None:
        return await get_or_create_default_project(db, user_id)
    project = (
        await db.execute(
            select(Project).where(Project.id == project_id, Project.user_id == user_id)
        )
    ).scalar_one_or_none()
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Project not found.")
    return project


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    archived: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        (
            await db.execute(
                select(Project)
                .where(
                    Project.user_id == current_user.id,
                    Project.archived_at.is_not(None) if archived else Project.archived_at.is_(None),
                )
                .order_by(Project.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    counts = await _counts(db, [p.id for p in rows])
    return ProjectListResponse(
        projects=[_to_response(p, counts.get(p.id, 0)) for p in rows],
        total=len(rows),
    )


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    payload: ProjectCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = Project(user_id=current_user.id, name=payload.name, description=payload.description)
    db.add(project)
    try:
        await db.commit()
    except IntegrityError:
        # 唯一索引不区分大小写；在数据库写入处处理冲突，才能避免先查询再插入的并发竞态。
        await db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"You already have a project named '{payload.name}'.",
        ) from None
    await db.refresh(project)
    logger.info("project_created", project_id=str(project.id))
    return _to_response(project)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    payload: ProjectUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = await resolve_project(db, current_user.id, project_id)

    if payload.name is not None:
        project.name = payload.name
    if payload.description is not None:
        project.description = payload.description
    if payload.archived is not None:
        project.archived_at = datetime.now(UTC) if payload.archived else None

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="You already have a project with that name."
        ) from None
    await db.refresh(project)
    counts = await _counts(db, [project.id])
    return _to_response(project, counts.get(project.id, 0))


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    corpus: CorpusLocator = Depends(get_corpus_locator),
    delete_checkpoint: CheckpointDeleter = Depends(get_checkpoint_deleter),
):
    """删除项目及其中全部会话。

    仍有会话执行时拒绝删除，避免后台 Worker 正使用的数据突然消失，使故障难以解释。"""
    project = await resolve_project(db, current_user.id, project_id)

    running_v1 = (
        await db.execute(
            select(func.count())
            .select_from(Session)
            .where(Session.project_id == project.id, Session.status == SessionStatus.RUNNING)
        )
    ).scalar_one()
    running_runs = (
        await db.execute(
            select(func.count())
            .select_from(ResearchRun)
            .where(
                ResearchRun.project_id == project.id,
                ResearchRun.status.in_(("RUNNING", "PENDING", "AWAITING_PLAN", "AWAITING_REVIEW")),
            )
        )
    ).scalar_one()
    if running_v1 or running_runs:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"{running_v1 + running_runs} run(s)/session(s) in this project are still running.",
        )

    session_ids = (
        (await db.execute(select(Session.id).where(Session.project_id == project.id)))
        .scalars()
        .all()
    )

    v2_run_ids = (
        (await db.execute(select(ResearchRun.id).where(ResearchRun.project_id == project.id)))
        .scalars()
        .all()
    )
    if v2_run_ids:
        # 多态代理日志没有外键，需要显式删除。
        await db.execute(delete(AgentLog).where(AgentLog.session_id.in_(v2_run_ids)))
        # 研究最终产物
        await db.execute(delete(ResearchArtifact).where(ResearchArtifact.run_id.in_(v2_run_ids)))
        # 先删除审核记录，其指向研究运行的外键为 RESTRICT。
        await db.execute(delete(Review).where(Review.run_id.in_(v2_run_ids)))
        # 研究运行记录
        await db.execute(delete(ResearchRun).where(ResearchRun.project_id == project.id))

    await db.delete(project)  # 级联删除会话及其日志、聊天和审计。
    await db.commit()

    for sid in session_ids:
        try:
            await delete_checkpoint(str(sid))
        except Exception as e:  # noqa: BLE001 — 用户记录已删除，清理异常只记录日志后继续
            logger.warning("checkpoint_cleanup_failed", session_id=str(sid), error=str(e))

    # 项目语料是独立 SQLite 文件，由定位器负责删除，不在路由里拼路径（docs/12 M10）。
    # 若只删业务行，文件及向量会永久成为无法访问的残留；还需清理 SQLite 辅助文件，不能遗漏 WAL。
    # 桌面使用共用语料库，其定位器按设计不执行项目级文件删除。
    corpus.delete(project_id)

    logger.info("project_deleted", project_id=str(project_id), sessions=len(session_ids))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
