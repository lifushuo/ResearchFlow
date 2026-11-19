"""连接研究引擎与领域持久化的适配层，不是另一套研究流程。

与会话 Worker 一样，通过 runner.run/resume 调用相同的图、端口、配置和检查点。依赖只从本模块指向引擎，引擎不感知业务数据库。

执行结果来自两处：RunOutcome 提供报告、编号来源和用量，LangGraph 检查点提供证据与冲突。通过三态读取器读取一次，再在事务中保存；此后 evidence 表是业务依据，检查点仅保留执行历史。

检查点读失败意味着证据数量未知，不是零条。persist_outcome 保存读取状态，不能生成无法解析引用的修订。它只依赖 db、run、outcome、state，不依赖 Celery、Redis 或 HTTP，便于直接验证持久化行为。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import metrics, run_lifecycle
from app.logconfig import bind_research_run_context
from app.models.research import ResearchRun
from app.models.user import User
from app.runtime import run_config_from_settings
from app.services import corpus_selection, model_routing
from app.services.run_config import (
    OVERRIDES_UNUSABLE,
    apply_demo_rule,
    preference_overrides,
    snapshot_overrides,
    usable_overrides,
)
from research_engine import citation_rate, events
from research_engine.checkpoint_read import CheckpointOutcome, read_checkpoint
from research_engine.prompt_composition import recording_provenance
from research_engine.runconfig import RunConfig
from research_engine.runner import RunOutcome

logger = structlog.get_logger()

# 与会话 Worker 共用用户密钥解密和回退路径，不再复制实现。
# 用户偏好契约位于 app/services/run_config.py；桌面端可导入服务层，不能依赖 app.workers 基础设施层。
from app.workers.pipeline_runner import _user_provider_keys  # noqa: E402

__all__ = [
    "PersistResult",
    "persist_outcome",
    "run_config_for_run",
    "provider_keys_for",
]


@dataclass
class PersistResult:
    """记录本次适配器写入和拒绝写入的内容。

    evidence_outcome 明确证据读取状态，供日志和验证使用；没有生成报告时 revision 为 None，例如执行失败或只到达计划关卡。"""

    status: str
    evidence_outcome: str
    source_count: int = 0
    evidence_count: int = 0
    revision_version: int | None = None
    claim_count: int = 0
    link_count: int = 0
    contradiction_count: int = 0


async def provider_keys_for(db: AsyncSession, owner_id: uuid.UUID) -> dict[str, str]:
    """获取本运行使用的用户自带密钥；与会话 Worker 共用解密和回退规则。"""
    return await _user_provider_keys(db, str(owner_id))


async def freeze_prompt_overrides(db: AsyncSession, run: ResearchRun) -> None:
    """仅在首次开始时冻结本运行的提示词覆盖配置。

    与写入 RUNNING 同一事务保存，避免出现已启动却没有配置快照的运行。是否首次开始由调用参数决定，不能看字段是否为 NULL，因为旧版运行也可能为 NULL；恢复旧任务时重新解析会改变后半段使用的指令。"""
    user = (await db.execute(select(User).where(User.id == run.owner_id))).scalar_one_or_none()
    frozen, status = snapshot_overrides(user)
    # 空配置统一保存为 None，不混用 {}；具体状态已有单独字段说明，避免调用方根据两种空值作出不同判断。
    run.effective_prompt_overrides = frozen or None
    run.prompt_overrides_status = status
    await db.flush()


def prompt_overrides_for_run(run: ResearchRun) -> tuple[dict[str, str], bool]:
    """只从运行记录读取冻结的提示词覆盖配置，两个主机遵循同一规则。

    恢复时不能重新读取用户当前偏好，否则暂停期间的设置修改会改变同一次研究的指令。快照不可用时全部退回项目默认提示词，不部分应用，也不重新解析偏好。
    返回覆盖配置和是否需要标记快照不可用；写事务由调用方负责，原快照内容始终保留以便排查。"""
    overrides, status = usable_overrides(run.effective_prompt_overrides)
    stamp = status == OVERRIDES_UNUSABLE and run.prompt_overrides_status != OVERRIDES_UNUSABLE
    return overrides, stamp


async def run_config_for_run(db: AsyncSession, run: ResearchRun) -> RunConfig:
    """构建本次研究的引擎配置，优先级为运行记录、用户设置、部署默认值。

    模型路由一旦在运行中冻结就始终优先使用，确保恢复前后由同一套模型完成报告。"""
    user = (await db.execute(select(User).where(User.id == run.owner_id))).scalar_one_or_none()

    routing = model_routing.resolve(
        session_routing=run.model_routing,
        user_routing=(user.model_routing if user else None),
    )
    if run.model_routing != routing:
        run.model_routing = routing
        await db.flush()

    base = run_config_from_settings()
    overrides = preference_overrides(user)

    # 从运行记录读取，不重新读取 user 偏好，规则见 prompt_overrides_for_run。
    # 桌面 sidecar._drive_run 也调用同一函数。
    prompt_overrides, unusable_stamp = prompt_overrides_for_run(run)
    if unusable_stamp:
        run.prompt_overrides_status = OVERRIDES_UNUSABLE
        await db.flush()

    overrides |= {"prompt_overrides": prompt_overrides, **row_overrides(run)}
    # 演示模式判断与会话和桌面运行共用 app/services/run_config.py 的实现，避免多处分支不一致。
    config, needs_stamp = apply_demo_rule(
        base, row_demo=bool(run.demo), host_is_scripted=base.llm_mode == "fake"
    )
    if needs_stamp:
        run.demo = True
        await db.flush()
    return replace(config, models=routing, **overrides)


async def persist_outcome(
    db: AsyncSession,
    run: ResearchRun,
    outcome: RunOutcome,
    *,
    saver=None,
    state: dict | None = None,
    prompt_provenance: dict[str, dict] | None = None,
) -> PersistResult:
    """保存一次图调用的结果；事务由调用方提交。

    可直接传入检查点 state，或传 saver 让本函数通过三态读取器获取。两者都没有时记为证据未知，不当成零。
    prompt_provenance 记录这次实际组成的提示词，需与前次合并：人工关卡前后可能分别组成规划和综合提示词，证据包必须保留两段。省略参数时不清空旧记录。"""
    # 必须在所有提前返回分支之前保存实际提示词来源。
    # 它描述真正执行过的内容，不受最终成功、失败或取消状态影响；否则停在计划关卡时会漏掉规划提示词。
    #
    # 已有用途的提示词保留首次执行值，与记录器首次写入优先的规则一致。
    if prompt_provenance:
        # 合并时先放本次记录，再用已有记录覆盖同名用途，保留最早实际执行的提示词。
        # 恢复不应重新组成旧用途；即使发生，也以原段执行时的内容为依据。
        run.effective_prompt_provenance = {
            **prompt_provenance,
            **(run.effective_prompt_provenance or {}),
        }

    # 用户取消的运行必须保持取消。当前取消不抢占引擎，在途调用稍后仍会返回结果，不能因此改回其他状态。
    #
    # ck_run_cancelled 要求状态与取消时间一致；取消后强行改状态会使整个事务回滚，连真实费用也丢失。
    # 先做判断才能既保留用户决定，又保存已经发生的用量。
    #
    # 取消到引擎发现停止之间实际消耗的 Token 仍要记录。
    # 除此之外不更新结论数据；运行自己的结果不能推翻用户取消决定。
    #
    # 会话的另外两处实现是 pipeline_runner._persist_outcome 和 sidecar._apply_outcome；修改此规则时三处必须一致。
    if run.status == "CANCELLED":
        await run_lifecycle.record_metrics(
            db,
            run,
            cost_usd=outcome.cost_usd,
            tokens_input=outcome.tokens_input,
            tokens_output=outcome.tokens_output,
            elapsed_seconds=outcome.elapsed_seconds,
        )
        logger.info(
            "outcome_discarded_run_cancelled",
            run_id=str(run.id),
            outcome_status=outcome.status,
            cost_usd=round(outcome.cost_usd, 4),
        )
        return PersistResult(status="CANCELLED", evidence_outcome="NOT_READ")

    if outcome.status == "failed":
        await run_lifecycle.record_failure(db, run, outcome.error)
        return PersistResult(status="FAILED", evidence_outcome="NOT_READ")

    # 计划审核关卡
    if outcome.status == "awaiting_plan":
        # 这里只保存模型提案：origin 标明模型生成，approved_at 等到 PLAN 审核批准时才填写。
        await run_lifecycle.record_plan(
            db,
            run,
            tasks=outcome.plan_tasks,
            outline_sections=outcome.plan_outline,
            origin="MODEL_PROPOSED",
        )
        await run_lifecycle.record_metrics(
            db,
            run,
            cost_usd=outcome.cost_usd,
            tokens_input=outcome.tokens_input,
            tokens_output=outcome.tokens_output,
        )
        await run_lifecycle.set_status(db, run, "AWAITING_PLAN")
        return PersistResult(status="AWAITING_PLAN", evidence_outcome="NOT_READ")

    # 从检查点读取证据，并保留三态读取结果
    evidence_outcome = "NOT_READ"
    values: dict = {}
    if state is not None:
        evidence_outcome, values = "READ", state
    elif saver is not None:
        read = await read_checkpoint(saver, str(run.id))
        if read.outcome is CheckpointOutcome.READ:
            evidence_outcome, values = "READ", read.values
        elif read.outcome is CheckpointOutcome.MISSING:
            evidence_outcome = "CHECKPOINT_MISSING"
        else:
            evidence_outcome = "CHECKPOINT_UNREADABLE"

    written = None
    if evidence_outcome == "READ":
        evidence = [e for e in (values.get("evidence") or []) if isinstance(e, dict)]
        # 没有 source_url 的证据没有来源身份，不能凭空补造。
        # 引擎编号也会跳过它，因此过滤不会丢失可引用来源；但过滤行为必须计数和记录日志。
        usable = [e for e in evidence if (e.get("source_url") or "").strip()]
        if len(usable) != len(evidence):
            logger.warning(
                "evidence_without_source_url_skipped",
                run_id=str(run.id),
                skipped=len(evidence) - len(usable),
            )
        # 所选文件模式只允许保存冻结选择范围内的来源和证据。
        # ScopedCorpus 应已保证此约束；若仍出现越界，说明存在绕过端口的路径，应明确失败，不能静默丢弃越界证据（RFC V3.1 §8.5）。
        bad = corpus_selection.violations(
            run.corpus_selection,
            [e["source_url"] for e in usable]
            + [s["url"] for s in outcome.sources or [] if isinstance(s, dict) and s.get("url")],
        )
        if bad:
            logger.error("selection_violation", run_id=str(run.id), urls=bad[:20], count=len(bad))
            await run_lifecycle.record_failure(
                db,
                run,
                f"selection_violation: {len(bad)} source(s) outside this run's selected files "
                "reached its evidence. Nothing from this run was kept as evidence.",
            )
            return PersistResult(status="FAILED", evidence_outcome=evidence_outcome)
        written = await run_lifecycle.record_evidence(
            db, run, evidence=usable, numbered_sources=outcome.sources
        )
        # 每次综合都重新保存核验记录；返工后的再次核验替换本次状态。
        run.grounding_report = values.get("grounding_report")

    # 报告修订、结论与证据关联
    revision_version = claim_count = link_count = 0
    report = outcome.report
    if report:
        result = await run_lifecycle.record_revision(
            db, run, report_markdown=report, evidence_index=written
        )
        revision_version = result.revision.version
        claim_count, link_count = result.claim_count, result.link_count

    # 冲突记录
    contradiction_count = 0
    if evidence_outcome == "READ":
        pairs = [c for c in (values.get("contradictions") or []) if isinstance(c, dict)]
        # 即使没有冲突对，也记录 detector_ran=True：这里的非语料路径已运行检测器。
        # 若调用方明确知道检测被跳过，应传 False，不能把未检测当成未发现。
        contradiction_count = await run_lifecycle.record_contradictions(
            db, run, pairs=pairs, evidence_index=written
        )

    # 运行指标与状态
    rate = None
    if outcome.status == "completed" and report:
        # 报告定稿时只测量一次，与会话 Worker 共用规则和函数。没有可测对象就保持 NULL。
        rate = citation_rate.resolution_rate(report, outcome.sources)
    await run_lifecycle.record_metrics(
        db,
        run,
        cost_usd=outcome.cost_usd,
        tokens_input=outcome.tokens_input,
        tokens_output=outcome.tokens_output,
        elapsed_seconds=outcome.elapsed_seconds,
        citation_resolution_rate=rate,
    )

    # 证据是否成功读取必须持久化，不能只写日志。
    # 证据包导出需要据此区分“实测零条”和“数量未知”。
    run.evidence_outcome = evidence_outcome

    status = run_lifecycle.OUTCOME_STATUS.get(outcome.status, "RUNNING")
    if status == "COMPLETED" and not report:
        # 声称完成却没有报告属于无效结果，应明确失败，不能把空运行标成已完成。
        await run_lifecycle.record_failure(db, run, "the graph completed without a report")
        status = "FAILED"
    else:
        await run_lifecycle.set_status(db, run, status)

    return PersistResult(
        status=status,
        evidence_outcome=evidence_outcome,
        source_count=written.source_count if written else 0,
        evidence_count=written.evidence_count if written else 0,
        revision_version=revision_version or None,
        claim_count=claim_count,
        link_count=link_count,
        contradiction_count=contradiction_count,
    )


def observe_persisted(run: ResearchRun, outcome: RunOutcome, result: PersistResult) -> None:
    """在提交成功之后，记录 persist_outcome 负责的终态指标。

    Prometheus 计数没有事务，提交前增加会把随后回滚的运行也算进去，且无法撤销。所以由主机驱动在 db.commit() 后调用。
    最终完成发生在报告批准接口，不由这里记录；app/metrics.py::_PERSISTED_TERMINAL 限定了此边界。"""
    metrics.observe_persisted_run(
        result.status,
        cost_usd=outcome.cost_usd,
        tokens_input=outcome.tokens_input,
        tokens_output=outcome.tokens_output,
        model_routing=run.model_routing,
    )


def lifecycle_event(result: PersistResult) -> dict:
    """把持久化结果转成主机应发布的生命周期事件。

    复用引擎 make_event 和既有 PLAN_READY、HITL_READY、COMPLETED、FAILED，保证两个主机和会话流识别相同事件，不另造一套消息协议。"""
    if result.status == "AWAITING_PLAN":
        return events.make_event("PLAN_READY", data={"run_id": None})
    if result.status == "AWAITING_REVIEW":
        return events.make_event(
            "HITL_READY",
            data={
                "revision_version": result.revision_version,
                "source_count": result.source_count,
                "evidence_count": result.evidence_count,
                "contradiction_count": result.contradiction_count,
            },
        )
    if result.status == "FAILED":
        return events.make_event("FAILED", data={"reason": "see error_message"})
    # 取消使用 FAILED 事件关闭现有事件流，不新增 CANCELLED 消息类型。
    # 两个主机的停止列表没有 CANCELLED，贸然新增会让流一直等待；同时也不能落入默认分支，把取消误报成完成。
    if result.status == "CANCELLED":
        return events.make_event("FAILED", data={"reason": "Research stopped by user."})
    return events.make_event(
        "COMPLETED",
        data={
            "revision_version": result.revision_version,
            "claim_count": result.claim_count,
            # 事件也带上三态读取结果，便于界面区分没有证据和证据情况未知。
            "evidence_outcome": result.evidence_outcome,
        },
    )


# 服务端 Worker 驱动入口
#
# 上面的函数不依赖主机；下面负责服务端的运行锁、数据库会话、Postgres 检查点和事件写入。
# 基本流程与 pipeline_runner._execute 一致，区别是读取 ResearchRun，并交给 persist_outcome 保存。


async def execute_run(
    run_id: str,
    *,
    resume: tuple[bool, str | None] | None = None,
    plan: dict | None = None,
) -> None:
    """驱动一次研究运行，调用引擎并将结果保存到领域表。"""
    # 先绑定运行日志标识，再初始化 Redis、加锁和查记录；这样提前失败的日志也能关联到本次运行。
    bind_research_run_context(run_id)

    from app import adapters
    from app.config import settings
    from app.db.base import AsyncSessionLocal, engine
    from app.db.redis import (
        acquire_session_lock,
        close_redis_pool,
        init_redis_pool,
        release_session_lock,
    )
    from research_engine import runner

    lock_token = uuid.uuid4().hex  # 这句生成一个随机令牌。
    await init_redis_pool()
    try:
        # Celery 可能重投消息，因此要防止重复执行；与会话 Worker 使用相同的锁规则与键空间。
        if not await acquire_session_lock(
            run_id, lock_token, ttl=settings.celery_task_timeout_seconds + 60
        ):
            logger.warning("run_lock_busy", run_id=run_id)
            return
        try:
            async with AsyncSessionLocal() as db:
                run = (
                    await db.execute(select(ResearchRun).where(ResearchRun.id == uuid.UUID(run_id)))
                ).scalar_one_or_none()
                if run is None:
                    logger.error("run_not_found", run_id=run_id)
                    return

                # 取消后不得再启动该运行的新研究。
                # 恢复消息可能晚于取消到达，必须先检查，否则既会重跑，还会把 CANCELLED 改为 RUNNING 并违反数据库约束。
                #
                # 已取消就是用户要求的终态，直接返回即可，不抛异常，也不再发布新消息。
                if run.status == "CANCELLED":
                    logger.info("run_start_skipped_cancelled", run_id=run_id)
                    return

                await run_lifecycle.set_status(db, run, "RUNNING")
                # 与 RUNNING 同一事务冻结提示词，仅在首次开始时执行。
                # 恢复时 resume/plan 已设置，应继承原快照，不能替换。
                if resume is None and plan is None:
                    await freeze_prompt_overrides(db, run)
                await db.commit()

                sink = adapters.agent_log_sink(db, run_id)
                ports = {
                    "provider_keys": await provider_keys_for(db, run.owner_id),
                    "event_sink": sink,
                    "cache": adapters.RedisCache(),
                    "run_config": await run_config_for_run(db, run),
                }
                await db.commit()

                if run.corpus_mode:
                    guard = await _corpus_port(db, run, ports)
                    if guard is not None:
                        await run_lifecycle.record_failure(db, run, guard)
                        await db.commit()
                        await sink(run_id, events.make_event("FAILED", data={"reason": guard}))
                        return

                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
                async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
                    await saver.setup()
                    # 记录器是同步上下文管理器，必须使用普通 with，不能放进 async with，否则图启动前就会报错。
                    # 它只包围图执行；退出时移除记录器，但已收集的数据仍可在外面使用。
                    with recording_provenance() as composed:
                        if plan is not None:
                            outcome = await runner.resume(
                                checkpointer=saver, session_id=run_id, plan=plan, **ports
                            )
                        elif resume is None:
                            outcome = await runner.run(
                                checkpointer=saver,
                                session_id=run_id,
                                user_id=str(run.owner_id),
                                query=run.question,
                                depth=run.depth,
                                **ports,
                            )
                        else:
                            approved, feedback = resume
                            outcome = await runner.resume(
                                checkpointer=saver,
                                session_id=run_id,
                                approved=approved,
                                feedback=feedback,
                                **ports,
                            )
                    # run 是执行前加载的旧 ORM 对象，取消可能已发生在数据库中。
                    # 保存前必须重新读取 cancelled_at，否则覆盖 CANCELLED 会触发约束并回滚真实用量。
                    await db.refresh(run, ["status", "cancelled_at"])
                    # 在保存器上下文内读取状态，因为退出后数据库连接会关闭。
                    result = await persist_outcome(
                        db, run, outcome, saver=saver, prompt_provenance=composed
                    )

                # 先保存、提交，再发布；客户端收到完成事件后不能读到旧状态。
                await db.commit()
                observe_persisted(run, outcome, result)
                await sink(run_id, lifecycle_event(result))
                logger.info(
                    "run_persisted",
                    run_id=run_id,
                    status=result.status,
                    evidence_outcome=result.evidence_outcome,
                    evidence=result.evidence_count,
                    claims=result.claim_count,
                )
        finally:
            await release_session_lock(run_id, lock_token)
    finally:
        await close_redis_pool()
        await engine.dispose()


def row_overrides(run: ResearchRun) -> dict:
    """统一把运行行字段转换为 RunConfig，服务端和桌面共用。

    请求存入数据库还不够，必须继续传到引擎配置。只安装语料端口不能代替 corpus_mode；检索器和工具从 RunConfig 决定是否访问网页。"""
    return {
        "skip_plan_gate": bool(run.skip_plan_gate),
        "topic_seeds": tuple(run.topic_seeds or ()),
        "outline_template": run.outline_template,
        "corpus_mode": bool(run.corpus_mode),
        # 创建时冻结，之后只从运行行读取，不重新解析。
        "corpus_allowed_ids": selection_ids_for_run(run),
        "offline": bool(run.offline),
    }


def selection_ids_for_run(run: ResearchRun) -> frozenset[str] | None:
    """从冻结选择返回运行允许读取的文档版本，两端共用。

    没有保存选择时为 None，包括网页运行和旧语料运行；后者仍读取原来整个项目语料。"""
    return corpus_selection.allowed_ids(run.corpus_selection)


def corpus_port_for(store, config):
    """构建运行使用的语料端口，有文件选择时严格限定范围。

    两种主机共用，避免仅一端执行选择限制（RFC V3.1 §8.4）。"""
    from research_engine.corpus import ScopedCorpus

    if config.corpus_allowed_ids is None:
        return store
    return ScopedCorpus(store, config.corpus_allowed_ids)


async def record_corpus_snapshot(db: AsyncSession, run: ResearchRun, store) -> None:
    """记录运行即将读取的语料身份和版本，两端共用。

    这是打开时状态快照，不是锁；运行期间语料仍可变化，旧版本保持可读，已有证据仍解析到原字节。不能声称整个运行期间语料完全没变。"""
    run.corpus_id, run.corpus_version = await store.identity()
    await db.flush()


async def _corpus_port(db: AsyncSession, run: ResearchRun, ports: dict) -> str | None:
    """连接语料存储；不能继续时返回原因。

    要求本地向量化防外发，且语料数据库必须已存在；缺少数据库意味着文档尚未摄取。返回原因让调用方统一处理失败。"""
    from app import adapters

    embedder = await adapters.embeddings_for(ports["provider_keys"])
    if not (embedder.model_id.startswith("ollama:") or embedder.model_id.startswith("local:")):
        return f"Corpus mode requires a local embedder (Ollama), but got {embedder.model_id}."

    store = await adapters.ServerCorpusLocator().for_project(
        run.project_id, keys=ports["provider_keys"]
    )
    if store is None:
        return f"Corpus database not found for project {run.project_id}. Ingest documents first."
    ports["corpus"] = corpus_port_for(store, ports["run_config"])
    await record_corpus_snapshot(db, run, store)
    return None
