"""研究运行的持久化生命周期，由服务端和桌面端共同使用。

研究引擎负责规划、执行、证据检查和综合，RunOutcome 只传递执行结果；本模块把结果保存为运行、计划、来源、证据、报告修订、结论、冲突、审核和最终产物。这里不依赖 FastAPI、鉴权、Redis 或加密，避免两种主机各写一套规则。

四条约束：检索到不等于引用过，来源编号不能编造；逐条证据核验结果必须来自引擎，未检查就保留 UNCHECKED；有效冲突必须有两端来源锚点，证据定位只能采用唯一精确匹配；只有 APPROVED 的 REPORT 审核能授权最终产物，统一由 app.authorization 判断。

返工只新增修订，不覆盖旧版；Review.sequence 保存审核顺序。"""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import authorization
from app.models.agent_log import AgentLog
from app.models.memory_chunk import MemoryChunk
from app.models.research import Contradiction, Evidence, ResearchPlan, ResearchRun, Source
from app.models.review import AuditEvent, ResearchArtifact, Review
from app.models.revision import Claim, ClaimEvidenceLink, Revision
from app.services import memory
from research_engine import citation_rate, document
from research_engine import claims as claim_rules
from research_engine.graph import _norm_url

# 将引擎 RunOutcome.status 映射为 research_runs.status。
# awaiting_approval 对应 AWAITING_REVIEW；取消只能经过 request_cancel，不能从消息内容推断。
OUTCOME_STATUS = {
    "awaiting_plan": "AWAITING_PLAN",
    "awaiting_approval": "AWAITING_REVIEW",
    "completed": "COMPLETED",
    "failed": "FAILED",
}


class LifecycleError(Exception):
    """不允许的领域状态变更；在实际写入之前抛出。"""


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# 1. 创建与更新研究运行


async def create_run(
    db: AsyncSession,
    *,
    owner_id: uuid.UUID,
    project_id: uuid.UUID,
    question: str,
    depth: str = "balanced",
    corpus_mode: bool = False,
    demo: bool = False,
    skip_plan_gate: bool = True,
    topic_seeds: list | None = None,
    outline_template: str | None = None,
    model_routing: dict | None = None,
    corpus_selection: dict | None = None,
    offline: bool = False,
) -> ResearchRun:
    """创建运行；owner_id 直接存于本表，权限校验不必跨表查询。"""
    run = ResearchRun(
        id=uuid.uuid4(),
        project_id=project_id,
        owner_id=owner_id,
        question=question,
        status="PENDING",
        depth=depth,
        corpus_mode=corpus_mode,
        demo=demo,
        skip_plan_gate=skip_plan_gate,
        topic_seeds=topic_seeds,
        outline_template=outline_template,
        model_routing=model_routing,
        corpus_selection=corpus_selection,
        offline=offline,
        cost_usd=0,
        tokens_input=0,
        tokens_output=0,
    )
    db.add(run)
    await db.flush()
    return run


async def set_status(db: AsyncSession, run: ResearchRun, status: str) -> None:
    """更新运行状态，但不允许绕过取消规则。

    ck_run_cancelled 要求取消状态与 cancelled_at 一致，取消必须走 request_cancel。任务一旦取消也不能再离开终态。
    这里提前拒绝非法更新，给调用方清楚的错误，避免后台恢复任务写入 RUNNING 后才撞上数据库约束。"""
    if status == "CANCELLED":
        raise LifecycleError("use request_cancel(); CANCELLED must carry cancelled_at")
    if run.status == "CANCELLED":
        raise LifecycleError(f"run {run.id} was cancelled; it cannot move to {status}")
    run.status = status
    await db.flush()


async def request_cancel(db: AsyncSession, run: ResearchRun, *, by: uuid.UUID) -> None:
    """将取消决定持久化到数据库，不依赖会过期的缓存键。"""
    run.status = "CANCELLED"
    run.cancelled_at = datetime.now(UTC)
    run.cancel_requested_by = by
    await db.flush()


async def record_failure(db: AsyncSession, run: ResearchRun, error: str | None) -> None:
    """保存执行失败；若用户已取消，则保留取消状态，但仍记录错误原因。

    强行把 CANCELLED 改为 FAILED 会违反数据库约束，使错误处理本身再次失败。这个检查集中在唯一的失败写入口，保证多个调用方遵循同一规则。"""
    if run.status == "CANCELLED":
        run.error_message = error
        await db.flush()
        return
    run.status = "FAILED"
    run.error_message = error
    await db.flush()


async def record_metrics(
    db: AsyncSession,
    run: ResearchRun,
    *,
    cost_usd: float,
    tokens_input: int,
    tokens_output: int,
    elapsed_seconds: float | None = None,
    citation_resolution_rate: float | None = None,
) -> None:
    """保存用量与耗时；未测量的 citation_resolution_rate 保留 NULL，不能写成 0。"""
    run.cost_usd = cost_usd
    run.tokens_input = tokens_input
    run.tokens_output = tokens_output
    if elapsed_seconds is not None:
        run.elapsed_seconds = elapsed_seconds
    if citation_resolution_rate is not None:
        run.citation_resolution_rate = citation_resolution_rate
    await db.flush()


# 2. 研究计划


async def record_plan(
    db: AsyncSession,
    run: ResearchRun,
    *,
    tasks: list[dict],
    outline_sections: list,
    origin: str = "MODEL_PROPOSED",
) -> ResearchPlan:
    """新增计划版本，永不覆盖已有提案。

    origin 区分模型提案与人工编辑；UNKNOWN 仅兼容尚未记录来源的历史数据，新记录不能使用它。"""
    if origin == "UNKNOWN":
        raise LifecycleError("UNKNOWN is for imported plans only; a native run knows")
    version = (
        await db.execute(
            select(func.coalesce(func.max(ResearchPlan.version), 0)).where(
                ResearchPlan.run_id == run.id
            )
        )
    ).scalar_one() + 1
    plan = ResearchPlan(
        id=uuid.uuid4(),
        run_id=run.id,
        version=version,
        tasks=tasks,
        outline_sections=outline_sections,
        origin=origin,
    )
    db.add(plan)
    await db.flush()
    return plan


async def approve_plan(db: AsyncSession, plan: ResearchPlan) -> None:
    """标记计划已批准；uq_plan_approved 保证每次运行最多有一个获批计划。"""
    plan.approved_at = datetime.now(UTC)
    await db.flush()


# 3. 来源与证据


@dataclass
class EvidenceWrite:
    """保存写入结果与查找索引，后续据此关联结论和冲突。"""

    sources_by_url: dict[str, uuid.UUID]
    # 引用编号 citation_index 对应该来源第一条证据；正文 [n] 通过它解析。
    evidence_by_index: dict[int, uuid.UUID]
    # 用来源和引用原文查找所有匹配证据；冲突只能在唯一匹配时进一步定位。
    evidence_by_quote: dict[tuple[uuid.UUID, str], list[uuid.UUID]]
    watermark: int
    source_count: int
    evidence_count: int


async def record_evidence(
    db: AsyncSession,
    run: ResearchRun,
    *,
    evidence: list[dict],
    numbered_sources: list[dict] | None = None,
) -> EvidenceWrite:
    """保存执行器返回的证据与来源。

    numbered_sources 来自综合器的编号结果。证据中出现但未被编号的来源保留 citation_index=NULL，不能把检索到误当成引用过；没有 source_url 的证据缺少来源身份，拒绝保存。

    返工会再次提交已有状态，因此按来源、任务、片段和关键事实，将输入与旧行一一配对，复用原 ID、序号及核验结果，只插入剩余条目。必须一一对应，不能用“有任意匹配就跳过”，否则会误吞执行器实际返回的重复条目。"""
    index_by_url: dict[str, int] = {}
    title_by_url: dict[str, str] = {}
    for entry in numbered_sources or []:
        if not isinstance(entry, dict) or not entry.get("url"):
            continue
        norm = _norm_url(entry["url"])
        if isinstance(entry.get("index"), int):
            index_by_url[norm] = entry["index"]
        if entry.get("title"):
            title_by_url[norm] = entry["title"]

    sources_by_url: dict[str, uuid.UUID] = {}
    evidence_by_index: dict[int, uuid.UUID] = {}
    evidence_by_quote: dict[tuple[uuid.UUID, str], list[uuid.UUID]] = {}
    watermark = 0

    # 按身份索引尚未配对的旧证据，优先匹配较早的记录。
    unpaired: dict[tuple, list[tuple[uuid.UUID, int]]] = {}
    for eid, source_id, task_id, digest, key_fact, sequence in (
        await db.execute(
            select(
                Evidence.id,
                Evidence.source_id,
                Evidence.task_id,
                Evidence.content_hash,
                Evidence.key_fact,
                Evidence.sequence,
            )
            .where(Evidence.run_id == run.id)
            .order_by(Evidence.sequence.asc(), Evidence.id.asc())
        )
    ).all():
        unpaired.setdefault((source_id, task_id, digest, key_fact), []).append((eid, sequence))
    base = max((seq for rows in unpaired.values() for _, seq in rows), default=0)
    inserted = 0

    for offset, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            continue
        url = (item.get("source_url") or "").strip()
        if not url:
            raise LifecycleError(
                f"evidence[{offset}] has no source_url; a Source cannot be invented for it"
            )
        norm = _norm_url(url)
        source_id = sources_by_url.get(norm)
        if source_id is None:
            source_id = (
                await db.execute(
                    select(Source.id).where(Source.run_id == run.id, Source.normalized_url == norm)
                )
            ).scalar_one_or_none()
        if source_id is None:
            source_id = uuid.uuid4()
            db.add(
                Source(
                    id=source_id,
                    run_id=run.id,
                    url=url,
                    normalized_url=norm,
                    title=title_by_url.get(norm) or item.get("source_title") or None,
                    kind="CORPUS" if url.startswith("corpus://") else "WEB",
                    # 执行器已记录内容来自网页正文还是搜索摘要，这里读取实测来源，不自行假设。
                    retrieval_status=(
                        "CORPUS_DOCUMENT"
                        if url.startswith("corpus://")
                        else item.get("retrieval_status") or "FETCHED"
                    ),
                    citation_index=index_by_url.get(norm),
                    corpus_document_id=(
                        url[len("corpus://") :] if url.startswith("corpus://") else None
                    ),
                    retrieved_at=datetime.now(UTC),
                )
            )
            await db.flush()
        sources_by_url[norm] = source_id

        snippet = item.get("snippet") or ""
        task_id = str(item["task_id"]) if item.get("task_id") is not None else None
        key_fact = item.get("key_fact") or None
        stored = unpaired.get((source_id, task_id, content_hash(snippet), key_fact))
        if stored:
            eid, sequence = stored.pop(0)
        else:
            inserted += 1
            sequence = base + inserted
            # 沿用引擎的逐条核验结果，判定规则见 verify_evidence_snippets。
            # snippet_unverified 表示核验发现虚构片段并清空；attestation_grade 表示片段匹配真实工具返回，类型为 FETCHED_BODY 或 SEARCH_SNIPPET。
            # 两个标记都没有就表示未检查，保留 UNCHECKED；语料库来源也不能因此免检。
            grade = item.get("attestation_grade")
            if url.startswith("corpus://") and grade:
                grade = "CORPUS_DOCUMENT"

            if item.get("snippet_unverified"):
                provenance_state, attested_against, attestation_run_at = (
                    "UNATTESTED",
                    None,
                    datetime.now(UTC),
                )
            elif grade:
                provenance_state, attested_against, attestation_run_at = (
                    "ATTESTED",
                    grade,
                    datetime.now(UTC),
                )
            else:
                provenance_state, attested_against, attestation_run_at = "UNCHECKED", None, None

            eid = uuid.uuid4()
            db.add(
                Evidence(
                    id=eid,
                    run_id=run.id,
                    source_id=source_id,
                    sequence=sequence,
                    task_id=task_id,
                    snippet=snippet,
                    content_hash=content_hash(snippet),
                    key_fact=key_fact,
                    provenance_state=provenance_state,
                    attested_against=attested_against,
                    attestation_run_at=attestation_run_at,
                )
            )
        watermark = max(watermark, sequence)
        idx = index_by_url.get(norm)
        if idx is not None and idx not in evidence_by_index:
            evidence_by_index[idx] = eid
        evidence_by_quote.setdefault((source_id, snippet.strip()[:500]), []).append(eid)

    await db.flush()
    return EvidenceWrite(
        sources_by_url=sources_by_url,
        evidence_by_index=evidence_by_index,
        evidence_by_quote=evidence_by_quote,
        watermark=watermark,
        source_count=len(sources_by_url),
        evidence_count=len([e for e in evidence if isinstance(e, dict)]),
    )


# 4—6. 报告修订、结论及证据关联


@dataclass
class RevisionWrite:
    revision: Revision
    claim_count: int
    link_count: int


async def record_revision(
    db: AsyncSession,
    run: ResearchRun,
    *,
    report_markdown: str,
    evidence_index: EvidenceWrite | None = None,
) -> RevisionWrite:
    """新增不可变的报告修订，同时保存结论和证据关联。

    返工按已有记录分配下一个 version，旧版保持原样。evidence_watermark 是综合时可见的最大证据序号，不是条数，序号有空缺也不影响。"""
    if not report_markdown:
        raise LifecycleError("a revision must carry the report it is a version of")

    version = (
        await db.execute(
            select(func.coalesce(func.max(Revision.version), 0)).where(Revision.run_id == run.id)
        )
    ).scalar_one() + 1
    watermark = (
        evidence_index.watermark
        if evidence_index is not None
        else (
            await db.execute(
                select(func.coalesce(func.max(Evidence.sequence), 0)).where(
                    Evidence.run_id == run.id
                )
            )
        ).scalar_one()
    )
    revision = Revision(
        id=uuid.uuid4(),
        run_id=run.id,
        version=version,
        report_markdown=report_markdown,
        # 哈希始终基于 Markdown 原文。审核哈希、产物哈希与证据包批准链都绑定这份字节。
        # 若改为对渲染结果算哈希，解析器变化就会使既有批准失效。
        report_hash=content_hash(report_markdown),
        # 结构化视图也从同一份原文解析；generated_at 由调用方传入，不在解析器里读时钟，保证同样输入得到同样文档。
        report_document=document.parse_markdown(
            report_markdown,
            metadata=document.ReportMetadata(
                run_id=str(run.id),
                revision_version=version,
                question=run.question,
                generated_at=datetime.now(UTC).isoformat(),
            ),
        ).model_dump(mode="json"),
        evidence_watermark=watermark,
    )
    db.add(revision)
    await db.flush()

    by_index = evidence_index.evidence_by_index if evidence_index else {}
    if not by_index:
        by_index = await _evidence_by_citation_index(db, run.id)

    claim_count = link_count = 0
    for position, text in enumerate(claim_rules.claim_lines(report_markdown)):
        claim = Claim(
            id=uuid.uuid4(),
            revision_id=revision.id,
            run_id=run.id,
            position=position,
            text=text,
            # 综合器输出的是文字，结论由文本提取；只有未来真正输出结构化结论时才能标 MODEL_STRUCTURED。
            extraction_method="DERIVED_FROM_REPORT",
            verification_state="UNCHECKED",
            verification_method="NOT_RUN",
            # 不能仅因文字相同，就假定当前结论与上一版是同一个断言；未观测到的关联不编造。
            lineage_id=None,
        )
        db.add(claim)
        claim_count += 1
        for marker in dict.fromkeys(claim_rules.extract_citations(text)):
            target = by_index.get(marker)
            if target is None:
                continue  # 引用编号无法解析到来源时，就不建立证据关联。
            db.add(
                ClaimEvidenceLink(
                    id=uuid.uuid4(),
                    run_id=run.id,
                    claim_id=claim.id,
                    evidence_id=target,
                    stance="SUPPORTS",
                    origin="CITATION_MARKER",
                )
            )
            link_count += 1
    await db.flush()
    return RevisionWrite(revision=revision, claim_count=claim_count, link_count=link_count)


async def _evidence_by_citation_index(db: AsyncSession, run_id) -> dict[int, uuid.UUID]:
    """从数据库重建“引用编号→第一条证据”的索引。

    返工的修订可能晚于证据事务写入，原内存索引已不存在。通过 IS NOT NULL 排除未引用来源。"""
    rows = (
        await db.execute(
            select(Source.citation_index, Evidence.id)
            .join(Evidence, Evidence.source_id == Source.id)
            .where(Source.run_id == run_id, Source.citation_index.isnot(None))
            .order_by(Evidence.sequence.asc(), Evidence.id.asc())
        )
    ).all()
    out: dict[int, uuid.UUID] = {}
    for index, evidence_id in rows:
        out.setdefault(index, evidence_id)
    return out


# 7. 证据冲突


async def record_contradictions(
    db: AsyncSession,
    run: ResearchRun,
    *,
    pairs: list[dict],
    evidence_index: EvidenceWrite | None = None,
    detector_ran: bool = True,
) -> int:
    """把检测器返回的成对引文保存为冲突。

    检测器没运行与确实没发现冲突是不同结论，detector_ran=False 时必须通过 detection_state 明确记录。"""
    if not detector_ran:
        db.add(
            Contradiction(
                id=uuid.uuid4(),
                run_id=run.id,
                detection_state="DETECTOR_UNAVAILABLE",
                review_state="UNREVIEWED",
            )
        )
        await db.flush()
        return 1

    by_url = dict(evidence_index.sources_by_url) if evidence_index else {}
    by_quote = dict(evidence_index.evidence_by_quote) if evidence_index else {}
    if not by_url:
        by_url = {
            row.normalized_url: row.id
            for row in (await db.execute(select(Source).where(Source.run_id == run.id))).scalars()
        }

    # 返工会重复提交冲突状态，因此也与旧记录一一配对，避免重复插入，并保留已有人工审核决定。
    unpaired = Counter(
        (
            await db.execute(
                select(
                    Contradiction.source_a_id,
                    Contradiction.source_b_id,
                    Contradiction.quote_a,
                    Contradiction.quote_b,
                    Contradiction.summary_a,
                    Contradiction.summary_b,
                    Contradiction.nature,
                ).where(
                    Contradiction.run_id == run.id,
                    Contradiction.detection_state != "DETECTOR_UNAVAILABLE",
                )
            )
        ).all()
    )

    written = 0
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        src_a = by_url.get(_norm_url(pair.get("source_a") or ""))
        src_b = by_url.get(_norm_url(pair.get("source_b") or ""))
        quote_a = pair.get("snippet_a") or None
        quote_b = pair.get("snippet_b") or None

        def refine(source_id, quote):
            """只有恰好一条证据携带该引用原文时，才返回证据定位。"""
            if source_id is None or not quote:
                return None
            hits = by_quote.get((source_id, quote.strip()[:500]), [])
            return hits[0] if len(hits) == 1 else None

        ev_a, ev_b = refine(src_a, quote_a), refine(src_b, quote_b)
        if ev_a is None or ev_b is None:
            ev_a = ev_b = None  # ck_contra_refine 要求成对定位；只找到一边不能算完整关联。
        detected = src_a is not None and src_b is not None and src_a != src_b
        summary_a, summary_b = pair.get("claim_a") or None, pair.get("claim_b") or None
        nature = pair.get("nature") or None

        written += 1
        key = (
            src_a if detected else None,
            src_b if detected else None,
            quote_a,
            quote_b,
            summary_a,
            summary_b,
            nature,
        )
        if unpaired[key]:
            unpaired[key] -= 1
            continue
        db.add(
            Contradiction(
                id=uuid.uuid4(),
                run_id=run.id,
                source_a_id=src_a if detected else None,
                source_b_id=src_b if detected else None,
                evidence_a_id=ev_a if detected else None,
                evidence_b_id=ev_b if detected else None,
                quote_a=quote_a,
                quote_b=quote_b,
                summary_a=summary_a,
                summary_b=summary_b,
                nature=nature,
                # 检测器尚未提供冲突维度，运行记录也不自行推断。
                dimension="UNCLASSIFIED",
                detection_state="DETECTED" if detected else "NOT_RUN",
                review_state="UNREVIEWED",
            )
        )
    await db.flush()
    return written


# 8. 人工审核


async def _next_sequence(db: AsyncSession, run_id) -> int:
    return (
        await db.execute(
            select(func.coalesce(func.max(Review.sequence), 0)).where(Review.run_id == run_id)
        )
    ).scalar_one() + 1


async def record_plan_review(
    db: AsyncSession,
    run: ResearchRun,
    plan: ResearchPlan,
    *,
    reviewer_id: uuid.UUID,
    decision: str = "APPROVED",
    feedback: str | None = None,
) -> Review:
    """保存某个计划版本的审核；revision_id 为空，不授权生成报告产物。"""
    review = Review(
        id=uuid.uuid4(),
        run_id=run.id,
        sequence=await _next_sequence(db, run.id),
        revision_id=None,
        plan_version_id=plan.id,
        reviewer_id=reviewer_id,
        gate="PLAN",
        decision=decision,
        feedback=feedback,
        # 这里计算计划的标准化序列化；会话路径将其当作不透明值。
        # 证据包校验器仍只绑定报告批准，不校验此计划内容。
        reviewed_hash=content_hash(f"{plan.version}|{plan.tasks}|{plan.outline_sections}"),
    )
    db.add(review)
    await db.flush()
    if decision == "APPROVED":
        await approve_plan(db, plan)
    await _audit(
        db, run, reviewer_id, "plan.approved" if decision == "APPROVED" else "plan.rejected"
    )
    return review


async def record_report_review(
    db: AsyncSession,
    run: ResearchRun,
    revision: Revision,
    *,
    reviewer_id: uuid.UUID,
    decision: str,
    feedback: str | None = None,
) -> Review:
    """保存某个报告修订的审核；哈希绑定审核者实际看到的原始字节。"""
    if decision not in ("APPROVED", "REWORK_REQUESTED", "REJECTED"):
        raise LifecycleError(f"unknown report decision {decision!r}")
    review = Review(
        id=uuid.uuid4(),
        run_id=run.id,
        sequence=await _next_sequence(db, run.id),
        revision_id=revision.id,
        plan_version_id=None,
        reviewer_id=reviewer_id,
        gate="REPORT",
        decision=decision,
        feedback=feedback,
        # revisions.report_markdown 不可变，因此审核始终绑定同一份正文，不会因下次返工而变化。
        reviewed_hash=revision.report_hash,
    )
    db.add(review)
    await db.flush()
    await _audit(
        db,
        run,
        reviewer_id,
        "review.approved" if decision == "APPROVED" else "review.rework_requested",
    )
    return review


async def _audit(db: AsyncSession, run: ResearchRun, actor_id, action: str) -> None:
    await db.execute(
        insert(AuditEvent).values(
            actor_id=actor_id,
            action=action,
            subject_type="research_run",
            subject_id=run.id,
            metadata_json={"native": True},
        )
    )


async def measure_citation_resolution(
    db: AsyncSession, run: ResearchRun, revision: Revision
) -> float | None:
    """测量获批报告的引用可解析比例；未测量时返回 NULL。

    报告在领域层批准后直接定稿，不再恢复引擎，因此不能依赖引擎 completed 分支来计算。测量必须发生在这里。
    统一使用 research_engine.citation_rate，避免页面、基准评测与 Judge 各算一套。没有引用意味着无可测对象，应保留 NULL，不能等同于全部引用失效。"""
    numbered = (
        (
            await db.execute(
                select(Source).where(Source.run_id == run.id, Source.citation_index.is_not(None))
            )
        )
        .scalars()
        .all()
    )
    rate = citation_rate.resolution_rate(
        revision.report_markdown, [{"index": s.citation_index} for s in numbered]
    )
    if rate is not None:
        run.citation_resolution_rate = rate
        await db.flush()
    return rate


# 9. 冻结最终产物


async def create_artifact(db: AsyncSession, run: ResearchRun) -> ResearchArtifact:
    """冻结获批记录；仅 APPROVED 的 REPORT 审核可进入。

    授权统一由 app.authorization 判断。保存的是冻结证据包，后续读取无需重新连接实时表来组装内容。"""
    review = await authorization.approving_report_review(db, run.id)
    if review is None:
        raise LifecycleError(
            "no APPROVED REPORT review for this run — a plan approval or a rework request "
            "does not authorize an artifact"
        )
    values = authorization.artifact_authorization_values(review)

    from app import run_bundle

    manifest = await run_bundle.assemble(db, run.id)
    if manifest is None:
        raise LifecycleError("the run has no assemblable bundle; refusing to freeze nothing")
    if manifest.report_hash != review.reviewed_hash:
        # 产物必须绑定人工批准的原文字节；不匹配说明证据包用了另一份修订，必须拒绝。
        raise LifecycleError(
            f"report_hash {manifest.report_hash[:12]}… does not match the approved "
            f"{review.reviewed_hash[:12]}…"
        )

    artifact = ResearchArtifact(
        id=uuid.uuid4(),
        owner_id=run.owner_id,
        run_id=run.id,
        project_id=run.project_id,
        revision_id=review.revision_id,
        format_version=manifest.bundle_version,
        payload=manifest.model_dump(),
        artifact_hash=manifest.bundle_hash,
        demo=run.demo,
        **values,
    )
    db.add(artifact)
    await db.flush()
    await _audit(db, run, review.reviewer_id, "artifact.created")
    return artifact


# 10. 归档、恢复和删除


async def archive_run(db: AsyncSession, run: ResearchRun) -> ResearchRun:
    """标记为已归档；重复操作结果不变。"""
    if run.archived_at is None:
        run.archived_at = datetime.now(UTC)
        await db.flush()
    return run


async def unarchive_run(db: AsyncSession, run: ResearchRun) -> ResearchRun:
    """将已归档运行恢复到活动历史列表；重复操作结果不变。"""
    if run.archived_at is not None:
        run.archived_at = None
        await db.flush()
    return run


async def delete_run(db: AsyncSession, run: ResearchRun) -> None:
    """永久删除运行及关联关系记录。

    PENDING、RUNNING、AWAITING_PLAN、AWAITING_REVIEW 等未终结状态不允许删除。审核与产物在同一事务删除。
    代理日志和项目记忆属于多态引用，没有外键级联，必须显式删除；删除运行后，其计划、来源、证据、修订、结论、批注、关联和冲突由外键级联清理。"""
    if run.status in ("PENDING", "RUNNING", "AWAITING_PLAN", "AWAITING_REVIEW"):
        raise LifecycleError(
            f"This run is still active ({run.status}). Cancel it or wait for it to finish before deleting."
        )

    # 1. 删除多态引用的代理日志 AgentLogs
    await db.execute(delete(AgentLog).where(AgentLog.session_id == run.id))

    # 同时显式删除项目记忆片段；它们也没有级联外键。
    # 否则运行删掉后，项目聊天仍可能引用已不存在的报告。桌面端没有记忆表，要先判断是否支持。
    if memory.is_available(db):
        await db.execute(delete(MemoryChunk).where(MemoryChunk.source_report_id == run.id))

    # 2. 删除本次运行的最终产物 ResearchArtifact
    await db.execute(delete(ResearchArtifact).where(ResearchArtifact.run_id == run.id))

    # 3. 删除审核 Review；其外键是 RESTRICT，必须先清理
    await db.execute(delete(Review).where(Review.run_id == run.id))

    # 4. 删除运行，级联清理计划、来源、证据、修订、结论、批注、关联和冲突
    await db.delete(run)
    await db.flush()
