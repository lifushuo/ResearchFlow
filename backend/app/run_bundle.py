"""从研究领域表组装可验证证据包的唯一入口。

由 run_lifecycle.create_artifact 冻结，结构与独立校验器沿用 research_engine。证据未成功从检查点恢复时拒绝组装，不能把未知数量写成零。
计划批准绝不能序列化为 approved；校验器将该动作视为报告授权，而数据库约束保护不到导出的 JSON。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.authorization import approval_chain
from app.models.agent_log import AgentLog
from app.models.research import Contradiction, Evidence, ResearchRun, Source
from app.models.revision import Revision
from app.services import corpus_selection
from research_engine import bundle as bundle_mod
from research_engine import verify_bundle

# 审核决定到证据包动作的唯一映射；必须一一对应，才能无歧义还原。
REVIEW_TO_BUNDLE_ACTION = {
    ("REPORT", "APPROVED"): "approved",
    ("REPORT", "REWORK_REQUESTED"): "rework_requested",
    # 旧会话没有 rejected 动作，但这里仍要为报告拒绝保留独立名称，不能破坏映射的一一对应。
    # 校验器只认可 approved，因此未知或拒绝动作都不能授权产物。
    ("REPORT", "REJECTED"): "rejected",
    ("PLAN", "APPROVED"): "plan_approved",
    ("PLAN", "REWORK_REQUESTED"): "plan_rework_requested",
    ("PLAN", "REJECTED"): "plan_rejected",
}


async def _trace(db: AsyncSession, run_id) -> list[dict]:
    """两个主机的 agent_logs 都按运行 ID 查询，兼容会话与研究运行。"""
    rows = (
        (
            await db.execute(
                select(AgentLog).where(AgentLog.session_id == run_id).order_by(AgentLog.id.asc())
            )
        )
        .scalars()
        .all()
    )
    return [row.payload for row in rows]


def _selection_trace(run: ResearchRun) -> list[dict]:
    """将冻结文件选择作为证据包首条轨迹；其他运行不添加。

    从数据库原样复制，不重新解析，供校验器检查每个语料来源是否在当时允许范围内。"""
    if not run.corpus_selection:
        return []
    entries = []
    report = run.grounding_report
    if report:
        # 只放数量、原因和状态；逐条结论列表由运行页面展示。
        entries.append(
            {"event": "grounding_validation", **{k: v for k, v in report.items() if k != "claims"}}
        )
    return [
        {
            "event": "corpus_selection",
            "resolved": run.corpus_selection.get("resolved") or [],
            "corpus_id": run.corpus_selection.get("corpus_id"),
            "corpus_version": run.corpus_selection.get("corpus_version"),
        }
    ] + entries


def _provenance_records(run: ResearchRun) -> list[dict] | None:
    """读取执行时保存的提示词来源，没有则返回 None。

    不能导出时重新调用 system_prompt，否则升级后的代码会伪造历史模型指令。按用途排序，确保重复组装得到相同字节。旧运行无法补采，继续使用 v1。"""
    stored = run.effective_prompt_provenance
    if not stored:
        return None
    records = []
    for purpose in sorted(stored):
        entry = dict(stored[purpose])
        # 在组装时对本包实际携带的文字计算哈希，不另存猜测值；内容事后被改会由 bundle_integrity 发现。
        entry["effective_prompt_sha256"] = bundle_mod.content_hash(entry["effective_prompt"])
        records.append(entry)
    return records


async def assemble_with_reason(
    db: AsyncSession, run_id
) -> tuple[bundle_mod.BundleManifest | None, str | None]:
    """从领域表组装完整证据包；无法组装时明确说明原因。"""
    run = (
        await db.execute(select(ResearchRun).where(ResearchRun.id == run_id))
    ).scalar_one_or_none()
    if run is None:
        return None, "RUN_ABSENT"

    # 证据未恢复时必须拒绝导出，不能凭空提供空证据列表并把未测量冒充实测零。
    #
    # 此限制来自实际缺陷：不可读检查点曾生成无法通过自身结论证据关联检查的包。
    if run.evidence_outcome in {"CHECKPOINT_MISSING", "CHECKPOINT_UNREADABLE"}:
        return None, "EVIDENCE_UNAVAILABLE"

    revision = (
        (
            await db.execute(
                select(Revision).where(Revision.run_id == run_id).order_by(Revision.version.desc())
            )
        )
        .scalars()
        .first()
    )
    if revision is None:
        return None, "NO_REVISION"

    sources = (
        (
            await db.execute(
                select(Source).where(Source.run_id == run_id).order_by(Source.citation_index)
            )
        )
        .scalars()
        .all()
    )
    evidence = (
        (
            await db.execute(
                select(Evidence)
                .where(Evidence.run_id == run_id)
                .order_by(Evidence.sequence.asc(), Evidence.id.asc())
            )
        )
        .scalars()
        .all()
    )
    by_source = {src.id: src for src in sources}

    evidence_dicts = [
        {
            "source_url": by_source[e.source_id].url if e.source_id in by_source else "",
            "source_title": (by_source[e.source_id].title or "")
            if e.source_id in by_source
            else "",
            "snippet": e.snippet,
            "key_fact": e.key_fact or "",
        }
        for e in evidence
    ]

    # 按引擎来源编号规则，从已存证据重建 sources，不复制另一套定义。
    # 若证据与运行原记录不一致，包内检查应揭示差异，而非悄悄修正。
    # 所选文件运行把文档版本和冻结选择放入既有 sources、trace 并纳入 bundle_hash，保持 v2 格式兼容（RFC V3.1 §8.6，D-12）。
    selected = {
        item["document_id"]: item
        for item in ((run.corpus_selection or {}).get("resolved") or [])
        if isinstance(item, dict) and item.get("document_id")
    }
    source_dicts = []
    for src in sources:
        snippets: list[str] = []
        for e in evidence:
            if e.source_id != src.id:
                continue
            text = (e.snippet or "").strip()
            if text and text not in snippets:
                snippets.append(text)
        entry = {
            "index": src.citation_index,
            "url": src.url,
            "title": src.title or "",
            "snippet": snippets[0] if snippets else "",
            "snippets": snippets,
        }
        doc = corpus_selection.document_id_of(src.url)
        if doc is not None and doc in selected:
            item = selected[doc]
            entry |= {
                key: item.get(key)
                for key in ("document_id", "doc_key", "version", "sha256", "path")
            }
        source_dicts.append(entry)

    contradictions = (
        (
            await db.execute(
                select(Contradiction)
                .where(Contradiction.run_id == run_id)
                .order_by(Contradiction.id.asc())
            )
        )
        .scalars()
        .all()
    )
    # 导出冲突的全部七个字段：两侧概括、引文、来源及冲突性质；保持检测器实际提供的来源粒度。
    contradiction_dicts = [
        {
            "claim_a": c.summary_a,
            "snippet_a": c.quote_a or "",
            "source_a": by_source[c.source_a_id].url if c.source_a_id in by_source else "",
            "claim_b": c.summary_b,
            "snippet_b": c.quote_b or "",
            "source_b": by_source[c.source_b_id].url if c.source_b_id in by_source else "",
            "nature": c.nature or "",
        }
        for c in contradictions
    ]

    # 按运行 sequence 收集审核链，不按 revision_id；计划审核没有修订 ID，按单个修订查询会漏掉计划批准。
    reviews = await approval_chain(db, run_id)
    approval_chain_dicts = []
    for r in reviews:
        action = REVIEW_TO_BUNDLE_ACTION.get((r.gate, r.decision))
        if action is None:  # pragma: no cover — AUDIT_MAP 的可逆性由专门测试保证
            raise ValueError(f"review {r.id} has no serialised action for {r.gate}/{r.decision}")
        # 在序列化层执行授权规则。
        # 只有报告批准能输出 approved，不能让计划批准在 JSON 文件中绕过数据库限制。
        if r.gate == "PLAN" and action == verify_bundle.REPORT_APPROVAL_ACTION:
            raise ValueError(
                f"review {r.id} is a PLAN approval and would serialize as a report approval"
            )
        approval_chain_dicts.append(
            {
                "action": action,
                "feedback": r.feedback,
                "draft_hash": r.reviewed_hash,
                "timestamp": r.created_at.isoformat(),
            }
        )

    provenance = _provenance_records(run)

    return (
        bundle_mod.assemble(
            session_id=str(run.id),
            query=run.question,
            report=revision.report_markdown,
            evidence=evidence_dicts,
            sources=source_dicts,
            contradictions=contradiction_dicts,
            models=run.model_routing or {},
            cost_usd=float(run.cost_usd),
            tokens_input=run.tokens_input,
            tokens_output=run.tokens_output,
            elapsed_seconds=float(run.elapsed_seconds) if run.elapsed_seconds else None,
            research_depth=run.depth,
            approval_chain=approval_chain_dicts,
            trace=_selection_trace(run) + await _trace(db, run.id),
            trace_available=True,
            demo=run.demo,
            prompt_provenance=provenance,
            # 从行记录复制配置状态，不重新计算；未配置与配置不可用虽都回退默认提示词，历史原因不能混淆。旧版 NULL 按 NONE 读取。
            prompt_overrides_status=(run.prompt_overrides_status or "NONE") if provenance else None,
        ),
        None,
    )


async def assemble(db: AsyncSession, run_id) -> bundle_mod.BundleManifest | None:
    """返回完整清单；无法组装就返回 None，绝不输出半个证据包。"""
    manifest, _ = await assemble_with_reason(db, run_id)
    return manifest


def verify(manifest: bundle_mod.BundleManifest):
    """直接导入随项目发布的独立校验器，不私自复制检查，否则只能证明本模块认可自己。"""
    return verify_bundle.verify(manifest)
