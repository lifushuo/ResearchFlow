"""研究证据包：可离线审计的导出格式，见 docs/12 M12。

JSON 内包含报告、Claim、带哈希的完整证据片段、来源、矛盾、模型路由、费用、
审批链和执行轨迹。verify_bundle.py 无需模型或网络即可核验。
这里只做纯数据组装，不依赖数据库、ORM 或宿主，服务端和桌面端共用。
Claim 提取来自 research_engine.claims，与图内引用校验和评测采用同一定义。

设计约定见 docs/reference/15-bundle-format.md：
- 哈希计算的是存储的证据片段，而不是可能随时变化的网页。
- 片段是完整存储文本，不是展示截断；EvidenceChunk 本身最多 500 字符。
- bundle_hash 覆盖除自身以外的所有本版本字段，包含轨迹及其可用性。
- trace_available 区分没有事件和宿主根本不支持持久化轨迹。
  当前服务端和桌面端均会写入 agent_logs，不能把桌面端误报成不支持。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from research_engine.claims import CITE_RE, claim_lines

# ── 数据结构 ──


class SnippetRecord(BaseModel):
    """一条带完整性哈希的证据片段。"""

    source_url: str
    source_title: str = ""
    snippet: str = Field(
        description=(
            "The COMPLETE stored evidence text (executor-extracted, 500-char cap). "
            "Not a display truncation — this is exactly what the citation judge ruled on."
        )
    )
    content_hash: str = Field(description="SHA-256 of snippet.encode('utf-8')")
    key_fact: str = ""


class ClaimRecord(BaseModel):
    """从报告正文提取的一条可核验 Claim。"""

    sentence: str
    citation_indices: list[int] = Field(
        default_factory=list,
        description="The [n] markers this sentence carries",
    )


class ApprovalRecord(BaseModel):
    """审批链中的一次人工决定。"""

    action: str = Field(
        description=(
            '"approved" or "rework_requested" at the report gate, "plan_approved" at the '
            "plan gate. Only a report approval authorizes anything: the verifier counts "
            "`approved` alone, and an artifact requires an APPROVED REPORT review."
        )
    )
    feedback: str | None = None
    draft_hash: str = Field(description="SHA-256 of the draft at decision time")
    timestamp: str = Field(description="ISO-8601 datetime")


class PromptProvenance(BaseModel):
    """记录某个用途实际使用的系统提示词，见证据包 v2 与范围冻结 §9。

    用途和策略保留普通字符串，不根据当前代码的注册表限制取值，
    否则新版本产物可能被旧版核验器错误拒绝。
    effective_prompt 保存模型实际收到的文本，不能仅靠角色名推测。"""

    purpose: str = Field(description="`<role>.<purpose>`, e.g. `planner.main`")
    role: str
    policy: Literal["OVERRIDABLE", "PROTECTED"]
    overridden: bool = Field(description="False when the shipped prompt ran unchanged")
    effective_prompt: str
    effective_prompt_sha256: str = Field(description="SHA-256 of `effective_prompt`")


class BundleManifest(BaseModel):
    """.bundle.json 数据结构，兼容版本 1 和 2。"""

    bundle_version: int = 1
    session_id: str
    query: str
    research_depth: str = "balanced"

    # 标识报告是否由脚本模型与固定检索数据生成，见 docs/17 §6.2。
    # 放在运行身份旁边，便于读者先判断结果性质；该字段受 bundle_hash 保护。
    demo: bool = False

    report: str
    report_hash: str

    claims: list[ClaimRecord] = Field(default_factory=list)
    evidence: list[SnippetRecord] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    contradictions: list[dict] = Field(default_factory=list)

    models: dict[str, str] = Field(default_factory=dict)
    cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    elapsed_seconds: float | None = None

    approval_chain: list[ApprovalRecord] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    trace_available: bool = True

    # ── v2 字段，见范围冻结 §9 ──
    # 保留默认值，让 v1 证据包和旧 research_artifacts.payload 继续可读。
    #
    # 仅在 v2 证据包中参与哈希；v1 必须排除后来新增的字段，避免旧产物核验失效。
    prompt_provenance: list[PromptProvenance] = Field(default_factory=list)

    # 直接复制运行记录的状态，不能由提示词来源反推：未配置覆盖与覆盖不可用可能产生相同默认提示词。
    prompt_overrides_status: str | None = None

    created_at: str = ""
    bundle_hash: str = ""


# ── 哈希计算 ──


def content_hash(text: str) -> str:
    """计算 UTF-8 文本的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: v2 新增字段，计算 v1 哈希时必须排除。
#:
#: 旧证据包的哈希必须长期有效；新增模型字段不能改变已发出或已冻结产物的哈希。
#:
#: 哈希字段按格式版本定义；将来增加 v3 时扩展此表。
_FIELDS_ADDED_BY_VERSION: dict[int, tuple[str, ...]] = {
    2: ("prompt_provenance", "prompt_overrides_status"),
}


def compute_bundle_hash(bundle: BundleManifest) -> str:
    """计算证据包声明版本所包含字段的 SHA-256，排除 bundle_hash 自身。

    使用键排序的标准 JSON，保证可由包内内容复算；排除后续版本才新增的字段。"""
    d = bundle.model_dump()
    d["bundle_hash"] = ""
    for version, fields in _FIELDS_ADDED_BY_VERSION.items():
        if bundle.bundle_version < version:
            for name in fields:
                d.pop(name, None)
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Claim 提取 ──


def _extract_citation_indices(sentence: str) -> list[int]:
    """提取句子中的 [n] 引用编号。"""
    out: list[int] = []
    for m in CITE_RE.finditer(sentence):
        out.extend(int(part.strip()) for part in m.group(1).split(","))
    return sorted(set(out))


# ── 证据包组装 ──


def assemble(
    *,
    session_id: str,
    query: str,
    report: str,
    evidence: list[dict],
    sources: list[dict],
    contradictions: list[dict] | None = None,
    models: dict[str, str] | None = None,
    cost_usd: float = 0.0,
    tokens_input: int = 0,
    tokens_output: int = 0,
    elapsed_seconds: float | None = None,
    research_depth: str = "balanced",
    approval_chain: list[dict] | None = None,
    trace: list[dict] | None = None,
    trace_available: bool = True,
    demo: bool = False,
    prompt_provenance: list[dict] | None = None,
    prompt_overrides_status: str | None = None,
) -> BundleManifest:
    """由会话数据组装完整证据包，不访问数据库或 ORM。

    是否记录 prompt_provenance 决定格式版本，与是否自定义提示词无关。
    有实际提示词记录就生成 v2；旧运行未记录时只能生成 v1，不能补造历史提示词。"""

    report_h = content_hash(report)

    snippet_records = [
        SnippetRecord(
            source_url=e.get("source_url", ""),
            source_title=e.get("source_title", ""),
            snippet=e.get("snippet", ""),
            content_hash=content_hash(e.get("snippet", "")),
            key_fact=e.get("key_fact", ""),
        )
        for e in evidence
    ]

    claims = [
        ClaimRecord(sentence=s, citation_indices=_extract_citation_indices(s))
        for s in claim_lines(report)
    ]

    approvals = [
        ApprovalRecord(
            action=a.get("action", ""),
            feedback=a.get("feedback"),
            draft_hash=a.get("draft_hash", ""),
            timestamp=a.get("timestamp") or a.get("created_at") or "",
        )
        for a in (approval_chain or [])
    ]

    bundle = BundleManifest(
        bundle_version=2 if prompt_provenance else 1,
        prompt_provenance=[PromptProvenance.model_validate(r) for r in (prompt_provenance or [])],
        prompt_overrides_status=prompt_overrides_status,
        session_id=session_id,
        query=query,
        research_depth=research_depth,
        demo=demo,
        report=report,
        report_hash=report_h,
        claims=claims,
        evidence=snippet_records,
        sources=sources,
        contradictions=contradictions or [],
        models=models or {},
        cost_usd=cost_usd,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        elapsed_seconds=elapsed_seconds,
        approval_chain=approvals,
        trace=trace or [],
        trace_available=trace_available,
        created_at=datetime.now(UTC).isoformat(),
        bundle_hash="",
    )
    bundle.bundle_hash = compute_bundle_hash(bundle)
    return bundle


def serialize(bundle: BundleManifest) -> str:
    """生成用于保存或导出的标准 JSON，保持可读且键顺序确定。"""
    return json.dumps(bundle.model_dump(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


#: 演示报告导出时添加的 Markdown 标记，PDF 渲染后也应可见，见 docs/17 §6.2。
#:
#: 服务端和桌面端共用此标记，防止脚本数据导出后看起来像真实研究。
#: 以前标记只在服务端私有模块中，曾导致桌面导出漏标。
#:
#: 不把标记插入证据包正文：批准时的 draft_hash 必须与 report_hash 一致。
#: 证据包通过受哈希保护的 demo 字段披露演示状态。
DEMO_STAMP_MD = (
    "> ## ⚠ DEMO — NOT REAL RESEARCH\n"
    ">\n"
    "> This report was produced with **scripted models and fixture sources** so the\n"
    "> product could be demonstrated without an API key. The citations below resolve to\n"
    "> real-looking references, but **nothing here was researched and nothing here is\n"
    "> verified**. Do not cite, share, or act on it.\n"
    "\n"
    "---\n\n"
)


def stamp_demo_md(report: str, *, demo: bool) -> str:
    """仅在 demo 为真时添加演示标记，两个宿主共用同一规则。"""
    return f"{DEMO_STAMP_MD}{report}" if demo else report


def render_model_attribution_md(
    model_routing: dict[str, str] | None, checker_accuracy: dict | None = None
) -> str:
    """生成 Markdown 模型披露页脚，供 .md 与 .pdf 导出使用。

    放在引擎层，供服务端和桌面端共用，避免桌面导入含 WeasyPrint 的服务端导出模块。
    由调用方追加到正文之外，不能改动已批准正文，否则会破坏审批哈希。
    没有解析过路由就返回空文本，不猜测默认模型。
    选定文件研究还会披露引用检查器已测得的准确性标签，见 D-24。"""
    if not model_routing:
        return ""
    lines = "\n".join(f"- **{role}** — `{route}`" for role, route in sorted(model_routing.items()))
    if checker_accuracy and checker_accuracy.get("label"):
        lines += f"\n\n{checker_accuracy['label'][0].upper()}{checker_accuracy['label'][1:]}."
    return f"\n\n---\n\n## Models used\n\n{lines}\n"
