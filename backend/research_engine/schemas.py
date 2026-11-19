"""智能体流水线的结构化输入输出契约（docs/architecture/04-agent-architecture.md §2）。

每个模型边界都由这里的模型校验。解析或校验失败即节点失败，不能静默回退（docs/11 §1 rule 2）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from research_engine.runconfig import get_run_config

TaskStatus = Literal["pending", "running", "passed", "failed"]


class ResearchTask(BaseModel):
    id: int | str = 0
    query: str = Field(min_length=3, description="A concrete, independently searchable query")
    rationale: str = ""
    status: TaskStatus = "pending"
    # 计划审核字段（internal/07 Phase 4）；缺省或空值兼容字段引入前的任务形状，不影响跳过关卡的研究。
    subtopics: list[str] = Field(default_factory=list)
    # False 表示审核者排除此任务，plan_gate_node 会在交给执行器前过滤；默认 True 与旧行为一致，保留全部任务。
    include: bool = True
    source_hint: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v):
        if isinstance(v, str):
            digits = "".join(c for c in v if c.isdigit())
            if digits:
                return int(digits)
        return v or 0


class OutlineSection(BaseModel):
    """报告建议结构中的一个章节（internal/07 Phase 4）。"""

    title: str = Field(min_length=1)
    description: str = ""


class PlannerOutput(BaseModel):
    tasks: list[ResearchTask]
    proposed_outline: list[OutlineSection] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _wrap_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"tasks": data}
        return data

    @field_validator("tasks")
    @classmethod
    def _bounded(cls, v: list[ResearchTask]) -> list[ResearchTask]:
        if not v:
            raise ValueError("planner must produce at least 1 task")
        # 配置只限制规划器主动提出的任务数；用户仍可在审核关卡手动新增任务（internal/07 Phase 4）。默认值 6 保持原有硬编码上限的行为。
        cap = get_run_config().max_planner_tasks
        if len(v) > cap:
            v = v[:cap]
        # 将 ID 规范为 1..n，稳定下游任务标记。
        for i, t in enumerate(v, start=1):
            t.id = i
        return v


class EvidenceChunk(BaseModel):
    task_id: int = 0
    # source_url 必须保持必填。若设默认值，无来源的块也能通过校验，让 evidence 非空，并误导 route_after_critic 和 _no_research_reason 放行无法引用的研究；兼容不规范输出应使用下方别名归一化，而非允许缺失来源。
    source_url: str
    source_title: str = ""
    snippet: str = Field("", max_length=500, description="Verbatim supporting text")
    key_fact: str = Field("", description="The claim this snippet supports")
    retrieved_at: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "source_url" not in data:
                for k in ("url", "link", "source"):
                    if k in data and isinstance(data[k], str):
                        data["source_url"] = data[k]
                        break
            if "key_fact" not in data:
                for k in ("fact", "claim", "title"):
                    if k in data and isinstance(data[k], str):
                        data["key_fact"] = data[k]
                        break
            if "snippet" not in data:
                for k in ("quote", "text", "content"):
                    if k in data and isinstance(data[k], str):
                        data["snippet"] = data[k]
                        break
        return data


class ExecutorOutput(BaseModel):
    task_id: int | str = 0
    evidence: list[EvidenceChunk] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _wrap_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"evidence": data}
        return data


class CriticVerdict(BaseModel):
    passed: bool
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    feedback_for_executor: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v):
        # 模型常以 0–100 百分比表达置信度，而非 0–1 概率；转换并钳制数值，避免良性的尺度差异丢弃有效评审或使研究崩溃。非数字回退到中性默认值。
        try:
            f = float(v)
        except TypeError, ValueError:
            return 0.5
        if f > 1.0:
            f = f / 100.0
        return min(max(f, 0.0), 1.0)

    @field_validator("feedback_for_executor")
    @classmethod
    def _feedback_required_on_fail(cls, v: str | None, info) -> str | None:
        # 未通过的评审必须给执行器提供可执行的改进意见。
        if info.data.get("passed") is False and not (v and v.strip()):
            return "Insufficient evidence; gather more independent, well-cited sources."
        return v


class Source(BaseModel):
    """界面渲染的一条编号引用（docs/05 §1，sessions.sources）。

    同一页面往往支持多个事实，执行器会分别提取逐字片段；若只保存一个，
    引用提示可能展示与当前句子无关的内容，违背悬停后可核查本条主张的承诺。
    snippets 保存该来源全部不同片段；snippet 仍保留首条，兼容旧 sessions.sources 行及旧客户端。"""

    index: int
    url: str
    title: str = ""
    snippet: str = ""
    snippets: list[str] = Field(default_factory=list)


class ContradictionPair(BaseModel):
    """两个来源间的一组直接冲突（docs/12 M11）。

    双方主张必须引自检测器实际看到的片段。contradictions.py 的校验器
    丢弃来源 URL 不在证据中的冲突对，防止虚构或注入的来源进入报告。"""

    claim_a: str = Field(
        max_length=400,
        description="One-sentence restatement of the first side of the conflict, in your own words",
    )
    snippet_a: str = Field(
        "",
        max_length=500,
        description="The VERBATIM quoted snippet text from the source that supports claim_a — copy it exactly as shown between the quotation marks, never a URL",
    )
    source_a: str = Field(
        description="The source URL for claim_a — copy the bare URL exactly as it appears after 'Source:' with no prefix, no 'Source:' label, no extra text"
    )
    claim_b: str = Field(
        max_length=400,
        description="One-sentence restatement of the second side of the conflict, in your own words",
    )
    snippet_b: str = Field(
        "",
        max_length=500,
        description="The VERBATIM quoted snippet text from the source that supports claim_b — copy it exactly as shown between the quotation marks, never a URL",
    )
    source_b: str = Field(
        description="The source URL for claim_b — copy the bare URL exactly as it appears after 'Source:' with no prefix, no 'Source:' label, no extra text"
    )
    nature: str = Field(
        "", max_length=400, description="One sentence: why they cannot both be true"
    )


class ContradictionReport(BaseModel):
    """检测器的结构化输出。pairs 超过上限时截断而非拒绝整批；展示十条真实冲突并舍弃两条，比因模型返回过多而让整个检查失败更有价值。"""

    pairs: list[ContradictionPair] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _wrap_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"pairs": data}
        return data

    @field_validator("pairs")
    @classmethod
    def _cap(cls, v: list[ContradictionPair]) -> list[ContradictionPair]:
        return v[:10]
