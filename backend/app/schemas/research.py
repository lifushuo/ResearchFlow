from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.models.session import SessionStatus
from app.services.chat_scope import ChatScope

# 请求结构


class ResearchStartRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    query: str = Field(..., min_length=10, max_length=2000)
    depth: str = Field(default="balanced", pattern="^(fast|balanced|comprehensive)$")
    # 研究所属项目；None 使用用户默认项目，新账号无需先建项目（docs/14 §7）。
    project_id: UUID | None = None
    # 本次模型选择优先，None 回退用户偏好再回退部署值。
    # 实际解析结果冻结到会话，恢复时沿用原模型（docs/12 M8）。
    model_routing: dict[str, str] | None = Field(
        default=None,
        description='Role → "provider:model". Omit to use your saved settings.',
    )
    # 仅语料库模式关闭网页检索，证据只来自已安装语料，查询向量化要求本地模型。
    # 但研究模型仍按角色路由接收问题和片段，本地向量化不代表整次研究离线（docs/12 M10）。
    corpus_mode: bool = False
    # 使用脚本模型和受控来源演示，无需真实供应商；保存演示身份，以便后续导出始终正确标记（docs/17 §6.2）。
    demo: bool = False

    # 研究设计关卡（internal/07 Phase 4）。
    # 这些字段必须存入会话，恢复时再传到 RunConfig；不能只在请求结构接受，却从不影响执行。

    # 用户要求计划至少覆盖的子主题；规划器还可扩展，审核者可在关卡编辑。
    topic_seeds: list[str] = Field(default_factory=list, max_length=20)
    # 引擎 outlines.TEMPLATES 中的模板 ID，由 outline-templates 接口提供；None 不强制报告结构。
    outline_template: str | None = Field(default=None, max_length=64)
    # True 是兼容脚本的默认值，不是产品表单默认。
    # 表单明确发送 false 以启用关卡，旧脚本省略字段时仍可自动执行，避免停在无法显示或恢复的暂停点。
    skip_plan_gate: bool = True

    @field_validator("topic_seeds")
    @classmethod
    def drop_blank_seeds(cls, v: list[str]) -> list[str]:
        # 过滤界面空白种子项，不能让三个空输入框变成规划器的三个硬性主题；planner_human 还会再次过滤。
        return [s.strip() for s in v if s and s.strip()]


class ApprovalRequest(BaseModel):
    approved: bool
    feedback: str | None = Field(default=None, max_length=1000)

    @field_validator("feedback")
    @classmethod
    def feedback_required_on_reject(cls, v: str | None, info) -> str | None:
        if info.data.get("approved") is False and (not v or not v.strip()):
            raise ValueError("Feedback is required when rejecting a draft.")
        return v


class PlanTaskSchema(BaseModel):
    """审核者可编辑的单项研究任务。

    与引擎任务字段对应，但不直接复用：客户端可传 include，不能设置引擎内部 status。"""

    id: int = 0
    query: str = Field(min_length=3, max_length=500)
    rationale: str = Field(default="", max_length=1000)
    subtopics: list[str] = Field(default_factory=list, max_length=20)
    # include=false 会在关卡过滤掉该任务，执行器不会收到，确保审核确实能删除研究项。
    include: bool = True
    source_hint: str | None = Field(default=None, max_length=200)


class OutlineSectionSchema(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)


class PlanDecisionRequest(BaseModel):
    """研究设计审核决定。

    None 表示未编辑、沿用提案；[] 表示删除全部内容。不能混淆，否则会忽略人工决定或执行没有研究内容的报告。"""

    # 人工上限 24 高于规划器默认 6，便于补充遗漏主题；仍需设上限，因为每项都可能增加付费检索。
    tasks: list[PlanTaskSchema] | None = Field(default=None, max_length=24)
    outline: list[OutlineSectionSchema] | None = Field(default=None, max_length=24)


class PlanResponse(BaseModel):
    session_id: UUID
    status: SessionStatus
    tasks: list[PlanTaskSchema] = Field(default_factory=list)
    outline: list[OutlineSectionSchema] = Field(default_factory=list)
    # 仍等待计划审核时为空；提交后保存实际决定，后续读取不再展示原始提案冒充最终计划。
    approved_at: datetime | None = None


class OutlineTemplateSchema(BaseModel):
    id: str
    label: str
    summary: str
    sections: list[OutlineSectionSchema]


class ChatRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    message: str = Field(..., min_length=1, max_length=4000)
    # 本轮问题可读取的范围（internal/07 Phase 5）。
    # 默认 report 保持旧客户端行为；corpus 只承诺检索不外发，不等于整轮没有网络调用，规则见 chat_scope.py。
    scope: ChatScope = "report"


# 响应结构


class ResearchStartResponse(BaseModel):
    session_id: UUID
    status: SessionStatus


class SourceSchema(BaseModel):
    index: int
    url: str
    title: str = ""
    snippet: str = ""
    # 返回该来源的全部逐字片段，而不只一条 snippet。
    # 引擎已生成、浏览器也会读取，但若响应结构没声明，Pydantic 会丢掉 snippets，导致多个不同结论的预览都只显示同一引文。
    snippets: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """历史列表的轻量记录，不携带报告正文（docs/05 §3）。"""

    session_id: UUID = Field(validation_alias="id")
    project_id: UUID
    status: SessionStatus
    prompt: str
    research_depth: str
    total_cost_usd: float
    total_tokens_input: int
    total_tokens_output: int
    elapsed_seconds: float | None = None
    rework_count: int = 0
    created_at: datetime
    archived_at: datetime | None = None
    # 演示标记要在列表摘要中提供，避免历史页面把演示与真实研究并排显示却无法区分。
    demo: bool = False
    corpus_mode: bool = False
    # 正文引用能解析到真实来源的比例。None 表示未测量或没有可测引用，0.0 才表示全失败。
    # 历史筛选也需要此指标，因此放在摘要而不只详情。
    citation_resolution_rate: float | None = None
    # 实际生成报告的模型路由，供历史列表筛选；路由解析前失败则为 None。
    model_routing: dict[str, str] | None = None

    model_config = {"from_attributes": True, "populate_by_name": True}


class SessionDetail(SessionSummary):
    draft_report: str | None = None
    final_report: str | None = None
    sources: list[SourceSchema] | None = None
    error_message: str | None = None
    updated_at: datetime
    # model_routing 已移到 SessionSummary，此处继承，不重复定义。
    # 数据早已冻结在会话行中，若响应模型没声明就会被静默丢弃。未解析或旧版缺失保持 None，不猜默认模型。
    model_routing: dict[str, str] | None = None

    # 明确披露用户配置了覆盖但旧会话路径未应用，不能让用户误以为报告使用了该指令。
    # 早于此字段的会话为 False，因为当时还不能设置覆盖。
    prompt_overrides_not_applied: bool = False


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
    total: int
    page: int
    limit: int


class ChatMessageSchema(BaseModel):
    id: UUID
    # 消息父对象从 0008 起可为报告或线程之一；本接口只返回报告消息，因此实践中该字段有值。
    session_id: UUID | None = None
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}
