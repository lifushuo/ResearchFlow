"""两个主机共用的运维计数与封闭标签集合。

使用私有 Prometheus 注册表，不暴露其他依赖偷偷注册的指标，测试也可独立计数。完成事件在报告批准接口记录，失败和取消在 persist_outcome 记录，不能假定一个入口覆盖所有终态。
不从创建时间推算执行耗时，因为其中包含人工审核等待；未测量就不提供该指标。
标签只能来自本文件固定表，禁止用户、项目、运行 ID、文件名、URL、问题和异常文本进入 labels，避免泄露及无限基数。"""

from __future__ import annotations

from collections.abc import Mapping

import prometheus_client
from prometheus_client import CollectorRegistry, Counter, Gauge

from research_engine.build_info import build_info
from research_engine.routing_rules import ENDPOINT_DEFINED_PROVIDERS

# 只统计真正结束的运行。人工关卡是暂停，不是结果；暂停时累计费用也不能重复计入。
RUN_OUTCOMES = ("completed", "failed", "cancelled")

# 费用可信度使用三态而非布尔值：有计价、无计价、配置未知。
# 路由为空时实际可能使用部署默认模型，但本模块不能读取 app.config 来猜。
COST_CONFIDENCE = ("priced", "unpriced", "unstamped")

TOKEN_DIRECTIONS = ("input", "output")

# 上传结果区分已存在而跳过、用户输入被拒绝、依赖不可用，与 corpus_ingest 的客户端语义一致。
INGEST_OUTCOMES = ("ingested", "skipped", "rejected", "unavailable")

# 完整指标契约：指标族、标签名及允许的标签值。
#:
# 键名采用 collect() 返回的指标族名；计数器族不含 _total，但公开采样名带该后缀。
DECLARED: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "research_run_outcomes": {"outcome": RUN_OUTCOMES},
    "research_run_cost_usd": {},
    "research_run_cost_confidence": {"confidence": COST_CONFIDENCE},
    "research_run_tokens": {"direction": TOKEN_DIRECTIONS},
    "corpus_ingest_documents": {"outcome": INGEST_OUTCOMES},
    "corpus_ingest_bytes": {},
    "research_build_info": {"version": (), "git_sha": ()},
}

# 构建身份标签是唯一不采用枚举的指标；一个进程对应一次构建，因此只产生一条序列。
IDENTITY_METRICS = frozenset({"research_build_info"})

# persist_outcome 只记录失败和取消，不含 COMPLETED；完成由批准接口记录，结构上避免重复计数。
# 若以后引擎也能完成运行，应同步调整此映射和对应回归检查。
_PERSISTED_TERMINAL = {"FAILED": "failed", "CANCELLED": "cancelled"}


def cost_confidence(model_routing: Mapping[str, str] | None) -> str:
    """评估本运行费用记录是否可信。

    自定义端点等供应商的 estimate_cost 返回 0，并不表示免费；必须标记无可靠计价。路由只按第一个冒号拆分，避免误读 ollama:qwen2.5:7b。"""
    if not model_routing:
        return "unstamped"
    for route in model_routing.values():
        if str(route).partition(":")[0] in ENDPOINT_DEFINED_PROVIDERS:
            return "unpriced"
    return "priced"


class Metrics:
    """管理独立指标注册表与七类指标。测试可新建隔离实例，生产使用模块默认实例。"""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._outcomes = Counter(
            "research_run_outcomes_total",
            "Research runs that reached a terminal state, by outcome. Counts finished runs, "
            "not started ones: a run waiting at a gate is not counted until it terminates.",
            ["outcome"],
            registry=self.registry,
        )
        self._cost = Counter(
            "research_run_cost_usd_total",
            "Estimated spend of terminated research runs, in USD. Under-reports whenever "
            'research_run_cost_confidence_total{confidence!="priced"} is moving: cost '
            "estimation returns 0 for endpoint-defined providers. Cap spend at the provider.",
            registry=self.registry,
        )
        self._confidence = Counter(
            "research_run_cost_confidence_total",
            "Terminated runs by how far their recorded cost can be trusted. 'unstamped' "
            "means the run carried no routing snapshot, so this is unknown rather than fine.",
            ["confidence"],
            registry=self.registry,
        )
        self._tokens = Counter(
            "research_run_tokens_total",
            "Tokens attributed to terminated research runs, by direction.",
            ["direction"],
            registry=self.registry,
        )
        self._ingest = Counter(
            "corpus_ingest_documents_total",
            "Corpus upload attempts by what happened to them.",
            ["outcome"],
            registry=self.registry,
        )
        self._ingest_bytes = Counter(
            "corpus_ingest_bytes_total",
            "Bytes actually indexed into a corpus. A skipped or refused upload indexed "
            "nothing and does not count here.",
            registry=self.registry,
        )
        info = build_info()
        Gauge(
            "research_build_info",
            "Which build is answering. Always 1; the labels carry the value.",
            ["version", "git_sha"],
            registry=self.registry,
        ).labels(version=info.version, git_sha=info.git_sha).set(1)

    # 研究运行指标

    def observe_completed_run(
        self,
        *,
        cost_usd,
        tokens_input: int,
        tokens_output: int,
        model_routing: Mapping[str, str] | None,
    ) -> None:
        """记录审核者已批准的运行，由 submit_report_review 调用。"""
        self._record_terminal(
            "completed",
            cost_usd=cost_usd,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model_routing=model_routing,
        )

    def observe_persisted_run(
        self,
        status: str,
        *,
        cost_usd,
        tokens_input: int,
        tokens_output: int,
        model_routing: Mapping[str, str] | None,
    ) -> None:
        """记录 persist_outcome 实际保存的终态；暂停状态或未知状态不增加计数。"""
        outcome = _PERSISTED_TERMINAL.get(status)
        if outcome is None:
            return
        self._record_terminal(
            outcome,
            cost_usd=cost_usd,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model_routing=model_routing,
        )

    def _record_terminal(
        self, outcome: str, *, cost_usd, tokens_input, tokens_output, model_routing
    ) -> None:
        """记录一条终态运行，费用值与可信度必须一起记录，避免把未知的零估价误读成免费。"""
        self._outcomes.labels(outcome=outcome).inc()
        # 数据库 cost_usd 是 Numeric，批准接口传入的是 Decimal。
        self._cost.inc(float(cost_usd or 0))
        self._confidence.labels(confidence=cost_confidence(model_routing)).inc()
        self._tokens.labels(direction="input").inc(int(tokens_input or 0))
        self._tokens.labels(direction="output").inc(int(tokens_output or 0))

    # 语料库指标

    def observe_ingest(self, outcome: str, *, byte_count: int = 0) -> None:
        """记录一次上传尝试；只有文档成功索引才计入 byte_count。

        未知结果直接拒绝，不能随意作为 labels 值，防止未来调用方把文件名或错误消息当标签。"""
        if outcome not in INGEST_OUTCOMES:
            raise ValueError(f"unknown ingest outcome {outcome!r}; declared: {INGEST_OUTCOMES}")
        self._ingest.labels(outcome=outcome).inc()
        if outcome == "ingested":
            self._ingest_bytes.inc(byte_count)

    # 公开指标

    def render(self) -> tuple[bytes, str]:
        """返回抓取响应正文及其内容类型。"""
        return (
            prometheus_client.generate_latest(self.registry),
            prometheus_client.CONTENT_TYPE_LATEST,
        )


_default = Metrics()


def observe_completed_run(
    *, cost_usd, tokens_input: int, tokens_output: int, model_routing: Mapping[str, str] | None
) -> None:
    _default.observe_completed_run(
        cost_usd=cost_usd,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        model_routing=model_routing,
    )


def observe_persisted_run(
    status: str,
    *,
    cost_usd,
    tokens_input: int,
    tokens_output: int,
    model_routing: Mapping[str, str] | None,
) -> None:
    _default.observe_persisted_run(
        status,
        cost_usd=cost_usd,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        model_routing=model_routing,
    )


def observe_ingest(outcome: str, *, byte_count: int = 0) -> None:
    _default.observe_ingest(outcome, byte_count=byte_count)


def render() -> tuple[bytes, str]:
    return _default.render()
