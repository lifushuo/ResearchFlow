"""以真实编译的 LangGraph StateGraph 实现研究流水线（docs/architecture/04-agent-architecture.md），
不自行编排循环，也不在校验失败后放行。

planner → executor（并行研究全部待办任务）→ critic（并行评审）
评审未通过且尚可重试 → executor；全部通过或重试耗尽 → contradiction_detector → synthesizer → hitl_gate。
hitl_gate 通过 interrupt 暂停；批准后进入 finalizer，拒绝后返回 synthesizer。
任何位置超出预算或时间限制均进入 failer。

研究按轮次推进，而非为每个子任务建立独立流水线（docs/12 M7）。每轮并行研究全部待办任务，
统一评审后在下一轮重试失败项。逐任务推进虽能让快任务提前重试，却会将重试编排移出图。
max_critic_loops 限制轮数，使同步等待开销有限，并让重试路径保持为编译图中的条件边。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from contextvars import ContextVar
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, model_validator

from research_engine import claims, contradictions, grounding, grounding_repair, outlines, prompts
from research_engine.concurrency import gather_or_cancel
from research_engine.corpus import parse_corpus_url
from research_engine.eval_trace import trace_span, traced_ainvoke, traced_node
from research_engine.events import emit
from research_engine.llm_factory import (
    estimate_cost,
    get_llm,
    served_model_id,
    text_of,
    token_counts,
)
from research_engine.prompt_composition import system_prompt
from research_engine.runconfig import get_run_config
from research_engine.schemas import (
    ContradictionReport,
    CriticVerdict,
    EvidenceChunk,
    ExecutorOutput,
    PlannerOutput,
    Source,
)
from research_engine.state import AgentState
from research_engine.token_usage import PaidRequestUncertain
from research_engine.tools import EXECUTOR_TOOLS

logger = structlog.get_logger()

_TOOLS_BY_NAME = {t.name: t for t in EXECUTOR_TOOLS}
#: 按研究深度限制每个任务的工具调用轮数和抓取页数。
#:
#: 一轮对应一次模型调用，是主要费用和等待来源。页面抓取限时 10 秒且可并发，
#: 托管路由上的模型响应却可能需要数分钟，因此减少模型调用次数才是主要优化手段。
#:
#: fast 和 comprehensive 曾都允许八轮、五页，仅在提示词中名称不同，
#: 并未兑现界面关于速度与成本的说明；此处限制使深度选择实际影响执行。
_DEPTH_LIMITS: dict[str, tuple[int, int]] = {
    # 每项研究深度对应单个任务的（最大工具轮数，最大抓取页数）。
    "fast": (3, 3),
    "balanced": (5, 5),
    "comprehensive": (8, 8),
}

#: 未知研究深度及循环回退时使用的轮数上限。
_MAX_TOOL_ROUNDS = 8

#: 限制单任务可打开的页数，达到后必须停止阅读并提交已有内容。
#:
#: 这是终止控制：模型可能持续搜索和抓取，却不主动总结提交。
#: 限制阅读次数可保证它带着已有来源进入提取步骤，而非耗尽轮数后仍无证据。
_MAX_PAGES_PER_TASK = 5


def _limits_for(depth: str | None) -> tuple[int, int]:
    """返回研究深度对应的 (rounds, pages)，未知值回退到 balanced。

    未知深度可能来自旧客户端；使用中等配置比拒绝任务或默默采用最昂贵配置更合适。"""
    return _DEPTH_LIMITS.get((depth or "").strip().lower(), _DEPTH_LIMITS["balanced"])


#: 强制提取阶段向模型提供的单个来源文本长度上限。
#:
#: 此处截断只限制请求大小，不缩小已读文本的核验范围；
#: verify_evidence_snippets 仍对完整 seen_text 检查逐字引文。
_FORCED_SUBMIT_SOURCE_CHARS = 6_000

#: 强制提取阶段展示的来源数量上限，优先选择文本最丰富的来源。
#:
#: 大量单行搜索摘要会掩盖真正含有可引用正文的网页；此限制用于聚焦内容，而非控制费用。
_FORCED_SUBMIT_SOURCES = 8


# ── 结构化输出辅助函数 ──────────────────────────────────────────────

# 记录 _structured 捕获的提供方错误。函数仍返回解析结果形状并保守失败（docs/11 §1），
# 调用方可读取真实原因，避免将配额耗尽误报为规划器未能生成有效任务列表。
_last_api_error: ContextVar[str | None] = ContextVar("last_api_error", default=None)
# 记录结构化输出解析失败的脱敏摘要（schema 校验错误类型与字段路径）。
# 没有它，"could not produce a valid task list" 会把模型输出格式问题掩盖成未知故障。
_last_parse_error: ContextVar[str | None] = ContextVar("last_parse_error", default=None)


def _looks_like_quota(msg: str) -> bool:
    """判断提供方错误是否表示费用或配额耗尽，此类错误无法靠重试恢复。

    匹配异常类名和消息，使引擎不依赖具体提供方的 SDK（docs/12 M6）。"""
    m = msg.lower()
    return any(
        marker in m
        for marker in (
            "resource_exhausted",
            "spending cap",
            "quota",
            "rate limit",
            "ratelimit",
            "429",
            "insufficient_quota",
        )
    )


async def _structured(role: str, messages: list, schema):
    """调用指定角色的模型，返回 (parsed_model_or_None, cost, in_tok, out_tok)。

    真实模型通过 with_structured_output(include_raw=True) 保留 usage_metadata 以计算费用；
    模拟模型直接返回 JSON 内容，由此处解析。"""
    with trace_span(
        f"structured.{role}", kind="structured_output", metadata={"role": role}
    ) as span:
        if span.writer is not None and span.writer.capture_content:
            try:
                span.observe(expected_schema=schema.model_json_schema())
            except Exception as exc:  # noqa: BLE001 — 数据结构观测不能改变解析流程
                span.observe(schema_observation_error=type(exc).__name__)
        result = await _structured_impl(role, messages, schema)
        span.add(schema_status="parsed" if result[0] is not None else "unparsed")
        if result[0] is None:
            span.status = "business_error"
        return result


async def _structured_impl(role: str, messages: list, schema):
    _last_api_error.set(None)
    _last_parse_error.set(None)
    model = get_llm(role)
    if get_run_config().llm_mode == "fake":
        resp = await traced_ainvoke(model, messages, role=role)
        cost = estimate_cost(resp, role)
        i, o = token_counts(resp)
        try:
            with trace_span(
                f"schema.{role}", kind="schema_validation", metadata={"strategy": "fake_json"}
            ):
                parsed = schema.model_validate_json(resp.content)
        except Exception as exc:
            parsed = None
            _last_parse_error.set(f"fake_json: {type(exc).__name__}: {str(exc)[:240]}")
        return parsed, cost, i, o

    route = get_run_config().model_for(role)

    # 注意：langchain-openai 1.6.6 的 with_structured_output(method="function_calling")
    # 默认就绑定 tool_choice=<schema 函数名>（base.py:2783），即已强制模型调用该函数。
    # 实验 gepa-ba26c622/e1448e62 的空响应发生在该强制已生效的前提下——是 DeepSeek
    # 服务端行为，不是「模型自主跳过工具调用」，因此这里不再注入 tool_choice 覆盖
    # 默认的函数级强制（注入 "required" 反而会把函数级强制放宽为任意工具）。
    if route.startswith("custom:deepseek"):
        structured = model.with_structured_output(
            schema,
            method="function_calling",
            include_raw=True,
        )
    else:
        structured = model.with_structured_output(schema, include_raw=True)
    try:
        result = await traced_ainvoke(structured, messages, role=role)
    except PaidRequestUncertain:
        raise
    except Exception as e:  # noqa: BLE001
        # with_structured_output 在字段不合规则或提供方拒绝请求时可能直接抛异常，
        # 而不仅设置 parsing_error。统一按无法解析处理，让节点保守失败（docs/11 §1），
        # 同时保留异常消息，避免把网络或配额错误误判为模型输出格式错误。
        _last_api_error.set(str(e)[:400])
        return None, 0.0, 0, 0
    raw = result.get("raw")
    cost = estimate_cost(raw, role) if raw is not None else 0.0
    i, o = token_counts(raw) if raw is not None else (0, 0)
    parsed = None if result.get("parsing_error") else result.get("parsed")
    parse_detail = ""
    if result.get("parsing_error"):
        # 解析失败摘要：类型 + 字段路径 + 失败摘要，脱敏截断；不保存完整模型响应。
        raw_error = str(result.get("parsing_error") or "")
        parse_detail = (
            f"{type(result.get('parsing_error')).__name__}: {' '.join(raw_error.split())[:240]}"
        )
    if parsed is None and raw is not None and getattr(raw, "content", None):
        content = raw.content
        if isinstance(content, str) and content.strip():
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                with trace_span(
                    f"schema.{role}", kind="schema_validation", metadata={"strategy": "raw_json"}
                ):
                    parsed = schema.model_validate_json(cleaned)
            except Exception as exc:
                parse_detail = f"raw_json: {type(exc).__name__}: {' '.join(str(exc).split())[:240]}"
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    try:
                        with trace_span(
                            f"schema.{role}",
                            kind="schema_validation",
                            metadata={"strategy": "json_object"},
                        ):
                            parsed = schema.model_validate_json(match.group(0))
                    except Exception as exc2:
                        parse_detail = f"json_object: {type(exc2).__name__}: {' '.join(str(exc2).split())[:240]}"
    if parsed is None:
        # 空文本恢复：function_calling 模式下模型可能只返回 tool_call（content 为空），
        # 先尝试从 tool_calls 参数直接通过 schema 校验；仍失败时记录响应形态——
        # reasoning-only 完成（有思维链、无 tool call）是 hybrid 推理模型已知的间歇行为，
        # 形态记录让下一次故障能区分「模型没答」与「答了但解析失败」。
        raw_tool_calls = getattr(raw, "tool_calls", None) if raw is not None else None
        if raw_tool_calls:
            for tool_call in raw_tool_calls:
                args = (
                    tool_call.get("args")
                    if isinstance(tool_call, dict)
                    else getattr(tool_call, "args", None)
                )
                try:
                    with trace_span(
                        f"schema.{role}",
                        kind="schema_validation",
                        metadata={"strategy": "tool_call_args"},
                    ):
                        parsed = schema.model_validate(args)
                    parse_detail = ""
                    break
                except Exception as exc:
                    parse_detail = (
                        f"tool_call_args: {type(exc).__name__}: {' '.join(str(exc).split())[:240]}"
                    )
        if parsed is None and not parse_detail:
            # invalid_tool_calls 是最后一个盲区：DeepSeek 返回畸形 tool_call
            # （参数截断/非法 JSON）时，langchain 不抛异常也不进 tool_calls，
            # 解析器看到的是 None——与真正的空消息无法区分，除非在此显式记录。
            invalid_tool_calls = (
                getattr(raw, "invalid_tool_calls", None) if raw is not None else None
            )
            reasoning_present = False
            has_content = False
            if raw is not None:
                reasoning_present = bool(
                    (getattr(raw, "additional_kwargs", None) or {}).get("reasoning_content")
                )
                has_content = bool(getattr(raw, "content", None))
            invalid_summary = "absent"
            if invalid_tool_calls:
                first = invalid_tool_calls[0]
                first_args = first.get("args") if isinstance(first, dict) else None
                first_error = first.get("error") if isinstance(first, dict) else None
                invalid_summary = (
                    f"present(count={len(invalid_tool_calls)}, "
                    f"args_len={len(first_args) if isinstance(first_args, str) else 0}, "
                    f"err={str(first_error)[:120]})"
                )
            parse_detail = (
                f"empty_response: content={'present' if has_content else 'empty'}; "
                f"tool_calls={'present' if raw_tool_calls else 'absent'}; "
                f"invalid_tool_calls={invalid_summary}; "
                f"reasoning_content={'present' if reasoning_present else 'absent'}"
            )
            parse_detail = (
                f"empty_response: content={'present' if has_content else 'empty'}; "
                f"tool_calls={'present' if raw_tool_calls else 'absent'}; "
                f"invalid_tool_calls={invalid_summary}; "
                f"reasoning_content={'present' if reasoning_present else 'absent'}"
            )
        # JSON Output mode 备用路径：绕过 function calling，用 response_format json_object
        # 重新请求同一模型。仅在 function_calling 返回空（无 content、无 tool_calls）时触发；
        # 结果仍经同一个 schema 校验，失败不伪造任务。备用路径尝试本身也记录到 parse_detail。
        if parsed is None and "empty_response" in parse_detail:
            json_bound = model.bind(response_format={"type": "json_object"})
            fallback_messages = list(messages) + [
                SystemMessage(content="Respond in valid JSON format matching the expected schema.")
            ]
            result_json = await traced_ainvoke(json_bound, fallback_messages, role=role)
            cost_f = estimate_cost(result_json, role) if result_json is not None else 0.0
            i_f, o_f = token_counts(result_json) if result_json is not None else (0, 0)
            cost += cost_f
            i += i_f
            o += o_f
            content_json = getattr(result_json, "content", "") or ""
            if isinstance(content_json, str) and content_json.strip():
                cleaned = content_json.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                try:
                    with trace_span(
                        f"schema.{role}",
                        kind="schema_validation",
                        metadata={"strategy": "json_mode_fallback"},
                    ):
                        parsed = schema.model_validate_json(cleaned)
                    parse_detail = "json_mode_fallback: recovered"
                except Exception as exc:
                    parse_detail = (
                        f"json_mode_fallback: {type(exc).__name__}: "
                        f"{' '.join(str(exc).split())[:240]}"
                    )
            else:
                parse_detail = f"json_mode_fallback: also returned empty content; {parse_detail}"
    if parsed is None and parse_detail:
        # 附带服务端响应元数据：finish_reason 与 reasoning token 是区分
        # 「推理耗尽输出预算」与「模型未调用工具」的关键证据（用户决策需要）。
        meta = getattr(raw, "response_metadata", None) or {}
        finish_reason = meta.get("finish_reason") if isinstance(meta, dict) else None
        usage = getattr(raw, "usage_metadata", None) or {}
        output_details = usage.get("output_token_details") or {}
        reasoning_tokens = output_details.get("reasoning")
        if finish_reason:
            parse_detail = f"{parse_detail}; finish_reason={finish_reason}"
        if reasoning_tokens:
            parse_detail = f"{parse_detail}; reasoning_tokens={reasoning_tokens}"
        _last_parse_error.set(parse_detail)
    return parsed, cost, i, o


def _acc(state: AgentState, cost: float, i: int, o: int) -> dict:
    return {
        "cost_usd": state.get("cost_usd", 0.0) + cost,
        "tokens_input": state.get("tokens_input", 0) + i,
        "tokens_output": state.get("tokens_output", 0) + o,
    }


# ── 图节点 ────────────────────────────────────────────────────────


async def planner_node(state: AgentState) -> dict:
    sid = state["session_id"]
    cfg = get_run_config()
    await emit(
        sid,
        "agent_log",
        agent="planner",
        message="Decomposing the query into tasks…",
        detail={
            "query": state["original_query"],
            "depth": state.get("research_depth", "balanced"),
            # 按 docs/04 要求，在研究花费发生前的规划阶段披露完整角色到模型路由映射。
            # 后续各节点启动事件中的 model 记录同一信息中对应角色的路由，不只在最终报告中展示。
            "models": dict(cfg.models),
            "model": cfg.model_for("planner"),
        },
    )
    messages = [
        SystemMessage(content=system_prompt("planner.main")),
        HumanMessage(
            content=prompts.planner_human(
                state["original_query"],
                state.get("research_depth", "balanced"),
                cfg.topic_seeds,
            )
        ),
    ]

    async def _attempt(attempt: int):
        _last_api_error.set(None)
        try:
            with trace_span(
                "planner.attempt",
                kind="attempt",
                metadata={
                    "role": "planner",
                    "attempt": attempt,
                    "strategy": "initial" if attempt == 1 else "retry_once",
                },
            ) as span:
                parsed, cost, i, o = await _structured("planner", messages, PlannerOutput)
                if parsed is None:
                    span.status = "business_error"
        except PaidRequestUncertain:
            raise
        except Exception as e:  # noqa: BLE001
            # 纵深防御：即使通常不抛异常的 _structured 意外失败，也应返回提供方错误而非掩盖原因。
            return None, 0.0, 0, 0, str(e)[:400]
        return parsed, cost, i, o, _last_api_error.get()

    parsed, cost, i, o, api_err = await _attempt(1)
    if api_err and _looks_like_quota(api_err):
        # 费用上限或配额耗尽不会在本次研究中自行恢复，重试只会延长等待；直接展示提供方原始消息。
        logger.error("planner_quota_exhausted", session_id=sid, error=api_err)
        return {
            "error": f"planner: provider error — {api_err}",
            **_acc(state, cost, i, o),
        }
    if parsed is None:
        parsed2, c2, i2, o2, api_err2 = await _attempt(2)  # 最多重试一次。
        cost, i, o = cost + c2, i + i2, o + o2
        parsed, api_err = parsed2, api_err2
    if parsed is None:
        if api_err:
            logger.error("planner_provider_error", session_id=sid, error=api_err)
            return {
                "error": f"planner: provider error — {api_err}",
                **_acc(state, cost, i, o),
            }
        # 解析失败摘要随错误浮出（类型/字段路径/截断摘要），让 Journal 能记录真实原因。
        parse_detail = _last_parse_error.get()
        detail_suffix = f" (parse detail: {parse_detail})" if parse_detail else ""
        return {
            "error": f"planner: could not produce a valid task list{detail_suffix}",
            **_acc(state, cost, i, o),
        }

    tasks = [t.model_dump() for t in parsed.tasks]
    # 用户显式选择的大纲模板优先于模型提案，否则模板选择器形同虚设。
    # 未选择模板时此处为空，保留规划器建议的结构。
    outline = outlines.sections_for(cfg.outline_template) or [
        s.model_dump() for s in parsed.proposed_outline
    ]
    await emit(
        sid,
        "agent_log",
        agent="planner",
        message=f"Created {len(tasks)} research tasks",
        detail={"tasks": tasks, "proposed_outline": outline},
    )
    return {
        "tasks": tasks,
        "proposed_outline": outline,
        "plan_approved": None,
        "evidence": [],
        "verdicts": {},
        "retries": {},
        "research_round": 0,
        **_acc(state, cost, i, o),
    }


# ── 并行研究轮次（docs/12 M7）───────────────────────────────────────


class _BudgetGuard:
    """同一轮并行任务共享的费用记账。

    并发与硬性费用上限存在冲突：N 个任务可能同时通过预算检查，然后同时花费。
    不预先保留预算时，只能在每次模型响应后立即累加费用，并在下一轮工具调用或派发前重新检查。
    因此超支最多来自已经发出的调用，无法彻底消除。
    原先顺序执行的 _over_budget 也只在任务间检查，只是窗口为一次调用；
    max_parallel_tasks=1 会恢复这一行为。"""

    def __init__(self, spent: float, limit: float) -> None:
        self._spent = spent
        self._limit = limit
        self._lock = asyncio.Lock()

    async def add(self, cost: float) -> None:
        async with self._lock:
            self._spent += cost

    def exceeded(self) -> bool:
        # 0 表示无限制，与 _over_budget 一致；必须先检查限制非零，否则零花费也会被判为超限并跳过全部任务。
        return bool(self._limit) and self._spent >= self._limit

    @property
    def spent(self) -> float:
        return self._spent


# 统一换行、弯引号等排版差异，避免误拒绝来源中真实存在的引文；此归一化不会放宽对伪造内容的检查。
_QUOTE_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _norm_text(text: str) -> str:
    """忽略空白和大小写，并将智能标点折叠为 ASCII，得到用于匹配的文本形式。"""
    return " ".join((text or "").translate(_QUOTE_FOLD).split()).lower()


def _norm_url(url: str) -> str:
    u = (url or "").strip().rstrip("/").lower()
    if u.startswith("https://"):
        u = u[8:]
    elif u.startswith("http://"):
        u = u[7:]
    if u.startswith("www."):
        u = u[4:]
    return u.rstrip("/")


def record_tool_output(
    seen: dict[str, str], tool_name: str, observation, grades: dict[str, str] | None = None
) -> None:
    """按 URL 累积工具实际返回的文本，这是执行器真正看到过什么的唯一记录。

    submit_evidence 的 source_url、source_title 和 snippet 都由模型编写，
    引擎随后只覆盖 task_id；缺少这份记录便无法核查引文。

    键保留工具报告的原始 URL，包括协议。匹配时才由 _seen_get 和 verify_evidence_snippets
    忽略协议及大小写；若存储时就归一化，会丢掉读者能直接打开的地址。
    此前片段修复曾将 a.example/paper 写回 source_url，导致下游链接失效。

    可选的 grades 记录来源是网页正文还是搜索摘要，供 Evidence.attested_against 使用。
    read_webpage 总是写入 FETCHED_BODY 并覆盖旧的 SEARCH_SNIPPET，因为正文的证据等级更高；
    web_search 使用 setdefault，不能让摘要降低已经抓取过正文的 URL 的等级。"""
    if tool_name == "read_webpage" and isinstance(observation, dict):
        body = observation.get("text") or ""
        for key in (observation.get("url"), observation.get("final_url")):
            if key:
                seen[key] = seen.get(key, "") + "\n" + body
                if grades is not None:
                    grades[key] = "FETCHED_BODY"
    elif tool_name == "web_search" and isinstance(observation, list):
        # 片段也可合法引用搜索摘要，因此搜索结果同样计入已见文本。
        for hit in observation:
            if isinstance(hit, dict) and hit.get("url"):
                key = hit["url"]
                seen[key] = seen.get(key, "") + "\n" + (hit.get("snippet") or "")
                if grades is not None:
                    grades.setdefault(key, "SEARCH_SNIPPET")


def _seen_get(seen: dict[str, str], url: str | None) -> str:
    """读取 URL 对应的已抓取文本；匹配时忽略协议、大小写和结尾斜杠。"""
    target = _norm_url(url or "")
    if not target:
        return ""
    return "\n".join(text for key, text in seen.items() if _norm_url(key) == target)


def _seen_grade(grades: dict[str, str], url: str | None) -> str | None:
    """按 _seen_get 的匹配规则读取 URL 的最高核验等级。

    同一 URL 既被搜索又被抓取时，FETCHED_BODY 优先于 SEARCH_SNIPPET，
    与 record_tool_output 的优先级一致；合并文本中的有效引文按更强的来源核验。"""
    target = _norm_url(url or "")
    if not target:
        return None
    matched = {g for key, g in grades.items() if _norm_url(key) == target}
    if "FETCHED_BODY" in matched:
        return "FETCHED_BODY"
    if "SEARCH_SNIPPET" in matched:
        return "SEARCH_SNIPPET"
    return None


def verify_evidence_snippets(
    evidence: list[dict], seen: dict[str, str], grades: dict[str, str] | None = None
) -> list[dict]:
    """清空未出现在工具实际返回文本中的片段。

    无法解析的引用会显示警告；伪造片段却可能指向真实 URL，并把从未出现的文字显示成引文，
    这更违背产品的可验证性承诺。真实运行曾将不在 PMC3543521 中的文字归给该论文，
    而界面因缺少逐字校验显示为正常引用。

    只清空 snippet，不删除整个证据块：key_fact 和 URL 仍可能有效；_number_sources
    会跳过空片段，使引用失去引文而不是展示伪造内容。返回未通过的证据块，供日志记录。

    若片段逐字存在于恰好一个其他已抓取来源中，则修正 URL，并用 snippet_reattributed
    保留模型原先声称的来源，让读者能看见修复过程。必须恰好命中一个来源；
    多页同时包含该句时不能猜测出处，仍须清空。写回的 URL 必须保留工具报告的完整协议。

    传入 grades 时，为通过核验的证据写入 attestation_grade（FETCHED_BODY 或 SEARCH_SNIPPET），
    使 record_evidence 能持久化 ATTESTED 和 attested_against，而不是继续标记 UNCHECKED。
    该参数保持可选以兼容现有调用和测试，生产调用始终传入。"""
    fabricated: list[dict] = []

    for chunk in evidence:
        snippet = chunk.get("snippet") or ""
        if not snippet.strip():
            continue

        claimed = chunk.get("source_url", "")
        norm_snip = _norm_text(snippet)
        if norm_snip in _norm_text(_seen_get(seen, claimed)):
            if grades is not None:
                grade = _seen_grade(grades, claimed)
                if grade:
                    chunk["attestation_grade"] = grade
            continue

        matches = [url for url, text in seen.items() if norm_snip in _norm_text(text)]
        # 重定向可能同时记录 url 与 final_url，多个键指向同一页面仍算一个来源，不构成出处歧义。
        if len({_norm_url(u) for u in matches}) == 1:
            chunk["source_url"] = matches[0]
            chunk["snippet_reattributed"] = claimed
            if grades is not None:
                grade = _seen_grade(grades, matches[0])
                if grade:
                    chunk["attestation_grade"] = grade
            continue

        chunk["snippet"] = ""
        chunk["snippet_unverified"] = True
        fabricated.append(chunk)
    return fabricated


def _task_key(task: dict) -> str:
    """评审结果与重试次数使用字符串键，因为检查点按 JSON 存储状态。"""
    return str(task.get("id"))


def _pending(state: AgentState, max_retries: int) -> list[dict]:
    """返回尚未通过评审、且重试次数未耗尽的待研究任务。"""
    verdicts = state.get("verdicts") or {}
    retries = state.get("retries") or {}
    out = []
    for task in state.get("tasks") or []:
        key = _task_key(task)
        if (verdicts.get(key) or {}).get("passed"):
            continue
        if retries.get(key, 0) >= max_retries:
            continue
        out.append(task)
    return out


class submit_evidence(BaseModel):
    """提交此任务最终收集到的证据。"""

    evidence: list[EvidenceChunk] = Field(default_factory=list, description="List of cited facts")

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return {"evidence": []}
        if isinstance(data, list):
            return {"evidence": data}
        if isinstance(data, dict):
            raw = data.get("evidence")
            if isinstance(raw, str):
                try:
                    data["evidence"] = json.loads(raw)
                except Exception:
                    data["evidence"] = []
        return data


def _parse_submission(args: Any) -> list[dict]:
    """将一次 submit_evidence 的参数校验为证据字典；不合格时抛出异常。

    校验前按配置截断片段（internal/07 Phase 3）。该限制只能收紧 EvidenceChunk.snippet
    自身的 max_length=500，不能放宽；配置默认值也为 500。"""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if isinstance(args, list):
        args = {"evidence": args}
    elif not isinstance(args, dict):
        args = {}

    raw_evidence = args.get("evidence")
    if isinstance(raw_evidence, str):
        try:
            args["evidence"] = json.loads(raw_evidence)
        except Exception:
            args["evidence"] = []

    snip_cap = get_run_config().snippet_max_chars
    for item in args.get("evidence", []) or []:
        if isinstance(item, dict) and isinstance(item.get("snippet"), str):
            item["snippet"] = item["snippet"][:snip_cap]
    return [e.model_dump() for e in submit_evidence.model_validate(args).evidence]


async def _forced_submit(
    sid: str, task: dict, seen_text: dict[str, str], guard: _BudgetGuard
) -> tuple[list[dict], float, int, int]:
    """模型未主动提交证据时，明确要求它从已有来源中提取。

    工具循环允许调用 submit_evidence，却不能保证模型会选择它。真实运行曾成功搜索和抓取，
    却持续搜索直到轮数耗尽，两次得到空证据，最后错误地建议用户检查正常工作的搜索提供方。
    模型不愿调用某个工具，不能成为丢弃引擎已获取文本的理由。

    此步骤只接收 seen_text，且返回结果仍由 verify_evidence_snippets 核验，
    因此模型记忆不能绕过来源校验成为引用。

    custom:auto/* 每次调用可能解析到不同模型；有些模型忽略指定的 tool_choice，
    只返回无工具调用的消息。因此先强制选择提交工具，不成功再采用结构化输出，
    以不同请求形状补救，而非原样重复。未抓取到任何来源时直接返回空列表，避免要求模型编造。"""
    sources = {url: text for url, text in seen_text.items() if (text or "").strip()}
    if not sources or guard.exceeded():
        return [], 0.0, 0, 0

    # 优先提供文本最丰富的来源，避免单行搜索命中挤掉可引用的网页正文。
    # 裁去尾部只缩小发送内容，verify_evidence_snippets 仍对完整 seen_text 核验。
    ranked = sorted(sources.items(), key=lambda kv: len(kv[1]), reverse=True)
    digest = json.dumps(
        {url: text[:_FORCED_SUBMIT_SOURCE_CHARS] for url, text in ranked[:_FORCED_SUBMIT_SOURCES]},
        indent=1,
        default=str,
    )
    messages = [
        SystemMessage(content=system_prompt("executor.main")),
        HumanMessage(
            content=(
                f"Task {task['id']}: {task['query']}\n\n"
                "Stop searching. Below is every source already fetched for this task, "
                "keyed by URL. Report the findings it supports now. "
                "Each snippet must be copied VERBATIM from the text of its own URL — a "
                "quotation that is not in its source is dropped. If a source supports "
                "nothing relevant, leave it out.\n\n"
                f"<untrusted_web_content>\n{digest}\n</untrusted_web_content>"
            )
        ),
    ]

    cost = 0.0
    i_tot = o_tot = 0
    evidence: list[dict] = []

    async def charge(delta: float, di: int, do: int) -> None:
        """每次调用只记账一次；预算检查接收本次增量，不能接收累计总额。

        两种提取机制可能先后运行；若提交累计数，第二次会重复计算第一次的花费，
        记录研究并未实际发生的支出，违背记录真实调用的原则。"""
        nonlocal cost, i_tot, o_tot
        cost += delta
        i_tot += di
        o_tot += do
        await guard.add(delta)

    for mechanism in ("tool_choice", "structured"):
        if guard.exceeded():
            break
        try:
            if mechanism == "tool_choice":
                # 优先复用主动提交路径的模式、校验和截断逻辑，使两条写入 evidence 的路径共用实现。
                model = get_llm("executor").bind_tools(
                    [submit_evidence], tool_choice="submit_evidence"
                )
                resp = await traced_ainvoke(model, messages, role="executor")
                await charge(estimate_cost(resp, "executor"), *token_counts(resp))
                calls = [
                    c
                    for c in (getattr(resp, "tool_calls", None) or [])
                    if c["name"] == "submit_evidence"
                ]
                if not calls:
                    logger.warning(
                        "forced_submit_returned_nothing",
                        session_id=sid,
                        task_id=task["id"],
                        mechanism=mechanism,
                        served_model=served_model_id(resp),
                        text=(text_of(resp) or "")[:200],
                    )
                    continue
                with trace_span(
                    "submit_evidence",
                    kind="tool",
                    inputs=calls[0]["args"],
                    metadata={
                        "tool_name": "submit_evidence",
                        "tool_call_id": calls[0].get("id", ""),
                    },
                ) as submit_span:
                    evidence = _parse_submission(calls[0]["args"])
                    submit_span.observe(output=evidence)
            else:
                parsed, c, di, do = await _structured("executor", messages, ExecutorOutput)
                await charge(c, di, do)
                if parsed is None:
                    logger.warning(
                        "forced_submit_returned_nothing",
                        session_id=sid,
                        task_id=task["id"],
                        mechanism=mechanism,
                        error=(_last_api_error.get() or "")[:200],
                    )
                    continue
                evidence = _parse_submission(
                    {"evidence": [e.model_dump() for e in parsed.evidence]}
                )
        except Exception as e:  # noqa: BLE001
            # 失败时保守处理并明确记录：提供方错误导致未能提取证据，不能说成来源没有支持任何结论。
            logger.warning(
                "forced_submit_failed",
                session_id=sid,
                task_id=task["id"],
                mechanism=mechanism,
                error=str(e)[:400],
            )
            continue

        if evidence:
            await emit(
                sid,
                "agent_log",
                agent="executor",
                message=(
                    f"Extracted {len(evidence)} evidence item(s) for task {task['id']} from "
                    f"the {len(sources)} source(s) already fetched — the model did not "
                    f"submit on its own"
                ),
                detail={
                    "task_id": task["id"],
                    "forced": True,
                    "mechanism": mechanism,
                    "evidence": evidence,
                },
            )
            return evidence, cost, i_tot, o_tot

    return [], cost, i_tot, o_tot


async def _research_one(state: AgentState, task: dict, guard: _BudgetGuard) -> dict:
    """为单个任务收集证据，返回 evidence、cost、tokens_in 和 tokens_out。"""
    sid = state["session_id"]
    feedback = ((state.get("verdicts") or {}).get(_task_key(task)) or {}).get(
        "feedback_for_executor"
    )

    await emit(
        sid,
        "agent_log",
        agent="executor",
        message=f"Researching: '{task['query']}'",
        detail={
            "task_id": task["id"],
            "query": task["query"],
            "feedback": feedback,
            "model": get_run_config().model_for("executor"),
        },
    )

    model = get_llm("executor").bind_tools(EXECUTOR_TOOLS + [submit_evidence])
    messages: list = [
        SystemMessage(content=system_prompt("executor.main")),
        HumanMessage(content=f"Task {task['id']}: {task['query']}"),
    ]
    if feedback:
        messages.append(HumanMessage(content=f"Previous attempt insufficient. Fix: {feedback}"))

    cost = 0.0
    i_tot = o_tot = 0
    evidence: list[dict] = []
    submitted = False
    # 按 URL 保存工具真正返回的文本，作为执行器已读内容的唯一记录；供 verify_evidence_snippets 核验。
    seen_text: dict[str, str] = {}
    # 记录各 URL 来自正文抓取还是搜索摘要，以便持久化引用具体经过哪种来源的核验。
    seen_grades: dict[str, str] = {}
    # 记录本任务已打开的页面；重复请求从 seen_text 返回，避免反复抓取同一页面并浪费模型轮次。
    fetched: set[str] = set()
    max_rounds, max_pages = _limits_for(state.get("research_depth"))
    for _round in range(max_rounds):
        if guard.exceeded():
            logger.warning("executor_budget_stop", session_id=sid, task_id=task["id"])
            break
        if len(fetched) >= max_pages:
            # 已获取足够来源，交给后续强制提取步骤生成证据。
            logger.info("executor_read_limit", session_id=sid, task_id=task["id"])
            break
        resp = await traced_ainvoke(model, messages, role="executor")
        round_cost = estimate_cost(resp, "executor")
        cost += round_cost
        await guard.add(round_cost)
        di, do = token_counts(resp)
        i_tot += di
        o_tot += do
        messages.append(resp)
        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            break
        is_done = False

        # 并发执行本轮工具调用。
        #
        # 顺序抓取会串行累积多个页面的超时；分散到多个模型轮次则还会增加昂贵的模型往返。
        # 同一轮并发抓取只承担最慢一次抓取的等待时间。
        # 结果仍按 tool_calls 顺序消费，使 seen_text、日志及 ToolMessage 与顺序版本保持一致，
        # 不受抓取完成先后影响。
        pending_calls: list[dict] = []
        for call in tool_calls:
            if call["name"] == "submit_evidence":
                try:
                    with trace_span(
                        "submit_evidence",
                        kind="tool",
                        inputs=call["args"],
                        metadata={
                            "tool_name": "submit_evidence",
                            "tool_call_id": call.get("id", ""),
                        },
                    ) as submit_span:
                        evidence = _parse_submission(call["args"])
                        submit_span.observe(output=evidence)
                    thought_snippet = text_of(resp) if resp else ""
                    await emit(
                        sid,
                        "agent_log",
                        agent="executor",
                        message=f"Submitted {len(evidence)} evidence item(s) for task {task['id']}",
                        detail={
                            "task_id": task["id"],
                            "thought": thought_snippet[:2000] if thought_snippet else None,
                            "evidence": evidence,
                        },
                    )
                    submitted = True
                    is_done = True
                    break
                except Exception as e:  # noqa: BLE001
                    # 明确记录参数校验失败；只写入工具观察会让反复传错参数与从未提交证据无法区分，两者处理方法不同。
                    logger.warning(
                        "submit_evidence_rejected",
                        session_id=sid,
                        task_id=task["id"],
                        forced=False,
                        error=str(e)[:400],
                    )
                    messages.append(
                        ToolMessage(
                            content=json.dumps(
                                f"submit_evidence validation error: {e}", default=str
                            ),
                            tool_call_id=call["id"],
                        )
                    )
                    continue
            pending_calls.append(call)

        if is_done:
            break

        async def _observe(call: dict, *, cached: bool) -> object:
            """返回一次工具调用的观察结果；不抛出异常，失败也作为观察结果交给模型。"""
            if cached:
                # seen_text 已包含该页面，重复抓取不会增加信息，却会挤占用于总结的模型轮次。
                url = (call.get("args") or {}).get("url")
                observation = {
                    "url": url,
                    "text": _seen_get(seen_text, url),
                    "error": None,
                    "note": "already read this page — quote it or move on",
                }
                with trace_span(call["name"], kind="tool_cache", inputs=call.get("args")) as span:
                    span.observe(output=observation)
                return observation
            tool = _TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                return f"unknown tool {call['name']}"
            try:
                with trace_span(
                    call["name"],
                    kind="tool",
                    inputs=call.get("args"),
                    metadata={"tool_name": call["name"], "tool_call_id": call.get("id", "")},
                ) as span:
                    observation = await tool.ainvoke(call["args"])
                    span.observe(output=observation)
                    if isinstance(observation, dict) and observation.get("error"):
                        span.status = "business_error"
                    return observation
            except Exception as e:  # noqa: BLE001
                return f"tool error: {e}"

        # 派发前逐次决定是否命中缓存，不能从整个批次共享的集合判断。
        # 否则将重复 URL 标记为缓存命中时，首次请求也会被误标记，整轮从空缓存读取且没有真正抓取。
        # 首次请求抓取，当前轮或之前轮次的重复请求才从 seen_text 读取。
        seen_before = set(fetched)
        dispatch: list[tuple[dict, bool]] = []
        for call in pending_calls:
            url = (call.get("args") or {}).get("url")
            # 参数尚由模型提供且未经校验；列表、数字等错误形状应交给工具模式拒绝并允许模型修正，不能先 strip 导致整个节点失败。
            if call["name"] == "read_webpage" and isinstance(url, str) and url.strip():
                norm = _norm_url(url)
                dispatch.append((call, norm in seen_before))
                seen_before.add(norm)
                fetched.add(norm)
            else:
                dispatch.append((call, False))

        observations = await asyncio.gather(
            *(_observe(call, cached=cached) for call, cached in dispatch)
        )

        for (call, _cached), observation in zip(dispatch, observations, strict=True):
            record_tool_output(seen_text, call["name"], observation, grades=seen_grades)

            tool_args = call.get("args") or {}
            search_query = tool_args.get("query") or tool_args.get("url") or ""
            thought_snippet = text_of(resp) if resp else ""

            msg_suffix = f': "{search_query}"' if search_query else ""
            await emit(
                sid,
                "agent_log",
                agent="executor",
                message=f"Used {call['name']}{msg_suffix}",
                detail={
                    "task_id": task["id"],
                    "tool": call["name"],
                    "args": tool_args,
                    "thought": thought_snippet[:2000] if thought_snippet else None,
                    "observation": str(observation)[:4000] if observation else None,
                },
            )
            messages.append(
                ToolMessage(content=json.dumps(observation, default=str), tool_call_id=call["id"])
            )

    # 工具轮数耗尽仍未提交时，显式请求提取；不能丢弃已经获取的文本，见 _forced_submit。
    if not submitted:
        evidence, fcost, fi, fo = await _forced_submit(sid, task, seen_text, guard)
        cost += fcost
        i_tot += fi
        o_tot += fo

    # 在下游将片段作为逐字引文展示前进行核验。
    #
    # 当前仅在真实模式启用，这是模拟数据的限制：脚本执行器可能未调用工具就提交证据，
    # seen_text 为空，会使全部模拟引文被清空。理想修复是让模拟来源和引用自洽；
    # 在此之前，CI 的模拟模式无法覆盖本检查的回归，而真实模型伪造引文的风险仍必须防护。
    fabricated = (
        verify_evidence_snippets(evidence, seen_text, grades=seen_grades)
        if get_run_config().llm_mode != "fake"
        else []
    )
    if fabricated:
        logger.warning(
            "evidence_snippet_unverified",
            session_id=sid,
            task_id=task["id"],
            count=len(fabricated),
            urls=[c.get("source_url", "") for c in fabricated][:5],
        )
        await emit(
            sid,
            "agent_log",
            agent="executor",
            message=(
                f"Dropped {len(fabricated)} snippet(s) that did not appear in the fetched "
                f"source — a quote we cannot find is not evidence"
            ),
            detail={"task_id": task["id"], "urls": [c.get("source_url", "") for c in fabricated]},
        )

    # 引用重新归属也属于修复，不能显示成未经改动的原始输出；显式报告来源变更，让用户知道发生了什么。
    repaired = [c for c in evidence if c.get("snippet_reattributed")]
    if repaired:
        logger.warning(
            "evidence_snippet_reattributed",
            session_id=sid,
            task_id=task["id"],
            count=len(repaired),
            moves=[(c["snippet_reattributed"], c.get("source_url", "")) for c in repaired][:5],
        )
        await emit(
            sid,
            "agent_log",
            agent="executor",
            message=(
                f"Re-attributed {len(repaired)} quotation(s) to the page they were actually "
                f"found on — the model cited a source that does not contain them"
            ),
            detail={
                "task_id": task["id"],
                "moves": [
                    {"claimed": c["snippet_reattributed"], "actual": c.get("source_url", "")}
                    for c in repaired
                ],
            },
        )

    for e in evidence:
        e["task_id"] = task["id"]

    await emit(
        sid,
        "agent_log",
        agent="executor",
        message=f"Gathered {len(evidence)} source(s) for task {task['id']}",
        detail={
            "task_id": task["id"],
            "source_count": len(evidence),
            "evidence": evidence,
        },
    )
    # sources 记录检索实际取得的来源数，与执行器从中提取的证据不同；据此区分检索失败和提取失败，见 _no_research_reason。
    return {
        "evidence": evidence,
        "cost": cost,
        "in": i_tot,
        "out": o_tot,
        "sources": len([t for t in seen_text.values() if (t or "").strip()]),
    }


async def executor_node(state: AgentState) -> dict:
    """并行研究全部待办任务，并以 max_parallel_tasks 限制并发数。"""
    sid = state["session_id"]
    cfg = get_run_config()
    pending = _pending(state, cfg.max_critic_loops)
    round_no = state.get("research_round", 0) + 1

    if len(pending) > 1:
        await emit(
            sid,
            "agent_log",
            agent="executor",
            message=f"Round {round_no}: researching {len(pending)} tasks in parallel",
            detail={
                "round": round_no,
                "task_ids": [t["id"] for t in pending],
                "max_parallel": cfg.max_parallel_tasks,
                "model": cfg.model_for("executor"),
            },
        )

    guard = _BudgetGuard(state.get("cost_usd", 0.0), cfg.max_cost_per_session_usd)
    semaphore = asyncio.Semaphore(max(1, cfg.max_parallel_tasks))

    async def bounded(task: dict) -> dict:
        async with semaphore:
            if guard.exceeded():
                logger.warning("executor_not_dispatched", session_id=sid, task_id=task["id"])
                return {"evidence": [], "cost": 0.0, "in": 0, "out": 0, "sources": 0}
            return await _research_one(state, task, guard)

    # 结果按参数顺序返回，保持后续合并确定性。
    # 使用 gather_or_cancel 而非 asyncio.gather，确保一项失败后其余任务不会继续研究，
    # 也不会在宿主关闭数据库会话后继续写入事件（research_engine/concurrency.py）。
    results = await gather_or_cancel(*(bounded(t) for t in pending))

    fresh = {_task_key(t): r["evidence"] for t, r in zip(pending, results, strict=True)}
    previous = state.get("evidence") or []

    # 按任务定义顺序重建证据列表，本轮任务使用新证据，其他任务保留旧证据；不能因完成顺序改变而导致相同研究的引用编号漂移。
    merged: list[dict] = []
    for task in state.get("tasks") or []:
        key = _task_key(task)
        if key in fresh:
            merged.extend(fresh[key])
        else:
            merged.extend(e for e in previous if str(e.get("task_id")) == key)

    cost = sum(r["cost"] for r in results)
    i_tot = sum(r["in"] for r in results)
    o_tot = sum(r["out"] for r in results)
    # 保留整个研究过程中观测到的最大来源数，不能让后续较差的检索结果抹掉此前成功检索的事实。
    sources_seen = max(state.get("sources_seen", 0), sum(r.get("sources", 0) for r in results))
    return {
        "evidence": merged,
        "research_round": round_no,
        "sources_seen": sources_seen,
        **_acc(state, cost, i_tot, o_tot),
    }


async def _criticize_one(
    state: AgentState, task: dict
) -> tuple[str, CriticVerdict, float, int, int]:
    task_evidence = [
        e for e in state.get("evidence", []) if str(e.get("task_id")) == _task_key(task)
    ]

    # 已知证据不足配置下限时直接保守判失败，不浪费模型调用；0 表示未启用下限（internal/07 Phase 3）。
    min_sources = get_run_config().min_sources_per_task
    if min_sources > 0 and len(task_evidence) < min_sources:
        verdict = CriticVerdict(
            passed=False,
            confidence=1.0,
            reasons=[f"only {len(task_evidence)} source(s) gathered, {min_sources} required"],
            feedback_for_executor=f"Gather at least {min_sources} sources before resubmitting.",
        )
        return _task_key(task), verdict, 0.0, 0, 0

    messages = [SystemMessage(content=system_prompt("critic.research"))]
    allowed = get_run_config().corpus_allowed_ids
    selection_note = ""
    if allowed is not None:
        # 在用户评审提示之后补充，不覆盖用户指令；只有一两个文件的指定范围无法满足两个独立来源的规则（internal/rfcs/V3.1 §9，AC-19）。
        messages.append(SystemMessage(content=system_prompt("critic.selection_addendum")))
        selection_note = (
            f"Selected files: {len(allowed)}. A passing task needs evidence from "
            f"{min(2, len(allowed))} different selected file(s).\n\n"
        )
    messages += [
        HumanMessage(
            content=f"{selection_note}Task: {task['query']}\n\n<untrusted_web_content>\n"
            f"{json.dumps(task_evidence, indent=2)}\n</untrusted_web_content>"
        ),
    ]
    parsed, cost, i, o = await _structured("critic", messages, CriticVerdict)
    if parsed is None:
        # 保守失败（docs/11 §1 rule 2）：无效评审结果算未通过，不能默认放行。
        parsed = CriticVerdict(
            passed=False,
            confidence=0.0,
            reasons=["critic output invalid — failing closed"],
            feedback_for_executor="Re-gather clearer, well-cited evidence.",
        )
    return _task_key(task), parsed, cost, i, o


async def critic_node(state: AgentState) -> dict:
    """并行评审本轮研究过的全部任务。"""
    sid = state["session_id"]
    cfg = get_run_config()
    pending = _pending(state, cfg.max_critic_loops)

    await emit(
        sid,
        "agent_log",
        agent="critic",
        message=f"Evaluating evidence for {len(pending)} task(s)…",
        detail={"task_ids": [t["id"] for t in pending], "model": cfg.model_for("critic")},
    )

    semaphore = asyncio.Semaphore(max(1, cfg.max_parallel_tasks))

    async def bounded(task: dict):
        async with semaphore:
            return await _criticize_one(state, task)

    results = await gather_or_cancel(*(bounded(t) for t in pending))

    verdicts = dict(state.get("verdicts") or {})
    retries = dict(state.get("retries") or {})
    cost, i_tot, o_tot = 0.0, 0, 0

    for key, verdict, c, i, o in results:
        verdicts[key] = verdict.model_dump()
        if not verdict.passed:
            retries[key] = retries.get(key, 0) + 1
        cost += c
        i_tot += i
        o_tot += o
        await emit(
            sid,
            "agent_log",
            agent="critic",
            message=(
                f"✅ PASS task {key}"
                if verdict.passed
                else f"❌ FAIL task {key} (retry {retries[key]})"
            ),
            detail={"task_id": key, **verdict.model_dump()},
        )

    return {
        "verdicts": verdicts,
        "retries": retries,
        **_acc(state, cost, i_tot, o_tot),
    }


# ── 引用忠实度核验（docs/12 M5）────────────────────────────────────
#
# 评测按主张所引来源的片段判断支持情况；综合器可能沿用执行器的 key_fact 改写，
# 写出逐字片段并不支持的主张。因此在进入人工审核前采用同一规则逐项检查：
# 支持则保留，不支持则移除引用标记并明确提示，避免空有引用造成可信假象。

# 提示必须以句末标点结束，否则评测按标点加空白切句时，会将提示并入下一句，使下一条有支持的主张被误判。
_VERIFIED_NOTE = " *(citation could not be verified)*."
# 切句必须与评测的 claims.split_sentences 和 claim_lines 完全一致，包括句首前瞻及缩写重组。
# 此处直接导入相同对象，不再复制正则；tests/test_claim_extraction_parity.py
# 同时检查对象身份和逐句提取结果，防止核验与评测面对不同片段。
_VSPLIT_RE = claims.SENTENCE_SPLIT_RE
_VABBREV_TAIL_RE = claims.ABBREV_TAIL_RE
_VLIST_MARKER_RE = claims.LIST_MARKER_RE
# Limitations 描述证据未覆盖的范围，是保留性说明而非事实主张；此处与评测 v3 一样排除该部分。
_VLIMITATIONS_HEADING_RE = claims.LIMITATIONS_HEADING_RE
# 以指代词开头的引用句在评测中被单独判断，前句的指代对象不在上下文中，会显得无据。
# 保持与评测相同的独立切句规则，并确定性移除此类引用。
_VDEICTIC_RE = re.compile(r"^(?:This|These|Those|That|It|Such)\b", re.I)
# 加粗标签前缀会让整句超出来源片段包含的文字；与指代句一样移除引用，综合器提示规则 6 也禁止这种写法。
_VLABEL_RE = re.compile(r"^\*\*[^*]+\*\*\s*:")
# 确定性预检：主张中的每个数字类词元都必须逐字出现在至少一个所引片段中。
# 小型本地核验器可能放行貌似合理的杜撰数字；对数字、百分比和年份做字面匹配可捕获该类偏差。
_VNUM_RE = re.compile(r"\d+(?:\.\d+)?%?|\b\d{4}\b")


def _claim_numbers(claim: str) -> list[str]:
    """提取主张中的数字类词元，排除引用标记。"""
    return _VNUM_RE.findall(re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", claim))


def _numbers_grounded(claim: str, snippets: str) -> bool:
    # 按词边界匹配，不能用 50% 为数字 5 提供依据，否则短数字会产生误匹配。
    return all(
        re.search(rf"(?<![0-9.]){re.escape(n)}(?![0-9.])", snippets) for n in _claim_numbers(claim)
    )


#: 此处独有的来源区块边界：除 Markdown 标题外，单独一行 Sources 也结束正文。
#: 不扩大共享的 claims.SOURCES_HEADING_RE，因为那会改变评测对主张的定义，
#: 需要同时递增 METRICS_VERSION；tests/test_claim_extraction_parity.py 固定此有意差异。
#:
#: 两条分支不能合并为可选的 #（#{0,6}）：可选标记会把以 Sources 开头的普通正文也当作边界，
#: 导致后续主张跳过核验而显得更可信。标题分支允许 Sources and References 等后缀，
#: 无标题标记的分支则必须匹配整行，确保它只是一行标签。
_VSOURCES_RE = re.compile(
    r"(?i)(?:"
    r"#{1,6}\s*(?:sources|references|citations|bibliography)\b"
    r"|"
    r"(?:sources|references|citations|bibliography)\s*:?\s*$"
    r")"
)


def _cited_claims(draft: str) -> list[str]:
    """提取草稿正文中带引用的事实句，排除来源列表。

    句子切分、列表标记清理、长度下限和引用模式均来自 research_engine.claims，
    与评测使用同一套定义，避免测量和检查面对不同的句子片段。

    此处有两个有意保留的差异：
    1. 单独一行的 Sources 也会结束正文（_VSOURCES_RE），而评测要求 Markdown 标题。
       只有整行仅含该标签才算边界，以 Sources 开头的普通句子仍是主张（issue #48）。
    2. 评测还会跳过引擎生成的 Conflicting evidence 区块；此处无需处理，
       因为 synthesizer_node 会在忠实度检查结束后才附加该区块。"""
    out: list[str] = []
    skipping = False
    for raw in (draft or "").splitlines():
        line = raw.strip()
        # 必须先判断来源边界，再跳过其他标题；否则 ## Sources 被 continue 提前跳过，来源列表会被误当主张，URL 中的数字又让全部书目被标成未核验。
        if _VSOURCES_RE.match(line):
            break
        if not line or line.startswith("#"):
            if line:
                skipping = bool(_VLIMITATIONS_HEADING_RE.match(line))
            continue
        if skipping:
            continue
        content = _VLIST_MARKER_RE.sub("", line)
        for sentence in claims.split_sentences(content):
            s = sentence.strip()
            if not claims.is_claim_sentence(s):
                continue
            if claims.CITE_RE.search(s):
                out.append(s)
    return out


async def _verifier_verdicts(
    claim_evidence: list[tuple[str, str]],
) -> tuple[list[bool], float, int, int]:
    """请求评审模型判断各主张是否受到对应证据支持，返回 (verdicts, cost, tokens_in, tokens_out)。

    真实模式调用模型；模拟模式在 fakes.py 中对全部主张回答 YES。
    提示词和逐行格式有意与 evals/harness.judge_citation_support 一致，
    使图内核验和测量该核验效果的评测遵循同一判定规则。"""
    llm = get_llm("critic")
    BATCH = 4
    verdicts: list[bool] = []
    cost = i_tot = o_tot = 0
    for start in range(0, len(claim_evidence), BATCH):
        batch = claim_evidence[start : start + BATCH]
        blocks = [
            f"Claim {i}: {claim}\nEvidence {i}:\n{snippets}"
            for i, (claim, snippets) in enumerate(batch, start=1)
        ]
        resp = await traced_ainvoke(
            llm,
            [
                SystemMessage(content=system_prompt("critic.citation_verify")),
                HumanMessage(content="\n\n".join(blocks)),
            ],
            role="critic",
        )
        cost += estimate_cost(resp, "critic")
        di, do = token_counts(resp)
        i_tot += di
        o_tot += do
        text = text_of(resp)
        ruled = {
            int(m.group(1)): m.group(2).upper() == "YES"
            for m in re.finditer(r"Claim\s+(\d+)\s*:\s*(YES|NO)", text, re.IGNORECASE)
        }
        for i in range(1, len(batch) + 1):
            # 核验器未作出判断时保留引用；只有明确判为不支持的主张才移除标记。
            verdicts.append(ruled.get(i, True))
    return verdicts, cost, i_tot, o_tot


async def _verify_citation_fidelity(
    sid: str, draft: str, sources: list[dict]
) -> tuple[str, float, int, int]:
    """逐项核对主张及其所引来源片段，去掉证据不支持的引用标记；返回草稿、费用及输入输出词元数。"""
    claims = _cited_claims(draft)
    if not claims or not sources:
        return draft, 0.0, 0, 0

    by_index = {s.get("index"): s for s in sources if isinstance(s, dict)}
    claim_evidence: list[tuple[str, str]] = []
    mechanically_unsupported: set[str] = set()
    for claim in claims:
        if _VDEICTIC_RE.match(claim) or _VLABEL_RE.match(claim):
            # 评测单独判断此句时，缺少指代对象或片段未包含标签会使它无据；这里确定性移除，补充综合器规则 5–6 的防护。
            mechanically_unsupported.add(claim)
            continue
        nums: list[int] = []
        for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", claim):
            nums.extend(int(p) for p in m.group(1).split(","))
        if any(by_index.get(n) is None for n in nums):
            # 指向不存在来源的标记没有引用任何证据；以空证据判断只能失败。
            mechanically_unsupported.add(claim)
            continue
        snippets = "\n".join(
            f"- {t}"
            for n in nums
            for s in [by_index.get(n)]
            if s
            for t in (s.get("snippets") or ([s["snippet"]] if s.get("snippet") else []))
        )
        # 在模型判定前先作确定性数字检查：片段未逐字包含的数字不可能受到支持，不能由模型的肯定回答补救。
        if not _numbers_grounded(claim, snippets):
            mechanically_unsupported.add(claim)
        claim_evidence.append((claim, snippets))

    todo = [(c, s) for c, s in claim_evidence if c not in mechanically_unsupported]
    try:
        verdicts, cost, i_tot, o_tot = await _verifier_verdicts(todo)
    except Exception as e:  # noqa: BLE001
        # 核验器失败时保留引用，仍由警告标记和评测揭示问题；但机械检查已确认无依据的引用仍须移除。
        logger.warning("citation_verify_unavailable", session_id=sid, error=str(e)[:200])
        verdicts, cost, i_tot, o_tot = [], 0.0, 0, 0

    ok_by_claim: dict[str, bool] = {}
    if verdicts:  # 核验失败时为空；未作出判断便不据此移除引用。
        for (claim, _), ok in zip(todo, verdicts, strict=True):
            ok_by_claim[claim] = ok

    result = draft
    stripped = 0
    for claim in claims:
        if ok_by_claim.get(claim, claim not in mechanically_unsupported):
            continue
        stripped += 1
        # 提示自身包含结束标点（见 _VERIFIED_NOTE），因此去掉主张末尾原有标点。
        cleaned = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", claim).strip().rstrip(".!?")
        result = result.replace(claim, cleaned + _VERIFIED_NOTE, 1)

    if stripped:
        await emit(
            sid,
            "agent_log",
            agent="synthesizer",
            message=f"Citation check: {stripped} claim(s) not supported by their snippets — markers removed",
            detail={"stripped": stripped, "checked": len(claims)},
        )
    return result, cost, i_tot, o_tot


async def contradiction_detector_node(state: AgentState) -> dict:
    """展示证据之间的冲突主张（docs/12 M11），不替用户裁决。

    对数量受限的片段集合只做一次有界调用，费用随研究规模增长，而非随来源数量的平方增长。
    使用评审器的模型路由，因为这是同类的零温度判断任务；不为每次研究的一次调用新增第六个角色。
    检测器不可用或结果无法解析时不展示任何冲突，并在智能体日志中说明；不能编造冲突。"""
    sid = state["session_id"]
    by_source = contradictions.group_snippets_by_source(state.get("evidence", []))
    if len(by_source) < 2:
        return {"contradictions": []}

    await emit(
        sid,
        "agent_log",
        agent="contradiction_detector",
        message="Checking evidence for conflicting claims…",
    )
    messages = [
        SystemMessage(content=system_prompt("critic.contradiction_detector")),
        HumanMessage(
            content=f"Research query: {state['original_query']}\n\n"
            f"{contradictions.build_detector_input(by_source)}"
        ),
    ]
    parsed, cost, i, o = await _structured("critic", messages, ContradictionReport)
    if parsed is None:
        reason = _last_api_error.get() or "unparseable response"
        await emit(
            sid,
            "agent_log",
            agent="contradiction_detector",
            message=f"Contradiction check unavailable ({reason}); none surfaced",
        )
        return {"contradictions": [], **_acc(state, cost, i, o)}

    normalized = contradictions.normalize_pairs(parsed.pairs, set(by_source))
    found = contradictions.validate_pairs(normalized, by_source)
    await emit(
        sid,
        "agent_log",
        agent="contradiction_detector",
        message=(
            f"Found {len(found)} conflicting claim pair(s)"
            if found
            else "No conflicting claims found"
        ),
    )
    return {"contradictions": found, **_acc(state, cost, i, o)}


def _number_sources(evidence: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """按首次出现的顺序建立编号来源列表及 URL 到编号的映射。

    综合器与冲突区块渲染器共用此实现（docs/12 M11），确保区块中的 [n] 与报告来源列表一致。"""
    sources: list[dict] = []
    seen: dict[str, int] = {}
    for e in evidence:
        url = e.get("source_url", "")
        if not url:
            continue
        snippet = (e.get("snippet") or "").strip()
        if url not in seen:
            n = len(sources) + 1
            seen[url] = n
            sources.append(
                Source(
                    index=n,
                    url=url,
                    title=e.get("source_title", ""),
                    snippet=snippet,
                    snippets=[snippet] if snippet else [],
                ).model_dump()
            )
        elif snippet:
            existing = sources[seen[url] - 1]
            if snippet not in existing["snippets"]:
                existing["snippets"].append(snippet)
            if not existing["snippet"]:
                existing["snippet"] = snippet
    return sources, seen


async def synthesizer_node(state: AgentState) -> dict:
    sid = state["session_id"]
    await emit(
        sid,
        "agent_log",
        agent="synthesizer",
        message="Compiling the report…",
        detail={"model": get_run_config().model_for("synthesizer")},
    )

    # 按不同证据 URL 建立编号来源列表，供草稿用 [n] 引用。
    #
    # 同一来源的全部不同片段都要保留（docs/12 M5，D3），而非只保留第一条。
    # 一页通常支持多个事实；只留一条会让引用提示展示与当前主张无关的文字。
    sources, seen = _number_sources(state.get("evidence", []))

    numbered_evidence = [
        {
            "n": seen.get(e.get("source_url", ""), 0),
            "snippet": e.get("snippet", ""),
            "url": e.get("source_url", ""),
        }
        for e in state.get("evidence", [])
    ]

    evidence_lines: list[str] = []
    for ev in numbered_evidence:
        # 只传 snippet，不传 key_fact：后者是执行器的转述，小型综合模型容易沿转述偏离原文；图内核验与评测都只认可逐字片段。
        evidence_lines.append(f'[{ev["n"]}] Snippet: "{ev["snippet"]}"')
        evidence_lines.append(f"    Source: {ev['url']}")
        evidence_lines.append("")  # 用空行分隔。
    evidence_text = "Evidence for citation:\n" + "\n".join(evidence_lines)

    messages = [
        SystemMessage(content=system_prompt("synthesizer.main")),
        HumanMessage(
            content=prompts.synthesizer_human(
                state["original_query"],
                evidence_text,
                state.get("human_feedback"),
                # 使用审核者编辑后的大纲；plan_gate_node 会覆盖提案。跳过审核关卡时为空，保留原有无关卡结构。
                state.get("proposed_outline"),
            )
        ),
    ]
    model = get_llm("synthesizer")
    resp = await traced_ainvoke(model, messages, role="synthesizer")
    draft = text_of(resp)
    cost = estimate_cost(resp, "synthesizer")
    i, o = token_counts(resp)

    # 引用修复：处理没有 [n] 标记的断言句。计数内联实现，避免引擎依赖 evals.metrics。
    _claim_re = re.compile(r"\[\d+\]")
    uncited = 0
    for _sentence in re.split(r"(?<=[.!?])\s+", draft):
        s = _sentence.strip()
        if len(s) < 15 or not re.search(r"[A-Za-z]", s):
            continue
        if not _claim_re.search(s):
            uncited += 1
    if uncited > 0:
        repair_messages = [
            SystemMessage(content=system_prompt("synthesizer.repair")),
            HumanMessage(
                content=f"Draft report with {uncited} uncited sentences:\n\n{draft}\n\n"
                f"Numbered evidence:\n{evidence_text}"
            ),
        ]
        repair_resp = await traced_ainvoke(model, repair_messages, role="synthesizer")
        draft = text_of(repair_resp)
        repair_cost = estimate_cost(repair_resp, "synthesizer")
        ri, ro = token_counts(repair_resp)
        cost += repair_cost
        i += ri
        o += ro
        resp = repair_resp  # 修复响应是最后一次实际返回的模型响应。
        await emit(
            sid,
            "agent_log",
            agent="synthesizer",
            message=f"Citation repair: fixed {uncited} uncited claims",
        )

    grounding_report = None
    allowed = get_run_config().corpus_allowed_ids
    if allowed is not None:
        # 指定文件的研究会检查全部主张，包括未引用者；失败内容必须重写移除，不能仅附提示。此分支替代只检查带引用句子的忠实度核验（internal/rfcs/V3.1-corpus-scope.md §9）。
        outcome = await grounding_repair.ground(
            draft,
            sources,
            question=state["original_query"],
            admissible=lambda url: (
                (loc := parse_corpus_url(url)) is not None and loc.doc_id in allowed
            ),
            budget=grounding.Budget(
                spent_usd=state.get("cost_usd", 0.0) + cost,
                tokens_in=state.get("tokens_input", 0) + i,
                started_at=state.get("started_at"),
            ),
        )
        draft, grounding_report = outcome.text, outcome.report
        cost += outcome.cost_usd
        i += outcome.tokens_in
        o += outcome.tokens_out
        await emit(
            sid,
            "agent_log",
            agent="synthesizer",
            message=(
                f"Grounding check: {grounding_report.get('claims_supported', 0)} claim(s) "
                f"supported by your selected files, {grounding_report.get('claims_rejected', 0)} "
                f"rejected, {grounding_report.get('claims_unmeasured', 0)} unmeasured"
            ),
            detail={k: v for k, v in grounding_report.items() if k != "claims"},
        )
    else:
        # 在修复后逐项核验剩余主张与其来源片段（docs/12 M5），同时捕获修复引入的偏差；无据主张移除引用并显示说明。
        draft, vcost, vi, vo = await _verify_citation_fidelity(sid, draft, sources)
        cost += vcost
        i += vi
        o += vo

    # 从已验证的冲突对确定性生成冲突区块，放在参考文献前（docs/12 M11）。在引用修复和核验之后插入，避免核验器评审说明性文字，也避免模型遗漏或擅自解决冲突；无冲突时不修改草稿。
    if state.get("contradictions"):
        block = contradictions.render_block(state["contradictions"], seen)
        draft = contradictions.insert_block(draft, block)
        await emit(
            sid,
            "agent_log",
            agent="synthesizer",
            message=f"Surfaced {len(state['contradictions'])} conflicting claim pair(s) in the report",
        )

    await emit(
        sid,
        "agent_log",
        agent="synthesizer",
        message=f"Draft compiled ({len(draft.split())} words, {len(sources)} sources)",
        detail={
            "word_count": len(draft.split()),
            "sources_count": len(sources),
            "preview": draft[:1200] + ("..." if len(draft) > 1200 else ""),
            # 记录提供方实际披露的模型 ID（llm_factory.served_model_id），区别于配置路由 model。模拟模式无 response_metadata，因此为 None；真实调用通常与配置相同，但 auto/* 路由别名可能每次由不同模型响应，不能将别名当成固定模型。
            "served_model": served_model_id(resp),
        },
    )
    result = {
        "draft_report": draft,
        "sources": sources,
        "human_feedback": None,
        **_acc(state, cost, i, o),
    }
    if grounding_report is not None:
        if grounding_report["status"] == "complete":
            # 对最终展示的全文（含冲突区块）计算校验值，使关卡和批准接口能够拒绝任何其他文本（§9.6）。
            grounding_report["validated_text_sha256"] = hashlib.sha256(
                draft.encode("utf-8")
            ).hexdigest()
        else:
            result["error"] = grounding_report["failure"]
        result["grounding_report"] = grounding_report
    return result


def plan_gate_node(state: AgentState) -> dict:
    """规划完成后暂停，让用户在搜索花费发生前编辑子主题与大纲（internal/07 Phase 4）。
    与 hitl_gate_node 一样，interrupt() 持久化检查点并挂起，恢复时携带审核决定。

    恢复参数为 {"tasks": [...], "outline": [...]}，两个字段均可省略；
    缺失表示未编辑，保留规划器的提议。任务的 include: false 会使执行器跳过它，
    从而让审核实际改变研究过程，而不只是改变显示内容。"""
    decision = interrupt(
        {
            "type": "PLAN_READY",
            "tasks": state.get("tasks", []),
            "proposed_outline": state.get("proposed_outline", []),
        }
    )
    tasks = decision.get("tasks") if decision.get("tasks") is not None else state.get("tasks", [])
    outline = (
        decision.get("outline")
        if decision.get("outline") is not None
        else state.get("proposed_outline", [])
    )
    tasks = [t for t in tasks if t.get("include", True)]
    return {"tasks": tasks, "proposed_outline": outline, "plan_approved": True}


def route_after_plan_gate(state: AgentState) -> str:
    # 审核后若所有子主题都被排除，则没有可研究任务，必须阻止无证据综合；否则模型会凭记忆写作，再装饰无法解析的引用。API 的 submit_plan_review 先行拒绝，此处防护旧检查点或绕过接口的恢复路径。
    if not state.get("tasks"):
        return "failer"
    return "executor"


def hitl_gate_node(state: AgentState) -> dict:
    """暂停等待人工审核；interrupt() 持久化检查点并挂起。"""
    refused = _grounding_refusal(state)
    if refused:
        # route_after_synthesizer 本应将这些情况送入 failer；其他路径意外到达这里时抛出异常，使研究失败而非停在批准关卡。
        raise RuntimeError(refused)
    decision = interrupt(
        {
            "type": "HITL_READY",
            "word_count": len((state.get("draft_report") or "").split()),
            "source_count": len(state.get("sources", [])),
            # M11：审核者批准草稿时也会面对尚未解决的冲突；关卡明确显示数量，避免遗漏。
            "contradiction_count": len(state.get("contradictions") or []),
            "cost_usd": round(state.get("cost_usd", 0.0), 4),
        }
    )
    # 恢复参数：decision = {"approved": bool, "feedback": str | None}。
    if decision.get("approved"):
        return {"approved": True}
    return {
        "approved": False,
        "human_feedback": decision.get("feedback"),
        "rework_count": state.get("rework_count", 0) + 1,
    }


def finalizer_node(state: AgentState) -> dict:
    return {"final_report": state.get("draft_report")}


def failer_node(state: AgentState) -> dict:
    # 路由不能写入状态，因此从同一状态重新计算触发限制的原因，向用户说明具体限制及超出数值，而非笼统提示预算或循环超限。
    return {
        "error": state.get("error")
        or _over_budget(state)
        or _no_research_reason(state)
        or "retry limit reached: every task exhausted its critic rounds"
    }


# ── 条件路由 ────────────────────────────────────────────────────


def _over_budget(state: AgentState) -> str | None:
    """返回触发限制的原因，未触发时返回 None；0 表示禁用该限制。

    所有限制均需显式启用（docs/04 §6）。旧实现将词元上限硬编码为 1_000_000，
    无法配置；openrouter/custom 未被定价，estimate_cost() 为 0.0，费用限制不起作用，
    因此该隐藏上限曾在 1,003,721 个词元时终止真实研究。
    返回具体原因而非布尔值，让用户直接知道哪项超限及超出多少。"""
    cfg = get_run_config()

    cost, cost_cap = state.get("cost_usd", 0.0), cfg.max_cost_per_session_usd
    if cost_cap and cost >= cost_cap:
        return f"cost ceiling reached: ${cost:.4f} of ${cost_cap:.2f}"

    tokens, token_cap = state.get("tokens_input", 0), cfg.max_input_tokens
    if token_cap and tokens >= token_cap:
        return f"input-token ceiling reached: {tokens:,} of {token_cap:,}"

    time_cap = cfg.max_wallclock_seconds
    elapsed = time.time() - state.get("started_at", time.time())
    if time_cap and elapsed >= time_cap:
        return f"time limit reached: {elapsed:.0f}s of {time_cap}s"

    return None


def _no_research_reason(state: AgentState) -> str | None:
    """说明研究为何无法在保持诚实的前提下生成报告。

    区分三种需要不同处理的情况：计划未选中任何任务、搜索后没有结果、
    获得来源却未提取出可引用的片段。混为一谈会让取消了全部子主题的用户去检查搜索服务，
    也会把执行器未提交证据误诊为正常工作的提供方失效。

    此函数为 route_after_plan_gate、route_after_planner 和 route_after_critic
    提供原因文本。路由函数不能写状态，因此这里从相同状态重新计算原因。"""
    if not state.get("tasks"):
        return (
            "no research tasks were selected — every subtopic was excluded at the design "
            "gate, so nothing was searched. Start a new run and keep at least one subtopic."
        )
    if not state.get("evidence"):
        seen = state.get("sources_seen", 0)
        if seen:
            return (
                f"no evidence was gathered — search worked and {seen} source(s) were "
                "fetched, but the executor model returned no usable quotation from any of "
                "them, so there is nothing a report could be built from. This is a model "
                "failure, not a retrieval one: try a different executor model in "
                "Settings → Models, then retry."
            )
        return (
            "no evidence was gathered — every research task finished without returning "
            "anything citable, and no source was successfully fetched, so there is nothing "
            "a report could be built from. Check that a search provider is reachable, "
            "then retry."
        )
    return None


def route_after_planner(state: AgentState) -> str:
    if state.get("error"):
        return "failer"
    # 规划器未生成任何任务时，既无计划可审核也无任务可执行；在此统一失败，防止跳过审核的路径生成无证据报告。
    if not state.get("tasks"):
        return "failer"
    # 默认启用计划审核，用户可显式跳过，而非需要主动开启（internal/07 Phase 4）。
    if get_run_config().skip_plan_gate:
        return "executor"
    return "plan_gate"


def route_after_critic(state: AgentState) -> str:
    """决定继续下一轮研究还是进入综合。

    任务通过评审或重试次数耗尽后退出待办集合；即便证据始终未通过评审，
    任务收集到的内容仍会保留，与原先逐任务执行的行为一致。"""
    if _over_budget(state):
        return "failer"
    if _pending(state, get_run_config().max_critic_loops):
        return "executor"
    # 全部任务结束却没有可引用内容时必须停止；否则综合器会凭训练记忆生成流畅报告，再被引用修复装饰成看似可验证的成果。
    if not state.get("evidence"):
        return "failer"
    return "contradiction_detector"


def _grounding_refusal(state: AgentState) -> str | None:
    """返回指定文件范围的草稿无法进入批准关卡的原因，符合条件时返回 None（§9.6）。

    判定同时依赖已记录的依据核验报告和被核验的原文，防止重试、恢复、
    回退或其他渲染路径将未经测量的主张送交批准。"""
    if get_run_config().corpus_allowed_ids is None:
        return None
    report = state.get("grounding_report") or {}
    if report.get("status") != "complete":
        return report.get("failure") or "grounding_unmeasured: the draft was never validated"
    if report.get("claims_unmeasured"):
        return "grounding_unmeasured: unmeasured claims remain"
    digest = hashlib.sha256((state.get("draft_report") or "").encode("utf-8")).hexdigest()
    if digest != report.get("validated_text_sha256"):
        return "grounding_unmeasured: the draft differs from the text that was validated"
    return None


def route_after_synthesizer(state: AgentState) -> str:
    return "failer" if _grounding_refusal(state) else "hitl_gate"


def route_after_gate(state: AgentState) -> str:
    if state.get("approved"):
        return "finalizer"
    return "synthesizer"  # 携带 human_feedback 返回重写。


# ── 构建图 ──────────────────────────────────────────────────────


def build_graph(checkpointer):
    g = StateGraph(AgentState)
    g.add_node("planner", traced_node("planner", planner_node))
    g.add_node("plan_gate", traced_node("plan_gate", plan_gate_node))
    g.add_node("executor", traced_node("executor", executor_node))
    g.add_node("critic", traced_node("critic", critic_node))
    g.add_node(
        "contradiction_detector", traced_node("contradiction_detector", contradiction_detector_node)
    )
    g.add_node("synthesizer", traced_node("synthesizer", synthesizer_node))
    g.add_node("hitl_gate", traced_node("hitl_gate", hitl_gate_node))
    g.add_node("finalizer", traced_node("finalizer", finalizer_node))
    g.add_node("failer", traced_node("failer", failer_node))

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        route_after_planner,
        {"executor": "executor", "plan_gate": "plan_gate", "failer": "failer"},
    )
    g.add_conditional_edges(
        "plan_gate", route_after_plan_gate, {"executor": "executor", "failer": "failer"}
    )
    g.add_edge("executor", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "executor": "executor",
            "contradiction_detector": "contradiction_detector",
            "failer": "failer",
        },
    )
    g.add_edge("contradiction_detector", "synthesizer")
    g.add_conditional_edges(
        "synthesizer", route_after_synthesizer, {"hitl_gate": "hitl_gate", "failer": "failer"}
    )
    g.add_conditional_edges(
        "hitl_gate", route_after_gate, {"finalizer": "finalizer", "synthesizer": "synthesizer"}
    )
    g.add_edge("finalizer", END)
    g.add_edge("failer", END)
    return g.compile(checkpointer=checkpointer)
