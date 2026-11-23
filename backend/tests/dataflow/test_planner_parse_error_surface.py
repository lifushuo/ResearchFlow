"""Planner 结构化解析失败必须把脱敏摘要浮出到节点错误（AGENTS.md 诚实规则）。

回归背景：第一次真实 GEPA 中 llm-judge-bias 的 Planner 结构化输出两次解析失败，
Journal 只留下通用消息，真正的 schema 校验错误无处可查。第二次真实 GEPA 中同一
错误再次出现，此时解析摘要（本轮新增）已能区分「模型没答」与「答了但解析失败」。
"""

from __future__ import annotations

from types import SimpleNamespace

from research_engine import graph
from research_engine.local import run_config_from_env
from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
from research_engine.runner import initial_state
from research_engine.schemas import PlannerOutput

_FILLER = "x" * 2000
_INVALID_TOOL_CALL_SOURCE = '{"tasks": [{"id": 1, "query": "未闭合的任务描'

_ROLES = ("planner", "executor", "critic", "synthesizer", "chat")


class _NoOpBindResult:
    """JSON mode 备用路径的空响应替身；不影响测试的 primary 路径断言。"""

    async def ainvoke(self, messages):
        return SimpleNamespace(
            content="",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "output_token_details": {}},
        )


class _BrokenPlanner:
    """结构化调用成功返回、但内容无法通过 PlannerOutput 校验的替身。"""

    def bind(self, **kwargs):
        return _NoOpBindResult()

    async def ainvoke(self, messages):
        return SimpleNamespace(
            content=(
                f'```json\n{{"tasks": "不是列表", "proposed_outline": [], "note": "{_FILLER}"}}\n```'
            ),
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 42, "output_tokens": 7, "output_token_details": {}},
        )

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        assert schema is PlannerOutput
        return self


class _EmptyContentWithValidToolCall:
    """function_calling 常见形态：content 为空，任务载荷在 tool_calls.args 里。"""

    def bind(self, **kwargs):
        return _NoOpBindResult()

    async def ainvoke(self, messages):
        return {
            "raw": SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "PlannerOutput",
                        "args": {
                            "tasks": [
                                {
                                    "id": 1,
                                    "query": "Search for TSP solver benchmarks in TSPLIB",
                                    "rationale": "r",
                                }
                            ],
                            "proposed_outline": [],
                        },
                    }
                ],
                invalid_tool_calls=[],
                additional_kwargs={},
            ),
            "parsed": None,
            "parsing_error": None,
        }

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        assert schema is PlannerOutput
        return self


class _ReasoningOnlyEmptyResponse:
    """hybrid 推理模型的间歇形态：只产出思维链，content 与 tool_calls 均为空。"""

    def bind(self, **kwargs):
        return _NoOpBindResult()

    async def ainvoke(self, messages):
        return {
            "raw": SimpleNamespace(
                content="",
                tool_calls=None,
                invalid_tool_calls=[],
                additional_kwargs={"reasoning_content": "thinking about the plan"},
            ),
            "parsed": None,
            "parsing_error": None,
        }

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        return self


class _InvalidToolCallResponse:
    """服务端返回畸形 tool_call（参数截断）：langchain 转入 invalid_tool_calls，不抛异常。"""

    def bind(self, **kwargs):
        return _NoOpBindResult()

    async def ainvoke(self, messages):
        return {
            "raw": SimpleNamespace(
                content="",
                tool_calls=None,
                invalid_tool_calls=[
                    {
                        "name": "PlannerOutput",
                        "args": _INVALID_TOOL_CALL_SOURCE,
                        "id": "call_1",
                        "error": "Expecting ',' delimiter",
                        "type": "invalid_tool_call",
                    }
                ],
                additional_kwargs={},
            ),
            "parsed": None,
            "parsing_error": None,
        }

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        return self


def _real_run_config(planner_route: str = "custom:deepseek-chat") -> RunConfig:
    return RunConfig(
        llm_mode="real",
        models={role: planner_route for role in _ROLES},
        provider_keys={"custom": "k", "custom_base_url": "https://api.deepseek.com/v1"},
        enforce_ssrf_guards=False,
    )


async def test_planner_error_surfaces_sanitized_parse_detail(monkeypatch) -> None:
    config = run_config_from_env(fake=True)
    token = set_run_config(config)
    try:
        monkeypatch.setattr(graph, "get_llm", lambda role: _BrokenPlanner())
        state = initial_state(
            session_id="parse-error-surface-test",
            user_id="u",
            query="研究长程记忆机制的测试问题。",
        )
        result = await graph.planner_node(state)
    finally:
        reset_run_config(token)

    error = result.get("error") or ""
    assert "could not produce a valid task list" in error
    # 脱敏摘要：错误类型与字段路径浮出（tasks 字段不是列表）。
    assert "parse detail:" in error
    assert "ValidationError" in error
    assert "tasks" in error
    # fail-closed：无效输出不产生任务，也不伪造规划。
    assert result.get("tasks") is None
    # 截断：2000 字符填充内容不得完整进入错误消息。
    assert _FILLER[:64] not in error
    assert len(error) < 700


async def test_empty_content_tool_call_args_are_recovered(monkeypatch) -> None:
    config = _real_run_config()
    token = set_run_config(config)
    try:
        monkeypatch.setattr(graph, "get_llm", lambda role: _EmptyContentWithValidToolCall())
        state = initial_state(
            session_id="tool-call-recovery-test",
            user_id="u",
            query="研究长程记忆机制的测试问题。",
        )
        result = await graph.planner_node(state)
    finally:
        reset_run_config(token)

    assert result.get("error") is None
    tasks = result.get("tasks") or []
    assert len(tasks) == 1
    assert tasks[0]["query"] == "Search for TSP solver benchmarks in TSPLIB"


async def test_reasoning_only_empty_response_is_recorded_with_shape(monkeypatch) -> None:
    config = _real_run_config()
    token = set_run_config(config)
    try:
        monkeypatch.setattr(graph, "get_llm", lambda role: _ReasoningOnlyEmptyResponse())
        state = initial_state(
            session_id="reasoning-only-test",
            user_id="u",
            query="研究长程记忆机制的测试问题。",
        )
        result = await graph.planner_node(state)
    finally:
        reset_run_config(token)

    error = result.get("error") or ""
    # 形态记录让下一次故障能区分「模型没答（reasoning-only）」与「答了但解析失败」。
    # JSON mode 备用路径也触发并失败，parse_detail 前缀记录了备用路径尝试。
    assert "empty_response" in error
    assert "json_mode_fallback" in error
    assert "tool_calls=absent" in error
    assert "invalid_tool_calls=absent" in error
    assert "reasoning_content=present" in error
    assert result.get("tasks") is None


async def test_invalid_tool_call_response_is_visible_in_the_error(monkeypatch) -> None:
    """服务端返回畸形 tool_call（截断/非法 JSON）时必须可见，而不是伪装成空消息。"""
    config = _real_run_config()
    token = set_run_config(config)
    try:
        monkeypatch.setattr(graph, "get_llm", lambda role: _InvalidToolCallResponse())
        state = initial_state(
            session_id="invalid-tool-call-test",
            user_id="u",
            query="研究长程记忆机制的测试问题。",
        )
        result = await graph.planner_node(state)
    finally:
        reset_run_config(token)

    error = result.get("error") or ""
    assert "invalid_tool_calls=present(count=1" in error
    assert "args_len=" in error
    assert "Expecting ',' delimiter" in error
    assert result.get("tasks") is None


class _RecordingStructured:
    """记录 with_structured_output 收到的 kwargs，并返回空文本响应。"""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        self.kwargs = kwargs
        return self

    def bind(self, **kwargs):
        return _NoOpBindResult()

    async def ainvoke(self, messages):
        return {
            "raw": SimpleNamespace(
                content="",
                tool_calls=None,
                invalid_tool_calls=[],
                additional_kwargs={},
                response_metadata={"finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "output_token_details": {},
                },
            ),
            "parsed": None,
            "parsing_error": None,
        }


async def test_deepseek_chat_route_records_finish_reason_without_tool_choice_injection(
    monkeypatch,
) -> None:
    """langchain 1.6.6 默认已强制 tool_choice=PlannerOutput 函数名（base.py:2783）。

    引擎不得再注入 "required"（会把函数级强制放宽为任意工具）；
    空响应是 DeepSeek 服务端在强制函数调用下的行为，只能靠元数据判定。"""
    recorder = _RecordingStructured()
    config = _real_run_config("custom:deepseek-chat")
    token = set_run_config(config)
    try:
        monkeypatch.setattr(graph, "get_llm", lambda role: recorder)
        state = initial_state(
            session_id="tool-choice-default-test",
            user_id="u",
            query="研究长程记忆机制的测试问题。",
        )
        result = await graph.planner_node(state)
    finally:
        reset_run_config(token)

    # 不注入 tool_choice——langchain 默认的函数级强制原样生效。
    assert recorder.kwargs.get("tool_choice") is None
    error = result.get("error") or ""
    # 服务端响应元数据随错误浮出，支撑「推理耗尽预算 vs 未调用工具」的判定。
    assert "finish_reason=stop" in error
    assert result.get("tasks") is None


async def test_non_deepseek_route_also_keeps_langchain_default(monkeypatch) -> None:
    recorder = _RecordingStructured()
    config = _real_run_config("custom:othermodel")
    token = set_run_config(config)
    try:
        monkeypatch.setattr(graph, "get_llm", lambda role: recorder)
        state = initial_state(
            session_id="no-force-test",
            user_id="u",
            query="研究长程记忆机制的测试问题。",
        )
        await graph.planner_node(state)
    finally:
        reset_run_config(token)

    # 非 deepseek 路由同样不注入 tool_choice（由 langchain 默认行为决定）。
    assert recorder.kwargs.get("tool_choice") is None
