"""JSON mode 备用路径：主路径空响应后 JSON 模式恢复的离线回归（零模型调用）。

覆盖三种场景：主路径成功不触发备用、主路径空 → 备用成功、全部失败。
"""

from __future__ import annotations

from types import SimpleNamespace

from research_engine import graph
from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
from research_engine.runner import initial_state

_ROLES = ("planner", "executor", "critic", "synthesizer", "chat")

_VALID_PLANNER_JSON = (
    '{"tasks": [{"id": 1, "query": "Search for TSP solver benchmarks in TSPLIB", '
    '"rationale": "r"}], "proposed_outline": []}'
)

_EMPTY_USAGE = {
    "input_tokens": 10,
    "output_tokens": 5,
    "output_token_details": {},
}


def _real_config(planner_route: str = "custom:deepseek-chat") -> RunConfig:
    return RunConfig(
        llm_mode="real",
        models={role: planner_route for role in _ROLES},
        provider_keys={"custom": "k", "custom_base_url": "https://api.deepseek.com/v1"},
        enforce_ssrf_guards=False,
    )


class _EmptyFunctionCallingStub:
    """主路径 function_calling 返回空响应（content 空、无 tool_calls）。"""

    async def ainvoke(self, messages):
        return {
            "raw": SimpleNamespace(
                content="",
                tool_calls=None,
                invalid_tool_calls=[],
                additional_kwargs={},
                response_metadata={"finish_reason": "stop"},
                usage_metadata=dict(_EMPTY_USAGE),
            ),
            "parsed": None,
            "parsing_error": None,
        }

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        return self


class _JsonModeSuccessModel:
    """主路径空 → JSON mode 备用路径返回合法 PlannerOutput JSON。"""

    def __init__(self) -> None:
        self.bind_kwargs: dict | None = None

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        return _EmptyFunctionCallingStub()

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return _JsonModeRunnable(_VALID_PLANNER_JSON)


class _JsonModeRunnable:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, messages):
        return SimpleNamespace(
            content=self._content,
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 80, "output_tokens": 40, "output_token_details": {}},
        )


class _AllEmptyModel:
    """主路径与 JSON mode 备用路径均返回空——全部失败。"""

    def __init__(self) -> None:
        self.json_mode_used = False

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        return _EmptyFunctionCallingStub()

    def bind(self, **kwargs):
        self.json_mode_used = True
        return _JsonModeRunnable("")


async def test_primary_function_calling_success_does_not_trigger_fallback(monkeypatch) -> None:
    """主路径正常返回时 JSON mode 备用路径不被触发。"""
    model = _JsonModeSuccessModel()
    # 覆写 with_structured_output 让主路径直接成功（不走空响应路径）。
    model.with_structured_output = lambda schema, include_raw=True, **kw: _WorkingStructuredStub()
    token = set_run_config(_real_config())
    try:
        monkeypatch.setattr(graph, "get_llm", lambda role: model)
        state = initial_state(
            session_id="primary-success-test", user_id="u", query="研究长程记忆机制的测试问题。"
        )
        result = await graph.planner_node(state)
    finally:
        reset_run_config(token)

    assert result.get("error") is None
    assert result.get("tasks")
    # JSON mode 备用路径未被调用。
    assert model.bind_kwargs is None


class _WorkingStructuredStub:
    async def ainvoke(self, messages):
        import json

        return {
            "raw": SimpleNamespace(
                content=json.dumps(
                    {"tasks": [{"id": 1, "query": "Search for TSP solver benchmarks in TSPLIB"}]}
                ),
                response_metadata={"finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "output_token_details": {},
                },
            ),
            "parsed": None,
            "parsing_error": None,
        }

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        return self


async def test_empty_primary_recovers_via_json_mode_fallback(monkeypatch) -> None:
    """主路径空响应后 JSON mode 备用路径恢复，产出合法任务。"""
    model = _JsonModeSuccessModel()
    token = set_run_config(_real_config())
    try:
        monkeypatch.setattr(graph, "get_llm", lambda role: model)
        state = initial_state(
            session_id="fallback-success-test", user_id="u", query="研究长程记忆机制的测试问题。"
        )
        result = await graph.planner_node(state)
    finally:
        reset_run_config(token)

    assert result.get("error") is None
    tasks = result.get("tasks") or []
    assert len(tasks) == 1
    assert tasks[0]["query"] == "Search for TSP solver benchmarks in TSPLIB"
    # JSON mode 备用路径被调用（bind 参数含 response_format）。
    assert model.bind_kwargs is not None
    assert model.bind_kwargs.get("response_format") == {"type": "json_object"}


async def test_all_paths_fail_records_both_attempts(monkeypatch) -> None:
    """主路径与 JSON mode 备用路径均空 → error 包含两次尝试的形态。"""
    model = _AllEmptyModel()
    token = set_run_config(_real_config())
    try:
        monkeypatch.setattr(graph, "get_llm", lambda role: model)
        state = initial_state(
            session_id="all-fail-test", user_id="u", query="研究长程记忆机制的测试问题。"
        )
        result = await graph.planner_node(state)
    finally:
        reset_run_config(token)

    error = result.get("error") or ""
    # 错误消息包含主路径空响应和 JSON mode 备用路径的失败摘要。
    assert "empty_response" in error
    assert "json_mode_fallback" in error
    assert result.get("tasks") is None
