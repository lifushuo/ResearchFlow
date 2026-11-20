"""由评测宿主显式启用的运行轨迹。

不依赖 evals 或 Web API；没有轨迹作用域时不做观测。
默认仅元数据，正文须宿主明确启用，并先脱敏、限额。
只观察故障，不改变控制流、重试、审批或预算。"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

import structlog
from langgraph.errors import GraphBubbleUp

from research_engine.response_diagnostics import safe_response_diagnostic
from research_engine.token_usage import invoke as usage_invoke
from research_engine.token_usage import usage_session
from research_engine.trace_content import configured_secrets, content_snapshot

_LOG = structlog.get_logger()
_ACTIVE: ContextVar[TraceRecorder | None] = ContextVar("rf_eval_trace", default=None)
_PARENT: ContextVar[str | None] = ContextVar("rf_eval_trace_parent", default=None)
_ATTEMPT: ContextVar[int | None] = ContextVar("rf_eval_trace_attempt", default=None)

# 只允许清单中的属性写入元数据，原始提示词和来源正文不能混入。
_SAFE_METADATA = frozenset(
    {
        "role",
        "tool_name",
        "tool_call_id",
        "configured_model",
        "served_model",
        "disclosed_alias",
        "input_tokens",
        "output_tokens",
        "schema_status",
        "parsing_error_type",
        "http_status",
        "attempt",
        "strategy",
        "llm_mode",
    }
)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))[:64] or "unknown"


def _safe_meta(meta: dict[str, Any] | None) -> dict[str, str | int | None]:
    clean: dict[str, str | int | None] = {}
    for name, value in (meta or {}).items():
        if name not in _SAFE_METADATA:
            continue
        if name in ("input_tokens", "output_tokens", "http_status", "attempt"):
            if type(value) is int and value >= 0:
                clean[name] = value
        elif isinstance(value, str):
            clean[name] = value[:128]
    return clean


class TraceRecorder:
    """每次调用独占创建一个轨迹文件，并安全分配并发序号。"""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        case_id: str,
        thread_id: str | None = None,
        capture_content: bool = False,
        redaction_secrets: tuple[str, ...] = (),
    ) -> None:
        self.trace_id = uuid.uuid4().hex
        self.run_id = str(run_id)
        self.case_id = str(case_id)
        self.thread_id = thread_id
        self.capture_content = capture_content
        self._content_bytes = 0
        self._secrets = tuple(redaction_secrets) + configured_secrets() if capture_content else ()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"{_safe_id(run_id)}-{_safe_id(case_id)}-{self.trace_id}"
        self.path = self.directory / f"{stem}.jsonl"
        self.bad_case_path = self.directory / f"{stem}.badcase.json"
        # x 模式拒绝追加或覆盖任何历史文件。
        self._stream = self.path.open("x", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0
        self._failures: list[dict[str, Any]] = []
        self._disabled = False
        self._closed = False
        self.bad_case_written = False

    @property
    def write_failed(self) -> bool:
        return self._disabled

    def emit(
        self,
        *,
        name: str,
        kind: str,
        status: str,
        span_id: str | None = None,
        parent_id: str | None = None,
        duration_ms: float | None = None,
        error_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        details: dict | None = None,
        response_diagnostic: dict | None = None,
    ) -> None:
        if self._disabled:
            return
        try:
            response_diagnostic = safe_response_diagnostic(response_diagnostic)
            # 写入和序号分配一起加锁，避免并发执行器事件交错。
            with self._lock:
                self._seq += 1
                event = {
                    "event_version": 2,
                    "trace_id": self.trace_id,
                    "run_id": self.run_id,
                    "case_id": self.case_id,
                    "thread_id": self.thread_id,
                    "seq": self._seq,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "kind": kind,
                    "name": name,
                    "status": status,
                    "span_id": span_id,
                    "parent_span_id": parent_id,
                    "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
                    "error_type": error_type[:128] if error_type else None,
                    "metadata": _safe_meta(metadata),
                }
                if response_diagnostic is not None:
                    event["response_diagnostic"] = response_diagnostic
                if self.capture_content:
                    detail = content_snapshot(details, secrets=self._secrets)
                    size = len(json.dumps(detail, ensure_ascii=False).encode())
                    if self._content_bytes + size > 4 * 1024 * 1024:
                        detail = {
                            "schema_version": 1,
                            "value": None,
                            "truncated": True,
                            "omitted_paths": ["$.trace_content_budget"],
                            "policy": "sanitized_bounded_content",
                        }
                    else:
                        self._content_bytes += size
                    event["details"] = detail
                self._stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                self._stream.flush()
                if status in ("error", "business_error", "cancelled"):
                    self._failures.append(
                        {
                            "span_id": span_id,
                            "name": name,
                            "kind": kind,
                            "symptom": status,
                            "error_type": error_type,
                        }
                    )
                if response_diagnostic is not None and response_diagnostic["category"] != "normal":
                    self._failures.append(
                        {
                            "span_id": span_id,
                            "name": name,
                            "kind": kind,
                            "symptom": "judge_response_diagnostic",
                            "response_diagnostic": response_diagnostic,
                        }
                    )
        except (OSError, ValueError) as exc:
            # 轨迹仅为观测；写入失败不能让研究失败。
            self._disabled = True
            _LOG.warning("agent_eval_trace_write_failed", error_type=type(exc).__name__)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        except OSError as exc:
            self._disabled = True
            _LOG.warning("agent_eval_trace_close_failed", error_type=type(exc).__name__)
        if not self._failures:
            return
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "trace_id": self.trace_id,
            "trace_file": self.path.name,
            "thread_id": self.thread_id,
            "root_cause_confirmed": False,
            "root_cause": "unknown",
            "failure_symptoms": self._failures,
            "note": "Automatically detected symptoms only; root cause requires evidence review.",
        }
        try:
            with self.bad_case_path.open("x", encoding="utf-8") as out:
                json.dump(payload, out, indent=2, ensure_ascii=False)
            self.bad_case_written = True
        except OSError as exc:
            self._disabled = True
            _LOG.warning("agent_eval_badcase_write_failed", error_type=type(exc).__name__)


def active_trace() -> TraceRecorder | None:
    return _ACTIVE.get()


@contextmanager
def eval_trace_run(
    *,
    case_id: str,
    run_id: str,
    directory: str | Path,
    enabled: bool = False,
    thread_id: str | None = None,
    capture_content: bool = False,
    redaction_secrets: tuple[str, ...] = (),
) -> Iterator[TraceRecorder | None]:
    """只为本次图执行开启观测，不改动已有数据。"""
    writer = None
    if enabled:
        try:
            writer = TraceRecorder(
                directory,
                run_id=run_id,
                case_id=case_id,
                thread_id=thread_id,
                capture_content=capture_content,
                redaction_secrets=redaction_secrets,
            )
        except OSError as exc:
            _LOG.warning("agent_eval_trace_start_failed", error_type=type(exc).__name__)
    # 内部作用域即使关闭或初始化失败，也不能借用外层案例的记录器。
    token = _ACTIVE.set(writer)
    parent_token = _PARENT.set(None)
    attempt_token = _ATTEMPT.set(None)
    try:
        yield writer
    finally:
        _PARENT.reset(parent_token)
        _ATTEMPT.reset(attempt_token)
        _ACTIVE.reset(token)
        if writer is not None:
            writer.close()


class Span:
    def __init__(self, writer: TraceRecorder | None) -> None:
        self.writer = writer
        self.status = "ok"
        self.metadata: dict[str, Any] = {}
        self.details: dict[str, Any] = {}
        self.response_diagnostic: dict | None = None

    def add(self, **metadata: Any) -> None:
        self.metadata.update(_safe_meta(metadata))

    def observe(self, **details: Any) -> None:
        if self.writer is not None and self.writer.capture_content:
            self.details.update(details)


@contextmanager
def trace_span(
    name: str,
    *,
    kind: str,
    metadata: dict[str, Any] | None = None,
    inputs: Any = None,
) -> Iterator[Span]:
    writer = _ACTIVE.get()
    span = Span(writer)
    if writer is None:
        yield span
        return
    span.metadata = _safe_meta(metadata)
    span.observe(input=inputs)
    if "attempt" not in span.metadata and _ATTEMPT.get() is not None:
        span.add(attempt=_ATTEMPT.get())
    span_id = uuid.uuid4().hex
    parent_id = _PARENT.get()
    started = time.perf_counter()
    writer.emit(
        name=name,
        kind=kind,
        status="start",
        span_id=span_id,
        parent_id=parent_id,
        metadata=span.metadata,
        details=span.details,
    )
    token = _PARENT.set(span_id)
    attempt_token = _ATTEMPT.set(span.metadata.get("attempt"))
    try:
        yield span
    except BaseException as exc:
        # LangGraph 的 GraphInterrupt 是暂停控制流，不是模型或工具错误。
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
            status = "cancelled"
        elif isinstance(exc, GraphBubbleUp):
            status = "interrupt"
        else:
            status = "error"
        try:
            span.add(http_status=getattr(exc, "status_code", None))
        except Exception as metadata_error:  # noqa: BLE001 — 保留原始异常
            _LOG.warning(
                "agent_eval_trace_metadata_failed", error_type=type(metadata_error).__name__
            )
        span.observe(error={"type": type(exc).__name__} if name == "llm.judge" else exc)
        writer.emit(
            name=name,
            kind=kind,
            status=status,
            span_id=span_id,
            parent_id=parent_id,
            duration_ms=(time.perf_counter() - started) * 1000,
            error_type=type(exc).__name__,
            metadata=span.metadata,
            details=span.details,
            response_diagnostic=span.response_diagnostic,
        )
        raise
    else:
        writer.emit(
            name=name,
            kind=kind,
            status=span.status,
            span_id=span_id,
            parent_id=parent_id,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata=span.metadata,
            details=span.details,
            response_diagnostic=span.response_diagnostic,
        )
    finally:
        _PARENT.reset(token)
        _ATTEMPT.reset(attempt_token)


def traced_node(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """包装真实 LangGraph 节点；未启用轨迹时保持原执行语义。"""
    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
            with usage_session(state.get("session_id") if isinstance(state, dict) else None):
                if _ACTIVE.get() is None:
                    return await fn(state, *args, **kwargs)
                with trace_span(name, kind="graph_node", inputs=state) as span:
                    result = await fn(state, *args, **kwargs)
                    span.observe(output=result)
                    if isinstance(result, dict) and result.get("error"):
                        span.status = "business_error"
                    return result

        return wrapped

    @wraps(fn)
    def wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
        if _ACTIVE.get() is None:
            return fn(state, *args, **kwargs)
        with trace_span(name, kind="graph_node", inputs=state) as span:
            result = fn(state, *args, **kwargs)
            span.observe(output=result)
            if isinstance(result, dict) and result.get("error"):
                span.status = "business_error"
            return result

    return wrapped


async def traced_ainvoke(
    model: Any, messages: Any, *, role: str, configured_route: str | None = None
) -> Any:
    """观测现有调用；仅当宿主明确启用时采集诊断正文。"""
    if _ACTIVE.get() is None:
        return await usage_invoke(model, messages, role=role, configured_route=configured_route)
    with trace_span(f"llm.{role}", kind="llm", metadata={"role": role}, inputs=messages) as span:
        # 请求的模型路由不一定是最终实际响应的模型。
        try:
            from research_engine.runconfig import get_run_config

            config = get_run_config()
            span.add(llm_mode=config.llm_mode)
            if config.llm_mode != "fake":
                span.add(configured_model=configured_route or config.model_for(role))
            span.observe(
                request_options={
                    name: getattr(model, name, None)
                    for name in ["temperature", "max_tokens", "top_p", "max_retries"]
                }
            )
        except Exception:  # noqa: BLE001 — 观测不能影响模型调用
            pass
        result = None
        try:
            result = await usage_invoke(
                model, messages, role=role, configured_route=configured_route
            )
        finally:
            if role == "judge":
                try:
                    from research_engine.response_diagnostics import current_observation, observe

                    observation = current_observation()
                    if observation is not None:
                        if result is not None:
                            observe("after_conversion", result)
                        content = getattr(result, "content", None)
                        span.response_diagnostic = observation.snapshot(
                            harness_text_chars=(len(content) if isinstance(content, str) else 0)
                            if result is not None
                            else "unavailable"
                        )
                except Exception as exc:
                    _LOG.warning("response_observation_unavailable", error_type=type(exc).__name__)
        if role != "judge":
            span.observe(output=result)
        try:
            raw = result.get("raw") if isinstance(result, dict) and "raw" in result else result
            from research_engine.llm_factory import served_model_id

            served = served_model_id(raw)
            if served and re.search(r"(^|[:/])auto(?:/|$)", served, re.IGNORECASE):
                span.add(disclosed_alias=served)
            elif served:
                span.add(served_model=served)
            usage = getattr(raw, "usage_metadata", None) or {}
            if isinstance(usage, dict):
                span.add(
                    input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens")
                )
            if isinstance(result, dict) and "parsed" in result:
                span.add(
                    schema_status=(
                        "parsed"
                        if result.get("parsed") is not None and not result.get("parsing_error")
                        else "unparsed"
                    )
                )
                if result.get("parsing_error") is not None:
                    span.add(parsing_error_type=type(result["parsing_error"]).__name__)
        except Exception as exc:  # noqa: BLE001 — 元数据采集不能影响模型调用
            _LOG.warning("agent_eval_trace_metadata_failed", error_type=type(exc).__name__)
        return result
