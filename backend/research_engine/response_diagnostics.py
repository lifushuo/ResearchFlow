"""单次模型响应跨 SDK 转换的本地观测，不记录正文。

能访问时观察已缓冲 HTTP JSON，否则观察已解析的 Chat Completion 对象。
这些边界均不能揭示供应商内部网关的实际处理。"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import structlog

from research_engine.trace_content import configured_secrets, content_snapshot

UNAVAILABLE = "unavailable"
_LOG = structlog.get_logger()
_ENABLED = ContextVar("response_diagnostic_models", default=False)
_ACTIVE = ContextVar("response_diagnostic_call", default=None)
_LIMITS = ("max_tokens", "max_completion_tokens", "max_output_tokens")
_FINISH = {
    "stop",
    "length",
    "max_tokens",
    "tool_calls",
    "function_call",
    "content_filter",
    "end_turn",
}


def read(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def count(value):
    return value if type(value) is int and value >= 0 else UNAVAILABLE


def identifier(value, *, secrets=()):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        return UNAVAILABLE
    known = configured_secrets() + tuple(secrets)
    if value in known:
        return UNAVAILABLE
    safe = content_snapshot(value, secrets=known).get("value")
    return value if safe == value else UNAVAILABLE


def content_shape(value):
    if isinstance(value, str):
        return "text", len(value), False
    if value is None:
        return "null", 0, False
    if isinstance(value, list):
        chars, nontext = 0, False
        for block in value:
            kind = read(block, "type")
            text = read(block, "text")
            if kind in {"text", "output_text"} and isinstance(text, str):
                chars += len(text)
            else:
                nontext = True
        return "list", chars, nontext
    return "other", UNAVAILABLE, True


@contextmanager
def diagnostic_models():
    """仅评测宿主所属客户端显式启用，普通研究模型不改变。"""
    token = _ENABLED.set(True)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def models_enabled():
    return _ENABLED.get()


@dataclass
class ResponseObservation:
    batch_id: str
    observation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    provider: dict = field(default_factory=dict)
    parsed: dict = field(default_factory=dict)
    sdk: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    conversion_error: bool = False
    conversion_error_type: str = UNAVAILABLE
    wire_observed: bool = False
    http_request_id: str = UNAVAILABLE
    _secrets: tuple[str, ...] = field(default=(), repr=False)

    def request_options(self, payload, api_key):
        value = api_key.get_secret_value() if hasattr(api_key, "get_secret_value") else api_key
        if isinstance(value, str) and value:
            self._secrets = (value,)
        for key in _LIMITS:
            value = payload.get(key)
            if type(value) is int and value >= 0:
                self.parameters[key] = value

    def _identifier(self, value):
        return identifier(value, secrets=self._secrets)

    def _provider_fields(self, response):
        choices = read(response, "choices")
        first = choices[0] if isinstance(choices, (list, tuple)) and choices else None
        message = read(first, "message")
        sentinel = object()
        content = read(message, "content", sentinel)
        kind, chars, nontext = (
            ("missing", UNAVAILABLE, UNAVAILABLE) if content is sentinel else content_shape(content)
        )
        finish = read(first, "finish_reason")
        usage = read(response, "usage")
        details = read(usage, "completion_tokens_details")
        return {
            "boundary": "openai_parsed_chat_completion",
            "request_id": self._identifier(read(response, "_request_id")),
            "response_id": self._identifier(read(response, "id")),
            "finish_reason": finish
            if isinstance(finish, str) and finish in _FINISH
            else UNAVAILABLE,
            "content_type": kind,
            "text_chars": chars,
            "nontext_present": True
            if nontext is True
            or read(message, "tool_calls")
            or read(message, "audio")
            or read(message, "refusal")
            else False
            if nontext is False
            else UNAVAILABLE,
            "output_tokens": count(read(usage, "completion_tokens")),
            "reasoning_tokens": count(read(details, "reasoning_tokens")),
        }

    def http_response(self, response):
        if response is None:
            return
        self.http_request_id = self._identifier(response.headers.get("x-request-id"))
        # 只检查已经缓冲的响应，不主动读流、不发送请求，也不保留原始字节；超大正文不采集。
        content = response.content
        if not isinstance(content, bytes) or len(content) > 4 * 1024 * 1024:
            return
        payload = json.loads(content)
        if not isinstance(payload, dict):
            return
        self.provider = self._provider_fields(payload)
        self.provider["boundary"] = "http_response_json"
        self.provider["request_id"] = self.http_request_id
        self.wire_observed = True

    def before_conversion(self, response):
        self.parsed = self._provider_fields(response)
        if not self.provider:
            self.provider = dict(self.parsed)

    def conversion_failed(self, error):
        self.conversion_error = True
        self.conversion_error_type = self._identifier(type(error).__name__)

    def after_conversion(self, message):
        kind, chars, nontext = content_shape(read(message, "content"))
        meta = read(message, "response_metadata")
        usage = read(message, "usage_metadata")
        finish = read(meta, "finish_reason")
        self.sdk = {
            "content_type": kind,
            "text_chars": chars,
            "nontext_present": bool(
                nontext or read(message, "tool_calls") or read(message, "invalid_tool_calls")
            ),
            "finish_reason": finish
            if isinstance(finish, str) and finish in _FINISH
            else UNAVAILABLE,
            "output_tokens": count(read(usage, "output_tokens")),
            "reasoning_tokens": count(read(read(usage, "output_token_details"), "reasoning")),
        }

    def snapshot(self, *, harness_text_chars=UNAVAILABLE):
        before = {
            key: self.provider.get(key, UNAVAILABLE)
            for key in (
                "boundary",
                "request_id",
                "response_id",
                "finish_reason",
                "content_type",
                "text_chars",
                "nontext_present",
                "output_tokens",
                "reasoning_tokens",
            )
        }
        if before["request_id"] == UNAVAILABLE:
            before["request_id"] = self.http_request_id
        after = {
            key: self.sdk.get(key, UNAVAILABLE)
            for key in (
                "content_type",
                "text_chars",
                "nontext_present",
                "finish_reason",
                "output_tokens",
                "reasoning_tokens",
            )
        }
        parsed = {key: self.parsed.get(key, UNAVAILABLE) for key in before}
        category, confirmed = "unknown", False
        finish = (
            before["finish_reason"]
            if before["finish_reason"] != UNAVAILABLE
            else after["finish_reason"]
        )
        if finish in {"length", "max_tokens"}:
            category, confirmed = "output_truncated", True
        elif (
            before["text_chars"] != UNAVAILABLE
            and before["text_chars"] > 0
            and (self.conversion_error or after["text_chars"] == 0)
        ):
            category, confirmed = "sdk_conversion_empty", True
        elif self.conversion_error:
            category, confirmed = "sdk_conversion_error", True
        elif harness_text_chars == 0 and (
            before["nontext_present"] is True
            or after["nontext_present"] is True
            or after["content_type"] == "list"
        ):
            category, confirmed = "nontext_response", True
        elif (
            before["text_chars"] == 0
            and after["text_chars"] == 0
            and finish in {"stop", "end_turn"}
        ):
            category, confirmed = "provider_empty", True
        elif type(harness_text_chars) is int and harness_text_chars > 0:
            category, confirmed = "normal", True
        return {
            "schema_version": 1,
            "observation_id": self.observation_id,
            "batch_id": self.batch_id,
            "provider": before,
            "provider_parsed": parsed,
            "sdk": after,
            "wire_response": "json_metadata_observed" if self.wire_observed else UNAVAILABLE,
            "request_parameters": {key: self.parameters.get(key, UNAVAILABLE) for key in _LIMITS},
            "harness_text_chars": harness_text_chars,
            "conversion_error": self.conversion_error,
            "conversion_error_type": self.conversion_error_type,
            "category": category,
            "classification_confirmed": confirmed,
            "root_cause_confirmed": False,
        }


def observe(action, *args):
    """观测故障不能抑制、替换或重试 SDK 调用。"""
    current = _ACTIVE.get()
    if current is not None:
        try:
            getattr(current, action)(*args)
        except Exception as exc:
            _LOG.warning("response_observation_unavailable", error_type=type(exc).__name__)


@contextmanager
def response_observation(batch_id):
    observation = (
        batch_id
        if isinstance(batch_id, ResponseObservation)
        else ResponseObservation(identifier(batch_id))
    )
    token = _ACTIVE.set(observation)
    try:
        yield observation
    finally:
        _ACTIVE.reset(token)


def current_observation():
    return _ACTIVE.get()


def safe_response_diagnostic(value):
    """重新投影持久化或不可信产物，不直接信任其中给出的诊断分类。"""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    observation = ResponseObservation(identifier(value.get("batch_id")))
    observation.observation_id = identifier(value.get("observation_id"))
    for name in ("provider", "parsed", "sdk"):
        source = value.get("provider_parsed" if name == "parsed" else name)
        if not isinstance(source, dict):
            continue
        target = {}
        for key in ("output_tokens", "reasoning_tokens", "text_chars"):
            target[key] = count(source.get(key))
        for key in ("request_id", "response_id"):
            target[key] = identifier(source.get(key))
        kind = source.get("content_type")
        target["content_type"] = (
            kind
            if isinstance(kind, str) and kind in {"text", "list", "null", "other", "missing"}
            else UNAVAILABLE
        )
        finish = source.get("finish_reason")
        target["finish_reason"] = (
            finish if isinstance(finish, str) and finish in _FINISH else UNAVAILABLE
        )
        target["nontext_present"] = (
            source.get("nontext_present")
            if type(source.get("nontext_present")) is bool
            else UNAVAILABLE
        )
        if name in {"provider", "parsed"}:
            target["boundary"] = (
                source.get("boundary")
                if source.get("boundary") in ("openai_parsed_chat_completion", "http_response_json")
                else UNAVAILABLE
            )
        setattr(observation, name, target)
    parameters = value.get("request_parameters")
    if isinstance(parameters, dict):
        observation.parameters = {key: count(parameters.get(key)) for key in _LIMITS}
    observation.conversion_error = value.get("conversion_error") is True
    observation.conversion_error_type = identifier(value.get("conversion_error_type"))
    observation.wire_observed = value.get("wire_response") == "json_metadata_observed"
    return observation.snapshot(harness_text_chars=count(value.get("harness_text_chars")))
