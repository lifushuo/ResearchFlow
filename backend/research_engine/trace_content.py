"""评测宿主显式启用的限额诊断正文。

只投影允许内容，不整包导出客户端、配置或请求头；先脱敏再截断，防止凭据截成半段后绕过匹配。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping

SECRET_NAME = re.compile(
    r"authorization|cookie|api.?key|password|passwd|secret|credential|access.?token|refresh.?token|private.?key",
    re.I,
)
SECRET_TEXT = re.compile(
    r"(?i)\b(?:bearer\s+\S+|sk-[a-z0-9_-]{8,}|eyJ[a-z0-9_-]+\.[a-z0-9_-]+\.[a-z0-9_-]+)",
    re.I,
)
EMAIL = re.compile(r"(?<![a-z0-9._%+-])[a-z0-9._%+-]{1,64}@[a-z0-9.-]{1,253}\.[a-z]{2,63}\b", re.I)
ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|password|passwd|secret|token|authorization|cookie)\b[\s\"']*[:=][\s\"']*)([^\s,;\"'}]+)"
)
URL = re.compile(r"https?://[^\s<>\"']+", re.I)


def _text(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if len(secret) >= 4:
            value = value.replace(secret, "[REDACTED]")
    value = SECRET_TEXT.sub("[REDACTED]", value)
    if "@" in value:
        value = EMAIL.sub("[REDACTED]", value)
    value = ASSIGNMENT.sub(r"\1[REDACTED]", value)

    # 签名 URL 与 userinfo 可能在任意参数名下携带凭据。
    def url(match):
        from urllib.parse import urlsplit, urlunsplit

        try:
            p = urlsplit(match[0])
            host = p.netloc.rsplit("@", 1)[-1]
            return urlunsplit((p.scheme, host, p.path, "[REDACTED]" if p.query else "", ""))
        except ValueError:
            return "[REDACTED_URL]"

    return URL.sub(url, value)


def content_snapshot(value, *, secrets=(), max_chars=4096, max_bytes=16000, max_depth=7):
    """返回可 JSON 序列化的内容及明确省略路径；不支持的对象不展开内部属性。"""
    omitted = []
    redacted = []

    def visit(obj, path, depth):
        if depth > max_depth:
            omitted.append(path)
            return "[DEPTH_LIMIT]"
        if obj is None or type(obj) in (bool, int):
            return obj
        if type(obj) is float:
            return obj if math.isfinite(obj) else None
        if isinstance(obj, str):
            text = _text(obj, tuple(secrets))
            if text != obj:
                redacted.append(path)
            if len(text) > max_chars:
                omitted.append(path)
                return text[:max_chars] + "[TRUNCATED]"
            return text
        if isinstance(obj, BaseException):
            import traceback

            frames = traceback.extract_tb(obj.__traceback__)[-12:] if obj.__traceback__ else []
            result = {
                "type": type(obj).__name__,
                "message": visit(str(obj), path + ".message", depth + 1),
                "frames": [
                    {
                        "file": f.filename.replace("\\", "/").rsplit("/", 1)[-1],
                        "function": f.name,
                        "line": f.lineno,
                    }
                    for f in frames
                ],
            }
            cause = obj.__cause__ or (obj.__context__ if not obj.__suppress_context__ else None)
            if cause is not None and cause is not obj:
                result["cause"] = visit(cause, path + ".cause", depth + 1)
            return result
        if isinstance(obj, Mapping):
            pairs = list(obj.items())[:40]
            if len(obj) > len(pairs):
                omitted.append(path + ".items")
            result = {}
            for key, child in pairs:
                key = str(key)
                # 不序列化供应商元数据整包或多模态字节。
                if SECRET_NAME.search(key) or key.lower() in {
                    "headers",
                    "response_metadata",
                    "additional_kwargs",
                    "image_url",
                    "audio",
                    "data",
                    "b64_json",
                }:
                    result[_text(key, tuple(secrets))[:120]] = "[REDACTED]"
                    redacted.append(path + ".field")
                else:
                    result[_text(key, tuple(secrets))[:120]] = visit(
                        child, path + "." + _text(key, tuple(secrets))[:120], depth + 1
                    )
            return result
        if isinstance(obj, (list, tuple)):
            if len(obj) > 40:
                omitted.append(path + ".items")
            return [visit(x, path + f"[{i}]", depth + 1) for i, x in enumerate(obj[:40])]
        # 消息元数据可能携带凭据。安全响应诊断另有允许清单，不能在这里整包导出消息。
        if hasattr(obj, "content") and hasattr(obj, "type"):
            return visit(
                {
                    "role": obj.type,
                    "content": obj.content,
                    "tool_calls": getattr(obj, "tool_calls", None),
                    "invalid_tool_calls": getattr(obj, "invalid_tool_calls", None),
                },
                path,
                depth + 1,
            )
        if hasattr(obj, "model_dump") and type(obj).__module__.startswith(
            "research_engine.schemas"
        ):
            return visit(obj.model_dump(), path, depth + 1)
        omitted.append(path)
        return {"unserialized_type": type(obj).__name__}

    try:
        data = visit(value, "$", 0)
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > max_bytes:
            omitted.append("$.byte_limit")
            # 预览仍须是有效 JSON，不能在序列化对象字段中间截断。
            data = {"preview": encoded.decode()[: max_bytes // 4]}
        return {
            "schema_version": 1,
            "value": data,
            "truncated": bool(omitted),
            "omitted_paths": omitted[:40],
            "redacted": bool(redacted),
            "policy": "sanitized_bounded_content",
            "projected_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    except Exception:
        return {
            "schema_version": 1,
            "value": None,
            "truncated": True,
            "omitted_paths": ["$.serialization_failed"],
            "policy": "sanitized_bounded_content",
        }


def configured_secrets() -> tuple[str, ...]:
    # 配置由宿主注入，引擎不在这里读取进程环境。
    from research_engine.llm_factory import _user_keys
    from research_engine.runconfig import get_run_config

    cfg = get_run_config()
    return tuple(
        str(v)
        for k, v in {**cfg.provider_keys, **(_user_keys.get() or {})}.items()
        if k != "custom_base_url" and v
    ) + tuple(v for v in [cfg.tavily_api_key, cfg.brave_api_key] if v)
