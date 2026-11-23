"""只统计响应实际披露的用量；缺失不补零，配置路由不冒充实际模型。"""

import asyncio
import hashlib
import json
import re
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

import structlog

from research_engine import events
from research_engine.runconfig import get_run_config
from research_engine.trace_content import configured_secrets, content_snapshot

_SESSION = ContextVar("model_usage_session", default=None)
# Optional paid-call gate. Default None preserves all existing behavior. When a
# host installs a gate, EVERY application-level model request charges one slot
# here — including Planner retries and the JSON fallback inside _structured.
_PAID_GATE: ContextVar = ContextVar("paid_call_gate", default=None)
_LOG = structlog.get_logger()


class PaidRequestUncertain(RuntimeError):
    """A classified request failed after dispatch; never retry automatically."""

    def __init__(self, slot_id, role, cause_type):
        self.slot_id, self.role, self.cause_type = slot_id, role, cause_type
        super().__init__(f"{role} request {slot_id} has uncertain outcome ({cause_type})")


@contextmanager
def gate_scope(gate):
    """Install a per-request paid-call gate for the enclosed awaited calls."""
    token = _PAID_GATE.set(gate)
    try:
        yield
    finally:
        _PAID_GATE.reset(token)


def count(value):
    return value if type(value) is int and 0 <= value < 10**12 else None


def identifier(value):
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+ -]{0,199}", value):
        if (
            not re.search(r"(^|[:/])auto(?:/|$)", value, re.I)
            and content_snapshot(value, secrets=configured_secrets(), max_chars=200)["value"]
            == value
        ):
            return value
    return None


@contextmanager
def usage_session(session_id):
    token = _SESSION.set(session_id)
    try:
        yield
    finally:
        _SESSION.reset(token)


def response_usage(response):
    raw = response.get("raw") if isinstance(response, dict) and "raw" in response else response
    usage = getattr(raw, "usage_metadata", None)
    meta = getattr(raw, "response_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    fallback = meta.get("token_usage") or meta.get("usage") or {}
    fallback = fallback if isinstance(fallback, dict) else {}
    return {
        "served_model": identifier(meta.get("model_name") or meta.get("model")),
        "input_tokens": count(usage.get("input_tokens", fallback.get("prompt_tokens"))),
        "output_tokens": count(usage.get("output_tokens", fallback.get("completion_tokens"))),
    }


async def invoke(model, messages, *, role, configured_route=None):
    """一次应用层调用对应一条持久事件；不估算 SDK 内部重试或未返回的用量。"""
    response, state = None, "failed"
    config = get_run_config()
    gate = _PAID_GATE.get()
    reservation = None
    if gate is not None and config.llm_mode == "real":
        fingerprint = hashlib.sha256(
            json.dumps(str(messages), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        reservation = gate.reserve(role, f"request:{role}", fingerprint)
        reservation.mark_sent()
    try:
        response = await model.ainvoke(messages)
        state = "completed"
        if reservation is not None:
            reservation.mark_completed()
        return response
    except asyncio.CancelledError:
        state = "cancelled"
        if reservation is not None:
            reservation.mark_unknown(reason="request cancelled")
        raise
    except Exception as exc:
        if reservation is not None:
            chain = []
            current = exc
            while current is not None and len(chain) < 6:
                chain.append(type(current).__name__)
                current = current.__cause__ or current.__context__
            reservation.mark_unknown(
                reason=f"{type(exc).__name__}: {str(exc)[:200]}",
                cause_types=chain,
                call_stack=[
                    {
                        "file": frame.filename.replace("\\", "/").split("/")[-1],
                        "function": frame.name,
                        "line": frame.lineno,
                    }
                    for frame in traceback.extract_tb(exc.__traceback__)[-8:]
                ],
            )
            raise PaidRequestUncertain(reservation.slot_id, role, type(exc).__name__) from exc
        raise
    finally:
        session = _SESSION.get()
        if session:
            try:
                detail = {
                    "call_id": uuid.uuid4().hex,
                    "role": identifier(role),
                    "configured_model": identifier(configured_route or config.models.get(role)),
                    "llm_mode": config.llm_mode,
                    "status": state,
                    **response_usage(response),
                }
                await events.emit(session, "llm_usage", agent=role, detail=detail)
            except Exception as exc:
                # 用量记录失败不能遮蔽原始模型异常，也不打印响应、密钥或请求内容。
                _LOG.warning("model_usage_write_failed", error_type=type(exc).__name__)


def summarize_usage(records, *, incomplete=False):
    """汇总已观测调用；父节点、等待、恢复批次均不能作为额外模型调用计数。"""
    groups, seen = {}, set()
    for record in records:
        if not isinstance(record, dict):
            continue
        ident = record.get("call_id")
        if isinstance(ident, str) and ident:
            if ident in seen:
                continue
            seen.add(ident)
        kind = "judge" if record.get("role") == "judge" else "research"
        mode = record.get("llm_mode")
        mode = mode if isinstance(mode, str) and mode in {"real", "fake"} else "unknown"
        actual = identifier(record.get("served_model")) if mode != "fake" else None
        route = identifier(record.get("configured_model")) if mode != "fake" else None
        key = (kind, mode, actual, route)
        row = groups.setdefault(
            key,
            {
                "kind": kind,
                "llm_mode": mode,
                "served_model": actual,
                "configured_model": route,
                "roles": [],
                "calls": 0,
                "failed_calls": 0,
                "missing_usage_calls": 0,
                "known_input_tokens": None,
                "known_output_tokens": None,
            },
        )
        role = identifier(record.get("role"))
        if role and role not in row["roles"]:
            row["roles"].append(role)
        row["calls"] += 1
        row["failed_calls"] += record.get("status") in ("failed", "cancelled", "error")
        incoming, outgoing = count(record.get("input_tokens")), count(record.get("output_tokens"))
        row["missing_usage_calls"] += incoming is None or outgoing is None
        for name, value in (("known_input_tokens", incoming), ("known_output_tokens", outgoing)):
            if value is not None:
                row[name] = (row[name] or 0) + value
    rows = list(groups.values())
    for row in rows:
        values = [
            row[k] for k in ("known_input_tokens", "known_output_tokens") if row[k] is not None
        ]
        row["known_total_tokens"] = sum(values) if values else None
        row["roles"].sort()
    rows.sort(key=lambda r: (r["kind"], r["served_model"] or "", r["configured_model"] or ""))
    result = {
        "schema_version": 1,
        "scope": "observed_application_calls",
        "models": rows,
        "calls": sum(row["calls"] for row in rows),
        "missing_usage_calls": sum(row["missing_usage_calls"] for row in rows),
        "collection_incomplete": incomplete or not rows,
    }
    for name in ("known_input_tokens", "known_output_tokens", "known_total_tokens"):
        values = [row[name] for row in rows if row[name] is not None]
        result[name] = sum(values) if values else None
    return result
