"""Optional judge observation, with bounded content only on explicit evaluation opt-in."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_engine.response_diagnostics import safe_response_diagnostic
from research_engine.trace_content import configured_secrets, content_snapshot

LOG = logging.getLogger(__name__)
_ACTIVE: ContextVar[JudgeBatchRecorder | None] = ContextVar("judge_batch_recorder", default=None)
_PATTERN = re.compile(r"Claim\s+(\d+)\s*:\s*(YES|NO)", re.IGNORECASE)
_FIELDS = frozenset(
    {
        "event",
        "judge_route",
        "status",
        "trace_available",
        "batch_no",
        "batch_start_zero_based",
        "expected_count",
        "judged_count",
        "yes_count",
        "no_count",
        "unjudged_count",
        "reason_hint",
        "elapsed_ms",
        "application_level_calls",
        "sdk_retry_attempts",
        "exception_type",
        "http_status",
        "response_chars",
        "recognizable_labels",
        "parsed_unique_in_range_labels",
        "out_of_range_labels",
        "duplicate_labels",
        "raw_response_saved",
        "response_received",
        "response_content_kind",
        "preexisting_judged_count",
        "served_model",
        "disclosed_alias",
        "wait_seconds",
        # P0 reliability: every wait says why it happened and how long it took, and a
        # resumed run discloses which batches were restored from disk rather than calling
        # the provider again. Old traces simply omit these fields.
        "wait_reason",
        "provider",
        "retry_after_seconds",
        "restored_from",
        "resumed_from",
        "restored_batches",
        "total_claims",
        "batch_size",
        "tokens_input",
        "tokens_output",
        "cost_usd",
        "response_diagnostic",
    }
)


def _warn(event: str, exc: Exception) -> None:
    # Exception strings and tracebacks may contain credentials, prompts or response bodies.
    LOG.warning("%s error_type=%s", event, type(exc).__name__)


def _status_code(exc: BaseException | None) -> int | None:
    if exc is None:
        return None
    try:
        objects = (exc, getattr(exc, "response", None))
        for obj in objects:
            for name in ("status_code", "status"):
                candidate = getattr(obj, name, None)
                if type(candidate) is int and 100 <= candidate <= 599:
                    return candidate
    except Exception as error:
        _warn("judge_trace_status_unavailable", error)
    return None


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)[:80] or "unknown"


def _content_kind(response: Any, received: bool) -> str:
    if not received:
        return "not_received"
    try:
        content = getattr(response, "content", None)
        return (
            "text" if isinstance(content, str) else "list" if isinstance(content, list) else "other"
        )
    except Exception as exc:
        _warn("judge_trace_content_kind_unavailable", exc)
        return "unavailable"


def token_usage(response: Any, field: str) -> int | None:
    usage = getattr(response, "usage_metadata", None)
    value = usage.get(field) if isinstance(usage, dict) else None
    return value if type(value) is int and value >= 0 else None


@dataclass
class JudgeBatchRecorder:
    directory: Path
    case_id: str
    phase: str
    judge_route: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex, init=False)
    capture_content: bool = False
    redaction_secrets: tuple[str, ...] = ()
    total_claims: int | None = None
    batch_size: int = 4
    _content_bytes: int = field(default=0, init=False)
    path: Path | None = field(default=None, init=False)
    failed: bool = field(default=False, init=False)
    _handle: Any = field(default=None, init=False, repr=False)
    response_problems: list[dict] = field(default_factory=list, init=False)
    response_bad_case_path: Path | None = field(default=None, init=False)
    _seq: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def open(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"judge-{_safe_id(self.phase)}-{_safe_id(self.case_id)}-{self.trace_id}"
        path = self.directory / f"{stem}.jsonl"
        self._handle = path.open("x", encoding="utf-8")
        self.path = path
        self.write(
            {
                "event": "case_start",
                "judge_route": self.judge_route,
                **(
                    {"total_claims": self.total_claims, "batch_size": self.batch_size}
                    if self.total_claims is not None
                    else {}
                ),
            }
        )

    def write(self, event: dict[str, Any]) -> None:
        try:
            with self._lock:
                if self.failed or self._handle is None:
                    return
                record = {
                    "schema_version": 1,
                    "trace_id": self.trace_id,
                    "run_id": self.run_id,
                    "case_id": self.case_id,
                    "phase": self.phase,
                    "seq": self._seq + 1,
                    "timestamp": datetime.now(UTC).isoformat(),
                    **{key: value for key, value in event.items() if key in _FIELDS},
                }
                if "response_diagnostic" in record:
                    record["response_diagnostic"] = safe_response_diagnostic(
                        record["response_diagnostic"]
                    )
                if self.capture_content and "details" in event:
                    detail = content_snapshot(
                        event["details"], secrets=self.redaction_secrets + configured_secrets()
                    )
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
                    record["details"] = detail
                self._handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self._handle.flush()
                self._seq += 1
                diagnostic = record.get("response_diagnostic")
                if diagnostic and (
                    record.get("unjudged_count", 0) or diagnostic["category"] != "normal"
                ):
                    self.response_problems.append(
                        {
                            "trace_id": self.trace_id,
                            "trace_file": self.path.name,
                            "seq": record["seq"],
                            "batch_no": record.get("batch_no"),
                            "response_diagnostic": diagnostic,
                        }
                    )
        except Exception as exc:
            self.failed = True
            _warn("judge_batch_trace_write_failed", exc)

    def close(self) -> None:
        if self.response_problems and self.path is not None and self.response_bad_case_path is None:
            try:
                target = self.path.with_suffix(".badcase.json")
                with target.open("x", encoding="utf-8") as out:
                    json.dump(
                        {
                            "schema_version": 1,
                            "run_id": self.run_id,
                            "case_id": self.case_id,
                            "trace_id": self.trace_id,
                            "trace_file": self.path.name,
                            "root_cause_confirmed": False,
                            "root_cause": "unknown",
                            "response_problems": self.response_problems,
                        },
                        out,
                        ensure_ascii=False,
                        indent=2,
                    )
                self.response_bad_case_path = target
            except Exception as exc:
                _warn("judge_response_badcase_write_failed", exc)
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception as exc:
                self.failed = True
                _warn("judge_batch_trace_close_failed", exc)
            finally:
                self._handle = None


@contextmanager
def trace_judge_case(
    *,
    case_id: str,
    trace_dir: str | Path | None,
    phase: str = "baseline",
    judge_route: str | None = None,
    run_id: str | None = None,
    capture_content: bool = False,
    redaction_secrets: tuple[str, ...] = (),
    total_claims: int | None = None,
    batch_size: int = 4,
) -> Iterator[JudgeBatchRecorder | None]:
    recorder = None
    if trace_dir:
        try:
            recorder = JudgeBatchRecorder(
                Path(trace_dir),
                str(case_id),
                phase,
                judge_route,
                run_id or uuid.uuid4().hex,
                capture_content=capture_content,
                redaction_secrets=redaction_secrets,
                total_claims=total_claims,
                batch_size=batch_size,
            )
            recorder.open()
        except Exception as exc:
            _warn("judge_batch_trace_open_failed", exc)
            if recorder is not None:
                recorder.close()
            recorder = None
    # Disabled and failed inner scopes must not inherit an outer case's recorder.
    token = _ACTIVE.set(recorder)
    status = "finished"
    error_type = None
    try:
        yield recorder
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        error_type = type(exc).__name__
        raise
    finally:
        try:
            if recorder is not None:
                recorder.write(
                    {
                        "event": "case_end",
                        "status": status,
                        "exception_type": error_type,
                        "trace_available": not recorder.failed,
                    }
                )
                recorder.close()
        finally:
            _ACTIVE.reset(token)


def judge_trace_artifacts(
    recorder: JudgeBatchRecorder | None, *, requested: bool, artifact_root: Path | None = None
) -> dict:
    if not requested:
        return {}
    if recorder is None or recorder.path is None:
        return {"judge_batch_trace_status": "unavailable"}
    shown = recorder.path
    if artifact_root is not None:
        try:
            shown = shown.relative_to(artifact_root)
        except ValueError:
            pass
    bad_case = recorder.response_bad_case_path
    if bad_case is not None and artifact_root is not None:
        try:
            bad_case = bad_case.relative_to(artifact_root)
        except ValueError:
            pass
    return {
        "judge_batch_trace": str(shown),
        "judge_batch_trace_status": "partial" if recorder.failed else "recorded",
        **({"judge_response_bad_case": str(bad_case)} if bad_case is not None else {}),
    }


def trace_judge_stage(event: str, **metadata) -> None:
    rec = _ACTIVE.get()
    if rec is None:
        return
    try:
        rec.write({"event": event, **metadata})
    except Exception as exc:
        rec.failed = True
        _warn("judge_stage_observer_failed", exc)


@contextmanager
def trace_judge_wait(
    batch_no: int,
    seconds: float,
    *,
    reason: str = "provider_throttle",
    provider: str | None = None,
    retry_after: float | None = None,
):
    """One *actual* wait, with its reason. No wait means no event at all."""
    trace_judge_stage(
        "wait_started",
        batch_no=batch_no,
        wait_seconds=seconds,
        wait_reason=reason,
        provider=provider,
        retry_after_seconds=retry_after,
    )
    started = time.perf_counter()
    status = "ok"
    # A controlled-clock host can set observed["elapsed_s"] so the recorded duration is
    # the one the throttle measured rather than a wall-clock near zero.
    observed: dict = {}
    try:
        yield observed
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        measured = observed.get("elapsed_s")
        elapsed_ms = (
            float(measured) * 1000
            if type(measured) in (int, float)
            else (time.perf_counter() - started) * 1000
        )
        trace_judge_stage(
            "wait_finished",
            batch_no=batch_no,
            status=status,
            wait_reason=reason,
            elapsed_ms=elapsed_ms,
        )


def trace_judge_restored_batch(
    *,
    batch_no: int,
    batch_start: int,
    expected_count: int,
    judged_count: int,
    unjudged_count: int,
    yes_count: int,
    no_count: int,
    status: str,
    restored_from: str,
    reason_hint: str | None = None,
    served_model: str | None = None,
    response_diagnostic: dict | None = None,
) -> None:
    """A batch restored from a previous run's atomic artifact. It costs no provider call.

    Kept in the same ordered event stream so `judge_runtime_inspect` can still prove
    batch numbering and claim coverage, while `application_level_calls=0` keeps it
    distinct from a batch this process actually judged.
    """
    rec = _ACTIVE.get()
    if rec is None:
        return
    try:
        rec.write(
            {
                "event": "batch_restored",
                "batch_no": batch_no,
                "batch_start_zero_based": batch_start,
                "expected_count": expected_count,
                "judged_count": judged_count,
                "unjudged_count": unjudged_count,
                "yes_count": yes_count,
                "no_count": no_count,
                "status": status,
                "reason_hint": reason_hint,
                "application_level_calls": 0,
                "sdk_retry_attempts": None,
                "elapsed_ms": 0.0,
                "restored_from": restored_from[:120],
                "served_model": served_model,
                **({"response_diagnostic": response_diagnostic} if response_diagnostic else {}),
            }
        )
    except Exception as exc:
        rec.failed = True
        _warn("judge_restored_batch_observer_failed", exc)


def trace_judge_batch(
    *,
    batch_no: int,
    batch_start: int,
    expected_count: int,
    verdict_by_claim: Mapping[int, bool],
    response: Any,
    text: str,
    error: BaseException | None,
    elapsed_ms: float,
    response_received: bool | None = None,
    preexisting_indices: set[int] | None = None,
    served_model: str | None = None,
    response_diagnostic: dict | None = None,
) -> None:
    """Observe final map coverage; inspect labels in memory without changing the parser."""
    rec = _ACTIVE.get()
    if rec is None:
        return
    try:
        local = [
            i
            for i in range(batch_start, batch_start + expected_count)
            if type(verdict_by_claim.get(i)) is bool
        ]
        labels = [int(number) for number, _ in _PATTERN.findall(text)]
        in_range = [n for n in labels if 1 <= n <= expected_count]
        count = len(local)
        received = response is not None if response_received is None else response_received
        status = "complete" if count == expected_count else "partial" if count else "unjudged"
        if isinstance(error, asyncio.CancelledError):
            status, reason = "cancelled", "evaluation_cancelled"
        elif error is not None:
            reason = "response_processing_exception" if received else "invoke_exception"
        elif count == expected_count:
            reason = None
        elif not text.strip():
            reason = "empty_or_nontext_response"
        elif count == 0:
            reason = "unparseable_response"
        else:
            reason = "partial_response"
        alias = isinstance(served_model, str) and bool(
            re.search(r"(^|[:/])auto(?:/|$)", served_model, re.I)
        )
        rec.write(
            {
                "event": "batch_finished",
                "batch_no": batch_no,
                "batch_start_zero_based": batch_start,
                "expected_count": expected_count,
                "judged_count": count,
                "yes_count": sum(verdict_by_claim[i] is True for i in local),
                "no_count": sum(verdict_by_claim[i] is False for i in local),
                "unjudged_count": expected_count - count,
                "status": status,
                "reason_hint": reason,
                "elapsed_ms": round(elapsed_ms, 3),
                "application_level_calls": 1,
                "sdk_retry_attempts": None,
                "exception_type": type(error).__name__ if error is not None else None,
                "http_status": _status_code(error),
                "response_chars": len(text),
                "recognizable_labels": len(labels),
                "parsed_unique_in_range_labels": len(set(in_range)),
                "out_of_range_labels": len(labels) - len(in_range),
                "duplicate_labels": len(in_range) - len(set(in_range)),
                "raw_response_saved": False,
                "tokens_input": token_usage(response, "input_tokens"),
                "tokens_output": token_usage(response, "output_tokens"),
                "cost_usd": None,
                "response_received": received,
                "response_content_kind": _content_kind(response, received),
                "preexisting_judged_count": sum(i in (preexisting_indices or set()) for i in local),
                "served_model": served_model
                if isinstance(served_model, str) and not alias
                else None,
                "disclosed_alias": served_model if alias else None,
                **({"response_diagnostic": response_diagnostic} if response_diagnostic else {}),
            }
        )
    except Exception as exc:
        rec.failed = True
        _warn("judge_batch_observer_failed", exc)
