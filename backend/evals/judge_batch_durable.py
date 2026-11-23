"""Durable, resumable Judge batches with cost protection.

Why this exists
---------------
A paid Judge batch once made 22 real calls whose results existed only in process
memory; a client-side timeout killed the shell while the process was still
running, and a later check ran before the final file was written. A separate
attempt made a paid call that could not be persisted because the output
directory was not writable. This module closes those gaps:

* every logical call gets an append-only ``CALL_STARTED`` record BEFORE the paid
  call, and a ``CALL_COMPLETED`` record with the raw response IMMEDIATELY after;
* each record is one fsync'd append, so a killed process cannot lose records
  already written;
* output writability is probed (write + read-back) before any paid call;
* a persistence failure stops the batch immediately (no further paid calls);
* a ``CALL_STARTED`` record with no ``CALL_COMPLETED`` is reported as UNKNOWN on
  resume and is NEVER silently re-called (that would double-bill);
* existing records are append-only; resume never edits or deletes evidence.

The module is stdlib-only so the same code runs in the evaluation container and
in local fake-Judge fault-injection tests. It does not call any model itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENT_STARTED = "CALL_STARTED"
EVENT_COMPLETED = "CALL_COMPLETED"
EVENT_UNMEASURED = "CALL_UNMEASURED"

STATUS_DONE = "DONE"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_UNMEASURED = "UNMEASURED"


class DurableBatchError(RuntimeError):
    """Base class for durable-batch failures."""


class PreflightError(DurableBatchError):
    """Output location is not usable; raised before any paid call."""


class PersistenceFailure(DurableBatchError):
    """A result could not be saved; no further paid call may be made."""


class UnknownCallError(DurableBatchError):
    """A call was started but never completed; re-calling needs an explicit decision."""


class CorruptLogError(DurableBatchError):
    """The ledger has a torn or unreadable line; fail closed before any call."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_hash(value: Any) -> str:
    """Deterministic hash of a JSON-serializable input fingerprint."""
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JudgeCallSpec:
    """One logical (case, variant, repeat) Judge call."""

    case_id: str
    variant: str
    repeat: int
    payload: Mapping[str, Any]

    @property
    def logical_id(self) -> str:
        return f"{self.case_id}|{self.variant}|{self.repeat}"

    @property
    def input_fingerprint(self) -> str:
        return canonical_hash(self.payload)


class DurableJudgeBatch:
    """Append-only Judge call ledger with preflight, resume, and fail-fast."""

    def __init__(self, log_path: str | Path, *, run_source: str, route: str | None = None) -> None:
        self.log_path = Path(log_path)
        self.incident_path = self.log_path.with_name(self.log_path.name + ".incidents.jsonl")
        self.run_source = run_source
        self.route = route

    # ---------------------------------------------------------------- preflight
    def preflight(self) -> dict[str, Any]:
        """Verify the output location can be written AND read before any paid call.

        Creates the parent directory if needed, writes a unique probe file with
        fsync, reads it back, then removes it. If an existing ledger is present,
        verifies it is readable and appendable. Raises PreflightError otherwise.
        """
        parent = self.log_path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PreflightError(f"cannot create output directory {parent}: {exc}") from exc
        if not parent.is_dir():
            raise PreflightError(f"output parent is not a directory: {parent}")

        probe_name = None
        try:
            fd, probe_name = tempfile.mkstemp(prefix=".judge-write-probe-", dir=str(parent))
            blob = b"judge-batch-write-probe\n"
            with os.fdopen(fd, "wb") as stream:
                stream.write(blob)
                stream.flush()
                os.fsync(stream.fileno())
            with open(probe_name, "rb") as stream:
                if stream.read() != blob:
                    raise PreflightError("write probe read-back mismatch")
        except PreflightError:
            raise
        except OSError as exc:
            raise PreflightError(f"output directory is not writable: {parent}: {exc}") from exc
        finally:
            if probe_name and os.path.exists(probe_name):
                try:
                    os.unlink(probe_name)
                except OSError:
                    pass

        if self.log_path.exists():
            try:
                with open(self.log_path, "rb") as stream:
                    stream.read(1)
                with open(self.log_path, "a", encoding="utf-8"):
                    pass
            except OSError as exc:
                raise PreflightError(
                    f"existing ledger is not readable/appendable: {self.log_path}: {exc}"
                ) from exc

        return {
            "log_path": str(self.log_path),
            "parent": str(parent),
            "writable": True,
            "readback_verified": True,
            "preflight_at": _utc_now(),
        }

    # ------------------------------------------------------------------ ledger
    def _append(self, record: Mapping[str, Any]) -> None:
        """One atomic append + fsync. Never edits existing bytes."""
        blob = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        try:
            fd = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                written = os.write(fd, blob)
                if written != len(blob):
                    raise OSError(f"short write: {written}/{len(blob)} bytes")
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise PersistenceFailure(f"failed to persist {record.get('event')}: {exc}") from exc

    def _incident(self, **fields: Any) -> None:
        """Best-effort separate incident record; never raises into the caller."""
        try:
            record = {"incident_at": _utc_now(), **fields}
            with open(self.incident_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            pass

    def load(self) -> list[dict[str, Any]]:
        """Read all ledger records in order; fail closed on a torn line."""
        if not self.log_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.log_path, encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except ValueError as exc:
                    raise CorruptLogError(f"torn ledger line {line_no} in {self.log_path}") from exc
                if (
                    not isinstance(record, dict)
                    or "event" not in record
                    or "logical_id" not in record
                ):
                    raise CorruptLogError(f"invalid ledger record at line {line_no}")
                records.append(record)
        return records

    def statuses(self) -> dict[str, str]:
        """Latest status per logical call, derived only from durable records."""
        state: dict[str, str] = {}
        for rec in self.load():
            lid = rec["logical_id"]
            event = rec["event"]
            if event == EVENT_COMPLETED:
                state[lid] = STATUS_DONE
            elif event == EVENT_STARTED:
                # Started with no later completed record = billed/unknown.
                state[lid] = STATUS_UNKNOWN
            elif event == EVENT_UNMEASURED:
                state.setdefault(lid, STATUS_UNMEASURED)
                if state[lid] != STATUS_DONE:
                    state[lid] = STATUS_UNMEASURED
        return state

    def pending(
        self,
        specs: Iterable[JudgeCallSpec],
        *,
        allow_unknown: bool = False,
        allow_unmeasured: bool = False,
    ) -> list[JudgeCallSpec]:
        """Return specs that still need a call; halt on ambiguous prior state."""
        state = self.statuses()
        todo: list[JudgeCallSpec] = []
        for spec in specs:
            status = state.get(spec.logical_id)
            if status == STATUS_DONE:
                continue
            if status == STATUS_UNKNOWN and not allow_unknown:
                raise UnknownCallError(
                    f"{spec.logical_id}: CALL_STARTED without CALL_COMPLETED; "
                    "status is UNKNOWN and will not be silently re-billed"
                )
            if status == STATUS_UNMEASURED and not allow_unmeasured:
                raise UnknownCallError(
                    f"{spec.logical_id}: previous call was unmeasured; "
                    "re-calling requires explicit operator authorization"
                )
            todo.append(spec)
        return todo

    def completed_results(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for rec in self.load():
            if rec["event"] == EVENT_COMPLETED:
                results[rec["logical_id"]] = rec
        return results

    # -------------------------------------------------------------------- run
    async def run(
        self,
        specs: Iterable[JudgeCallSpec],
        judge_call: Callable[[JudgeCallSpec], Any],
        *,
        allow_unknown: bool = False,
        allow_unmeasured: bool = False,
    ) -> dict[str, Any]:
        """Execute pending calls durably.

        ``judge_call(spec)`` may be sync or async and must return a JSON-serializable
        dict. A raised exception is recorded as CALL_UNMEASURED and the batch
        continues. A persistence failure aborts the batch immediately.
        """
        preflight = self.preflight()
        spec_list = list(specs)
        todo = self.pending(
            spec_list, allow_unknown=allow_unknown, allow_unmeasured=allow_unmeasured
        )
        report = {
            "schema_version": 1,
            "run_source": self.run_source,
            "route": self.route,
            "preflight": preflight,
            "total_logical_calls": len(spec_list),
            "skipped_already_done": len(spec_list) - len(todo),
            "attempted_this_run": 0,
            "completed_this_run": 0,
            "unmeasured_this_run": 0,
            "stopped_reason": None,
        }
        for spec in todo:
            # 1) durable intent record BEFORE any paid call
            self._append(
                {
                    "event": EVENT_STARTED,
                    "logical_id": spec.logical_id,
                    "case_id": spec.case_id,
                    "variant": spec.variant,
                    "repeat": spec.repeat,
                    "input_fingerprint": spec.input_fingerprint,
                    "route": self.route,
                    "run_source": self.run_source,
                    "recorded_at": _utc_now(),
                }
            )
            report["attempted_this_run"] += 1
            try:
                raw_result = judge_call(spec)
                if inspect.isawaitable(raw_result):
                    raw_result = await raw_result
                if not isinstance(raw_result, Mapping):
                    raise TypeError("judge_call must return a mapping")
            except Exception as exc:  # timeout / parsing / provider error
                self._append(
                    {
                        "event": EVENT_UNMEASURED,
                        "logical_id": spec.logical_id,
                        "case_id": spec.case_id,
                        "variant": spec.variant,
                        "repeat": spec.repeat,
                        "input_fingerprint": spec.input_fingerprint,
                        "route": self.route,
                        "run_source": self.run_source,
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                        "recorded_at": _utc_now(),
                    }
                )
                report["unmeasured_this_run"] += 1
                continue

            # 2) durable result record IMMEDIATELY after Judge returns
            try:
                self._append(
                    {
                        "event": EVENT_COMPLETED,
                        "logical_id": spec.logical_id,
                        "case_id": spec.case_id,
                        "variant": spec.variant,
                        "repeat": spec.repeat,
                        "input_fingerprint": spec.input_fingerprint,
                        "route": self.route,
                        "run_source": self.run_source,
                        "result": dict(raw_result),
                        "recorded_at": _utc_now(),
                    }
                )
            except PersistenceFailure as exc:
                self._incident(
                    logical_id=spec.logical_id,
                    case_id=spec.case_id,
                    variant=spec.variant,
                    repeat=spec.repeat,
                    phase="result_persist_failed",
                    error=str(exc),
                    status=STATUS_UNKNOWN,
                )
                report["stopped_reason"] = (
                    f"persistence failure after Judge returned: {spec.logical_id}"
                )
                raise
            report["completed_this_run"] += 1
        report["final_statuses"] = self.statuses()
        return report


def run_batch_sync(
    log_path: str | Path,
    specs: Iterable[JudgeCallSpec],
    judge_call: Callable[[JudgeCallSpec], Any],
    *,
    run_source: str,
    route: str | None = None,
    allow_unknown: bool = False,
    allow_unmeasured: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper for synchronous callers."""
    batch = DurableJudgeBatch(log_path, run_source=run_source, route=route)
    return asyncio.run(
        batch.run(specs, judge_call, allow_unknown=allow_unknown, allow_unmeasured=allow_unmeasured)
    )
