"""Persistent, race-safe paid-call budget gate for GEPA-style evaluations.

The distinction this module enforces:

* an **evaluation** is one logical case/observation (e.g. one of the 64 GEPA evals);
* a **paid call** is one underlying model request (Planner / Judge / Reflection).
  One evaluation can trigger several paid calls. Budgets are expressed in paid
  calls per role, plus an overall paid-call cap; the evaluation count is recorded
  separately and never used as if it were the paid-call budget.

Design guarantees
-----------------
* Quota pre-occupation is atomic across processes: the quota is a set of numbered
  slot files created with O_CREAT|O_EXCL. A caller that cannot create a slot is
  denied; two concurrent callers can never occupy the same slot.
* Reservations are append-only. State transitions are separate event lines, so a
  response-before-commit crash leaves an UNKNOWN slot, never a silent retry.
* UNKNOWN slots (request sent, no durable completion) block further calls until a
  human explicitly acknowledges them. Retries consume a NEW slot: no free retry.
* A process-level OS lock prevents a stale shell timeout from being followed by a
  second overlapping run. A lock held by a live process cannot be taken over.
* Every decision, including denial, is written to a JSONL trace before returning.
* Preflight verifies output writability (write + read-back), prior run state, and
  remaining balance BEFORE any request is sent.

This module is stdlib-only and never calls a model itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROLE_PLANNER = "planner"
ROLE_JUDGE = "judge"
ROLE_REFLECTION = "reflection"
ROLES = (ROLE_PLANNER, ROLE_JUDGE, ROLE_REFLECTION)

STATUS_RESERVED = "RESERVED"
STATUS_SENT = "SENT"
STATUS_COMPLETED = "COMPLETED"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_FAILED = "FAILED"

TERMINAL = {STATUS_COMPLETED, STATUS_UNKNOWN, STATUS_FAILED}


class BudgetError(RuntimeError):
    """Base class for budget-gate failures."""


class RequestOutcomeUnknown(BudgetError):
    """The original classified request failed; distinct from blocking its successor."""

    def __init__(self, slot_id, role, cause_type):
        self.slot_id, self.role, self.cause_type = slot_id, role, cause_type
        super().__init__(
            f"{role} {slot_id}: outcome unknown after {cause_type}; automatic retry forbidden"
        )


class BudgetExhausted(BudgetError):
    """No slot remains for the requested role or overall cap."""


class BudgetPreflightError(BudgetError):
    """Output/balance/history check failed before any request."""


class UnknownCallBlocked(BudgetError):
    """An UNKNOWN prior request must be resolved by a human before continuing."""


class RunAlreadyActive(BudgetError):
    """Another live process holds the execution lock."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


@dataclass(frozen=True)
class BudgetConfig:
    """Authorized paid-call quotas. Evaluation count is tracked separately."""

    run_id: str
    quotas: Mapping[str, int]  # role -> max paid model calls
    total_cap: int  # max paid model calls across all roles
    planned_evaluations: int  # e.g. 64; informational, NOT a call budget
    source: str = "user-authorization"

    def validate(self) -> None:
        missing = set(ROLES) - set(self.quotas)
        if missing:
            raise BudgetPreflightError(f"missing quota for roles: {sorted(missing)}")
        for role in ROLES:
            q = self.quotas[role]
            if not isinstance(q, int) or q < 0:
                raise BudgetPreflightError(f"invalid quota for {role}: {q}")
        if not isinstance(self.total_cap, int) or self.total_cap < 0:
            raise BudgetPreflightError(f"invalid total cap: {self.total_cap}")
        if sum(self.quotas[r] for r in ROLES) > self.total_cap:
            raise BudgetPreflightError("sum of role quotas exceeds total cap")
        if not self.run_id:
            raise BudgetPreflightError("run_id is required")

    def config_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "quotas": dict(self.quotas),
            "total_cap": self.total_cap,
            "planned_evaluations": self.planned_evaluations,
            "source": self.source,
        }


class PaidCallBudget:
    """Persistent quota ledger. Instantiate once per authorized run."""

    def __init__(self, root: str | Path, config: BudgetConfig) -> None:
        self.root = Path(root)
        self.config = config
        self.slots_dir = self.root / "slots"
        self.trace_path = self.root / "budget_trace.jsonl"
        self.config_path = self.root / "budget_config.json"
        self.lock_path = self.root / "execution.lock"

    # ----------------------------------------------------------------- helpers
    def _trace(self, **fields: Any) -> None:
        record = {"at": _now(), "run_id": self.config.run_id, **fields}
        with open(self.trace_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _slot_path(self, role: str, index: int) -> Path:
        return self.slots_dir / f"{role}-{index:06d}.json"

    def _events_path(self, role: str, index: int) -> Path:
        return self.slots_dir / f"{role}-{index:06d}.events.jsonl"

    def _append_event(self, role: str, index: int, status: str, **fields: Any) -> None:
        event = {"at": _now(), "status": status, **fields}
        path = self._events_path(role, index)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._trace(event="slot_state", role=role, slot=index, status=status)

    # ---------------------------------------------------------------- preflight
    def preflight(self) -> dict[str, Any]:
        """Verify dir writable/readable, config persisted, run state and balance."""
        self.config.validate()
        probe = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.slots_dir.mkdir(parents=True, exist_ok=True)
            fd, probe = _mkstemp_readback(self.root)
        except OSError as exc:
            raise BudgetPreflightError(f"output directory unusable: {exc}") from exc

        cfg_disk = None
        if self.config_path.exists():
            try:
                cfg_disk = json.loads(self.config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise BudgetPreflightError(f"budget config unreadable: {exc}") from exc
            if cfg_disk != json.loads(json.dumps(self.config.config_dict(), ensure_ascii=False)):
                raise BudgetPreflightError("existing budget config differs from authorized config")
        else:
            _atomic_write(self.config_path, self.config.config_dict())

        state = self.state()
        if state["unknown"]:
            raise UnknownCallBlocked(
                f"UNKNOWN paid calls must be resolved by a human: {state['unknown']}"
            )
        if state["remaining_total"] <= 0:
            raise BudgetExhausted("no paid-call budget remains")
        self._trace(event="preflight", probe_ok=True, state=state)
        return {"writable": True, "readback_ok": True, "state": state}

    # ------------------------------------------------------------------- state
    def state(self) -> dict[str, Any]:
        used: dict[str, list[int]] = {r: [] for r in ROLES}
        statuses: dict[str, str] = {}
        for path in sorted(self.slots_dir.glob("*.json")):
            if path.is_symlink():
                raise BudgetPreflightError(f"symlink in slots dir: {path}")
            name = path.name[:-5]
            role, _, idx = name.rpartition("-")
            try:
                index = int(idx)
            except ValueError:
                continue
            if role not in used:
                continue
            used[role].append(index)
            events_path = self._events_path(role, index)
            status = STATUS_RESERVED
            if events_path.exists():
                for line in events_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        status = json.loads(line)["status"]
            statuses[f"{role}-{index:06d}"] = status
        unknown = sorted(k for k, v in statuses.items() if v == STATUS_UNKNOWN)
        sent = sorted(k for k, v in statuses.items() if v == STATUS_SENT)
        used_by_role = {r: len(v) for r, v in used.items()}
        total_used = sum(used_by_role.values())
        return {
            "used_by_role": used_by_role,
            "used_total": total_used,
            "remaining_by_role": {
                r: max(0, self.config.quotas[r] - used_by_role[r]) for r in ROLES
            },
            "remaining_total": max(0, self.config.total_cap - total_used),
            "planned_evaluations": self.config.planned_evaluations,
            "unknown": unknown + sent,  # SENT without completion is equally unknown
            "statuses": statuses,
        }

    # ------------------------------------------------------------------ reserve
    def reserve(
        self, role: str, purpose: str, input_fingerprint: str | None = None
    ) -> CallReservation:
        """Atomically occupy one paid-call slot, or raise BudgetExhausted."""
        if role not in ROLES:
            raise BudgetError(f"unknown role: {role}")
        self.slots_dir.mkdir(parents=True, exist_ok=True)
        # Fail closed for EVERY path, including direct per-request charges from
        # token_usage.gate_scope (which do not go through paid_call's preflight).
        prior = self.state()
        if prior["unknown"]:
            self._trace(event="denied", reason="unknown_prior_call", role=role)
            raise UnknownCallBlocked(
                f"UNKNOWN paid calls must be resolved by a human first: {prior['unknown']}"
            )
        quota = self.config.quotas[role]
        # Try every slot up to the quota; O_EXCL makes occupation race-safe.
        for index in range(1, quota + 1):
            path = self._slot_path(role, index)
            reservation = {
                "slot": index,
                "role": role,
                "run_id": self.config.run_id,
                "purpose": purpose,
                "input_fingerprint": input_fingerprint,
                "reserved_at": _now(),
                "owner_pid": os.getpid(),
                "owner_host": socket.gethostname(),
            }
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            except OSError as exc:
                raise BudgetError(f"cannot create reservation slot: {exc}") from exc
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(reservation, stream, ensure_ascii=False, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise BudgetError(f"cannot persist reservation: {exc}") from exc
            state = self.state()
            if state["used_total"] > self.config.total_cap:
                path.unlink(missing_ok=True)
                self._trace(event="denied", reason="total_cap", role=role, purpose=purpose)
                raise BudgetExhausted("overall paid-call cap reached")
            self._trace(
                event="reserved",
                role=role,
                slot=index,
                purpose=purpose,
                input_fingerprint=input_fingerprint,
            )
            return CallReservation(self, role, index, reservation)
        self._trace(event="denied", reason="role_quota", role=role, purpose=purpose)
        raise BudgetExhausted(f"paid-call quota exhausted for role={role}")

    def acknowledge_unknown(self, slot_id: str, *, decision: str, evidence: str) -> None:
        """Human decision for an UNKNOWN slot; records no fabricated response."""
        role, _, idx = slot_id.rpartition("-")
        try:
            index = int(idx)
        except ValueError as exc:
            raise BudgetError(f"invalid slot id: {slot_id}") from exc
        if role not in ROLES:
            raise BudgetError(f"invalid slot role: {role}")
        if decision not in ("resolved_completed", "abandoned"):
            raise BudgetError("decision must be resolved_completed or abandoned")
        if not evidence.strip():
            raise BudgetError("evidence is required to resolve an UNKNOWN call")
        self._append_event(
            role,
            index,
            STATUS_COMPLETED if decision == "resolved_completed" else STATUS_FAILED,
            human_decision=True,
            decision=decision,
            evidence=evidence,
        )

    # --------------------------------------------------------------------- lock
    def execution_lock(self) -> ExecutionLock:
        return ExecutionLock(self)

    def paid_call(self, role: str, purpose: str, call, input_data: Any = None):
        """Synchronous helper: preflight -> reserve -> call -> complete, fail-closed.

        The whole post-SENT window is guarded: a call that actually reached the
        provider but fails afterwards (e.g. a non-JSON result fingerprint) MUST
        be marked UNKNOWN, never left at SENT. A bare SENT slot is treated as
        UNKNOWN by state() and would block every later paid call in the run.
        """
        self.preflight()  # blocks on UNKNOWN prior calls, unwritable dirs, no balance
        fp = _sha(input_data) if input_data is not None else None
        reservation = self.reserve(role, purpose, fp)
        reservation.mark_sent()
        try:
            result = call()
            # SDK responses (Pydantic models, LangChain messages) are not
            # necessarily JSON-serializable. Hashing the result is an audit
            # nicety; it must never strand a billed call at SENT.
            try:
                result_fp = _sha(result)
            except TypeError, ValueError:
                result_fp = f"unserializable:{type(result).__name__}"
        except BaseException as exc:
            reservation.mark_unknown(reason=f"{type(exc).__name__}: {str(exc)[:300]}")
            raise
        reservation.mark_completed(result_fingerprint=result_fp)
        return result

    async def paid_call_async(self, role: str, purpose: str, call, input_data: Any = None):
        self.preflight()
        fp = _sha(input_data) if input_data is not None else None
        reservation = self.reserve(role, purpose, fp)
        reservation.mark_sent()
        try:
            result = await call()
            try:
                result_fp = _sha(result)
            except TypeError, ValueError:
                result_fp = f"unserializable:{type(result).__name__}"
        except BaseException as exc:
            reservation.mark_unknown(reason=f"{type(exc).__name__}: {str(exc)[:300]}")
            raise
        reservation.mark_completed(result_fingerprint=result_fp)
        return result


def _mkstemp_readback(root: Path) -> tuple[str, None]:
    import tempfile

    fd, probe = tempfile.mkstemp(prefix=".budget-probe-", dir=str(root))
    blob = b"paid-budget-probe\n"
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        with open(probe, "rb") as stream:
            if stream.read() != blob:
                raise OSError("write probe read-back mismatch")
    finally:
        os.unlink(probe)
    return "", None


@dataclass
class CallReservation:
    budget: PaidCallBudget
    role: str
    index: int
    record: dict[str, Any]
    _done: bool = False

    @property
    def slot_id(self) -> str:
        return f"{self.role}-{self.index:06d}"

    def mark_sent(self) -> None:
        if self._done:
            raise BudgetError("reservation already terminal")
        self.budget._append_event(self.role, self.index, STATUS_SENT)

    def mark_completed(self, **fields: Any) -> None:
        self.budget._append_event(self.role, self.index, STATUS_COMPLETED, **fields)
        self._done = True

    def mark_failed(self, **fields: Any) -> None:
        self.budget._append_event(self.role, self.index, STATUS_FAILED, **fields)
        self._done = True

    def mark_unknown(self, **fields: Any) -> None:
        """Request may have been billed but no durable response: never auto-retry."""
        self.budget._append_event(self.role, self.index, STATUS_UNKNOWN, **fields)
        self._done = True


class ExecutionLock:
    """OS-level execution lock; dies with the process, never with a lease timer."""

    def __init__(self, budget: PaidCallBudget) -> None:
        self.budget = budget
        self.handle = None

    def __enter__(self) -> ExecutionLock:
        path = self.budget.lock_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise BudgetPreflightError("execution lock is a symlink")
        self.handle = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                if not path.stat().st_size:
                    self.handle.write(b"0")
                    self.handle.flush()
                    os.fsync(self.handle.fileno())
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            self.budget._trace(event="lock_denied", reason=str(exc))
            raise RunAlreadyActive("another live evaluation process holds the lock") from exc
        previous = None
        marker = path.with_suffix(".owner.json")
        if marker.exists():
            try:
                previous = json.loads(marker.read_text(encoding="utf-8"))
            except OSError, ValueError:
                previous = {"state": "UNREADABLE"}
        owner = {
            "run_id": self.budget.config.run_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": _now(),
            "previous_owner": previous,
        }
        _atomic_write(marker, owner)
        self.budget._trace(event="lock_acquired", owner=owner)
        return self

    def __exit__(self, *_) -> None:
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None
            self.budget._trace(event="lock_released")
