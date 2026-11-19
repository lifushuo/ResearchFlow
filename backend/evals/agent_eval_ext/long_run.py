"""P0 reliability for long multi-agent report evaluations (docs/08 §"长任务可靠性").

The 420 s worker kill of ``quantum-error-correction`` lost a finished research draft
because the graph's only checkpoint was an in-process ``MemorySaver``, and the judge's
result existed only in memory until the very last line of ``evals.harness``. Eight
unconditional 15 s inter-batch waits (and a 60 s wait after the last query) consumed
the budget while the worker had no stage of its own to stop at.

This module supplies the missing pieces and deliberately reuses the existing ones:
``app.services.evaluation_jobs.update_status`` for atomic replacement,
``app.services.evaluation_closure.core.input_hash`` for canonical fingerprints,
``judge_runtime``'s event names and ``judge_labels``' batch-local parsing. It changes
no score. A batch that was not judged stays unjudged and out of the denominator; a
batch that was saved is never re-requested.

It is stdlib-only (plus those two pure-stdlib app services), so the controlled-clock
tests can exercise it without a provider, a key, or a database.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from app.services.evaluation_closure.core import input_hash
from app.services.evaluation_jobs import update_status

SCHEMA_VERSION = 1
#: Bump when the judge prompt, label policy, batching or throttle semantics change.
#: A resumed run refuses batches recorded under a different version instead of
#: silently mixing two scoring definitions.
JUDGE_CONFIG_VERSION = "rf-judge-reliability/2"
JUDGE_BATCH_SIZE = 4

RESEARCH_DRAFT_READY = "research_draft_ready"
RESEARCH_FAILED = "research_failed"
JUDGE_PARTIAL = "judge_partial"
JUDGE_COMPLETED = "judge_completed"
JUDGE_FAILED = "judge_failed"
ARTIFACT_SAVE_FAILED = "artifact_save_failed"
STATES = frozenset(
    {
        RESEARCH_DRAFT_READY,
        RESEARCH_FAILED,
        JUDGE_PARTIAL,
        JUDGE_COMPLETED,
        JUDGE_FAILED,
        ARTIFACT_SAVE_FAILED,
    }
)

#: Non-zero process exits the worker can name without guessing from a traceback.
EXIT_JUDGE_PARTIAL = 3
EXIT_JUDGE_FAILED = 4
EXIT_ARTIFACT_SAVE_FAILED = 5

BATCH_STATUSES = frozenset({"complete", "partial", "unjudged"})
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024


class ReliabilityError(RuntimeError):
    """A reliability precondition that must be reported, not worked around."""


class CooperativeStop(ReliabilityError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class StageDeadlineExceeded(ReliabilityError):
    """A named stage used up its own budget; the run stops cooperatively."""

    def __init__(self, stage: str, budget_s: float) -> None:
        super().__init__(f"{stage} deadline exceeded ({budget_s:.2f}s)")
        self.stage = stage
        self.budget_s = budget_s


class ResumeRejected(ReliabilityError):
    """Saved progress cannot be trusted for this run; no silent re-scoring."""


class ArtifactConflict(ReliabilityError):
    """A per-batch result already exists; never overwrite saved evidence."""


class ArtifactSaveFailed(ReliabilityError):
    """Progress could not be written; the run must say so rather than lose it silently."""


class ChildStillRunning(Exception):
    """The child did not exit inside its cooperative-stop grace period."""


# ── Injectable clock / sleep / wait_for ────────────────────────────────────────────
# Module-level functions, not captured references: tests replace these three, and the
# existing `fast_judge` fixture replaces `asyncio.sleep` on the asyncio module itself
# (which `_sleep` resolves at call time), so no real time passes in any test.


def _monotonic() -> float:
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _wait_for(awaitable, timeout: float):
    return asyncio.wait_for(awaitable, timeout)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)[:80] or "unknown"


def _number(value: Any, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(value, "").strip() if isinstance(value, str) else value
    if raw in (None, ""):
        return default
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise ReliabilityError(f"invalid numeric setting {value!r}") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ReliabilityError(f"out-of-range numeric setting {value!r}")
    return parsed


def stop_grace_seconds() -> float:
    value = _number("RF_EVAL_STOP_GRACE_S", 15.0, minimum=0.1)
    if value > 60:
        raise ReliabilityError("stop grace must not exceed 60 seconds")
    return value


def worker_hard_cap(max_cases: int) -> int:
    return 28800 if max_cases == 0 else min(7200, max(420, max_cases * 360))


# ── Explicit stage deadlines ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Deadlines:
    """Research, judge, overall, single-request and stop-grace budgets in seconds.

    The worker's 420 s hard cap is unchanged. The overall budget is deliberately
    *below* it so a cooperative SIGTERM plus progress flush has room to finish
    before the worker would have to force-kill the process.
    """

    research_s: float = 240.0
    judge_s: float = 390.0
    total_s: float = 390.0
    request_s: float = 45.0
    stop_grace_s: float = 15.0
    hard_cap_s: float = 420.0
    judge_policy: str = "shared_remaining"

    @classmethod
    def from_env(cls, *, case_count: int = 1) -> Deadlines:
        hard_cap = _number(
            "RF_EVAL_WORKER_HARD_CAP_S", float(worker_hard_cap(case_count)), minimum=60.0
        )
        cases = _number("RF_EVAL_WORKER_CASES", float(case_count), minimum=1.0)
        if hard_cap > 28800 or cases > 20 or not cases.is_integer():
            raise ReliabilityError("invalid worker deadline scope")
        total = _number("RF_EVAL_TOTAL_DEADLINE_S", min(390.0 * cases, hard_cap - 30), minimum=1.0)
        values = cls(
            research_s=_number(
                "RF_EVAL_RESEARCH_DEADLINE_S", min(240.0 * cases, hard_cap - 30), minimum=1.0
            ),
            judge_s=_number("RF_EVAL_JUDGE_DEADLINE_S", total, minimum=1.0),
            total_s=_number(
                "RF_EVAL_TOTAL_DEADLINE_S", min(390.0 * cases, hard_cap - 30), minimum=1.0
            ),
            request_s=_number("RF_EVAL_JUDGE_REQUEST_TIMEOUT_S", 45.0, minimum=1.0),
            stop_grace_s=stop_grace_seconds(),
            hard_cap_s=hard_cap,
            judge_policy="fixed"
            if os.environ.get("RF_EVAL_JUDGE_DEADLINE_S")
            else "shared_remaining",
        )
        if (
            values.research_s > values.total_s
            or values.judge_s > values.total_s
            or values.request_s > values.judge_s
        ):
            raise ReliabilityError("stage deadlines must fit inside the overall deadline")
        if values.total_s + values.stop_grace_s > values.hard_cap_s:
            raise ReliabilityError("overall deadline plus stop grace exceeds the worker hard cap")
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "research_s": self.research_s,
            "judge_s": self.judge_s,
            "total_s": self.total_s,
            "request_s": self.request_s,
            "stop_grace_s": self.stop_grace_s,
            "worker_hard_cap_s": self.hard_cap_s,
            "judge_policy": self.judge_policy,
            "runtime_policy_version": "shared-deadline/1",
        }


@dataclass
class Budget:
    """Elapsed time per stage on one clock; every check is against the overall cap too."""

    deadlines: Deadlines
    started: float | None = None
    spent: dict[str, float] = field(default_factory=lambda: {"research": 0.0, "judge": 0.0})
    clock: Callable[[], float] | None = None
    case_deadline: float | None = None

    def now(self) -> float:
        return (self.clock or _monotonic)()

    def __post_init__(self) -> None:
        if self.started is None:
            self.started = self.now()

    def spend(self, stage: str, seconds: float) -> None:
        self.spent[stage] = self.spent.get(stage, 0.0) + max(0.0, seconds)

    def start_case(self, remaining_cases: int) -> None:
        # Reserve an equal share for each remaining arm; unused time rolls forward.
        left = max(0.0, self.deadlines.total_s - (self.now() - self.started))
        self.case_deadline = self.now() + left / max(1, remaining_cases)

    def remaining(self, stage: str) -> float:
        total_left = self.deadlines.total_s - (self.now() - self.started)
        if self.case_deadline is not None:
            total_left = min(total_left, self.case_deadline - self.now())
        stage_left = getattr(self.deadlines, f"{stage}_s", self.deadlines.total_s) - self.spent.get(
            stage, 0.0
        )
        return max(0.0, min(total_left, stage_left))

    def check(self, stage: str) -> float:
        left = self.remaining(stage)
        if left <= 0:
            raise StageDeadlineExceeded(stage, getattr(self.deadlines, f"{stage}_s"))
        return left

    def request_timeout(self) -> float:
        return min(self.deadlines.request_s, self.remaining("judge"))


async def run_request(awaitable, *, budget: Budget):
    """One judge request, capped by the remaining judge/overall budget."""
    try:
        timeout = budget.check("judge")
    except StageDeadlineExceeded:
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise
    timeout = min(timeout, budget.deadlines.request_s)
    started = budget.now()
    try:
        return await _timed_call(awaitable, timeout)
    except TimeoutError as exc:
        raise StageDeadlineExceeded("judge_request", timeout) from exc
    finally:
        budget.spend("judge", budget.now() - started)


async def with_deadline(awaitable, *, budget: Budget, stage: str):
    """Cooperative cancellation of one stage, never a process kill.

    `asyncio.wait_for` cancels only the awaited coroutine; the harness then persists
    whatever it has. A hard kill remains the worker's last resort, not this function.
    """
    try:
        timeout = budget.check(stage)
    except StageDeadlineExceeded:
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise
    started = budget.now()
    try:
        return await _timed_call(awaitable, timeout)
    except TimeoutError as exc:
        raise StageDeadlineExceeded(stage, timeout) from exc
    finally:
        budget.spend(stage, budget.now() - started)


# ── Provider throttle, Retry-After, and every wait's reason ────────────────────────


def query_interval() -> float:
    """Optional pacing between two *real* research queries; 0 = none.

    Replaces the harness's unconditional 60 s sleep after every query, including the
    last one. It is only ever applied when another query actually follows.
    """
    return _number("RF_EVAL_QUERY_MIN_INTERVAL_S", 0.0)


def throttle_interval(provider: str) -> float:
    """Minimum seconds between two judge requests for one provider; 0 = none.

    Per-provider override wins (`RF_EVAL_JUDGE_THROTTLE_GOOGLE_S`), then a global
    `RF_EVAL_JUDGE_MIN_INTERVAL_S`, then no wait at all. The provider is the route's
    prefix, so an operator can configure its actual limit without a universal sleep.
    """
    key = "RF_EVAL_JUDGE_THROTTLE_" + re.sub(r"[^A-Z0-9]", "_", provider.upper()) + "_S"
    return _number(key, _number("RF_EVAL_JUDGE_MIN_INTERVAL_S", 0.0))


def retry_after_seconds(exc: BaseException | None) -> float | None:
    """Honour a provider's `Retry-After` (seconds or HTTP date), bounded and disclosed."""
    if exc is None:
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None) or {}
    raw = None
    for name, value in headers.items() if hasattr(headers, "items") else []:
        if str(name).lower() == "retry-after":
            raw = value
            break
    if raw is None:
        raw = getattr(exc, "retry_after", None)
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except TypeError, ValueError:
        try:
            when = parsedate_to_datetime(str(raw))
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            seconds = (when - datetime.now(UTC)).total_seconds()
        except TypeError, ValueError, OverflowError:
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def http_status(exc: BaseException | None) -> int | None:
    """Reuse the judge trace's existing status extraction — one home for this rule."""
    from .judge_runtime import _status_code

    return _status_code(exc)


@dataclass(frozen=True)
class WaitRecord:
    reason: str
    requested_s: float
    actual_s: float
    provider: str | None
    batch_no: int | None
    retry_after_s: float | None
    at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "requested_s": round(self.requested_s, 6),
            "elapsed_s": round(self.actual_s, 6),
            "provider": self.provider,
            "after_batch_no": self.batch_no,
            "retry_after_s": self.retry_after_s,
            "at": self.at,
        }


class Throttle:
    """Wait only before the *next* request, and only as long as actually needed."""

    def __init__(
        self,
        *,
        provider: str,
        min_interval_s: float,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self.provider = provider
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.sleep = sleep
        self.records: list[WaitRecord] = []
        self._last_request_at: float | None = None
        self._retry_after_until: float | None = None
        self.blocked_reason: str | None = None

    def now(self) -> float:
        return (self.clock or _monotonic)()

    async def _do_sleep(self, seconds: float) -> None:
        await _until_stopped((self.sleep or _sleep)(seconds))

    def note_request_started(self) -> None:
        self._last_request_at = self.now()

    def note_rate_limit(self, exc: BaseException, *, status: int | None = None) -> float | None:
        """A 429 paces the *following* batch; the failed batch is never re-requested."""
        if (status if status is not None else http_status(exc)) != 429:
            return None
        seconds = retry_after_seconds(exc)
        if seconds is None:
            seconds = self.min_interval_s
        if seconds > _number("RF_EVAL_JUDGE_RETRY_AFTER_MAX_S", 60.0):
            # A local waiting cap cannot shorten the provider's cooldown.
            self.blocked_reason = "retry_after_exceeds_wait_limit"
        self._retry_after_until = self.now() + seconds
        return seconds

    def required(self) -> tuple[float, str, float | None]:
        now = self.now()
        interval_due = (
            self._last_request_at + self.min_interval_s
            if self._last_request_at is not None
            else None
        )
        retry_due = self._retry_after_until
        due = max([d for d in (interval_due, retry_due) if d is not None], default=None)
        if due is None or due <= now:
            self._retry_after_until = None
            return 0.0, "not_required", None
        reason = (
            "retry_after"
            if retry_due is not None and (interval_due is None or retry_due >= interval_due)
            else "provider_throttle"
        )
        return due - now, reason, (retry_due - now if retry_due is not None else None)

    async def wait_before_next(
        self,
        *,
        batch_no: int | None,
        budget: Budget | None = None,
        reason_override: str | None = None,
        planned: tuple[float, str, float | None] | None = None,
    ) -> WaitRecord:
        # `planned` comes from the caller's single `required()` call, so the traced
        # wait_seconds, the sleep and the WaitRecord can never disagree.
        seconds, reason, retry_after = planned if planned is not None else self.required()
        if budget is not None:
            seconds = min(seconds, budget.remaining("judge"))
        started = self.now()
        if seconds > 0:
            await self._do_sleep(seconds)
        actual = self.now() - started
        resolved_reason = reason_override or (reason if seconds > 0 else "not_required")
        record = WaitRecord(
            reason=resolved_reason,
            requested_s=seconds,
            actual_s=actual,
            provider=self.provider,
            batch_no=batch_no,
            retry_after_s=retry_after,
            at=_utcnow(),
        )
        self.records.append(record)
        return record

    async def wait_fixed(self, seconds: float, *, reason: str) -> WaitRecord:
        """A wait requested explicitly by a host policy (e.g. between two queries)."""
        started = self.now()
        if seconds > 0:
            await self._do_sleep(seconds)
        actual = self.now() - started
        record = WaitRecord(
            reason=reason if seconds > 0 else "not_required",
            requested_s=seconds,
            actual_s=actual,
            provider=self.provider,
            batch_no=None,
            retry_after_s=None,
            at=_utcnow(),
        )
        self.records.append(record)
        return record


# ── Cooperative stop ───────────────────────────────────────────────────────────────


class StopRequest:
    def __init__(self) -> None:
        self.requested = False
        self.reason: str | None = None
        self._event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def request(self, reason: str) -> None:
        if not self.requested:
            self.requested = True
            self.reason = reason
            if self._event is not None:
                # signal.signal handlers must wake the selector, not only mark a Future ready.
                self._loop.call_soon_threadsafe(self._event.set)

    async def wait(self) -> None:
        if self.requested:
            return
        if self._event is None:
            self._event = asyncio.Event()
            self._loop = asyncio.get_running_loop()
        await self._event.wait()


async def _timed_call(awaitable, timeout: float):
    task = asyncio.ensure_future(awaitable)
    try:
        return await _wait_for(_until_stopped(task), timeout)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _until_stopped(awaitable):
    ctx = current()
    if ctx is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    stopped = asyncio.create_task(ctx.stop.wait())
    try:
        done, _ = await asyncio.wait([task, stopped], return_when=asyncio.FIRST_COMPLETED)
        # Keep a response that completed concurrently with the stop: it may already be billed.
        if task in done:
            return task.result()
        raise CooperativeStop(ctx.stop.reason or "cooperative_stop")
    finally:
        for pending in [task, stopped]:
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, stopped, return_exceptions=True)


def install_stop_handlers(stop: StopRequest) -> list[tuple[int, Any]]:
    """SIGTERM/SIGINT set a flag; the running batch is awaited and its progress saved."""
    previous: list[tuple[int, Any]] = []
    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if not isinstance(sig, int):
            continue
        try:
            previous.append((sig, signal.getsignal(sig)))
            signal.signal(sig, lambda _s, _f, n=name: stop.request(f"signal_{n}"))
        except ValueError, OSError:
            previous.pop()
    return previous


def restore_stop_handlers(previous: list[tuple[int, Any]]) -> None:
    for sig, handler in previous:
        try:
            signal.signal(sig, handler)
        except ValueError, OSError:
            pass


def stop_child(
    *,
    send: Callable[[str], None],
    wait: Callable[[float | None], int],
    grace_s: float,
) -> dict[str, Any]:
    """SIGTERM first; SIGKILL only if the child ignores its grace period.

    `wait` raises `ChildStillRunning` on timeout. Injected callables keep this
    testable on Windows, where `os.killpg` does not exist in the worker runtime.
    """
    send("SIGTERM")
    try:
        code = wait(grace_s)
        return {"mode": "cooperative", "forced": False, "exit_code": code}
    except ChildStillRunning:
        send("SIGKILL")
        return {"mode": "forced", "forced": True, "exit_code": wait(None)}


def child_waiter(process: subprocess.Popen) -> Callable[[float | None], int]:
    def wait(timeout: float | None) -> int:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ChildStillRunning from exc

    return wait


# ── Atomic, write-once progress artifacts ──────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ResumeRejected(f"missing_or_unsafe:{path.name}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ResumeRejected(f"unreadable:{path.name}") from exc
    if not isinstance(doc, dict):
        raise ResumeRejected(f"invalid_root:{path.name}")
    return doc


def atomic_create(path: Path, data: dict[str, Any]) -> None:
    """Exclusive, atomic creation of one completed batch — never overwrite evidence."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)  # fails with FileExistsError if the batch is already saved
        except FileExistsError as exc:
            raise ArtifactConflict(f"batch already saved: {path.name}") from exc
        except AttributeError, NotImplementedError, OSError:
            # No hard links on this filesystem: exclusive create still forbids
            # overwriting; a torn file is rejected by the resume validator.
            with path.open("x", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
    finally:
        tmp.unlink(missing_ok=True)


def persist(action, *args, **kwargs):
    """Wrap one artifact write so a failed save is named, never swallowed."""
    try:
        return action(*args, **kwargs)
    except (OSError, TypeError, ValueError, ArtifactConflict) as exc:
        raise ArtifactSaveFailed(
            f"{getattr(action, '__name__', 'save')}: {type(exc).__name__}"
        ) from exc


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    """`input_hash` with a deterministic fallback for a state value JSON cannot sort."""
    try:
        return input_hash(value)
    except TypeError, ValueError, OverflowError:
        fallback = json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
        )
        return sha256_text(fallback)


def build_config(
    *,
    run_config: Any,
    judge_route: str | None,
    judge_prompt: str,
    case_id: str,
    query: str,
    depth: str,
    prompt_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The original run configuration, allowlisted so no key can enter a fingerprint."""
    from evals.judge_config import judge_generation_fingerprint

    models = getattr(run_config, "models", {}) or {}
    return {
        **judge_generation_fingerprint(),
        "schema_version": SCHEMA_VERSION,
        "judge_config_version": JUDGE_CONFIG_VERSION,
        "llm_mode": getattr(run_config, "llm_mode", None),
        "models": {role: models.get(role) for role in sorted(models)},
        "limits": {
            "max_parallel_tasks": getattr(run_config, "max_parallel_tasks", None),
            "max_critic_loops": getattr(run_config, "max_critic_loops", None),
            "max_wallclock_seconds": getattr(run_config, "max_wallclock_seconds", None),
            "max_input_tokens": getattr(run_config, "max_input_tokens", None),
            "max_cost_per_session_usd": getattr(run_config, "max_cost_per_session_usd", None),
        },
        "judge_route": judge_route,
        "judge_prompt_sha256": sha256_text(judge_prompt),
        "judge_batch_size": JUDGE_BATCH_SIZE,
        "case_id": case_id,
        "query": query,
        "depth": depth,
        "phase": "candidate" if prompt_overrides is not None else "baseline",
        "prompt_overrides_sha256": canonical_hash(prompt_overrides or {}),
    }


@dataclass
class ResumeState:
    draft: dict[str, Any]
    batches: dict[int, dict[str, Any]]
    progress: dict[str, Any]

    @property
    def restored_batches(self) -> int:
        return len(self.batches)


class ProgressStore:
    """One directory per case; drafts are reproducible, batches are write-once."""

    def __init__(
        self,
        directory: Path,
        *,
        config: dict[str, Any],
        source_dir: Path | None = None,
        ledger_root: Path | None = None,
    ) -> None:
        self.dir = directory
        self.source_dir = source_dir
        self.config = config
        self.fingerprint = input_hash(config)
        self.status: str | None = None
        self.reason: str | None = None
        self.save_failed = False
        self.waits: list[WaitRecord] = []
        self.resume: ResumeState | None = None
        self.lineage_id = secrets.token_hex(16)
        self.ledger_root = ledger_root or directory.parent / ".judge-ledger"
        self.lease = None

    @property
    def batches_dir(self) -> Path:
        return self.dir / "batches"

    def preflight(self) -> None:
        """Actually write before any model call — an existing directory is not proof."""
        if any((self.dir / "drafts").glob("*.json")) or any(self.batches_dir.glob("batch-*.json")):
            raise ArtifactConflict("progress destination already contains evidence")
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.ledger_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.ledger_root, prefix=".write-check-") as probe:
            probe.write(b"batch ledger write check")
            probe.flush()
        (self.batches_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.dir / "drafts").mkdir(mode=0o700, parents=True, exist_ok=True)
        self.save_progress(status=None, reason=None, extra={"preflight": _utcnow()})

    def save_progress(
        self, *, status: str | None, reason: str | None, extra: dict[str, Any] | None = None
    ) -> None:
        if status is not None and status not in STATES:
            raise ReliabilityError(f"unknown reliability status {status!r}")
        self.status = status or self.status
        self.reason = reason
        doc = {
            "schema_version": SCHEMA_VERSION,
            "judge_config_version": JUDGE_CONFIG_VERSION,
            "config_fingerprint": self.fingerprint,
            "config": self.config,
            "lineage_id": self.lineage_id,
            "status": self.status,
            "reason": self.reason,
            "updated_at": _utcnow(),
            "waits": [w.as_dict() for w in self.waits],
            "save_failed": self.save_failed,
            **(extra or {}),
        }
        update_status(self.dir / "progress.json", doc)

    def note_wait(self, record: WaitRecord) -> None:
        self.waits.append(record)

    def draft_path(self, case_id: str) -> Path:
        return self.dir / "drafts" / f"{_safe_id(case_id)}.json"

    def save_draft(self, case_id: str, doc: dict[str, Any]) -> None:
        path = self.draft_path(case_id)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "config_fingerprint": self.fingerprint,
            "judge_config_version": JUDGE_CONFIG_VERSION,
            "saved_at": _utcnow(),
            **doc,
        }
        payload["evidence_sha256"] = canonical_hash(payload.get("evidence") or [])
        update_status(path, payload)

    def save_batch(self, batch_no: int, doc: dict[str, Any]) -> None:
        """A *completed* batch is canonical and write-once; an unfinished one is a
        diagnostic attempt that a later resume must re-score, never replay.

        Naming the two apart means the resume validator can glob the canonical
        files only, without ever overwriting a partial attempt's evidence.
        """
        payload = {
            "schema_version": SCHEMA_VERSION,
            "batch_no": batch_no,
            "judge_config_version": JUDGE_CONFIG_VERSION,
            "config_fingerprint": self.fingerprint,
            **doc,
        }
        if self.lease is not None:
            payload["batch_id"] = self.lease.batch_id
            payload["scoring_rule_version"] = "claim-labels/v5"
            payload["batching_version"] = "fixed-four/v1"
            if doc.get("status") == "complete":
                self.lease.commit(payload)
            else:
                self.lease.record(
                    "incomplete",
                    reason=doc.get("reason_hint"),
                    response_received=doc.get("response_received"),
                )
        stem = f"batch-{batch_no:04d}"
        if doc.get("status") == "complete":
            atomic_create(self.batches_dir / f"{stem}.json", payload)
        else:
            atomic_create(self.batches_dir / f"{stem}.attempt-{secrets.token_hex(4)}.json", payload)

    def save_final(self, *, status: str, reason: str | None, extra: dict[str, Any]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "reason": reason,
            "config_fingerprint": self.fingerprint,
            **extra,
        }
        update_status(self.dir / "final-status.json", payload)

    def load(self, *, case_id: str) -> ResumeState:
        """Validate the report and judge configuration before reusing any saved score."""
        source = self.source_dir or self.dir
        progress = _read_json(source / "progress.json")
        self.lineage_id = progress.get("lineage_id") or sha256_text(str(source.resolve()))
        if (
            progress.get("schema_version") != SCHEMA_VERSION
            or progress.get("judge_config_version") != JUDGE_CONFIG_VERSION
            or progress.get("config_fingerprint") != self.fingerprint
            or progress.get("config") != self.config
        ):
            raise ResumeRejected("run_or_judge_configuration_changed")
        draft = _read_json(source / "drafts" / f"{_safe_id(case_id)}.json")
        report = draft.get("draft_report")
        sources = draft.get("sources")
        claim_lines = draft.get("claim_lines")
        if (
            draft.get("schema_version") != SCHEMA_VERSION
            or draft.get("case_id") != case_id
            or draft.get("config_fingerprint") != self.fingerprint
            or draft.get("judge_config_version") != JUDGE_CONFIG_VERSION
            or draft.get("state") != RESEARCH_DRAFT_READY
            or not isinstance(report, str)
            or not isinstance(sources, list)
            or not isinstance(claim_lines, list)
            or draft.get("report_sha256") != sha256_text(report)
            or draft.get("sources_sha256") != canonical_hash(sources)
            or draft.get("claims_sha256") != canonical_hash(claim_lines)
            or (
                "evidence_sha256" in draft
                and draft["evidence_sha256"] != canonical_hash(draft.get("evidence") or [])
            )
        ):
            raise ResumeRejected("report_or_sources_changed")
        batches: dict[int, dict[str, Any]] = {}
        source_batches = source / "batches"
        if source_batches.is_dir() and not source_batches.is_symlink():
            # Only canonical, fully judged batches are "finished". `*.attempt-*.json`
            # records (429 / empty reply / cancelled / timeout) stay on disk as
            # evidence but are deliberately re-scored on the next run.
            canonical = re.compile(r"^batch-(\d{4})\.json$")
            for path in sorted(source_batches.glob("batch-*.json")):
                match = canonical.fullmatch(path.name)
                if match is None:
                    continue
                row = _read_json(path)
                expected = int(match.group(1))
                self.validate_batch(row, expected, draft)
                batches[expected] = row
        self.resume = ResumeState(draft=draft, batches=batches, progress=progress)
        return self.resume

    def load_for_rejudge(self, *, case_id: str) -> ResumeState:
        """只复用已校验草稿；改变生成预算后，新评分从零开始且不写回旧目录。"""
        source = self.source_dir
        if source is None:
            raise ResumeRejected("rejudge_requires_a_source")
        progress = _read_json(source / "progress.json")
        original = progress.get("config")
        if not isinstance(original, dict):
            raise ResumeRejected("invalid_source_configuration")
        # 只允许改变输出预算，研究、Prompt、Judge 路由及规则必须一致。
        before = {k: v for k, v in original.items() if k != "judge_generation"}
        after = {k: v for k, v in self.config.items() if k != "judge_generation"}
        if before != after:
            raise ResumeRejected("rejudge_changes_more_than_output_budget")
        old = ProgressStore(source, config=original).load(case_id=case_id)
        draft = {
            **old.draft,
            "config_fingerprint": self.fingerprint,
            "saved_at": _utcnow(),
            "research_reused_from": {
                "config_fingerprint": progress["config_fingerprint"],
                "report_sha256": old.draft["report_sha256"],
            },
        }
        self.resume = ResumeState(draft=draft, batches={}, progress=progress)
        return self.resume

    def validate_batch(self, row: dict, expected: int, draft: dict) -> None:
        report_hash = draft["report_sha256"]
        sources_hash = draft["sources_sha256"]
        claim_lines = draft["claim_lines"]
        claims_count = len(claim_lines)
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("judge_config_version") != JUDGE_CONFIG_VERSION
            or row.get("config_fingerprint") != self.fingerprint
            or row.get("report_sha256") != report_hash
            or row.get("sources_sha256") != sources_hash
            or type(row.get("batch_no")) is not int
            or row["batch_no"] != expected
            or row.get("status") != "complete"
            or row.get("unjudged_count") != 0
            or type(row.get("expected_count")) is not int
            or not 1 <= row["expected_count"] <= JUDGE_BATCH_SIZE
            or not isinstance(row.get("verdicts"), dict)
            or len(row["verdicts"]) != row["expected_count"]
            or type(row.get("application_level_calls")) is not int
        ):
            raise ResumeRejected(f"invalid_batch:batch-{expected:04d}.json")
        start = (expected - 1) * JUDGE_BATCH_SIZE
        end = min(start + row["expected_count"], claims_count)
        if (
            start >= claims_count
            or end - start != row["expected_count"]
            or row.get("claim_sha256") != canonical_hash(claim_lines[start:end])
            or any(
                not isinstance(k, str) or not k.isdigit() or not start <= int(k) < end
                for k in row["verdicts"]
            )
        ):
            raise ResumeRejected(f"batch_out_of_range:batch-{expected:04d}.json")
        verdicts = list(row["verdicts"].values())
        if any(not isinstance(v, dict) or type(v.get("supported")) is not bool for v in verdicts):
            raise ResumeRejected(f"invalid_verdict:batch-{expected:04d}.json")
        yes = sum(v["supported"] for v in verdicts)
        counts = {
            "judged_count": len(verdicts),
            "unjudged_count": 0,
            "yes_count": yes,
            "no_count": len(verdicts) - yes,
        }
        if any(type(row.get(k)) is not int or row[k] != n for k, n in counts.items()):
            raise ResumeRejected(f"invalid_batch_counts:batch-{expected:04d}.json")
        if (
            row.get("scoring_rule_version", "claim-labels/v5") != "claim-labels/v5"
            or row.get("batching_version", "fixed-four/v1") != "fixed-four/v1"
        ):
            raise ResumeRejected("batch_rule_version_changed")
        if "batch_id" in row and row["batch_id"] != self.batch_identity(draft, expected):
            raise ResumeRejected("batch_identity_changed")

    def batch_lease(self, batch_no: int):
        from .batch_ledger import BatchLease

        draft = (
            self.resume.draft
            if self.resume is not None
            else _read_json(self.draft_path(str(self.config["case_id"])))
        )
        return BatchLease(self.ledger_root, self.batch_identity(draft, batch_no))

    def batch_identity(self, draft: dict, batch_no: int) -> str:
        start = (batch_no - 1) * JUDGE_BATCH_SIZE
        identity = {
            "lineage": self.lineage_id,
            "config": self.fingerprint,
            "report": draft["report_sha256"],
            "sources": draft["sources_sha256"],
            "evidence": canonical_hash(draft.get("evidence") or []),
            "claims": canonical_hash(draft["claim_lines"][start : start + JUDGE_BATCH_SIZE]),
            "batch_no": batch_no,
            "scoring_rule_version": "claim-labels/v5",
            "batching_version": "fixed-four/v1",
        }
        return canonical_hash(identity)

    def restore_commit(self, batch_no: int, payload: dict) -> None:
        draft = (
            self.resume.draft
            if self.resume is not None
            else _read_json(self.draft_path(str(self.config["case_id"])))
        )
        self.validate_batch(payload, batch_no, draft)
        if self.lease is None or payload.get("batch_id") != self.lease.batch_id:
            raise ResumeRejected("ledger_batch_identity_changed")
        atomic_create(self.batches_dir / f"batch-{batch_no:04d}.json", payload)

    def request_started(self) -> None:
        if self.lease is not None:
            self.lease.record("request_started")

    def response_received(self) -> None:
        if self.lease is not None:
            self.lease.record("response_received")

    def adopt_resume(self) -> int:
        """Copy validated, already-saved batches into this run so a later resume is self-contained.

        This copies evidence; it never re-requests a batch. Exclusive creation means a
        concurrent writer cannot silently replace what another process measured.
        """
        if self.resume is None or self.source_dir is None:
            return 0
        atomic_create(self.draft_path(str(self.config["case_id"])), self.resume.draft)
        adopted = 0
        for batch_no, row in sorted(self.resume.batches.items()):
            path = self.batches_dir / f"batch-{batch_no:04d}.json"
            if path.exists():
                continue
            atomic_create(path, row)
            adopted += 1
        return adopted


# ── Per-process context (no new `run_one` arguments for existing callers/tests) ──────


@dataclass
class ReliabilityContext:
    root: Path
    deadlines: Deadlines
    budget: Budget
    throttle: Throttle
    stop: StopRequest
    enabled: bool = True
    resume_root: Path | None = None
    rejudge: bool = False
    cases: dict[str, ProgressStore] = field(default_factory=dict)
    last_judge_status: str | None = None
    exit_code: int = 0
    store: ProgressStore | None = None  # the case store currently being judged
    stop_handlers: list[tuple[int, Any]] = field(default_factory=list)
    research_reused: int = 0
    batches_restored: int = 0
    batches_persisted: int = 0
    batches_completed: int = 0
    notes: list[str] = field(default_factory=list)
    remaining_cases: int = 1

    def note(self, message: str) -> None:
        self.notes.append(message)

    def store_for(self, config: dict[str, Any]) -> ProgressStore:
        case_id = str(config["case_id"])
        relative = Path("cases") / _safe_id(case_id)
        key = case_id
        if config.get("phase") == "candidate":
            suffix = "candidate-" + str(config["prompt_overrides_sha256"])
            relative /= suffix
            key += ":" + suffix
        if key not in self.cases:
            source = self.resume_root / relative if self.resume_root else None
            ledger = (
                Path(os.environ["RF_EVAL_BATCH_LEDGER_ROOT"])
                if os.environ.get("RF_EVAL_BATCH_LEDGER_ROOT")
                else (self.resume_root or self.root).parent / ".judge-ledger"
            )
            self.cases[key] = ProgressStore(
                self.root / relative, config=config, source_dir=source, ledger_root=ledger
            )
        return self.cases[key]

    @property
    def waits(self) -> list[WaitRecord]:
        rows = list(self.throttle.records)
        for store in self.cases.values():
            rows.extend(w for w in store.waits if w not in rows)
        return rows

    @property
    def save_failed(self) -> bool:
        return any(store.save_failed for store in self.cases.values())


def build_context(
    *,
    root: Path,
    judge_provider: str,
    resume_root: Path | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], Any] | None = None,
    case_count: int = 1,
) -> ReliabilityContext:
    if resume_root is not None:
        destination, source = root.resolve(), resume_root.resolve()
        if (
            destination == source
            or destination.is_relative_to(source)
            or source.is_relative_to(destination)
        ):
            raise ResumeRejected("resume_requires_a_separate_output_directory")
    deadlines = Deadlines.from_env(case_count=case_count)
    return ReliabilityContext(
        root=root,
        deadlines=deadlines,
        budget=Budget(deadlines=deadlines, clock=clock),
        throttle=Throttle(
            provider=judge_provider,
            min_interval_s=throttle_interval(judge_provider),
            clock=clock,
            sleep=sleep,
        ),
        stop=StopRequest(),
        resume_root=resume_root,
        remaining_cases=case_count,
    )


def preflight_root(root: Path) -> None:
    """Actually write into the progress root before the first model call."""
    if (root / "final-status.json").exists() or any(root.glob("cases/**/drafts/*.json")):
        raise ArtifactConflict("progress root already contains evidence; use a new directory")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    update_status(root / "write-check.json", {"schema_version": SCHEMA_VERSION, "ok": True})


def write_root_status(
    root: Path, *, status: str, reason: str | None, extra: dict[str, Any]
) -> None:
    if status not in STATES:
        raise ReliabilityError(f"unknown reliability status {status!r}")
    update_status(
        root / "final-status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "reason": reason,
            "updated_at": _utcnow(),
            **extra,
        },
    )


def read_root_status(root: Path) -> dict[str, Any] | None:
    """The worker reads this bounded, self-describing record; never a raw traceback."""
    try:
        row = _read_json(root / "final-status.json")
    except ResumeRejected:
        return None
    if row.get("schema_version") != SCHEMA_VERSION or row.get("status") not in STATES:
        return None
    return row


_ACTIVE: ContextVar[ReliabilityContext | None] = ContextVar("rf_long_run", default=None)


def current() -> ReliabilityContext | None:
    return _ACTIVE.get()


_TOKENS: list[Any] = []


def activate(ctx: ReliabilityContext | None) -> None:
    """Set the run context without reindenting `_main`'s existing query loops."""
    _TOKENS.append(_ACTIVE.set(ctx))
    if ctx is not None:
        ctx.stop_handlers = install_stop_handlers(ctx.stop)


def deactivate() -> None:
    while _TOKENS:
        token = _TOKENS.pop()
        ctx = _ACTIVE.get()
        _ACTIVE.reset(token)
        if ctx is not None:
            restore_stop_handlers(ctx.stop_handlers)
            ctx.stop_handlers = []


@contextmanager
def active(ctx: ReliabilityContext | None):
    token = _ACTIVE.set(ctx)
    try:
        yield ctx
    finally:
        _ACTIVE.reset(token)


def reliability_section(
    ctx: ReliabilityContext, *, status: str, reason: str | None
) -> dict[str, Any]:
    """Additive top-level result section; every legacy field stays where it was."""
    waits = ctx.waits
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "judge_config_version": JUDGE_CONFIG_VERSION,
        "deadlines": ctx.deadlines.as_dict(),
        "throttle": {
            "provider": ctx.throttle.provider,
            "min_interval_s": ctx.throttle.min_interval_s,
        },
        "waits": [w.as_dict() for w in waits],
        "total_wait_s": round(sum(w.actual_s for w in waits), 6),
        "batches_persisted": ctx.batches_persisted,
        "batches_completed": ctx.batches_completed,
        "batches_restored": ctx.batches_restored,
        "research_reused": ctx.research_reused,
        "stop_requested": ctx.stop.requested,
        "stop_reason": ctx.stop.reason,
        "save_failed": ctx.save_failed,
        "cases": {
            case_id: {
                "status": store.status,
                "config_fingerprint": store.fingerprint,
                "batches": len(store.resume.batches) if store.resume else 0,
            }
            for case_id, store in sorted(ctx.cases.items())
        },
        "notes": ctx.notes,
    }
