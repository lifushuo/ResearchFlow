"""Manual metadata-only spans; not instrumented into LangGraph or provider calls."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from .audit import classify_error

logger = logging.getLogger(__name__)


class TraceWriter:
    def __init__(self, path: str | Path, *, run_id: str, case_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.case_id = case_id
        self.trace_id = uuid.uuid4().hex
        self.seq = 0
        self._lock = Lock()
        self._parent: ContextVar[str | None] = ContextVar("agent_eval_span", default=None)
        self._enabled = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A recorder owns one new file; opening an old run as a trace is an error.
            with self.path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            raise
        except OSError as exc:
            self._disable(exc)

    def _disable(self, exc: OSError) -> None:
        self._enabled = False
        logger.warning("agent_eval_trace_write_failed error_type=%s", type(exc).__name__)

    def write(
        self,
        *,
        node: str,
        kind: str,
        status: str,
        duration_ms: float | None = None,
        error_type: str | None = None,
        http_status: int | None = None,
        span_id: str | None = None,
        parent_id: str | None = None,
    ) -> dict | None:
        with self._lock:
            if not self._enabled:
                return None
            self.seq += 1
            event = {
                "event_version": 1,
                "trace_id": self.trace_id,
                "run_id": self.run_id,
                "case_id": self.case_id,
                "seq": self.seq,
                "span_id": span_id,
                "parent_id": parent_id,
                "utc": datetime.now(UTC).isoformat(),
                "node": node,
                "kind": kind,
                "status": status,
                "duration_ms": duration_ms,
                "error_type": error_type,
                "error_class": classify_error(error_type, http_status=http_status)
                if error_type
                else None,
                "http_status": http_status,
            }
            try:
                with self.path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(event, ensure_ascii=False) + "\n")
            except OSError as exc:
                # Observability must not turn successful work into an application failure.
                self._disable(exc)
                return None
            return event

    @contextmanager
    def span(self, node: str, *, kind: str = "graph_node") -> Iterator[None]:
        span_id = uuid.uuid4().hex
        parent_id = self._parent.get()
        token = self._parent.set(span_id)
        self.write(node=node, kind=kind, status="start", span_id=span_id, parent_id=parent_id)
        started = time.perf_counter()
        status, error_type = "ok", None
        try:
            yield
        except BaseException as exc:
            error_type = type(exc).__name__
            status = "error" if isinstance(exc, Exception) else "cancelled"
            raise
        finally:
            self.write(
                node=node,
                kind=kind,
                status=status,
                span_id=span_id,
                parent_id=parent_id,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                error_type=error_type,
            )
            self._parent.reset(token)
