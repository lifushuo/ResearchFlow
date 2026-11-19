"""Controlled fault re-execution on the actual compiled graph, never historical replay."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from .artifacts import save_artifact

_KINDS = frozenset({"none", "planner_invalid_schema", "planner_rate_limit", "planner_upstream_522"})
_STATUSES = frozenset({"failed", "awaiting_approval", "awaiting_plan", "completed"})
_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
_REPLAY_LOCK = threading.Lock()


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    query: str
    fault_kind: str
    expected_status: str
    expected_planner_calls: int | None
    error_contains: str | None
    depth: str = "balanced"

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _ID_RE.fullmatch(self.case_id):
            raise ValueError("case_id must be a safe ID with <=64 characters")
        if not isinstance(self.query, str) or not self.query.strip() or len(self.query) > 2000:
            raise ValueError("query must be a nonempty <=2000 character string")
        if self.depth not in ("fast", "balanced", "comprehensive"):
            raise ValueError("unsupported research depth")
        if not isinstance(self.fault_kind, str) or self.fault_kind not in _KINDS:
            raise ValueError("unknown fault kind")
        if not isinstance(self.expected_status, str) or self.expected_status not in _STATUSES:
            raise ValueError("unknown expected terminal status")
        if self.expected_planner_calls is not None and (
            type(self.expected_planner_calls) is not int or self.expected_planner_calls < 0
        ):
            raise ValueError("expected planner_calls must be nonnegative integer or null")
        if self.error_contains is not None and (
            not isinstance(self.error_contains, str)
            or not self.error_contains
            or len(self.error_contains) > 150
        ):
            raise ValueError("error_contains must be a short nonempty string or null")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReplayCase:
        if (
            not isinstance(raw, dict)
            or type(raw.get("schema_version")) is not int
            or raw["schema_version"] != 1
        ):
            raise ValueError("replay case requires schema_version=1")
        if set(raw) - {"schema_version", "case_id", "query", "depth", "fault", "expected"}:
            raise ValueError("unknown replay case fields")
        fault, expected = raw.get("fault"), raw.get("expected")
        if not isinstance(fault, dict) or not isinstance(expected, dict):
            raise ValueError("fault and expected must be objects")
        if set(fault) != {"kind"} or set(expected) - {"status", "planner_calls", "error_contains"}:
            raise ValueError("unknown fault or expected fields")
        return cls(
            raw.get("case_id"),
            raw.get("query"),
            fault.get("kind"),
            expected.get("status"),
            expected.get("planner_calls"),
            expected.get("error_contains"),
            raw.get("depth", "balanced"),
        )

    def as_dict(self) -> dict:
        return {
            "schema_version": 1,
            "case_id": self.case_id,
            "query": self.query,
            "depth": self.depth,
            "fault": {"kind": self.fault_kind},
            "expected": {
                "status": self.expected_status,
                "planner_calls": self.expected_planner_calls,
                "error_contains": self.error_contains,
            },
        }


def load_case(path: str | Path) -> tuple[ReplayCase, str]:
    data = Path(path).read_bytes()
    return ReplayCase.from_dict(json.loads(data.decode("utf-8-sig"))), hashlib.sha256(
        data
    ).hexdigest()


def observed_status(state: dict[str, Any]) -> str:
    interruptions = state.get("__interrupt__") or []
    if interruptions:
        if not isinstance(interruptions, (list, tuple)) or len(interruptions) != 1:
            return "unknown"
        value = getattr(interruptions[0], "value", None)
        if not isinstance(value, dict):
            return "unknown"
        if not isinstance(value.get("type"), str):
            return "unknown"
        return {"PLAN_READY": "awaiting_plan", "HITL_READY": "awaiting_approval"}.get(
            value.get("type"), "unknown"
        )
    if state.get("error"):
        return "failed"
    if isinstance(state.get("final_report"), str) and state["final_report"].strip():
        return "completed"
    return "unknown"


def score_case(
    case: ReplayCase, *, status: str, planner_calls: int | None, error: str | None
) -> dict:
    if planner_calls is not None and (type(planner_calls) is not int or planner_calls < 0):
        raise ValueError("observed planner_calls must be a nonnegative integer or null")
    assertions = {
        "status": status == case.expected_status,
        "planner_calls": None
        if case.expected_planner_calls is None
        else planner_calls == case.expected_planner_calls,
        "error_contains": None
        if case.error_contains is None
        else case.error_contains in (error or ""),
    }
    return {
        "passed": all(value is not False for value in assertions.values()),
        "assertions": assertions,
    }


class _FaultPlanner:
    """Count actual application invocations; the healthy delegate is the engine's fake model."""

    def __init__(self, kind: str, delegate: Any = None) -> None:
        self.kind, self.delegate, self.calls = kind, delegate, 0

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.kind == "none":
            return await self.delegate.ainvoke(messages, **kwargs)
        if self.kind == "planner_invalid_schema":
            from langchain_core.messages import AIMessage

            return AIMessage(content='{"tasks": "invalid-list"}')
        if self.kind == "planner_rate_limit":
            raise RuntimeError("simulated 429 rate limit")
        if self.kind == "planner_upstream_522":
            raise RuntimeError("simulated 522 upstream failure")
        raise AssertionError("unsupported replay fault")


async def run_replay(
    case: ReplayCase,
    *,
    trace_dir: str | Path,
    run_id: str | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run in a dedicated process; reject overlapping replays before changing the factory."""
    if not isinstance(case, ReplayCase):
        raise ValueError("case must be a validated ReplayCase")
    run_id = run_id or f"replay-{uuid.uuid4().hex[:12]}"
    if not isinstance(run_id, str) or not _ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be a safe ID with <=64 characters")
    if not _REPLAY_LOCK.acquire(blocking=False):
        raise RuntimeError("Another replay is active; use sequential cases or separate processes")
    try:
        from langgraph.checkpoint.memory import MemorySaver

        from research_engine import events, llm_factory
        from research_engine import graph as graph_module
        from research_engine.cache import NullCache, reset_cache, set_cache
        from research_engine.corpus import NoCorpus, reset_corpus, set_corpus
        from research_engine.eval_trace import eval_trace_run
        from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
        from research_engine.runner import initial_state

        from .trace_inspect import inspect_trace

        async def discard_event(_identity, _event):
            pass

        # A fake model alone does not isolate inherited database-backed host ports.
        with ExitStack() as scope:
            scope.callback(
                reset_run_config,
                set_run_config(
                    RunConfig(llm_mode="fake", skip_plan_gate=True, max_parallel_tasks=1)
                ),
            )
            scope.callback(llm_factory.reset_user_keys, llm_factory.set_user_keys({}))
            scope.callback(reset_cache, set_cache(NullCache()))
            scope.callback(reset_corpus, set_corpus(NoCorpus()))
            scope.callback(events.reset_emitter, events.set_emitter(discard_event))
            original_get_llm = graph_module.get_llm
            delegate = original_get_llm("planner") if case.fault_kind == "none" else None
            model = _FaultPlanner(case.fault_kind, delegate)

            def get_model(role: str) -> Any:
                return model if role == "planner" else original_get_llm(role)

            scope.enter_context(patch.object(graph_module, "get_llm", side_effect=get_model))
            graph = graph_module.build_graph(MemorySaver())
            config = {"configurable": {"thread_id": run_id}}
            execution_error_type = None
            state = {}
            with eval_trace_run(
                case_id=case.case_id,
                run_id=run_id,
                thread_id=run_id,
                directory=trace_dir,
                enabled=True,
                capture_content=os.environ.get("RF_EVAL_CONTENT_TRACE") == "1",
            ) as trace:
                try:
                    state = await graph.ainvoke(
                        initial_state(
                            session_id=run_id,
                            user_id="replay-fixture-user",
                            query=case.query,
                            depth=case.depth,
                        ),
                        config,
                    )
                    if not isinstance(state, dict):
                        raise TypeError("graph returned non-object state")
                    status = observed_status(state)
                    error = state.get("error")
                    if error is not None and not isinstance(error, str):
                        raise TypeError("graph returned a non-string error")
                except Exception as exc:
                    # An unhandled graph crash is distinct from its expected failed terminal route.
                    status, error, execution_error_type = (
                        "execution_error",
                        str(exc),
                        type(exc).__name__,
                    )
        inspection = None
        if trace is not None and not trace.write_failed:
            try:
                inspection = inspect_trace(trace.path)
            except OSError, ValueError:
                pass
        available = bool(
            inspection
            and inspection["trace_complete"]
            and inspection["case_id"] == case.case_id
            and inspection["run_id"] == run_id
        )
        score = score_case(case, status=status, planner_calls=model.calls, error=error)
        score["assertions"]["trace_available"] = available
        terminal = {
            "failed": "failer",
            "awaiting_approval": "hitl_gate",
            "awaiting_plan": "plan_gate",
            "completed": "finalizer",
        }.get(status)
        score["assertions"]["terminal_trace"] = bool(
            available and terminal and inspection["node_terminals"].get(terminal)
        )
        score["passed"] = all(value is not False for value in score["assertions"].values())

        def shown(path):
            if path is None:
                return None
            if artifact_root is not None:
                try:
                    return str(path.resolve().relative_to(Path(artifact_root).resolve()))
                except ValueError:
                    pass
            return str(path)

        return {
            "schema_version": 1,
            "replay_type": "controlled_simulation",
            "historical_replay": False,
            "run_id": run_id,
            "case_id": case.case_id,
            "fault_kind": case.fault_kind,
            "llm_mode": "fake",
            "status": status,
            "planner_calls": model.calls,
            "error_present": bool(error),
            "execution_error_type": execution_error_type,
            **score,
            "case_spec_sha256": hashlib.sha256(
                json.dumps(case.as_dict(), sort_keys=True).encode()
            ).hexdigest(),
            "trace_path": shown(trace.path) if trace else None,
            "trace_status": "recorded" if available else "partial" if trace else "unavailable",
            "bad_case_path": shown(trace.bad_case_path)
            if trace and trace.bad_case_written
            else None,
            "trace_inspection": inspection,
            "sdk_retry_attempts": None,
            "note": "Fixed synthetic fault, actual compiled graph; not historical provider-response replay. Passing validates expected handling, not research success.",
        }
    finally:
        _REPLAY_LOCK.release()


def write_result_exclusive(path: str | Path, result: dict[str, Any]) -> None:
    save_artifact(result, Path(path))
