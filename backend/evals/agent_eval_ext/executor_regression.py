"""Fixed compiled-graph Executor contracts with scripted models and HTTP transport only."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from .artifacts import check_output_path, save_artifact
from .replay import _REPLAY_LOCK
from .replay_suite import _artifact, _clean_environment
from .trace_inspect import inspect_trace

CASES = (
    "search-read-submit",
    "malformed-url",
    "blocked-url",
    "tool-timeout",
    "duplicate-read",
    "duplicate-submit",
    "cancel",
)
URL = "https://example.org/article"
PAGE = "Controlled observations show that three files were indexed successfully."


async def run_case(case_id: str, root: Path) -> dict:
    if case_id not in CASES:
        raise ValueError("unknown Executor case")
    if not _REPLAY_LOCK.acquire(blocking=False):
        raise RuntimeError("use isolated processes for Executor regression")
    try:
        import httpx
        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.checkpoint.memory import MemorySaver

        from research_engine import events, graph, llm_factory, tools
        from research_engine.cache import NullCache, reset_cache, set_cache
        from research_engine.corpus import NoCorpus, reset_corpus, set_corpus
        from research_engine.eval_trace import eval_trace_run
        from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
        from research_engine.runner import initial_state

        observations, requested, http_calls = [], [], []
        entered, settled = asyncio.Event(), asyncio.Event()

        def call(name, args, number):
            return {"name": name, "args": args, "id": f"call-{number}"}

        def read(url, n):
            return call("read_webpage", {"url": url}, n)

        submit = call(
            "submit_evidence",
            {
                "evidence": [
                    {
                        "source_url": URL,
                        "source_title": "Controlled source",
                        "snippet": PAGE,
                        "key_fact": PAGE,
                    }
                ]
            },
            9,
        )
        turns = {
            "search-read-submit": [
                [call("web_search", {"query": "controlled files"}, 1)],
                [read(URL, 2)],
                [submit],
            ],
            "malformed-url": [[read([URL], 1)], [read(URL, 2)], [submit]],
            "blocked-url": [[read("http://127.0.0.1/private", 1)], [read(URL, 2)], [submit]],
            "tool-timeout": [[read("https://example.org/timeout", 1)], [read(URL, 2)], [submit]],
            "duplicate-read": [[read(URL, 1)], [read(URL, 2)], [submit]],
            "duplicate-submit": [[read(URL, 1)], [submit, {**submit, "id": "duplicate-submit"}]],
            "cancel": [[read("https://example.org/slow", 1)]],
        }[case_id]

        class Planner:
            def with_structured_output(self, schema, **kwargs):
                self.schema = schema
                return self

            async def ainvoke(self, messages, **kwargs):
                body = {
                    "tasks": [{"id": 1, "query": "controlled files", "rationale": "fixed contract"}]
                }
                return {
                    "parsed": self.schema.model_validate(body),
                    "raw": AIMessage(content=json.dumps(body)),
                    "parsing_error": None,
                }

        class Executor:
            index = 0

            def bind_tools(self, definitions, **kwargs):
                return self

            async def ainvoke(self, messages, **kwargs):
                observations[:] = [m.content for m in messages if isinstance(m, ToolMessage)]
                if self.index >= len(turns):
                    raise AssertionError("unexpected extra model invocation")
                batch = turns[self.index]
                self.index += 1
                requested.extend(c["name"] for c in batch)
                return AIMessage(content="", tool_calls=batch)

        async def search(query, max_results):
            assert query == "controlled files"
            return [{"title": "Controlled source", "url": URL, "snippet": ""}]

        async def transport(request):
            http_calls.append(request.url.path)
            if request.url.path == "/timeout":
                raise httpx.ReadTimeout("controlled timeout", request=request)
            if request.url.path == "/slow":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    settled.set()
            assert request.url.path == "/article"
            return httpx.Response(
                200,
                text=f"<html><title>Controlled source</title><p>{PAGE}</p></html>",
                headers={"content-type": "text/html"},
            )

        async def discard(*args, **kwargs):
            pass

        executor, planner = Executor(), Planner()
        run_id = "executor-" + uuid.uuid4().hex[:12]
        client = httpx.AsyncClient
        cfg = RunConfig(llm_mode="real", skip_plan_gate=True, max_parallel_tasks=1)
        with ExitStack() as scope:
            scope.callback(reset_run_config, set_run_config(cfg))
            scope.callback(llm_factory.reset_user_keys, llm_factory.set_user_keys({}))
            scope.callback(reset_cache, set_cache(NullCache()))
            scope.callback(reset_corpus, set_corpus(NoCorpus()))
            scope.callback(events.reset_emitter, events.set_emitter(discard))
            scope.enter_context(
                patch.object(
                    graph,
                    "get_llm",
                    side_effect=lambda role: {"planner": planner, "executor": executor}[role],
                )
            )
            scope.enter_context(patch.object(tools, "search", side_effect=search))
            scope.enter_context(
                patch.object(
                    httpx,
                    "AsyncClient",
                    side_effect=lambda **kw: client(transport=httpx.MockTransport(transport), **kw),
                )
            )
            scope.enter_context(
                patch.object(
                    socket,
                    "getaddrinfo",
                    return_value=[
                        (
                            socket.AF_INET,
                            socket.SOCK_STREAM,
                            socket.IPPROTO_TCP,
                            "",
                            ("93.184.216.34", 443),
                        )
                    ],
                )
            )
            compiled = graph.build_graph(MemorySaver())
            config = {"configurable": {"thread_id": run_id}}
            with eval_trace_run(
                case_id=case_id,
                run_id=run_id,
                thread_id=run_id,
                directory=root / "traces",
                enabled=True,
                capture_content=os.environ.get("RF_EVAL_CONTENT_TRACE") == "1",
            ) as trace:
                task = asyncio.create_task(
                    compiled.ainvoke(
                        initial_state(
                            session_id=run_id,
                            user_id="executor-regression",
                            query="controlled files",
                            depth="fast",
                        ),
                        config,
                        interrupt_after=["executor"],
                    )
                )
                try:
                    if case_id == "cancel":
                        await asyncio.wait_for(entered.wait(), 5)
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            status, state = "cancelled", {}
                        else:
                            raise AssertionError("cancellation was swallowed")
                    else:
                        state = await task
                        snapshot = await compiled.aget_state(config)
                        status = (
                            "executor_checkpoint" if "executor" not in snapshot.next else "unknown"
                        )
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

        inspection = inspect_trace(trace.path)
        raw_events = [json.loads(line) for line in trace.path.read_text().splitlines()]
        submissions = sum(
            e["kind"] == "tool" and e["name"] == "submit_evidence" and e["status"] != "start"
            for e in raw_events
        )
        expected_http = 2 if case_id == "tool-timeout" else 1
        expected_names = [c["name"] for batch in turns for c in batch]
        error_marker = {
            "malformed-url": "tool error",
            "blocked-url": "blocked",
            "tool-timeout": "timeout",
        }.get(case_id)
        assertions = {
            "requested_sequence": requested == expected_names,
            "http_count": len(http_calls) == expected_http,
            "trace_complete": inspection["trace_complete"],
            "executor_observed": bool(inspection["node_terminals"].get("executor")),
            "status": status == ("cancelled" if case_id == "cancel" else "executor_checkpoint"),
            "submission_count": submissions == (0 if case_id == "cancel" else 1),
            "evidence": len(state.get("evidence", [])) == (0 if case_id == "cancel" else 1),
            "verbatim_snippet": case_id == "cancel" or state["evidence"][0]["snippet"] == PAGE,
            "observation_delivered": not error_marker
            or any(error_marker in str(o) for o in observations),
            "cancel_settled": case_id != "cancel" or settled.is_set(),
        }
        return {
            "schema_version": 1,
            "kind": "controlled_executor_case",
            "case_id": case_id,
            "run_id": run_id,
            "status": status,
            "passed": all(assertions.values()),
            "assertions": assertions,
            "model_kind": "scripted",
            "engine_llm_mode": "real",
            "http_transport": "httpx.MockTransport",
            "paid_api_calls": 0,
            "live_agent_accuracy_measured": False,
            "http_calls": len(http_calls),
            "model_requested_tool_names": requested,
            "tool_submission_count": submissions,
            "trace_path": trace.path.relative_to(root).as_posix(),
            "trace_inspection": inspection,
            "bad_case_path": trace.bad_case_path.relative_to(root).as_posix()
            if trace.bad_case_written
            else None,
            "note": "Real compiled Planner/Executor and tool validation, HTTP parsing, SSRF and quotation branches; scripted providers, mock DNS/HTTP, no real 10s wait or SDK retry measurement. Stops before Critic.",
        }
    finally:
        _REPLAY_LOCK.release()


def verify_case(path: Path, case_id: str) -> dict:
    detail = json.loads(path.read_text(encoding="utf-8"))
    if detail.get("case_id") != case_id or detail.get("kind") != "controlled_executor_case":
        raise ValueError("Executor result identity mismatch")
    assertions = detail.get("assertions")
    expected_assertions = {
        "requested_sequence",
        "http_count",
        "trace_complete",
        "executor_observed",
        "status",
        "submission_count",
        "evidence",
        "verbatim_snippet",
        "observation_delivered",
        "cancel_settled",
    }
    if (
        not isinstance(assertions, dict)
        or set(assertions) != expected_assertions
        or any(type(v) is not bool for v in assertions.values())
    ):
        raise ValueError("invalid Executor assertions")
    if type(detail.get("passed")) is not bool or detail["passed"] is not all(assertions.values()):
        raise ValueError("Executor result contradicts assertions")
    trace = _artifact(path.parent, detail.get("trace_path"))
    inspected = inspect_trace(trace)
    if (
        not inspected["trace_complete"]
        or inspected != detail.get("trace_inspection")
        or inspected["case_id"] != case_id
        or inspected["run_id"] != detail.get("run_id")
    ):
        raise ValueError("Executor trace mismatch")
    bad = detail.get("bad_case_path")
    if bad:
        record = json.loads(_artifact(path.parent, bad).read_text(encoding="utf-8"))
        if any(record.get(k) != inspected[k] for k in ("case_id", "run_id", "trace_id")):
            raise ValueError("Executor Bad Case mismatch")
    return detail


def verify_suite(path: Path, suite: dict) -> list[dict]:
    rows = suite.get("cases")
    if (
        suite.get("kind") != "controlled_executor_suite"
        or type(suite.get("schema_version")) is not int
        or suite["schema_version"] != 1
        or suite.get("live_agent_accuracy_measured") is not False
        or not isinstance(rows, list)
        or any(not isinstance(r, dict) for r in rows)
        or len(rows) != len(CASES)
        or {r.get("case_id") for r in rows} != set(CASES)
    ):
        raise ValueError("invalid Executor suite")
    details = []
    for row in rows:
        if type(row.get("process_exit_code")) is not int:
            raise ValueError("unmeasured Executor process exit code")
        result = _artifact(path.parent, row.get("result_path"))
        if hashlib.sha256(result.read_bytes()).hexdigest() != row.get("result_sha256"):
            raise ValueError("Executor result bytes changed")
        detail = verify_case(result, row["case_id"])
        if type(row.get("passed")) is not bool or row["passed"] is not (
            row.get("process_exit_code") == 0
            and detail["passed"] is True
            and row.get("failure_type") is None
        ):
            raise ValueError("inconsistent Executor suite verdict")
        details.append(detail)
    if (
        type(suite.get("total")) is not int
        or suite["total"] != len(rows)
        or type(suite.get("passed_count")) is not int
        or suite.get("passed_count") != sum(r["passed"] for r in rows)
        or suite.get("all_passed") is not all(r["passed"] for r in rows)
    ):
        raise ValueError("inconsistent Executor totals")
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.out_dir)
    try:
        root.mkdir(parents=True, exist_ok=False)
        check_output_path(root / "result.json" if args.case else root / "suite.json")
        if args.case:
            result = asyncio.run(run_case(args.case, root))
            save_artifact(result, root / "result.json")
            return 0 if result["passed"] else 1
        rows = []
        for name in CASES:
            path = root / name / "result.json"
            code, detail, failure = None, None, None
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        __package__ + ".executor_regression",
                        "--case",
                        name,
                        "--out-dir",
                        str(root / name),
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    env=_clean_environment(),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    shell=False,
                )
                code = process.returncode
                detail = verify_case(path, name)
            except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
                failure = type(exc).__name__
            rows.append(
                {
                    "case_id": name,
                    "process_exit_code": code,
                    "failure_type": failure,
                    "passed": code == 0
                    and failure is None
                    and detail is not None
                    and detail["passed"] is True,
                    "result_path": f"{name}/result.json" if detail else None,
                    "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                    if detail
                    else None,
                }
            )
        result = {
            "schema_version": 1,
            "kind": "controlled_executor_suite",
            "total": len(rows),
            "cases": rows,
            "passed_count": sum(r["passed"] for r in rows),
            "all_passed": all(r["passed"] for r in rows),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "live_agent_accuracy_measured": False,
        }
        save_artifact(result, root / "suite.json")
        print(json.dumps(result))
        return 0 if result["all_passed"] else 1
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Executor regression error: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
