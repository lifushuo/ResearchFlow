"""Read-only console summary for v0.2 graph/runtime traces (no model calls)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def inspect_trace(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    events = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            data = json.loads(line)
            if not isinstance(data, dict) or "status" not in data:
                raise ValueError(f"Malformed trace event at line {lineno}")
            events.append(data)
    if events:
        identity = (events[0].get("trace_id"), events[0].get("run_id"), events[0].get("case_id"))
        if not all(identity):
            raise ValueError("Trace identity is missing")
        for seq, event in enumerate(events, 1):
            if event.get("event_version") != 2 or event.get("seq") != seq:
                raise ValueError("Unsupported trace version or broken event sequence")
            if (event.get("trace_id"), event.get("run_id"), event.get("case_id")) != identity:
                raise ValueError("Mixed trace/case identities")
            if not isinstance(event.get("metadata"), dict):
                raise ValueError("Trace metadata must be an object")
    terminal = [ev for ev in events if ev.get("status") != "start"]
    bad = [ev for ev in terminal if ev.get("status") in ("error", "business_error", "cancelled")]
    node_names = [ev.get("name") for ev in terminal if ev.get("kind") == "graph_node"]
    model_names = [ev.get("name") for ev in terminal if ev.get("kind") == "llm"]
    tool_names = [ev.get("name") for ev in terminal if ev.get("kind") == "tool"]
    token_in = [
        ev.get("metadata", {}).get("input_tokens") for ev in terminal if ev.get("kind") == "llm"
    ]
    token_out = [
        ev.get("metadata", {}).get("output_tokens") for ev in terminal if ev.get("kind") == "llm"
    ]
    valid_in = [n for n in token_in if type(n) is int and n >= 0]
    valid_out = [n for n in token_out if type(n) is int and n >= 0]
    starts = [ev.get("span_id") for ev in events if ev.get("status") == "start"]
    ends = [ev.get("span_id") for ev in terminal]
    return {
        "trace_file": path.name,
        "event_count": len(events),
        "case_id": events[0].get("case_id") if events else None,
        "run_id": events[0].get("run_id") if events else None,
        "trace_id": events[0].get("trace_id") if events else None,
        "thread_id": events[0].get("thread_id") if events else None,
        "trace_complete": bool(events) and Counter(starts) == Counter(ends),
        "node_terminals": dict(Counter(node_names)),
        "node_successes": dict(
            Counter(
                ev.get("name")
                for ev in terminal
                if ev.get("kind") == "graph_node" and ev.get("status") == "ok"
            )
        ),
        "llm_calls": len(model_names),
        "tool_calls": len(tool_names),
        "tool_call_names": dict(Counter(tool_names)),
        "llm_input_tokens_observed": sum(valid_in) if valid_in else None,
        "llm_output_tokens_observed": sum(valid_out) if valid_out else None,
        "llm_calls_without_input_usage": len(token_in) - len(valid_in),
        "llm_calls_without_output_usage": len(token_out) - len(valid_out),
        "interrupts": sum(ev.get("status") == "interrupt" for ev in terminal),
        "failure_symptoms": [
            {
                "name": ev.get("name"),
                "kind": ev.get("kind"),
                "status": ev.get("status"),
                "error_type": ev.get("error_type"),
            }
            for ev in bad
        ],
        "note": "Observed tokens exclude calls without provider usage; judge runs outside this trace.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect a v0.2 ResearchFlow trace (read-only)")
    p.add_argument("path", type=Path, help="Path to one .jsonl trace")
    args = p.parse_args()
    print(json.dumps(inspect_trace(args.path), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
