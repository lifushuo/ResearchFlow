"""Deterministic tool trajectory evaluator modeled on AgentEvals' matching modes."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _freeze_call(call: dict[str, Any], check_args: bool) -> str:
    import json

    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ValueError("every tool call must be {'name': str, 'args': object}")
    payload = {"name": call["name"]}
    if check_args:
        payload["args"] = call.get("args")
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def check_trajectory(
    expected: list[dict[str, Any]],
    observed: list[dict[str, Any]],
    *,
    mode: str = "strict",
    check_args: bool = True,
    forbidden_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Subset=observed is submultiset of expected; superset=expected is submultiset of observed."""
    if mode not in {"strict", "unordered", "subset", "superset"}:
        raise ValueError("mode must be strict/unordered/subset/superset")
    e = [_freeze_call(v, check_args) for v in expected]
    o = [_freeze_call(v, check_args) for v in observed]
    ce, co = Counter(e), Counter(o)
    forbidden = sorted({v["name"] for v in observed if v["name"] in (forbidden_tools or [])})
    match = {
        "strict": lambda: e == o,
        "unordered": lambda: ce == co,
        "subset": lambda: all(co[k] <= ce[k] for k in co),
        "superset": lambda: all(ce[k] <= co[k] for k in ce),
    }[mode]()
    return {
        "passed": match and not forbidden,
        "mode": mode,
        "check_args": check_args,
        "forbidden_called": forbidden,
        "expected_count": len(e),
        "observed_count": len(o),
        "note": "fixture trajectory evaluation; runtime capture must be integrated separately",
    }
