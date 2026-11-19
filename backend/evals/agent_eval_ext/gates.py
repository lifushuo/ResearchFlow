"""Example new gate, independent of unchanged official check_release_criteria."""

from __future__ import annotations


def evaluate_gate(
    audit: dict,
    *,
    min_completion: float = 0.90,
    min_support: float = 0.95,
    min_coverage: float = 0.98,
) -> dict:
    result: dict[str, dict] = {}
    for name, actual, threshold in (
        ("completion", audit["completion_rate_recomputed"], min_completion),
        ("judge_support", audit["judge"]["support_among_judged"], min_support),
        ("judge_coverage", audit["judge"]["coverage"], min_coverage),
    ):
        result[name] = {
            "actual": actual,
            "threshold": threshold,
            "status": "unmeasured"
            if actual is None
            else ("pass" if actual >= threshold else "fail"),
        }
    return {
        "passed": all(x["status"] == "pass" for x in result.values()),
        "checks": result,
        "important": "This is a proposed new reliability gate, not the official release criteria.",
    }
