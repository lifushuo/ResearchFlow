"""Validate and summarize recorded judge calls, without loading the judge or its keys."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from research_engine.response_diagnostics import safe_response_diagnostic

from .audit import sha256_file


def summarize(path: Path | str, *, total_claims: int | None = None, batch_size: int = 4) -> dict:
    path = Path(path)
    events = []
    identity = None
    batches = []
    ended = False
    active_batch = None
    active_wait = None
    explicit_batches = False
    restored_batches = 0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if isinstance(event, dict) and "response_diagnostic" in event:
            event["response_diagnostic"] = safe_response_diagnostic(event["response_diagnostic"])
        if (
            not isinstance(event, dict)
            or type(event.get("schema_version")) is not int
            or event["schema_version"] != 1
        ):
            raise ValueError(f"line {lineno}: unsupported judge trace schema")
        current = tuple(event.get(key) for key in ("trace_id", "run_id", "case_id", "phase"))
        if any(not isinstance(value, str) or not value for value in current):
            raise ValueError(f"line {lineno}: missing judge trace identity")
        if identity is not None and current != identity:
            raise ValueError(f"line {lineno}: mixed judge trace identities")
        identity = current
        if type(event.get("seq")) is not int or event["seq"] != len(events) + 1:
            raise ValueError(f"line {lineno}: nonsequential judge trace")
        kind = event.get("event")
        if kind == "case_start" and event.get("total_claims") is not None:
            planned = event["total_claims"]
            if type(planned) is not int or planned < 0:
                raise ValueError("invalid full claim count")
            if total_claims is not None and total_claims != planned:
                raise ValueError("full input does not match trace plan")
            total_claims = planned
            batch_size = event.get("batch_size", batch_size)
        if ended or (not events and kind != "case_start") or (events and kind == "case_start"):
            raise ValueError(f"line {lineno}: invalid case event order")
        if kind == "batch_started":
            if (
                active_batch
                or active_wait
                or any(
                    type(event.get(k)) is not int
                    for k in ["batch_no", "batch_start_zero_based", "expected_count"]
                )
                or event.get("batch_no") != len(batches) + 1
                or event.get("batch_start_zero_based") != sum(b["expected_count"] for b in batches)
                or event["expected_count"] < 1
            ):
                raise ValueError(f"line {lineno}: invalid batch start")
            active_batch = event
            explicit_batches = True
        elif kind == "wait_started":
            value = event.get("wait_seconds")
            if (
                active_batch
                or active_wait
                or not batches
                or event.get("batch_no") != batches[-1]["batch_no"]
                or type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"line {lineno}: invalid wait start")
            active_wait = event
        elif kind == "wait_finished":
            value = event.get("elapsed_ms")
            if (
                not active_wait
                or event.get("batch_no") != active_wait["batch_no"]
                or event.get("status") not in {"ok", "cancelled", "error"}
                or type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"line {lineno}: invalid wait finish")
            active_wait = None
        elif kind in ("batch_finished", "batch_restored"):
            # A resumed run restores saved batches at their original positions, including
            # holes after an earlier failed batch that needs a new provider call.
            # but must never be counted as an application-level call in this process.
            restored = kind == "batch_restored"
            if restored and (
                active_wait
                or active_batch
                or not isinstance(event.get("restored_from"), str)
                or not event["restored_from"]
            ):
                raise ValueError(f"line {lineno}: invalid restored batch")
            if active_wait or (
                not restored
                and explicit_batches
                and (
                    not active_batch
                    or any(
                        event.get(k) != active_batch.get(k)
                        for k in ["batch_no", "batch_start_zero_based", "expected_count"]
                    )
                )
            ):
                raise ValueError(f"line {lineno}: mismatched batch finish")
            active_batch = None
            counts = ("expected_count", "judged_count", "unjudged_count", "yes_count", "no_count")
            if any(type(event.get(key)) is not int or event[key] < 0 for key in counts):
                raise ValueError(f"line {lineno}: invalid claim counts")
            if not (
                event["expected_count"] > 0
                and event["judged_count"] + event["unjudged_count"] == event["expected_count"]
                and event["yes_count"] + event["no_count"] == event["judged_count"]
            ):
                raise ValueError(f"line {lineno}: inconsistent claim counts")
            if event.get("batch_no") != len(batches) + 1 or event.get(
                "batch_start_zero_based"
            ) != sum(b["expected_count"] for b in batches):
                raise ValueError(f"line {lineno}: invalid batch position")
            expected_status = (
                "complete"
                if not event["unjudged_count"]
                else "unjudged"
                if not event["judged_count"]
                else "partial"
            )
            if event.get("status") not in (expected_status, "cancelled"):
                raise ValueError(f"line {lineno}: inconsistent batch status")
            if (
                event.get("application_level_calls") != (0 if restored else 1)
                or event.get("sdk_retry_attempts") is not None
            ):
                raise ValueError(f"line {lineno}: unsupported attempt accounting")
            batches.append(event)
            if restored:
                restored_batches += 1
        elif kind == "case_end":
            if active_batch or active_wait:
                raise ValueError(f"line {lineno}: case ended with unfinished phase")
            if type(event.get("trace_available")) is not bool:
                raise ValueError(f"line {lineno}: invalid trace availability")
            ended = True
        elif kind != "case_start":
            raise ValueError(f"line {lineno}: unknown judge trace event")
        events.append(event)
    complete = bool(events) and ended and events[-1]["trace_available"]
    expected = sum(e["expected_count"] for e in batches) if events else None
    judged = sum(e["judged_count"] for e in batches) if events else None
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("invalid batch size")
    if total_claims is not None and (
        type(total_claims) is not int or total_claims < (expected or 0)
    ):
        raise ValueError("observed claims exceed full input")
    total_batches = (
        (total_claims + batch_size - 1) // batch_size if total_claims is not None else None
    )
    successful = sum(e["status"] == "complete" for e in batches)
    starts = [e for e in events if e["event"] == "batch_started"]
    requested = len(starts) if starts else sum(e["application_level_calls"] for e in batches)
    responses = [e for e in batches if e["event"] != "batch_restored"]
    received = (
        sum(e["response_received"] is True for e in responses)
        if all(type(e.get("response_received")) is bool for e in responses)
        else None
    )
    measurements = [
        {
            k: e.get(k)
            for k in (
                "batch_no",
                "status",
                "elapsed_ms",
                "tokens_input",
                "tokens_output",
                "cost_usd",
                "http_status",
                "exception_type",
                "response_diagnostic",
                "served_model",
            )
        }
        for e in responses
    ]
    return {
        "trace_file": str(path),
        "input_sha256": sha256_file(path),
        "trace_id": identity[0] if identity else None,
        "run_id": identity[1] if identity else None,
        "case_id": identity[2] if identity else None,
        "phase": identity[3] if identity else None,
        "trace_complete": complete,
        "judge_route": events[0].get("judge_route") if events else None,
        "llm_mode": events[0].get("llm_mode") if events else None,
        "case_status": events[-1].get("status") if ended else None,
        "case_exception_type": events[-1].get("exception_type") if ended else None,
        "observed_batches": len(batches),
        "total_claims": total_claims,
        "total_batches": total_batches,
        "successful_batches": successful,
        "attempted_batches": len(batches) + bool(active_batch),
        "new_requested_batches": requested,
        "responses_received": received,
        "successful_requests": received,
        "valid_result_batches": successful,
        "judged_claims_total": judged,
        "unjudged_claims_total": total_claims - judged
        if total_claims is not None and judged is not None
        else None,
        "full_judge_coverage": judged / total_claims
        if total_claims and judged is not None
        else None,
        "attempted_judge_coverage": judged / expected if expected else None,
        "scoring_complete": bool(complete and total_claims and judged == total_claims),
        "batch_measurements": measurements,
        "known_input_tokens": sum(
            e["tokens_input"] for e in responses if type(e.get("tokens_input")) is int
        ),
        "known_output_tokens": sum(
            e["tokens_output"] for e in responses if type(e.get("tokens_output")) is int
        ),
        "batches_missing_usage": sum(
            e.get("tokens_input") is None or e.get("tokens_output") is None for e in responses
        ),
        "wait_observations": [
            {k: e.get(k) for k in ("batch_no", "wait_reason", "elapsed_ms")}
            for e in events
            if e["event"] == "wait_finished"
        ],
        "restored_batches": restored_batches,
        "application_level_calls": sum(e["application_level_calls"] for e in batches)
        if complete
        else None,
        "observed_application_level_calls": sum(e["application_level_calls"] for e in batches)
        if events
        else None,
        "sdk_retry_attempts": None,
        "served_models": sorted({e["served_model"] for e in batches if e.get("served_model")}),
        "disclosed_aliases": sorted(
            {e["disclosed_alias"] for e in batches if e.get("disclosed_alias")}
        ),
        "expected_claims": expected,
        "judged_claims": judged,
        "unjudged_claims": expected - judged if events else None,
        "observed_judge_coverage": judged / expected if expected else None,
        "judge_coverage": judged / total_claims if complete and total_claims else None,
        "problem_batches": [
            {
                key: e.get(key)
                for key in (
                    "batch_no",
                    "status",
                    "reason_hint",
                    "exception_type",
                    "http_status",
                    "unjudged_count",
                    "out_of_range_labels",
                    "duplicate_labels",
                    "preexisting_judged_count",
                    "response_diagnostic",
                )
            }
            for e in batches
            if e["status"] != "complete"
            or e.get("reason_hint")
            or any(
                e.get(key)
                for key in ("out_of_range_labels", "duplicate_labels", "preexisting_judged_count")
            )
        ],
        "scope": "Finished application-level calls only; SDK HTTP retries and killed in-flight calls are unobserved",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
