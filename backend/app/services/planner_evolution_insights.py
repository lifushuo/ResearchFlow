"""Read-only GEPA experiment evidence projection for the owner-facing P2 screen.

Only observed journal events are presented. This is not a replay of GEPA's
internal Pareto selection: parent selection and merge internals are not logged.
Raw prompts, queries, tool output and reflection text never leave this module.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.services.evaluation_jobs import _read

MAX_EVENTS = 600
MAX_PUBLIC_EVENTS = 120
VALID_SPLITS = frozenset({"train", "validation", "holdout"})
EVENT_KINDS = frozenset(
    {
        "started",
        "final_cases_loaded",
        "observation",
        "reflection_returned",
        "candidate_frozen",
        "completed",
        "interrupted",
        "validation_retry_round_0",
        "validation_retry_round_1",
        "unmeasured_live_run",
    }
)


def _valid_sha(value: object) -> str | None:
    if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return value
    return None


def _safe_key(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _score(row: dict) -> float | None:
    try:
        count = row.get("required_count")
        covered = row.get("covered_indices")
        if row.get("error") or type(count) is not int or count < 1 or not isinstance(covered, list):
            return None
        if any(type(i) is not int or i < 1 or i > count for i in covered):
            return None
        if len(set(covered)) != len(covered):
            return None
        return len(covered) / count
    except TypeError, ValueError:
        return None


def project_journal(experiment_dir: Path) -> dict[str, Any]:
    """Make a bounded, prompt-free summary of actual, on-disk journal events."""
    journal = experiment_dir / "journal"
    if experiment_dir.is_symlink() or journal.is_symlink() or not journal.is_dir():
        return _empty()
    paths = sorted(journal.glob("event-*.json"))
    if len(paths) > MAX_EVENTS:
        # Fail closed: do not give a falsely complete search history.
        return {**_empty(), "error": "journal_event_limit_exceeded"}

    events: list[dict] = []
    candidates: dict[str, dict] = {}
    token_totals = {
        "planner_input": None,
        "planner_output": None,
        "judge_input": None,
        "judge_output": None,
        "reflection_input": None,
        "reflection_output": None,
    }
    counts = defaultdict(int)
    previous_seq = 0
    experiment_context: dict[str, Any] = {"dataset_id": None, "model_routes": None}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            return {**_empty(), "error": "invalid_journal_path"}
        try:
            doc = _read(path)
        except OSError, ValueError:
            return {**_empty(), "error": "invalid_journal_event"}
        seq = doc.get("sequence")
        if type(seq) is not int or seq != previous_seq + 1 or path.name != f"event-{seq:05d}.json":
            return {**_empty(), "error": "journal_sequence_inconsistent"}
        previous_seq = seq
        kind = doc.get("kind")
        if kind not in EVENT_KINDS:
            return {**_empty(), "error": "unknown_journal_event"}
        payload = doc.get("payload")
        if not isinstance(payload, dict):
            return {**_empty(), "error": "invalid_journal_payload"}
        counts[kind] += 1
        safe: dict[str, Any] = {
            "sequence": seq,
            "kind": kind,
            "recorded_at": doc.get("recorded_at"),
        }
        if kind == "started":
            dataset = payload.get("dataset")
            if isinstance(dataset, dict) and isinstance(dataset.get("dataset_id"), str):
                experiment_context["dataset_id"] = dataset["dataset_id"][:80]
            raw_routes = payload.get("routes")
            if isinstance(raw_routes, dict):
                # Provider:model identifiers only; never a URL, environment variable, or credential.
                import re

                def clean_route(value: object) -> str | None:
                    if not isinstance(value, str) or len(value) > 128 or "://" in value:
                        return None
                    return (
                        value if re.fullmatch(r"[A-Za-z0-9._-]+:[A-Za-z0-9._/+-]+", value) else None
                    )

                # Current journals nest Planner under pipeline; retain older flat records.
                experiment_context["model_routes"] = {
                    "planner": clean_route(
                        (raw_routes.get("pipeline") or raw_routes).get("planner")
                    ),
                    "judge": clean_route(raw_routes.get("judge")),
                    "reflection": clean_route(raw_routes.get("reflection")),
                }
        elif kind == "final_cases_loaded":
            if type(payload.get("simulated")) is not bool or payload.get("case_count") != 9:
                return {**_empty(), "error": "invalid_final_load_event"}
            safe.update(simulated=payload["simulated"], case_count=9)
        elif kind == "observation":
            row = payload.get("observation")
            if not isinstance(row, dict):
                return {**_empty(), "error": "invalid_observation"}
            sha = _valid_sha(payload.get("prompt_sha256"))
            split = row.get("split")
            if sha is None or split not in VALID_SPLITS:
                return {**_empty(), "error": "invalid_observation_identity"}
            score = _score(row)
            quality = None
            try:
                from evals.agent_eval_ext.planner_evolution.experiment import Observation
                from evals.agent_eval_ext.planner_evolution.metrics_v3 import case_score_v3

                quality = case_score_v3(Observation.model_validate(row))
            except ValueError:
                pass
            safe.update(
                {
                    "prompt_sha256": sha,
                    "split": split,
                    "case_key": _safe_key(row.get("case_id")),
                    "coverage": score,
                    "scores": quality,
                    "error_type": str(row.get("error")).split(":")[1].strip()[:80]
                    if row.get("error") and ":" in str(row.get("error"))
                    else ("observation_unmeasured" if row.get("error") else None),
                    "measured": score is not None,
                    "task_count": len(row.get("task_queries", []))
                    if isinstance(row.get("task_queries"), list)
                    else None,
                }
            )
            group = candidates.setdefault(
                sha,
                {
                    "sha256": sha,
                    "samples": 0,
                    "measured": 0,
                    "errors": 0,
                    "splits": {},
                    "scores": [],
                },
            )
            group["samples"] += 1
            if score is None:
                group["errors"] += 1
            else:
                group["measured"] += 1
                group["scores"].append(score)
            sp = group["splits"].setdefault(split, {"count": 0, "measured": 0, "score_total": 0.0})
            sp.setdefault("quality_scores", []).append(quality)
            sp["count"] += 1
            if score is not None:
                sp["measured"] += 1
                sp["score_total"] += score
            for prefix in ("planner", "judge"):
                amounts = row.get(f"{prefix}_tokens")
                if isinstance(amounts, (list, tuple)) and len(amounts) == 2:
                    for idx, key in enumerate((f"{prefix}_input", f"{prefix}_output")):
                        if type(amounts[idx]) is int and 0 <= amounts[idx] < 1_000_000_000:
                            token_totals[key] = (token_totals[key] or 0) + amounts[idx]
        elif kind == "reflection_returned":
            safe["response_sha256"] = _valid_sha(payload.get("response_sha256"))
        elif kind == "candidate_frozen":
            safe["candidate_sha256"] = _valid_sha(payload.get("sha256"))
            safe["accepted_for_holdout"] = payload.get("accepted_for_holdout") is True
        elif kind == "completed":
            gate = payload.get("holdout_gate")
            safe["holdout_passed"] = gate.get("passed") if isinstance(gate, dict) else None
        elif kind == "interrupted":
            # Exception class only; never expose raw SDK response or traceback.
            safe["error_type"] = str(payload.get("error_type", "unknown"))[:64]
            safe["slot_id"] = str(payload["slot_id"])[:64] if payload.get("slot_id") else None
            safe["role"] = (
                payload.get("role")
                if payload.get("role") in ("planner", "judge", "reflection")
                else None
            )
            safe["cause_type"] = (
                str(payload["cause_type"])[:64] if payload.get("cause_type") else None
            )
        events.append(safe)

    grouped = []
    for group in candidates.values():
        split_stats = {
            k: {
                "scores": (
                    {
                        key: round(
                            sum(v[key] for v in s["quality_scores"]) / len(s["quality_scores"]), 4
                        )
                        for key in (
                            "coverage",
                            "specificity",
                            "plan_quality",
                            "redundancy",
                            "quality",
                        )
                    }
                    if s.get("quality_scores") and all(v is not None for v in s["quality_scores"])
                    else None
                ),
                "count": s["count"],
                "measured": s["measured"],
                "mean_coverage": round(s["score_total"] / s["measured"], 4)
                if s["measured"]
                else None,
            }
            for k, s in group["splits"].items()
        }
        grouped.append(
            {
                "sha256": group["sha256"],
                "samples": group["samples"],
                "measured": group["measured"],
                "errors": group["errors"],
                "splits": split_stats,
            }
        )
    grouped.sort(key=lambda item: (-item["samples"], item["sha256"]))
    return {
        "available": bool(events),
        "error": None,
        "event_count": len(events),
        "events": events[-MAX_PUBLIC_EVENTS:],
        "events_truncated": len(events) > MAX_PUBLIC_EVENTS,
        "candidates": grouped,
        "observed_tokens": token_totals,
        "dataset_id": experiment_context["dataset_id"],
        "model_routes": experiment_context["model_routes"],
        "observation_count": counts["observation"],
        "reflection_count": counts["reflection_returned"],
        "holdout_observed": any(
            e.get("split") == "holdout"
            or (e["kind"] == "final_cases_loaded" and e.get("simulated") is False)
            for e in events
        ),
        "pareto_parent_selection": "not_recorded",
        "reflection_text": "not_recorded",
        "token_scope": "planner_and_judge_only_if_reported;_reflection_usage_not_included",
    }


def _empty() -> dict[str, Any]:
    return {
        "available": False,
        "error": None,
        "event_count": 0,
        "events": [],
        "events_truncated": False,
        "candidates": [],
        "observed_tokens": {},
        "dataset_id": None,
        "model_routes": None,
        "observation_count": 0,
        "reflection_count": 0,
        "holdout_observed": False,
        "pareto_parent_selection": "not_recorded",
        "reflection_text": "not_recorded",
        "token_scope": "planner_and_judge_only_if_reported;_reflection_usage_not_included",
    }
