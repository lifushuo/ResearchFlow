"""Read-only report-eval JSON adapter and transparent score audit.

IMPORTANT: Citation support is measured over JUDGED claims only. Unjudged != NO.
All scores in this module describe existing results, never re-grade claims.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def load_run(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_run(data)
    return data


def validate_run(data: Any) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Not an existing ResearchFlow report evaluation JSON (missing results[])")
    if data.get("mode", "report") != "report":
        raise ValueError("This adapter supports report evaluations only")
    if any(not isinstance(row, dict) for row in data["results"]):
        raise ValueError("results[] must contain objects")
    ids = [r.get("id") for r in data["results"]]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("results[] must have unique nonempty case ids")
    for row in data["results"]:
        if type(row.get("completed")) is not bool:
            raise ValueError(f"{row['id']}: completed must be boolean")
        claims = row.get("claim_verdicts")
        if claims is not None and (
            not isinstance(claims, list) or any(not isinstance(c, dict) for c in claims)
        ):
            raise ValueError(f"{row['id']}: claim_verdicts must be a list of objects")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_signature(data: dict[str, Any], dataset_path: str | Path | None = None) -> dict[str, Any]:
    method = data.get("method") or {}
    recorded_sha = method.get("query_set_sha256")
    if recorded_sha is not None and (
        not isinstance(recorded_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", recorded_sha)
    ):
        raise ValueError("method.query_set_sha256 must be a SHA256 digest")
    supplied_sha = sha256_file(dataset_path) if dataset_path else None
    if recorded_sha and supplied_sha and recorded_sha.lower() != supplied_sha:
        raise ValueError("Supplied dataset does not match the run's recorded SHA256")
    return {
        "llm_mode": data.get("llm_mode"),
        "eval_mode": data.get("mode"),
        "models": data.get("models"),
        "judge_route": (data.get("judge") or {}).get("route"),
        "metrics_version": method.get("metrics_version"),
        "retriever": method.get("retriever"),
        "query_set_path": method.get("query_set"),
        # Hashing today's file does not establish which bytes a historical run read.
        "query_set_sha256": recorded_sha.lower() if recorded_sha else None,
        "supplied_dataset_sha256": supplied_sha,
        "prompt_snapshot_sha256": method.get("prompt_snapshot_sha256"),
        "limits": method.get("limits") or {},
    }


def report_audit(data: dict[str, Any], *, dataset_path: str | Path | None = None) -> dict[str, Any]:
    validate_run(data)
    yes = no = unjudged = 0
    per_case = []
    failures = []
    missing_records = []
    for row in data["results"]:
        if row["completed"] and row.get("claim_verdicts") is None:
            missing_records.append(row["id"])
        cy = cn = cu = 0
        for claim in row.get("claim_verdicts") or []:
            # Only an explicit boolean verdict with judged=true is a measurement.
            if claim.get("judged") is True and claim.get("supported") is True:
                cy += 1
            elif claim.get("judged") is True and claim.get("supported") is False:
                cn += 1
            else:
                cu += 1
        yes += cy
        no += cn
        unjudged += cu
        if row.get("completed") is not True:
            failures.append(
                {
                    "case_id": row["id"],
                    "error": row.get("error"),
                    "category": classify_error(row.get("error")),
                }
            )
        per_case.append(
            {
                "case_id": row["id"],
                "completed": row.get("completed") is True,
                "latency_s": row.get("latency_s"),
                "source_count": row.get("source_count"),
                "uncited_claim_count": row.get("uncited_claim_count"),
                "yes": cy,
                "no": cn,
                "unjudged": cu,
                "judge_coverage": (cy + cn) / (cy + cn + cu) if cy + cn + cu else None,
                "error": row.get("error"),
            }
        )
    total = yes + no + unjudged
    judged = yes + no
    warnings = []
    if "candidate" in data:
        warnings.append(
            "Only the top-level shipped baseline is audited; nested candidate results are not selected."
        )
    if missing_records:
        warnings.append(
            "Completed reports lack claim records; overall judge coverage is unmeasured."
        )
    if judged >= 20 and no == 0:
        warnings.append(
            "Judge returned only YES for all measured claims; calibrate against labeled negative controls."
        )
    if unjudged:
        warnings.append(
            "Unjudged claims were excluded from citation-support denominator; inspect judge batch logs."
        )
    if any((r.get("tokens") or 0) > 0 and r.get("cost_usd") == 0 for r in data["results"]):
        warnings.append(
            "0.0 cost with nonzero token counts may mean unpriced provider, not zero actual spend."
        )
    # No true source corpus is in the report JSON: an evidence-fidelity error cannot be verified here.
    warnings.append(
        "Report JSON lacks cited evidence snippets and full runtime spans; root cause is not provable from this artifact alone."
    )
    return {
        "signature": run_signature(data, dataset_path),
        "aggregate_reported": data.get("aggregate"),
        "case_count": len(data["results"]),
        "completed": sum(r["completed"] for r in per_case),
        "completion_rate_recomputed": sum(r["completed"] for r in per_case) / len(per_case)
        if per_case
        else None,
        "judge": {
            "yes": yes,
            "no": no,
            "unjudged": unjudged,
            "coverage": judged / total if total and not missing_records else None,
            "cases_missing_claim_records": missing_records,
            "support_among_judged": yes / judged if judged else None,
            "support_aggregation": "claim_weighted; official aggregate may average per-report rates",
            "supported_among_all_cited_claims": yes / total if total else None,
            "note": "last ratio counts unjudged in denominator, not official citation_support_rate",
        },
        "per_case": per_case,
        "bad_cases": failures,
        "warnings": warnings,
    }


def classify_error(message: str | None, *, http_status: int | None = None) -> str:
    """Conservative observable symptom labels, never definitive root-cause claims."""
    msg = (message or "").lower()
    if http_status == 429 or "rate limit" in msg or re.search(r"\b429\b", msg):
        return "rate_limit"
    if http_status in (502, 503, 504, 522, 524) or any(
        f"{x}" in msg for x in (" 502", " 503", " 504", " 522", " 524")
    ):
        return "upstream_or_gateway_unavailable"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "permission" in msg or "forbidden" in msg or "unauthorized" in msg:
        return "authorization_error"
    if "schema" in msg or "validation" in msg or "pydantic" in msg or "valid task list" in msg:
        return "structured_output_or_validation"
    if "tool" in msg and any(word in msg for word in ("failed", "error", "exception")):
        return "tool_error"
    if "retriev" in msg or "search" in msg:
        return "retrieval_error"
    return "unknown"
