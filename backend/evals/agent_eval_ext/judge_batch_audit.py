"""Infer judge batch coverage from saved verdict order, without observing requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import save_artifact
from .audit import load_run, sha256_file, validate_run


def analyze_report(payload: dict[str, Any], *, batch_size: int = 4) -> dict[str, Any]:
    """Group saved claims under an explicit batch-size assumption (docs/08 §5)."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    validate_run(payload)
    rows = []
    total = judged = total_batches = complete_batches = 0
    missing_records = []
    warnings = []
    for report in payload["results"]:
        records = report.get("claim_verdicts")
        if records is None and report["completed"]:
            missing_records.append(report["id"])
        claims = records or []
        # A flag alone is insufficient: keep the same measurement contract as report_audit.
        measured = [v.get("judged") is True and type(v.get("supported")) is bool for v in claims]
        invalid = [
            i
            for i, v in enumerate(claims, 1)
            if v.get("judged") is True and type(v.get("supported")) is not bool
        ]
        batches = []
        for start in range(0, len(claims), batch_size):
            chunk = measured[start : start + batch_size]
            count = sum(chunk)
            status = "complete" if count == len(chunk) else "unjudged" if not count else "partial"
            total_batches += 1
            complete_batches += status == "complete"
            batches.append(
                {
                    "batch_number": start // batch_size + 1,
                    "status": status,
                    "is_full_batch": len(chunk) == batch_size,
                    "claim_positions": list(range(start + 1, start + len(chunk) + 1)),
                    "unjudged_positions": [start + i for i, ok in enumerate(chunk, 1) if not ok],
                    "judged_count": count,
                    "unjudged_count": len(chunk) - count,
                    "root_cause": None if status == "complete" else "unknown",
                }
            )
        total += len(claims)
        judged += sum(measured)
        if invalid:
            warnings.append(
                f"{report['id']}: judged flags without boolean verdicts are unmeasured."
            )
        record_status = "missing" if records is None else "present" if claims else "empty"
        rows.append(
            {
                "case_id": report["id"],
                "completed": report["completed"],
                "pipeline_error": report.get("error"),
                "claim_records_status": record_status,
                "total_claims": len(claims),
                "unjudged_claims": len(claims) - sum(measured),
                "invalid_verdict_positions": invalid,
                "problem_batches": [b for b in batches if b["status"] != "complete"],
            }
        )
    problem_batches = [b for row in rows for b in row["problem_batches"]]
    if missing_records:
        warnings.append(
            "Completed reports lack claim records; overall judge coverage is unmeasured."
        )
    if "candidate" in payload:
        warnings.append(
            "Only the top-level baseline is audited; nested candidate results are not selected."
        )
    return {
        "batch_size": batch_size,
        "batch_assignment": "inferred_from_saved_verdict_order",
        "actual_requests_observed": False,
        "total_claims": total,
        "judged_claims": judged,
        "unjudged_claims": total - judged,
        "judge_coverage": judged / total if total and not missing_records else None,
        "cases_missing_claim_records": missing_records,
        "total_batches": total_batches,
        "complete_batches": complete_batches,
        "incomplete_batches": len(problem_batches),
        "full_unjudged_batches": sum(b["status"] == "unjudged" for b in problem_batches),
        "partial_batches": sum(b["status"] == "partial" for b in problem_batches),
        "reports": rows,
        "warnings": warnings,
        "interpretation_limit": (
            "Batch membership assumes saved verdict order and the supplied batch size match the "
            "historical harness. These are inferred groups, not observed API requests. Historical "
            "JSON lacks raw batch replies, status codes, parse errors and retry history; API/parser "
            "root causes and actual HTTP attempt counts cannot be established."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="New analysis file; refuses overwrite")
    parser.add_argument("--batch-size", type=int, default=4, help="Assumed historical batch size")
    args = parser.parse_args()
    result = analyze_report(load_run(args.input), batch_size=args.batch_size)
    result["input_sha256"] = sha256_file(args.input)
    if args.out:
        save_artifact(result, args.out)
        print(f"Created new artifact: {args.out}")
    else:
        print(json.dumps({"extension_schema_version": 1, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
