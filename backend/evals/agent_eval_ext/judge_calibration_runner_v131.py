"""Bounded live Judge calibration using the 24 original v0.1 handwritten cases.

This uses ResearchFlow's original judge_citation_support implementation and the
original v0.1 calibration scorer. It never reuses historical predictions as new scores.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .calibrate_live import run as invoke_real_judge
from .calibration import load_calibration, score_calibration

INPUT = Path(__file__).resolve().parent / "fixtures" / "judge_calibration.jsonl"
EXPECTED_SHA256 = "78a778a8315688568c7845ada2099d1c422f1bce65b1e89448c9174cae526800"


def validate_source() -> list[dict]:
    if INPUT.is_symlink() or hashlib.sha256(INPUT.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("original_judge_dataset_checksum_mismatch")
    rows = load_calibration(INPUT)
    if (
        len(rows) != 24
        or sum(x["split"] == "dev" for x in rows) != 16
        or sum(x["split"] == "holdout" for x in rows) != 8
        or sum(x["expected_supported"] for x in rows) != 12
    ):
        raise ValueError("original_judge_dataset_schema_mismatch")
    return rows


async def execute(*, limit: int, judge_route: str) -> dict:
    # Deferred: this module is imported by evaluation_pipeline_v13._preflight.
    from app.services.evaluation_pipeline_v13 import online_lane_enabled

    if os.getenv("LLM_MODE") != "real" or not online_lane_enabled():
        raise RuntimeError("live_judge_disabled")
    if not judge_route or ":" not in judge_route:
        raise ValueError("pinned_independent_judge_required")
    original = validate_source()
    if type(limit) is not int or not 0 <= limit <= 24:
        raise ValueError("invalid_calibration_limit")
    chosen = original[:limit] if limit else original
    observed = await invoke_real_judge(INPUT, judge_route, len(chosen), None)
    if [x["case_id"] for x in observed] != [x["case_id"] for x in chosen]:
        raise ValueError("calibration_case_identity_mismatch")
    metrics = score_calibration(observed)
    return {
        "status": "measured" if metrics["judged"] > 0 else "unmeasured",
        "dataset_id": "judge-calibration",
        "dataset_sha256": EXPECTED_SHA256,
        "evidence_kind": "real_independent_llm_judge_on_handwritten_controls",
        "real_model_executed": True,
        "split_counts": {s: sum(x["split"] == s for x in observed) for s in ("dev", "holdout")},
        "metrics": metrics,
        "cases": observed,  # PRIVATE worker-owned volume only; API never returns claim/evidence
    }
