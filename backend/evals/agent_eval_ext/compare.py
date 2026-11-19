"""Paired comparison with explicit confounder disclosure (never a fake causal claim)."""

from __future__ import annotations

from typing import Any

from .audit import report_audit


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_dataset: str | None = None,
    candidate_dataset: str | None = None,
) -> dict[str, Any]:
    b = report_audit(baseline, dataset_path=baseline_dataset)
    c = report_audit(candidate, dataset_path=candidate_dataset)
    sb, sc = b["signature"], c["signature"]
    blockers: list[str] = []
    warnings: list[str] = []
    ids_b = {r["id"] for r in baseline["results"]}
    ids_c = {r["id"] for r in candidate["results"]}
    if ids_b != ids_c:
        blockers.append("different case_id sets; compare only common ids descriptively")
    for field in ("metrics_version", "eval_mode", "llm_mode"):
        if sb[field] != sc[field]:
            blockers.append(f"{field} differs: {sb[field]!r} vs {sc[field]!r}")
    if sb["query_set_sha256"] is None or sc["query_set_sha256"] is None:
        blockers.append("dataset SHA256 was not captured in both run artifacts")
        warnings.append(
            "Dataset content SHA unavailable: equal paths do NOT prove equal dataset content"
        )
    elif sb["query_set_sha256"] != sc["query_set_sha256"]:
        blockers.append("dataset content SHA256 differs")
    if sb["models"] != sc["models"]:
        blockers.append("research model routing differs")
    if sb["judge_route"] != sc["judge_route"]:
        blockers.append(
            "judge model differs; citation support scores are not calibrated on the same ruler"
        )
    if sb["retriever"] != sc["retriever"]:
        blockers.append("retriever setting differs")
    if sb["limits"] != sc["limits"]:
        blockers.append("run limits/concurrency differ")
    for name in (
        "models",
        "judge_route",
        "metrics_version",
        "retriever",
        "limits",
        "llm_mode",
        "eval_mode",
    ):
        if not sb[name] or not sc[name]:
            blockers.append(f"{name} missing; equality of unknown values is not a control")
    if not sb["prompt_snapshot_sha256"] or not sc["prompt_snapshot_sha256"]:
        blockers.append("prompt snapshot SHA256 unavailable in one or both runs")
    elif sb["prompt_snapshot_sha256"] != sc["prompt_snapshot_sha256"]:
        warnings.append(
            "Prompt snapshots differ; treat the prompt change as the intervention, not a code-only comparison"
        )
    warnings.append(
        "Live web evidence may drift over time even if query set and model configs match"
    )
    rb = {r["id"]: r for r in baseline["results"]}
    rc = {r["id"]: r for r in candidate["results"]}
    changes = []
    for case_id in sorted(ids_b & ids_c):
        before, after = rb[case_id], rc[case_id]
        b_ok, c_ok = before.get("completed") is True, after.get("completed") is True
        status = (
            "recovered"
            if not b_ok and c_ok
            else ("regressed" if b_ok and not c_ok else "unchanged")
        )
        changes.append(
            {
                "case_id": case_id,
                "baseline_completed": b_ok,
                "candidate_completed": c_ok,
                "status": status,
                "baseline_error": before.get("error"),
                "candidate_error": after.get("error"),
            }
        )
    br = b["completion_rate_recomputed"]
    cr = c["completion_rate_recomputed"]
    return {
        "is_controlled_comparison": not blockers and sb["query_set_sha256"] is not None,
        "blockers": blockers,
        "warnings": warnings,
        "signatures": {"baseline": sb, "candidate": sc},
        "only_in_baseline": sorted(ids_b - ids_c),
        "only_in_candidate": sorted(ids_c - ids_b),
        "baseline": {"completion_rate": br, "judge": b["judge"]},
        "candidate": {"completion_rate": cr, "judge": c["judge"]},
        "descriptive_completion_delta": (cr - br) if cr is not None and br is not None else None,
        "case_changes": changes,
        "statement": "Descriptive comparison only; do not attribute deltas to a code change without matched controls."
        if blockers or sb["query_set_sha256"] is None
        else "Configuration fingerprints match; random and time-varying effects may still require repeats.",
    }
