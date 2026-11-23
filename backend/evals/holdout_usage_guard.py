"""Purpose-based access guard for the frozen GEPA v3 Planner holdout.

Why this exists
---------------
`planner-holdout-gepa-v3-20261123-v1` is a set of real ResearchFlow report
benchmark questions. Historically they were used for report-quality evals and
four earlier GEPA experiments pinned them as a report A/B dataset. They are
therefore *Planner-independent* (no GEPA journal ever scored them as Planner
cases) but NOT virgin samples. Reusing them for a fresh report A/B acceptance
would make the final acceptance dual-use and meaningless.

The 0444 file mode and hashes prevent silent editing; they do NOT prevent a task
from reading the holdout for the wrong purpose. This module enforces the purpose
contract in code and fails closed BEFORE any model call.

Rules (frozen with the dataset; changing this module requires a new holdout id):
- Allowed purpose: `planner_final_ab`, only when the candidate is frozen and
  `allow_single_final_ab=True`.
- Denied purposes: `report_ab`, `search`, `candidate_selection`, `prompt_adjustment`,
  `calibration`. Any attempt raises and is recorded with a reason.
- Fingerprints are verified against a sidecar manifest before every decision;
  a missing/tampered manifest fails closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HOLDOUT_ID = "planner-holdout-gepa-v3-20261123-v1"
CONTENT_SHA256 = "2a9d296b1628595a04a6e1c2c3736bedf02ca1cfb918ba973888ee29d95e8b3d"
FILE_SHA256 = "32bdcc45dd4e4ea689bc4fecfdf73c0a0a8dd7c66c2cf695a87804c4c121376e"

ALLOWED_PURPOSES = {"planner_final_ab"}
DENIED_PURPOSES = {"report_ab", "search", "candidate_selection", "prompt_adjustment", "calibration"}

_HERE = Path(__file__).resolve().parents[2]  # repo root
POLICY_PATH = _HERE / "internal" / "gepa-v3-acceptance-20261123" / "holdout-usage-policy.json"
HOLDOUT_PATH = _HERE / "internal" / "gepa-v3-acceptance-20261123" / (
    "holdout-gepa-v3-20261123-v1.json"
)


class HoldoutUsageError(RuntimeError):
    """Raised before execution when a task would misuse the frozen holdout."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class UsageDecision:
    allowed: bool
    holdout_id: str
    purpose: str
    reason: str
    fingerprint_verified: bool


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def verify_fingerprints() -> dict[str, Any]:
    """Verify the frozen dataset and manifest; fail closed on any mismatch."""
    if not HOLDOUT_PATH.exists() or not POLICY_PATH.exists():
        raise HoldoutUsageError("holdout_fingerprint_missing: frozen files not found")
    raw = HOLDOUT_PATH.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if data.get("dataset_id") != HOLDOUT_ID:
        raise HoldoutUsageError("holdout_id_mismatch")
    file_hash = _sha256_bytes(raw)
    if file_hash != FILE_SHA256:
        raise HoldoutUsageError(f"holdout_file_hash_mismatch: {file_hash}")
    content = {"dataset_id": data["dataset_id"], "cases": data["cases"]}
    content_hash = _sha256_bytes(_canonical(content).encode("utf-8"))
    if content_hash != CONTENT_SHA256:
        raise HoldoutUsageError(f"holdout_content_hash_mismatch: {content_hash}")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("content_sha256") != CONTENT_SHA256 or policy.get("file_sha256") != FILE_SHA256:
        raise HoldoutUsageError("holdout_policy_fingerprint_mismatch")
    return {"content_sha256": content_hash, "file_sha256": file_hash, "case_count": len(data["cases"])}


def _holdout_query_set() -> frozenset[str]:
    data = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8-sig"))
    return frozenset(" ".join(c["query"].split()) for c in data["cases"])


def _same_query_set(texts: Iterable[str]) -> bool:
    incoming = frozenset(" ".join(t.split()) for t in texts if isinstance(t, str))
    return len(incoming) == 9 and incoming == _holdout_query_set()


def refers_to_holdout(
    path: str | Path | None = None,
    *,
    dataset_id: str | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
) -> bool:
    """Detect whether a dataset reference is the frozen holdout.

    Catches: explicit id/path/hash, content hash, and (importantly) an OS-path
    copy of the nine questions — file permissions alone cannot stop that.
    """
    if dataset_id == HOLDOUT_ID:
        return True
    if path is not None:
        p = Path(path).resolve()
        if p == HOLDOUT_PATH.resolve():
            return True
        if p.exists() and p.suffix == ".json":
            try:
                d = json.loads(p.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                d = None
            if isinstance(d, dict):
                if d.get("dataset_id") == HOLDOUT_ID:
                    return True
                if d.get("content_sha256") == CONTENT_SHA256:
                    return True
    if rows is not None:
        rows_list = list(rows)
        canonical = _sha256_bytes(_canonical(rows_list).encode("utf-8"))
        if canonical == CONTENT_SHA256:
            return True
    return False


def shares_questions_with_holdout(texts: Iterable[str]) -> bool:
    """Explicit opt-in check: does a text set reuse the holdout's nine questions?

    NOT used by the automatic path guard, because the legacy report A/B set
    historically shares these questions. Call this only when authorizing a
    *new* task that claims to be independent, so dual use cannot be asserted.
    """
    return _same_query_set(texts)


def assert_holdout_usage(
    purpose: str,
    *,
    path: str | Path | None = None,
    dataset_id: str | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
    candidate_frozen: bool = False,
    allow_single_final_ab: bool = False,
    audit_log: str | Path | None = None,
) -> UsageDecision:
    """Fail closed if the frozen holdout would be used outside its sole allowed purpose."""
    is_holdout = refers_to_holdout(path, dataset_id=dataset_id, rows=rows)
    purpose_norm = (purpose or "").strip().lower()

    if not is_holdout:
        decision = UsageDecision(True, HOLDOUT_ID, purpose_norm,
                                 "not_the_frozen_holdout; no holdout restriction applies", False)
        _record(decision, audit_log)
        return decision

    fp = verify_fingerprints()
    if purpose_norm not in ALLOWED_PURPOSES | DENIED_PURPOSES:
        raise HoldoutUsageError(f"unknown_holdout_purpose: {purpose_norm}")
    if purpose_norm in DENIED_PURPOSES:
        decision = UsageDecision(False, HOLDOUT_ID, purpose_norm,
                                 f"DENIED: {purpose_norm} is not an allowed use of {HOLDOUT_ID}; "
                                 "holdout is reserved for one pre-agreed final Planner A/B", True)
        _record(decision, audit_log)
        raise HoldoutUsageError(decision.reason)
    if not (candidate_frozen and allow_single_final_ab):
        decision = UsageDecision(False, HOLDOUT_ID, purpose_norm,
                                 "DENIED: planner_final_ab requires candidate_frozen and "
                                 "allow_single_final_ab; search/candidate selection must not read holdout", True)
        _record(decision, audit_log)
        raise HoldoutUsageError(decision.reason)
    decision = UsageDecision(True, HOLDOUT_ID, purpose_norm,
                             f"ALLOWED: one pre-agreed final Planner A/B on {HOLDOUT_ID} "
                             f"(case_count={fp['case_count']})", True)
    _record(decision, audit_log)
    return decision


def _record(decision: UsageDecision, audit_log: str | Path | None) -> None:
    if audit_log is None:
        return
    p = Path(audit_log)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "holdout_id": decision.holdout_id,
        "purpose": decision.purpose,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "fingerprint_verified": decision.fingerprint_verified,
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
