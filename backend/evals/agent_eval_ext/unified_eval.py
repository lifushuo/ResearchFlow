"""Non-destructive, versioned ResearchFlow Evaluation Result adapters (v0.7).

Read existing report/retrieval/grounding/replay-suite artifacts. Never re-score
original Judge verdicts and never turn missing measurements into zero or PASS.
This module uses the Python standard library and imports replay_suite only when
validating an existing replay dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import save_artifact

SCHEMA_VERSION = 1
DIMENSIONS = ("outcome", "reliability", "retrieval", "grounding")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def fingerprint(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class _SourceJSON(dict):
    def __init__(self, obj: dict, content: bytes):
        super().__init__(obj)
        self.byte_sha256 = hashlib.sha256(content).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    content = Path(path).read_bytes()
    obj = json.loads(content.decode("utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return _SourceJSON(obj, content)


def save_new(path: str | Path, obj: dict[str, Any]) -> None:
    """Create-only. Never overwrite an old evaluation artifact."""
    save_artifact(obj, Path(path))


def _number(value: Any, *, fraction: bool = False) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        if not math.isfinite(value):
            return None
    except OverflowError:
        return None
    if fraction and not 0 <= value <= 1:
        return None
    return float(value)


def metric(
    value: Any, *, unit: str = "ratio", note: str = "", valid: bool = True, count: int | None = None
) -> dict[str, Any]:
    if type(valid) is not bool or (count is not None and (type(count) is not int or count < 0)):
        raise ValueError("valid must be boolean and metric count a nonnegative integer or null")
    number = _number(value, fraction=unit == "ratio")
    status = "invalid" if not valid else "unmeasured" if number is None else "measured"
    return {"status": status, "value": number, "unit": unit, "n": count, "note": note}


def _missing(reason: str) -> dict[str, Any]:
    return {"status": "unmeasured", "reason": reason, "metrics": {}, "cases": []}


def _source(path: str | Path, kind: str, raw: dict | None = None) -> dict[str, Any]:
    # Basename + SHA: avoid leaking a local workstation's absolute directories.
    digest = raw.byte_sha256 if isinstance(raw, _SourceJSON) else fingerprint(path)
    return {"kind": kind, "basename": Path(path).name, "sha256": digest}


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Resolve and verify a dataset wrapper + all v0.6 Replay Case bytes."""
    src = Path(path).resolve()
    wrapper_bytes = src.read_bytes()
    raw = json.loads(wrapper_bytes.decode("utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("dataset must be an object")
    if set(raw) != {"schema_version", "dataset_id", "dataset_version", "split", "replay_manifest"}:
        raise ValueError("dataset manifest requires exact v1 fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported dataset schema_version")
    if not isinstance(raw["dataset_id"], str) or not ID_PATTERN.fullmatch(raw["dataset_id"]):
        raise ValueError("invalid dataset_id")
    if type(raw["dataset_version"]) is not int or raw["dataset_version"] < 1:
        raise ValueError("invalid dataset_version")
    if raw["split"] not in ("regression", "dev", "holdout"):
        raise ValueError("split must be regression/dev/holdout")
    manifest_name = raw["replay_manifest"]
    if not isinstance(manifest_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", manifest_name
    ):
        raise ValueError("replay_manifest must be a local JSON filename")
    manifest = (src.parent / manifest_name).resolve()
    if manifest.parent != src.parent:
        raise ValueError("replay manifest resolves outside dataset directory")
    from .replay_suite import _manifest_snapshot

    manifest_bytes, cases = _manifest_snapshot(manifest)
    return {
        "schema_version": 1,
        "dataset_id": raw["dataset_id"],
        "dataset_version": raw["dataset_version"],
        "split": raw["split"],
        "manifest_path": manifest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "wrapper_sha256": hashlib.sha256(wrapper_bytes).hexdigest(),
        "cases": [
            {"case_id": case.case_id, "case_sha256": digest} for case, _, _, digest, _ in cases
        ],
        "_wrapper_bytes": wrapper_bytes,
        "_wrapper_name": src.name,
        "_manifest_bytes": manifest_bytes,
        "_case_bytes": {name: content for _, name, _, _, content in cases},
    }


def freeze_dataset(dataset: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    wrapper = json.loads(dataset["_wrapper_bytes"].decode("utf-8-sig"))
    snapshots = {
        dataset["_wrapper_name"]: dataset["_wrapper_bytes"],
        wrapper["replay_manifest"]: dataset["_manifest_bytes"],
        **dataset["_case_bytes"],
    }
    for name, content in snapshots.items():
        with (directory / name).open("xb") as stream:
            stream.write(content)
    return directory / wrapper["replay_manifest"]


def _validate_replay(raw: dict[str, Any], dataset: dict[str, Any] | None) -> None:
    from .replay_suite import _validate_suite

    try:
        _validate_suite(raw)
    except ValueError as exc:
        raise ValueError(f"inconsistent replay suite: {exc}") from exc
    if dataset is not None:
        required = {(x["case_id"], x["case_sha256"]) for x in dataset["cases"]}
        observed = {(x["case_id"], x["case_sha256"]) for x in raw["cases"]}
        if observed != required or raw["manifest_sha256"] != dataset["manifest_sha256"]:
            raise ValueError("replay suite and dataset case bytes/manifest do not match")


def adapt_replay(path: str | Path, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = load_json(path)
    _validate_replay(raw, dataset)
    from .replay_suite import _verify_saved_suite

    _verify_saved_suite(Path(path), raw)
    cases = raw["cases"]
    n = len(cases)
    success = sum(x["passed"] for x in cases)
    return {
        "status": "measured",
        "source": _source(path, "replay_suite", raw),
        "integrity_verified": True,
        "contract": "controlled_fault_handling_not_task_completion",
        "sample_count": n,
        "revision_supplied": raw.get("revision_supplied"),
        "revision_verified": False,  # v0.6 records an unverified label
        "manifest_sha256": raw["manifest_sha256"],
        "metrics": {
            "case_pass_rate": metric(
                success / n, count=n, note="All Replay cases, including expected Agent failures"
            )
        },
        "cases": [
            {
                "case_id": x["case_id"],
                "case_sha256": x["case_sha256"],
                "test_result": "pass" if x["passed"] else "fail",
                "agent_status": x.get("status"),
                "fault_kind": x.get("fault_kind"),
                "result_verified": x["result_verified"],
            }
            for x in cases
        ],
    }


def adapt_report(path: str | Path) -> dict[str, Any]:
    # Reuse v0.1's Judge accounting to preserve official result semantics.
    from .audit import report_audit, validate_run

    raw = load_json(path)
    validate_run(raw)
    for key in ("method", "judge", "models", "aggregate"):
        if key in raw and not isinstance(raw[key], dict):
            raise ValueError(f"report {key} must be an object")
    if raw.get("mode") != "report":
        raise ValueError("report adapter only accepts mode=report, not memory/chat")
    for case in raw["results"]:
        if type(case.get("completed")) is not bool:
            raise ValueError("report case completed must be a boolean")
        claims = case.get("claim_verdicts")
        if claims is not None and (
            not isinstance(claims, list) or any(not isinstance(x, dict) for x in claims)
        ):
            raise ValueError("malformed report claim_verdicts")
    audit = report_audit(raw)
    rows = audit["per_case"]
    durations = [
        float(row["latency_s"])
        for row in rows
        if row["completed"] and _number(row["latency_s"]) is not None and row["latency_s"] >= 0
    ]
    lat95 = None
    completed_count = sum(row["completed"] for row in rows)
    if durations and len(durations) == completed_count:
        ordered = sorted(durations)
        lat95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
    return {
        "status": "measured" if rows else "unmeasured",
        "source": _source(path, "report", raw),
        "contract": "report_quality_not_controlled_fault_handling",
        "sample_count": len(rows),
        "signature": audit["signature"],
        "metrics": {
            "completion_rate": metric(audit["completion_rate_recomputed"], count=len(rows)),
            "judge_coverage": metric(
                audit["judge"]["coverage"],
                count=None
                if audit["judge"]["cases_missing_claim_records"]
                else sum(audit["judge"][k] for k in ("yes", "no", "unjudged")),
            ),
            "citation_support_among_judged": metric(
                audit["judge"]["support_among_judged"],
                count=audit["judge"]["yes"] + audit["judge"]["no"],
                note="Claim-weighted conditional support; unjudged excluded; not the official mean across reports",
            ),
            "latency_p95_completed_s": metric(
                lat95,
                unit="seconds",
                count=len(durations),
                note="Nearest-rank research-only P95; Judge time excluded; unmeasured if any completed latency is missing",
            ),
            "api_cost_usd": metric(
                None,
                unit="usd",
                note="Original pricing not verified; a reported zero is not free API usage",
            ),
        },
        "judge_counts": {key: audit["judge"][key] for key in ("yes", "no", "unjudged")},
        "reported_aggregate": raw.get("aggregate"),
        "completed_latency_missing": completed_count - len(durations),
        "agent_status_scope": "Legacy report-eval completion flag; final human approval is not measured",
        "cases": [
            {
                "case_id": x["case_id"],
                "test_result": "unmeasured",  # no golden answer for full report quality
                "agent_status": "completed" if x["completed"] else "failed",
                "completion": x["completed"],
                "judge_coverage": x["judge_coverage"],
            }
            for x in rows
        ],
        "warnings": audit["warnings"]
        + [
            "Legacy Judge parser out-of-range Claim labels are an OPEN defect; this adapter does not fix or rescore them."
        ],
    }


def adapt_retrieval(path: str | Path) -> dict[str, Any]:
    raw = load_json(path)
    if (
        type(raw.get("retrieval_metrics_version")) is not int
        or raw["retrieval_metrics_version"] < 1
        or not isinstance(raw.get("queries"), list)
    ):
        raise ValueError("not a ResearchFlow retrieval evaluation JSON")
    queries = raw["queries"]
    ids = [q.get("id") if isinstance(q, dict) else None for q in queries]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("retrieval queries need unique nonempty IDs")
    if raw.get("status") != "measured" or not queries:
        return {
            **_missing("retrieval source is not measured or has no queries"),
            "source": _source(path, "retrieval", raw),
        }
    agg = raw.get("aggregate")
    if not isinstance(agg, dict):
        raise ValueError("missing retrieval aggregate")
    if isinstance(agg.get("overall"), dict):
        scores, flavor = agg["overall"], "corpus_overall"
        query_scores = queries
    elif isinstance(agg.get("hit_cap"), dict):
        after = [
            v for k, v in agg["hit_cap"].items() if k.startswith("after") and isinstance(v, dict)
        ]
        if len(after) != 1:
            raise ValueError("ambiguous selection retrieval after-cap metrics")
        scores, flavor = after[0], "selection_after_cap"
        query_scores = [q.get("with_cap", {}) for q in queries]
    else:
        raise ValueError("unsupported retrieval aggregate layout")
    for block in (agg, scores):
        if "queries" in block and (
            type(block["queries"]) is not int or block["queries"] != len(queries)
        ):
            raise ValueError("retrieval query count mismatch")
    names = (
        "recall@1",
        "recall@3",
        "recall@5",
        "precision@1",
        "precision@5",
        "ndcg@5",
        "ndcg@10",
        "passage_recall",
        "span_accuracy",
    )
    metrics = {}
    for key in names:
        if key not in scores:
            continue
        samples = []
        for query in query_scores:
            if not isinstance(query, dict):
                raise ValueError("retrieval per-query score must be an object")
            if key.startswith(("recall@", "precision@")):
                family, k = key.split("@")
                block = query.get(family)
                value = block.get("@" + k) if isinstance(block, dict) else None
            else:
                value = query.get(key)
            if _number(value, fraction=True) is not None:
                samples.append(value)
        metrics[key.replace("@", "_at_")] = metric(
            scores[key],
            count=len(samples) if samples else None,
            note="Recorded aggregate; n counts available per-query observations for this metric, not all queries",
        )
    return {
        "status": "measured",
        "source": _source(path, "retrieval", raw),
        "contract": flavor,
        "sample_count": len(queries),
        "signature": {
            "metrics_version": raw["retrieval_metrics_version"],
            "dataset_fingerprint_recorded": raw.get("dataset_fingerprint"),
            "dataset_version": raw.get("dataset_version"),
            "embedding_model": raw.get("embedding_model"),
            "retriever": raw.get("retriever"),
            "top_k": raw.get("top_k"),
        },
        "metrics": metrics,
        "cases": [],
    }


def adapt_grounding(path: str | Path) -> dict[str, Any]:
    raw = load_json(path)
    if raw.get("eval") != "grounding-benchmark" or not isinstance(raw.get("result"), dict):
        raise ValueError("not a ResearchFlow grounding-benchmark result")
    result = raw["result"]
    status = result.get("status")
    count = result.get("claims")
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("grounding claims must be a nonnegative integer or null")
    if status == "measured" and not count:
        raise ValueError("measured grounding requires a nonempty claim set")
    reference = raw.get("reference_labels") or {}
    if not isinstance(reference, dict):
        raise ValueError("grounding reference labels must be an object")
    dimension_status = (
        "measured" if status == "measured" else "invalid" if status == "invalid" else "unmeasured"
    )
    metrics = {}
    for key in (
        "false_accept",
        "false_accept_wilson_upper",
        "false_reject",
        "unmeasured_share",
        "contradicted_recall",
    ):
        n = None
        if key.startswith("false_accept"):
            block = result.get("false_accept_counts") or {}
            if not isinstance(block, dict):
                raise ValueError("invalid false-accept counts")
            n = block.get("negatives_measured")
        elif key == "false_reject":
            block = result.get("false_reject_counts") or {}
            if not isinstance(block, dict):
                raise ValueError("invalid false-reject counts")
            n = block.get("positives_measured")
        elif key == "unmeasured_share":
            n = count
        else:
            confusion = result.get("confusion") or {}
            if not isinstance(confusion, dict):
                raise ValueError("invalid grounding confusion")
            row = confusion.get("contradicted") or {}
            if not isinstance(row, dict):
                raise ValueError("invalid contradicted counts")
            values = [row.get(k) for k in ("supported", "not_supported", "contradicted")]
            if all(type(v) is int and v >= 0 for v in values):
                n = sum(values)
        value = result.get(key) if dimension_status != "unmeasured" else None
        metrics[key] = metric(
            value,
            valid=dimension_status != "invalid",
            count=n,
            note="Recorded conditional rate; source validity retained; no unified release gate",
        )
    return {
        "status": dimension_status,
        "source_status": status,
        "source": _source(path, "grounding", raw),
        "contract": "grounding_benchmark_not_report_citation_support",
        "sample_count": count,
        "signature": {
            "set_version": raw.get("set_version"),
            "reference_rubric_sha256": reference.get("rubric_sha256"),
            "reference_kind": reference.get("kind"),
            "critic_routing": raw.get("critic_routing"),
            "routing_is_alias": raw.get("routing_is_alias"),
        },
        "metrics": metrics,
        "cases": [],
    }


def build_unified(
    *,
    dataset: dict[str, Any] | None = None,
    replay_suite: str | Path | None = None,
    report: str | Path | None = None,
    retrieval: str | Path | None = None,
    grounding: str | Path | None = None,
) -> dict[str, Any]:
    if not any((replay_suite, report, retrieval, grounding)):
        raise ValueError("at least one evaluation artifact is required")
    dimensions = {k: _missing("artifact_not_supplied") for k in DIMENSIONS}
    if replay_suite:
        dimensions["reliability"] = adapt_replay(replay_suite, dataset)
    if report:
        dimensions["outcome"] = adapt_report(report)
    if retrieval:
        dimensions["retrieval"] = adapt_retrieval(retrieval)
    if grounding:
        dimensions["grounding"] = adapt_grounding(grounding)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "researchflow_unified_evaluation",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": (
            {
                k: dataset[k]
                for k in (
                    "dataset_id",
                    "dataset_version",
                    "split",
                    "wrapper_sha256",
                    "manifest_sha256",
                    "cases",
                )
            }
            if dataset
            else None
        ),
        "dimensions": dimensions,
        "cross_source_pairing_verified": False,
        "not_a_single_score": True,
        "semantics": "Different dimensions have different sample universes; never average rates across dimensions.",
    }


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _replay_cases(dimension):
    if (
        dimension.get("contract") != "controlled_fault_handling_not_task_completion"
        or dimension.get("integrity_verified") is not True
    ):
        raise ValueError("replay dimension lacks validated provenance")
    if not _sha(dimension.get("manifest_sha256")):
        raise ValueError("missing replay manifest fingerprint")
    source = dimension.get("source")
    if not isinstance(source, dict) or not _sha(source.get("sha256")):
        raise ValueError("missing replay source fingerprint")
    cases = dimension.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("replay cases must be nonempty")
    mapping = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid replay case")
        ident = case.get("case_id")
        if (
            not isinstance(ident, str)
            or not ID_PATTERN.fullmatch(ident)
            or ident in mapping
            or not _sha(case.get("case_sha256"))
        ):
            raise ValueError("missing, duplicate or invalid replay identity/fingerprint")
        if case.get("test_result") not in ("pass", "fail"):
            raise ValueError("invalid replay verdict")
        if type(case.get("result_verified")) is not bool or (
            case["test_result"] == "pass" and not case["result_verified"]
        ):
            raise ValueError("passing replay must have verified artifacts")
        mapping[ident] = case
    metrics = dimension.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("case_pass_rate"), dict):
        raise ValueError("missing replay pass-rate metric")
    metric_row = metrics["case_pass_rate"]
    count = dimension.get("sample_count")
    if (
        type(count) is not int
        or count != len(cases)
        or metric_row.get("status") != "measured"
        or type(metric_row.get("n")) is not int
        or metric_row["n"] != count
    ):
        raise ValueError("inconsistent replay sample count")
    value = _number(metric_row.get("value"), fraction=True)
    expected = sum(c["test_result"] == "pass" for c in cases) / count
    if value is None or not math.isclose(value, expected, abs_tol=1e-12):
        raise ValueError("inconsistent replay pass rate")
    return mapping


def _dataset_identity(dataset):
    if not isinstance(dataset, dict):
        raise ValueError("invalid unified dataset")
    if (
        not isinstance(dataset.get("dataset_id"), str)
        or not ID_PATTERN.fullmatch(dataset["dataset_id"])
        or type(dataset.get("dataset_version")) is not int
        or dataset["dataset_version"] < 1
        or dataset.get("split") not in ("regression", "dev", "holdout")
    ):
        raise ValueError("invalid unified dataset identity")
    if not all(_sha(dataset.get(k)) for k in ("wrapper_sha256", "manifest_sha256")):
        raise ValueError("missing dataset fingerprints")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("missing dataset cases")
    mapping = {}
    for row in cases:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("case_id"), str)
            or not ID_PATTERN.fullmatch(row["case_id"])
            or row["case_id"] in mapping
            or not _sha(row.get("case_sha256"))
        ):
            raise ValueError("invalid dataset case fingerprint")
        mapping[row["case_id"]] = row["case_sha256"]
    return mapping


def compare_unified(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    for run in (baseline, candidate):
        if (
            not isinstance(run, dict)
            or run.get("kind") != "researchflow_unified_evaluation"
            or type(run.get("schema_version")) is not int
            or run["schema_version"] != 1
        ):
            raise ValueError("both inputs must be unified evaluation v1")
        dims = run.get("dimensions")
        if (
            not isinstance(dims, dict)
            or set(dims) != set(DIMENSIONS)
            or any(not isinstance(v, dict) for v in dims.values())
        ):
            raise ValueError("malformed unified dimensions")

    def blocked(reason):
        return {
            "comparable": False,
            "reason": reason,
            "regressions": None,
            "recoveries": None,
            "quality_gate_passed": None,
            "scope": "deterministic_replay_contract_only",
            "code_revision_verified": False,
        }

    old, new = baseline["dimensions"]["reliability"], candidate["dimensions"]["reliability"]
    if old.get("status") != "measured" or new.get("status") != "measured":
        return blocked("replay_dimension_not_measured")
    b_map, c_map = _replay_cases(old), _replay_cases(new)
    if set(b_map) != set(c_map):
        return blocked("replay_case_set_mismatch")
    if any(b_map[k]["case_sha256"] != c_map[k]["case_sha256"] for k in b_map):
        return blocked("replay_case_bytes_mismatch")
    if old["manifest_sha256"] != new["manifest_sha256"]:
        return blocked("replay_manifest_changed")
    if baseline.get("dataset") is None or candidate.get("dataset") is None:
        return blocked("unified_dataset_not_supplied")
    expected_b = _dataset_identity(baseline["dataset"])
    expected_c = _dataset_identity(candidate["dataset"])
    for expected, actual, ds, dim in (
        (expected_b, b_map, baseline["dataset"], old),
        (expected_c, c_map, candidate["dataset"], new),
    ):
        if (
            expected != {k: v["case_sha256"] for k, v in actual.items()}
            or ds["manifest_sha256"] != dim["manifest_sha256"]
        ):
            raise ValueError("normalized replay and dataset disagree")
    if baseline["dataset"] != candidate["dataset"]:
        return blocked("unified_dataset_not_identical")
    changes = [
        {
            "case_id": k,
            "baseline": b_map[k]["test_result"],
            "candidate": c_map[k]["test_result"],
            "change": "regressed"
            if b_map[k]["test_result"] == "pass" and c_map[k]["test_result"] == "fail"
            else "recovered"
            if b_map[k]["test_result"] == "fail" and c_map[k]["test_result"] == "pass"
            else "unchanged",
        }
        for k in sorted(b_map)
    ]
    return {
        "comparable": True,
        "scope": "deterministic_replay_contract_only",
        "regressions": sum(x["change"] == "regressed" for x in changes),
        "recoveries": sum(x["change"] == "recovered" for x in changes),
        "quality_gate_passed": all(x["candidate"] == "pass" for x in changes),
        "other_dimensions": "not_paired_or_gated; same code label does not prove a true code revision A/B",
        "code_revision_verified": False,
        "artifact_verification_repeated": False,
        "note": "Comparison validates normalized recorded contracts. Source chains were checked at collection; files are not re-opened here.",
        "changes": changes,
    }
