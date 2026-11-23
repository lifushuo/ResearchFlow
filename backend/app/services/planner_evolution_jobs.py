"""Planner GEPA 前端作业队列（独立于正式研究、人工发布）。

Web 端不能传命令或文件路径；由离线/在线评测 Worker 从评测卷认领。
这里保存调度状态，不会在 API 进程中调用模型。
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Literal

from app.services.evaluation_jobs import _admission, _read, now, update_status

JOB_ID = re.compile(r"^gepa-[a-f0-9]{24}$")
UUID_RE = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$")
PAID_CONSENT = "I_CONFIRM_GEPA_MODEL_API_COST"
MAX_JOBS = 40
TERMINAL = frozenset(
    {"succeeded", "quality_rejected", "failed", "cancelled", "pending_final", "no_candidate"}
)


class EvolutionJobError(ValueError):
    pass


class EvolutionJobBusy(EvolutionJobError):
    pass


def _base(root: Path) -> Path:
    path = root / "planner-evolution" / "jobs"
    if any(p.is_symlink() for p in (root / "planner-evolution", path)):
        raise EvolutionJobError("unsafe_job_storage")
    return path


def _folder(root: Path, job_id: str) -> Path:
    if not JOB_ID.fullmatch(job_id):
        raise EvolutionJobError("invalid_job_id")
    path = _base(root) / job_id
    if path.is_symlink():
        raise EvolutionJobError("unsafe_job_directory")
    return path


def _owned(root: Path, owner: str, job_id: str):
    path = _folder(root, job_id)
    if not path.is_dir():
        raise KeyError("job_not_found")
    req = _read(path / "request.json")
    if req.get("owner") != owner or req.get("job_id") != job_id:
        raise KeyError("job_not_found")
    return path, req, _read(path / "status.json")


def _rows(root: Path):
    base = _base(root)
    if not base.is_dir():
        return []
    result = []
    for path in base.iterdir():
        if path.is_symlink() or not path.is_dir() or not JOB_ID.fullmatch(path.name):
            continue
        try:
            req, status = _read(path / "request.json"), _read(path / "status.json")
            if req.get("job_id") == path.name and status.get("job_id") == path.name:
                result.append((path, req, status))
        except OSError, ValueError, KeyError:
            continue
    return sorted(result, key=lambda row: row[1].get("created_at", ""), reverse=True)


def _public(req: dict, status: dict) -> dict:
    # Never return raw Prompt, research question, provider credential or subprocess stdout.
    return {
        "job_id": req["job_id"],
        "task_kind": req.get("task_kind", "search"),
        "source_job_id": req.get("source_job_id"),
        "mode": req["mode"],
        "metric_version": req.get("metric_version", "v1"),
        "max_evals": req["max_evals"],
        "proposals": req["proposals"],
        "seed": req["seed"],
        "frozen": req.get("frozen"),
        "dataset_preset": req.get("dataset_preset"),
        "request_cap": (req.get("budget") or {}).get("total_cap"),
        "dataset_sha256": req.get("dataset_sha256"),
        "holdout_sha256": req.get("holdout_sha256"),
        "state": status["state"],
        "phase": status.get("phase"),
        "error_type": status.get("error_type"),
        "quality_gate_passed": status.get("quality_gate_passed"),
        "created_at": req["created_at"],
        "updated_at": status["updated_at"],
    }


def _paid_allowed(owner: str) -> bool:
    allow = {x.strip() for x in os.getenv("RF_GEPA_ALLOWED_OWNERS", "").split(",") if x.strip()}
    return os.getenv("RF_GEPA_LIVE_ENABLED", "0") == "1" and owner in allow


def _holdout_fingerprint(dataset: dict) -> str:
    """Stable under changes to train cases; changing train must not reset holdout use."""
    from app.services.evaluation_closure import input_hash

    cases = dataset.get("cases")
    if not isinstance(cases, list):
        raise EvolutionJobError("invalid_holdout_dataset")
    holdout = sorted(
        (case for case in cases if isinstance(case, dict) and case.get("split") == "holdout"),
        key=lambda case: case.get("case_id", ""),
    )
    if not holdout:
        raise EvolutionJobError("holdout_required")
    return input_hash({"holdout": holdout})


def _has_consumed_holdout(root: Path, existing: list, fingerprint: str) -> bool:
    """Web live admission: block repeating a disclosed holdout on another candidate.

    Old live jobs without the fingerprint are checked against their dataset snapshots.
    This guard applies to the web queue; external CLI invocations require separate governance.
    """
    from app.services.planner_evolution_insights import project_journal

    for path, req, status in existing:
        if req.get("mode") != "live":
            continue
        previous = req.get("holdout_sha256")
        if not previous:
            snapshot = path / "dataset.json"
            if snapshot.is_file() and not snapshot.is_symlink():
                previous = _holdout_fingerprint(_read(snapshot))
        if previous != fingerprint:
            continue
        audit = project_journal(root / "planner-evolution" / "experiments" / req["job_id"])
        if audit["holdout_observed"] or audit["error"]:
            return True
        # A terminal result without usable trace cannot prove the holdout was not read.
        # A clean quality rejection before holdout is NOT an exposure and may retry.
        if status.get("state") in ("succeeded", "quality_rejected") and not audit["available"]:
            return True
    return False


def create_job(
    root: Path,
    owner: str,
    *,
    request_id: str,
    mode: Literal["demo", "live"],
    max_evals: int,
    proposals: int,
    seed: int,
    consent: str | None,
    metric_version: str = "v3",
    dataset_preset: str = "full",
    request_quotas: dict | None = None,
) -> dict[str, Any]:
    if not UUID_RE.fullmatch(request_id):
        raise EvolutionJobError("invalid_request_id")
    if mode not in ("demo", "live") or type(max_evals) is not int or type(proposals) is not int:
        raise EvolutionJobError("invalid_job_options")
    if metric_version not in ("v1", "v2", "v3"):
        raise EvolutionJobError("invalid_metric_version")
    if not (4 <= max_evals <= 200 and 1 <= proposals <= 8 and 0 <= seed <= 2**31 - 1):
        raise EvolutionJobError("evaluation_budget_out_of_range")
    if mode == "live" and (not _paid_allowed(owner) or consent != PAID_CONSENT):
        raise EvolutionJobError("live_gepa_not_authorized")
    if mode == "demo" and consent:
        raise EvolutionJobError("demo_does_not_require_paid_consent")
    if metric_version != "v3":
        raise EvolutionJobError("web_search_requires_v3")
    if dataset_preset not in ("smoke", "full"):
        raise EvolutionJobError("invalid_dataset_preset")

    job_id = "gepa-" + hashlib.sha256((owner + "/" + request_id).encode()).hexdigest()[:24]
    base = _base(root)
    base.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = _folder(root, job_id)
    with _admission(root):
        if path.exists():
            _, old, status = _owned(root, owner, job_id)
            if any(
                old.get(k) != v
                for k, v in (
                    ("mode", mode),
                    ("metric_version", metric_version),
                    ("max_evals", max_evals),
                    ("proposals", proposals),
                    ("seed", seed),
                    ("dataset_preset", dataset_preset),
                    ("requested_quotas", request_quotas),
                )
            ):
                raise EvolutionJobError("idempotency_conflict")
            return _public(old, status)
        existing = _rows(root)
        if len(existing) >= MAX_JOBS:
            raise EvolutionJobBusy("evolution_history_full")
        if any(r.get("owner") == owner and s["state"] not in TERMINAL for _, r, s in existing):
            raise EvolutionJobBusy("one_active_evolution_per_user")
        if sum(s["state"] not in TERMINAL for _, _, s in existing) >= 2:
            raise EvolutionJobBusy("evolution_queue_full")
        from app.services.evaluation_closure import input_hash
        from app.services.planner_gepa_contract import freeze, search_dataset
        from evals.agent_eval_ext.closure_observer import read
        from evals.agent_eval_ext.long_run import atomic_create
        from evals.agent_eval_ext.planner_evolution.__main__ import DEMO_DATASET
        from research_engine.prompts import PLANNER_PROMPT_V2

        if mode == "live":
            from app.services.planner_evolution_center import merged_dataset

            merged = merged_dataset(root)
            if merged.get("error") or not merged.get("dataset"):
                raise EvolutionJobError("reviewed_dataset_unavailable")
            source = merged["dataset"]
        else:
            source = read(DEMO_DATASET)
        dataset = search_dataset(source, dataset_preset)
        validations = sum(c["split"] == "validation" for c in dataset["cases"])
        if max_evals < validations + 2:
            raise EvolutionJobError("insufficient_budget_for_validation")
        frozen = freeze(mode, dataset, max_evals, proposals, seed)
        quotas = {"planner": 4 * max_evals, "judge": max_evals, "reflection": proposals}
        if request_quotas is not None:
            if set(request_quotas) != set(quotas) or any(
                type(v) is not int or v < 0 or v > quotas[k] for k, v in request_quotas.items()
            ):
                raise EvolutionJobError("invalid_request_quotas")
            quotas = dict(request_quotas)
        budget = {
            "run_id": job_id,
            "quotas": quotas,
            "total_cap": sum(quotas.values()),
            "planned_evaluations": max_evals,
            "source": "web-explicit-consent" if mode == "live" else "offline-simulation",
        }
        dataset_sha = input_hash(dataset)
        holdout_sha = None
        created = now()
        req = {
            "schema_version": 1,
            "job_id": job_id,
            "owner": owner,
            "request_id": request_id,
            "mode": mode,
            "metric_version": metric_version,
            "dataset_preset": dataset_preset,
            "requested_quotas": request_quotas,
            "search_only": True,
            "frozen": frozen,
            "budget": budget,
            "max_evals": max_evals,
            "proposals": proposals,
            "seed": seed,
            "dataset_sha256": dataset_sha,
            "holdout_sha256": holdout_sha,
            "created_at": created,
        }
        status = {"job_id": job_id, "state": "queued", "phase": "queued", "updated_at": created}
        path.mkdir(mode=0o700)
        try:
            atomic_create(path / "budget.json", budget)
            atomic_create(path / "frozen-config.json", frozen)
            atomic_create(path / "baseline.json", {"planner": PLANNER_PROMPT_V2})
            # Dataset snapshot is immutable per job; never read a changing active dataset mid-search.
            if dataset is not None:
                from evals.agent_eval_ext.long_run import atomic_create

                atomic_create(path / "dataset.json", dataset)
            with (path / "request.json").open("x", encoding="utf-8") as f:
                import json

                json.dump(req, f, ensure_ascii=False)
            update_status(path / "status.json", status)
        except BaseException:
            for item in path.iterdir():
                if item.is_file():
                    item.unlink()
            path.rmdir()
            raise
        return _public(req, status)


def list_jobs(root: Path, owner: str) -> list[dict]:
    return [_public(req, status) for _, req, status in _rows(root) if req.get("owner") == owner]


def get_job(root: Path, owner: str, job_id: str) -> dict:
    _, req, status = _owned(root, owner, job_id)
    return _public(req, status)


def cancel_job(root: Path, owner: str, job_id: str) -> dict:
    with _admission(root):
        path, req, status = _owned(root, owner, job_id)
        if status["state"] in TERMINAL:
            return _public(req, status)
        status["state"] = "cancelled" if status["state"] == "queued" else "cancelling"
        status["phase"] = "cancel_requested"
        status["updated_at"] = now()
        update_status(path / "status.json", status)
        return _public(req, status)


def claim_next(root: Path, mode: str):
    with _admission(root):
        for path, req, status in reversed(_rows(root)):
            if req["mode"] != mode or status["state"] != "queued":
                continue
            status.update(state="running", phase="starting", updated_at=now())
            update_status(path / "status.json", status)
            return path, req
    return None


def finish_job(
    root: Path, path: Path, *, state: str, error_type: str | None = None, gate: bool | None = None
) -> None:
    if state not in TERMINAL:
        raise EvolutionJobError("invalid_terminal_state")
    with _admission(root):
        status = _read(path / "status.json")
        if status["state"] == "cancelled":
            return
        if status["state"] == "cancelling":
            state = "cancelled"
        if status["state"] not in ("running", "cancelling"):
            raise EvolutionJobError("job_not_running")
        status.update(
            state=state,
            phase="finished",
            error_type=error_type,
            quality_gate_passed=gate,
            updated_at=now(),
        )
        update_status(path / "status.json", status)


def recover_interrupted(root: Path, mode: str) -> int:
    n = 0
    with _admission(root):
        for path, req, status in _rows(root):
            if req["mode"] != mode or status["state"] not in ("running", "cancelling"):
                continue
            status.update(
                state="failed",
                phase="interrupted",
                error_type="worker_restarted_no_auto_retry",
                updated_at=now(),
            )
            update_status(path / "status.json", status)
            n += 1
    return n


def _verify_result_binding(req: dict, result: dict) -> None:
    """Check result contents against pinned job inputs (integrity, not an HMAC)."""
    if req.get("task_kind") == "planner_final_ab":
        from app.services.planner_final_jobs import verify_result

        return verify_result(req, result)
    from difflib import unified_diff

    from app.services.evaluation_closure import input_hash
    from evals.agent_eval_ext.long_run import sha256_text
    from research_engine.prompts import PLANNER_PROMPT_V2

    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise EvolutionJobError("result_schema_mismatch")
    if result.get("evidence_kind") != (
        "live_planner"
        if req["mode"] == "live"
        else "fake_planner"
        if req.get("search_only")
        else "scripted_demo"
    ):
        raise EvolutionJobError("result_evidence_mode_mismatch")
    dataset = result.get("dataset")
    if not isinstance(dataset, dict) or result.get("dataset_sha256") != input_hash(dataset):
        raise EvolutionJobError("result_dataset_hash_mismatch")
    if (req["mode"] == "live" or req.get("search_only")) and result["dataset_sha256"] != req.get(
        "dataset_sha256"
    ):
        raise EvolutionJobError("result_job_dataset_mismatch")
    if req.get("frozen"):
        frozen = req["frozen"]
        if (
            result.get("metric_version") != "v3"
            or result.get("metric_rules_sha256") != frozen["metric_rules_sha256"]
            or result.get("routes") != frozen["routes"]
            or result.get("limits") != frozen["limits"]
            or result.get("judge_settings") != frozen["judge_settings"]
            or not result.get("preexperiment_no_holdout")
            or result.get("baseline", {}).get("sha256") != frozen["baseline_sha256"]
        ):
            raise EvolutionJobError("result_frozen_protocol_mismatch")
    baseline = result.get("baseline")
    if not isinstance(baseline, dict):
        raise EvolutionJobError("result_baseline_missing")
    baseline_text = baseline.get("override") or PLANNER_PROMPT_V2
    if not isinstance(baseline_text, str) or sha256_text(baseline_text) != baseline.get("sha256"):
        raise EvolutionJobError("result_baseline_hash_mismatch")
    candidate = result.get("candidate")
    if candidate is None:
        if result.get("prompt_diff"):
            raise EvolutionJobError("result_unexpected_diff")
    else:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("override"), str):
            raise EvolutionJobError("result_invalid_candidate")
        text = candidate["override"]
        if sha256_text(text) != candidate.get("sha256"):
            raise EvolutionJobError("result_candidate_hash_mismatch")
        expected_diff = "\n".join(
            unified_diff(
                baseline_text.splitlines(),
                text.splitlines(),
                fromfile="baseline",
                tofile="candidate",
                lineterm="",
            )
        )
        if result.get("prompt_diff") != expected_diff:
            raise EvolutionJobError("result_prompt_diff_mismatch")


def job_result(root: Path, owner: str, job_id: str) -> dict:
    path, _req, status = _owned(root, owner, job_id)
    result_path = path.parent.parent / "experiments" / job_id / "experiment.json"
    if status["state"] not in TERMINAL or not result_path.is_file() or result_path.is_symlink():
        return {"job_id": job_id, "available": False}
    if result_path.stat().st_size > 8 * 1024 * 1024:
        raise EvolutionJobError("experiment_result_oversized")
    import json

    result = json.loads(result_path.read_text(encoding="utf-8"))
    _verify_result_binding(_req, result)
    if _req.get("task_kind") == "planner_final_ab":
        from app.services.planner_final_jobs import result_view
        from app.services.planner_gepa_contract import budget_view

        return {
            **result_view(_req, status, result),
            "budget": budget_view(result_path.parent, _req),
        }
    gate = result.get("holdout", {}).get("gate", {})
    from app.services.planner_gepa_contract import budget_view, quality_comparison

    return {
        "comparison": quality_comparison(result),
        "rejection_reasons": result.get("search", {}).get("rejection_reasons"),
        "outcome": status["state"],
        "search_only": result.get("preexperiment_no_holdout", False),
        "budget": budget_view(result_path.parent, _req),
        "routes": result.get("routes"),
        "job_id": job_id,
        "available": True,
        "quality_gate_passed": gate.get("passed"),
        "holdout": {
            k: gate.get(k)
            for k in ("passed", "reasons", "delta", "baseline_coverage", "candidate_coverage")
        },
        "baseline_sha256": result.get("baseline", {}).get("sha256"),
        "candidate_sha256": (result.get("candidate") or {}).get("sha256"),
        "prompt_diff": (result.get("prompt_diff") or "")[:20000],
        "validation_gain": result.get("search", {}).get("validation_gain"),
        "evidence_kind": result.get("evidence_kind"),
    }


def mark_phase(root: Path, path: Path, phase: str) -> None:
    with _admission(root):
        status = _read(path / "status.json")
        if status["state"] != "running":
            return
        status["phase"] = phase[:64]
        status["updated_at"] = now()
        update_status(path / "status.json", status)


def job_audit(root: Path, owner: str, job_id: str) -> dict[str, Any]:
    """Bounded, read-only search timeline; strict owner check before artifact reads."""
    _path, req, status = _owned(root, owner, job_id)
    from app.services.planner_evolution_insights import project_journal
    from app.services.planner_gepa_contract import budget_view

    experiment_dir = root / "planner-evolution" / "experiments" / job_id
    audit = project_journal(experiment_dir)
    result_integrity = "not_available"
    result_versions = None
    artifact = experiment_dir / "experiment.json"
    if status["state"] in TERMINAL and artifact.is_file() and not artifact.is_symlink():
        try:
            import json

            if artifact.stat().st_size <= 8 * 1024 * 1024:
                result = json.loads(artifact.read_text(encoding="utf-8"))
                _verify_result_binding(req, result)
                result_integrity = "verified"
                optimizer = result.get("optimizer") or {}
                result_versions = {
                    "optimizer": optimizer.get("name") if optimizer.get("name") == "GEPA" else None,
                    "optimizer_version": str(optimizer.get("version", ""))[:32],
                    "metric_version": str(result.get("metric_version", ""))[:80],
                    "baseline_sha256": (result.get("baseline") or {}).get("sha256"),
                    "candidate_sha256": (result.get("candidate") or {}).get("sha256"),
                    "report_dataset_sha256": result.get("report_dataset_sha256"),
                }
            else:
                result_integrity = "mismatch"
        except OSError, ValueError, KeyError, TypeError:
            result_integrity = "mismatch"
    return {
        "job_id": job_id,
        "task_kind": req.get("task_kind", "search"),
        "source_job_id": req.get("source_job_id"),
        "frozen": req.get("frozen"),
        "budget": budget_view(experiment_dir, req),
        "mode": req["mode"],
        "state": status["state"],
        "phase": status.get("phase"),
        "dataset_sha256": req.get("dataset_sha256"),
        "holdout_sha256": req.get("holdout_sha256"),
        "limits": {
            "max_evals": req["max_evals"],
            "proposals": req["proposals"],
            "seed": req["seed"],
        },
        "journal": audit,
        "result_integrity": result_integrity,
        "result_versions": result_versions,
        "holdout_policy": "one_web_live_exposure_per_frozen_holdout",
        "full_report_ab": "manual_not_started_by_gepa_worker",
    }
