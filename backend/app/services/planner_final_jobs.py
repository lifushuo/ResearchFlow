"""Final Planner A/B tasks on the existing GEPA queue; never publish or search."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from app.services import planner_evolution_jobs as jobs
from app.services.evaluation_closure import input_hash
from app.services.evaluation_jobs import _admission, _read, now, update_status
from app.services.planner_gepa_contract import generation_settings, request_timeouts
from evals import holdout_usage_guard as guard
from evals.agent_eval_ext.long_run import atomic_create, sha256_text
from evals.agent_eval_ext.planner_evolution.experiment import Case, Observation, checked_prompt
from evals.agent_eval_ext.planner_evolution.metrics_v3 import gate_v3, rules_hash

KIND = "planner_final_ab"
CONSENT = "I_CONFIRM_ONE_PLANNER_FINAL_AB_API_COST"
QUOTAS = {"planner": 72, "judge": 18, "reflection": 0}


def read_artifact(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise jobs.EvolutionJobError("invalid_final_artifact")
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise jobs.EvolutionJobError("invalid_final_artifact")
    return value


def fake_dataset():
    # Dedicated simulation fixture, never a substitute for real holdout evidence.
    return {
        "dataset_id": "fake-planner-final-v3",
        "cases": [
            {
                "case_id": f"fake-final-{i}",
                "query": f"Compare offline simulation methods {i}",
                "depth": "fast",
                "required_aspects": ["named comparison", "measured outcome"],
            }
            for i in range(1, 10)
        ],
    }


def content_hash(data):
    return input_hash({"dataset_id": data["dataset_id"], "cases": data["cases"]})


def as_cases(data):
    cases = [
        Case(
            case_id=c["case_id"],
            group="planner-final",
            split="holdout",
            query=c["query"],
            depth=c["depth"],
            required_aspects=c["required_aspects"],
        )
        for c in data["cases"]
    ]
    if len(cases) != 9 or len({c.case_id for c in cases}) != 9:
        raise ValueError("final_requires_nine_distinct_cases")
    return cases


def protocol_metadata():
    """Admission reads the sidecar only; no questions or annotations are loaded."""
    manifest = guard.HOLDOUT_PATH.with_name(guard.HOLDOUT_PATH.stem + ".manifest.json")
    m = _read(manifest)
    if (
        m.get("dataset_id") != guard.HOLDOUT_ID
        or m.get("content_sha256") != guard.CONTENT_SHA256
        or m.get("file_sha256") != guard.FILE_SHA256
        or not guard.HOLDOUT_PATH.is_file()
        or guard.HOLDOUT_PATH.is_symlink()
        or not guard.POLICY_PATH.is_file()
        or guard.POLICY_PATH.is_symlink()
    ):
        raise jobs.EvolutionJobError("final_protocol_unavailable")
    return m


def live_allowed(owner):
    return jobs._paid_allowed(owner) and os.getenv("RF_GEPA_FINAL_LIVE_ENABLED", "0") == "1"


def exposure_path(root):
    folder = root / "planner-evolution" / "final-exposures"
    if folder.is_symlink():
        raise jobs.EvolutionJobError("unsafe_final_exposure_storage")
    return folder / (guard.CONTENT_SHA256 + ".json")


def source_experiment(root, owner, source_id):
    _, req, status = jobs._owned(root, owner, source_id)
    if req.get("task_kind") == KIND or status["state"] != "pending_final":
        raise jobs.EvolutionJobError("source_not_ready_for_final")
    path = root / "planner-evolution" / "experiments" / source_id
    if path.is_symlink():
        raise jobs.EvolutionJobError("unsafe_source_experiment")
    result = read_artifact(path / "experiment.json")
    jobs._verify_result_binding(req, result)
    if (
        result.get("metric_version") != "v3"
        or result.get("search", {}).get("accepted_candidate") is not True
        or not result.get("candidate")
        or req["frozen"]["min_gain"] != 0.05
        or req["frozen"]["metric_rules_sha256"] != rules_hash()
    ):
        raise jobs.EvolutionJobError("source_has_no_frozen_v3_candidate")
    return req, result


def preview(root, owner, source_id):
    req, result = source_experiment(root, owner, source_id)
    reason = None
    try:
        protocol_metadata()
    except OSError, ValueError:
        reason = "final_protocol_unavailable"
    if exposure_path(root).exists():
        reason = "formal_holdout_already_reserved"
    if req["mode"] != "live":
        reason = "live_final_requires_real_source"
    if not live_allowed(owner):
        reason = "administrator_has_not_enabled_paid_final"
    return {
        "source_job_id": source_id,
        "candidate_sha256": result["candidate"]["sha256"],
        "baseline_sha256": result["baseline"]["sha256"],
        "fake_enabled": True,
        "live_enabled": reason is None,
        "live_disabled_reason": reason,
        "case_count": 9,
        "evaluations": 18,
        "request_cap": 90,
        "quotas": QUOTAS,
        "min_gain": 0.05,
        "routes": result["routes"],
        "full_report_ab": "separate_manual_gate",
    }


def create_final_job(
    root, owner, source_id, *, request_id, mode="demo", consent=None, request_quotas=None
):
    if not jobs.UUID_RE.fullmatch(request_id) or mode not in ("demo", "live"):
        raise jobs.EvolutionJobError("invalid_final_request")
    if mode == "live" and (not live_allowed(owner) or consent != CONSENT):
        raise jobs.EvolutionJobError("paid_final_not_authorized")
    if mode == "demo" and consent:
        raise jobs.EvolutionJobError("fake_final_does_not_require_paid_consent")
    quotas = dict(QUOTAS)
    if request_quotas is not None:
        if set(request_quotas) != set(QUOTAS) or any(
            type(v) is not int or v < 0 or v > QUOTAS[k] for k, v in request_quotas.items()
        ):
            raise jobs.EvolutionJobError("invalid_request_quotas")
        quotas = dict(request_quotas)
    job_id = "gepa-" + hashlib.sha256((owner + "/" + request_id).encode()).hexdigest()[:24]
    jobs._base(root).mkdir(parents=True, exist_ok=True, mode=0o700)
    with _admission(root):
        path = jobs._folder(root, job_id)
        if path.exists():
            _, old, status = jobs._owned(root, owner, job_id)
            if any(
                old.get(k) != v
                for k, v in {
                    "task_kind": KIND,
                    "source_job_id": source_id,
                    "mode": mode,
                    "requested_quotas": request_quotas,
                }.items()
            ):
                raise jobs.EvolutionJobError("idempotency_conflict")
            return jobs._public(old, status)
        existing = jobs._rows(root)
        if (
            len(existing) >= jobs.MAX_JOBS
            or sum(s["state"] not in jobs.TERMINAL for _, _, s in existing) >= 2
        ):
            raise jobs.EvolutionJobBusy("evolution_queue_full")
        if any(r.get("owner") == owner and s["state"] not in jobs.TERMINAL for _, r, s in existing):
            raise jobs.EvolutionJobBusy("one_active_evolution_per_user")
        parent, source = source_experiment(root, owner, source_id)
        frozen = json.loads(json.dumps(parent["frozen"]))
        if mode == "live":
            if parent["mode"] != "live":
                raise jobs.EvolutionJobError("live_final_requires_real_source")
            from evals.agent_eval_ext.planner_evolution.__main__ import _live_routes

            _, actual_routes = _live_routes(
                source["routes"]["judge"], source["routes"]["reflection"]
            )
            if actual_routes != source["routes"]:
                raise jobs.EvolutionJobError("final_model_routes_changed")
            protocol_metadata()
            if exposure_path(root).exists():
                raise jobs.EvolutionJobError("formal_holdout_already_reserved")
            dataset_sha, dataset_id = guard.CONTENT_SHA256, guard.HOLDOUT_ID
            routes = source["routes"]
        else:
            from research_engine.runconfig import ROLES

            dataset_sha, dataset_id = content_hash(fake_dataset()), "fake-planner-final-v3"
            routes = {
                "pipeline": {r: "fake:planner" for r in ROLES},
                "judge": "fake:judge",
                "reflection": "fake:reflection",
            }
        frozen.update(
            routes=routes,
            dataset_sha256=dataset_sha,
            candidate_sha256=source["candidate"]["sha256"],
            request_timeouts=request_timeouts(),
            limits={
                "evaluations": 18,
                "proposals": 0,
                "seed": parent["seed"],
                "minimum_gain": 0.05,
            },
        )
        budget = {
            "run_id": job_id,
            "quotas": quotas,
            "total_cap": sum(quotas.values()),
            "planned_evaluations": 18,
            "source": "web-explicit-final-consent" if mode == "live" else "offline-simulation",
        }
        req = {
            "schema_version": 1,
            "job_id": job_id,
            "owner": owner,
            "mode": mode,
            "task_kind": KIND,
            "source_job_id": source_id,
            "source_sha256": input_hash(source),
            "metric_version": "v3",
            "search_only": False,
            "dataset_preset": dataset_id,
            "max_evals": 18,
            "proposals": 0,
            "seed": parent["seed"],
            "requested_quotas": request_quotas,
            "frozen": frozen,
            "budget": budget,
            "dataset_sha256": dataset_sha,
            "holdout_sha256": dataset_sha if mode == "live" else None,
            "created_at": now(),
        }
        path.mkdir(mode=0o700)
        atomic_create(path / "source-experiment.json", source)
        atomic_create(path / "frozen-config.json", frozen)
        atomic_create(path / "budget.json", budget)
        if mode == "live":
            # Permanent reservation, global to the frozen holdout rather than per user.
            # Cancellation/crash does not release it or silently allow another exposure.
            atomic_create(
                exposure_path(root),
                {
                    "job_id": job_id,
                    "owner": owner,
                    "candidate_sha256": frozen["candidate_sha256"],
                    "holdout_sha256": dataset_sha,
                    "reserved_at": req["created_at"],
                },
            )
        atomic_create(path / "request.json", req)
        status = {
            "job_id": job_id,
            "state": "queued",
            "phase": "queued",
            "updated_at": req["created_at"],
        }
        update_status(path / "status.json", status)
        return jobs._public(req, status)


def validate_inputs(root, folder, req):
    """Shared API/Worker/CLI checks. Must not read the formal questions."""
    if (
        _read(folder / "request.json") != req
        or req.get("task_kind") != KIND
        or req.get("mode") not in ("demo", "live")
        or jobs._folder(root, req["job_id"]).resolve() != folder.resolve()
    ):
        raise ValueError("final_request_changed")
    f = req["frozen"]
    source = read_artifact(folder / "source-experiment.json")
    parent, original = source_experiment(root, req["owner"], req["source_job_id"])
    if input_hash(original) != req["source_sha256"]:
        raise ValueError("final_parent_experiment_changed")
    if (
        _read(folder / "frozen-config.json") != f
        or _read(folder / "budget.json") != req["budget"]
        or input_hash(source) != req["source_sha256"]
        or f["metric_rules_sha256"] != rules_hash()
        or f["min_gain"] != 0.05
        or f["metric_version"] != "v3"
        or f["judge_settings"] != original["judge_settings"]
        or f["generation"] != parent["frozen"]["generation"]
        or f["limits"]
        != {"evaluations": 18, "proposals": 0, "seed": req["seed"], "minimum_gain": 0.05}
        or f["generation"] != generation_settings(f["judge_settings"]["output_tokens"])
    ):
        raise ValueError("final_frozen_inputs_changed")
    if set(f["request_timeouts"]) != {"planner", "judge", "reflection"} or any(
        type(value) is not int or not 1 <= value <= 600 for value in f["request_timeouts"].values()
    ):
        raise ValueError("final_timeout_invalid")
    for arm in ("baseline", "candidate"):
        text = source[arm]["override"]
        if not isinstance(text, str) or sha256_text(checked_prompt(text)) != f[arm + "_sha256"]:
            raise ValueError("final_prompt_changed")
    quotas = req["budget"]["quotas"]
    if (
        set(quotas) != set(QUOTAS)
        or any(type(v) is not int or not 0 <= v <= QUOTAS[k] for k, v in quotas.items())
        or req["budget"]["total_cap"] != sum(quotas.values())
        or req["budget"]["run_id"] != req["job_id"]
        or req["budget"]["planned_evaluations"] != 18
    ):
        raise ValueError("final_budget_invalid")
    if req["mode"] == "live":
        if not live_allowed(req["owner"]):
            raise ValueError("paid_final_not_authorized")
        from evals.agent_eval_ext.planner_evolution.__main__ import _live_routes

        _, actual_routes = _live_routes(f["routes"]["judge"], f["routes"]["reflection"])
        if actual_routes != f["routes"]:
            raise ValueError("final_model_routes_changed")
        claim = _read(exposure_path(root))
        if (
            claim["job_id"] != req["job_id"]
            or claim["candidate_sha256"] != f["candidate_sha256"]
            or claim["holdout_sha256"] != guard.CONTENT_SHA256
            or req["dataset_sha256"] != guard.CONTENT_SHA256
        ):
            raise ValueError("final_exposure_binding_changed")
    else:
        if req["dataset_sha256"] != content_hash(fake_dataset()):
            raise ValueError("fake_final_fixture_changed")
    return source


def verify_result(req, result):
    f = req["frozen"]
    if (
        result.get("schema_version") != 1
        or result.get("task_kind") != KIND
        or result.get("source_job_id") != req["source_job_id"]
        or result.get("source_sha256") != req["source_sha256"]
        or result.get("metric_version") != "v3"
        or result.get("metric_rules_sha256") != f["metric_rules_sha256"]
        or result.get("evidence_kind")
        != ("live_planner" if req["mode"] == "live" else "fake_planner")
        or content_hash(result["dataset"]) != req["dataset_sha256"]
        or result.get("dataset_sha256") != req["dataset_sha256"]
        or result.get("release_ready") is not False
        or result["routes"] != f["routes"]
        or result["limits"] != f["limits"]
        or result["request_timeouts"] != f["request_timeouts"]
        or result["judge_settings"] != f["judge_settings"]
    ):
        raise ValueError("final_result_binding_mismatch")
    for arm in ("baseline", "candidate"):
        if (
            sha256_text(result[arm]["override"]) != f[arm + "_sha256"]
            or result[arm]["sha256"] != f[arm + "_sha256"]
        ):
            raise ValueError("final_result_prompt_mismatch")
    arms = [
        [Observation.model_validate(row) for row in result["holdout"][arm]]
        for arm in ("baseline", "candidate")
    ]
    gate = gate_v3(
        as_cases(result["dataset"]), *arms, 0.05, require_live_evidence=req["mode"] == "live"
    )
    if gate != result["holdout"]["gate"]:
        raise ValueError("final_result_gate_mismatch")
    if req["mode"] == "live":
        from evals.harness import validate_judge

        for row in arms[0] + arms[1]:
            validate_judge(
                f"{f['routes']['judge'].partition(':')[0]}:{row.served_judge}",
                {**f["routes"]["pipeline"], "reflection": f["routes"]["reflection"]},
            )


def result_view(req, status, result):
    from app.services.planner_gepa_contract import quality_comparison

    gate = result["holdout"]["gate"]
    comparison = quality_comparison(
        {
            **result,
            "search": {
                "validation_" + arm: result["holdout"][arm] for arm in ("baseline", "candidate")
            },
        }
    )
    return {
        "job_id": req["job_id"],
        "available": True,
        "task_kind": KIND,
        "source_job_id": req["source_job_id"],
        "outcome": status["state"],
        "comparison": comparison,
        "evidence_kind": result["evidence_kind"],
        "quality_gate_passed": gate["passed"],
        "release_ready": False,
        "full_report_ab": "separate_manual_gate",
        "baseline_sha256": result["baseline"]["sha256"],
        "candidate_sha256": result["candidate"]["sha256"],
        "holdout": {
            "passed": gate["passed"],
            "reasons": gate["reasons"],
            "delta": gate["quality_delta"],
            "baseline_coverage": gate["coverage_baseline"],
            "candidate_coverage": gate["coverage_candidate"],
        },
    }
