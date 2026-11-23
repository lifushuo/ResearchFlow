"""Frozen inputs and read-only budget/score projections for the existing GEPA queue."""

import json
import os
import re
from pathlib import Path
from statistics import mean

from app.services.evaluation_closure import input_hash
from evals.agent_eval_ext.long_run import sha256_text
from evals.agent_eval_ext.planner_evolution.experiment import Dataset, Observation
from evals.agent_eval_ext.planner_evolution.judge_prompt import PLANNER_JUDGE_PROMPT_V2
from evals.agent_eval_ext.planner_evolution.metrics_v3 import case_score_v3, rules_hash
from evals.paid_call_budget import BudgetConfig, PaidCallBudget
from research_engine.prompts import PLANNER_PROMPT_V2
from research_engine.runconfig import ROLES


def request_timeouts():
    raw = os.environ.get("RF_GEPA_JUDGE_TIMEOUT_SECONDS", "240")
    if not raw.isascii() or not raw.isdigit() or not 1 <= int(raw) <= 600:
        raise ValueError("RF_GEPA_JUDGE_TIMEOUT_SECONDS must be an integer in 1..600")
    return {"planner": 90, "judge": int(raw), "reflection": 90}


def generation_settings(output):
    from research_engine.llm_factory import _ROLE_CONFIG

    return {
        "planner": dict(_ROLE_CONFIG["planner"]),
        "judge": {**_ROLE_CONFIG["critic"], "max_tokens": output},
        "reflection": {**_ROLE_CONFIG["planner"], "max_tokens": 4096},
    }


def freeze(mode, dataset, max_evals, proposals, seed):
    if mode == "live":
        from evals.agent_eval_ext.planner_evolution.__main__ import _live_routes
        from evals.judge_config import judge_max_output_tokens

        _, routes = _live_routes(None, None)  # configuration only, never invokes a model
        for route in [routes["pipeline"]["planner"], routes["judge"], routes["reflection"]]:
            if route.partition(":")[0] not in {"custom", "openai", "openrouter", "ollama"}:
                raise ValueError("strict_budget_provider_not_supported")
        output = judge_max_output_tokens()
    else:
        routes = {
            "pipeline": {r: "fake:planner" for r in ROLES},
            "judge": "fake:judge",
            "reflection": "fake:reflection",
        }
        output = 8192
    return {
        "metric_version": "v3",
        "metric_rules_sha256": rules_hash(),
        "min_gain": 0.05,
        "baseline_sha256": sha256_text(PLANNER_PROMPT_V2),
        "dataset_sha256": input_hash(dataset),
        "routes": routes,
        "judge_settings": {
            "rubric_sha256": input_hash(PLANNER_JUDGE_PROMPT_V2),
            "output_tokens": output,
        },
        "limits": {
            "evaluations": max_evals,
            "proposals": proposals,
            "seed": seed,
            "minimum_gain": 0.05,
        },
        "generation": generation_settings(output),
        "request_timeouts": request_timeouts(),
    }


def search_dataset(source, preset):
    raw = {**source, "cases": [c for c in source["cases"] if c["split"] in {"train", "validation"}]}
    if preset == "smoke":
        raw["cases"] = [
            c
            for split in ("train", "validation")
            for c in [r for r in raw["cases"] if r["split"] == split][:2]
        ]
    raw["dataset_id"] = source["dataset_id"][:60] + "-search-" + preset
    return Dataset.model_validate(raw).model_dump()


def budget_view(experiment_dir: Path, request: dict):
    expected = request.get("budget")
    if expected is None:
        return None
    folder = experiment_dir / "paid-budget"
    if not folder.exists():
        return {
            "configured": expected,
            "used_total": 0,
            "used_by_role": {"planner": 0, "judge": 0, "reflection": 0},
            "remaining_total": expected["total_cap"],
            "unknown": [],
            "token_scope": "reported_planner_judge_only; reflection_not_recorded",
            "simulated": request["mode"] == "demo",
        }
    try:
        state = PaidCallBudget(folder, BudgetConfig(**expected)).state()
        details = []
        for slot in state["unknown"]:
            events = folder / "slots" / (slot + ".events.jsonl")
            records = (
                [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
                if events.is_file()
                else []
            )
            reason = next(
                (e.get("reason", "") for e in reversed(records) if e["status"] == "UNKNOWN"), ""
            )
            cause = reason.split(":", 1)[0]
            details.append(
                {
                    "slot_id": slot,
                    "role": slot.rsplit("-", 1)[0],
                    "cause_type": cause if re.fullmatch(r"[A-Za-z0-9_]{1,80}", cause) else None,
                }
            )
        state["unknown_details"] = details
        return {
            "configured": expected,
            **state,
            "token_scope": "reported_planner_judge_only; reflection_not_recorded",
            "simulated": request["mode"] == "demo",
        }
    except OSError, ValueError, KeyError:
        return {
            "configured": expected,
            "error": "budget_unreadable",
            "used_total": None,
            "remaining_total": None,
            "unknown": None,
            "simulated": request["mode"] == "demo",
        }


def quality_comparison(result):
    search = result.get("search", {})
    arms = {}
    for name in ("baseline", "candidate"):
        rows = search.get("validation_" + name, [])
        scores = []
        for row in rows:
            try:
                score = case_score_v3(Observation.model_validate(row)) if row else None
            except ValueError:
                score = None
            scores.append(score)
        arms[name] = (
            {
                key: mean(s[key] for s in scores)
                for key in ("coverage", "specificity", "plan_quality", "redundancy", "quality")
            }
            if scores and all(s is not None for s in scores)
            else None
        )
    return {
        **arms,
        "quality_delta": arms["candidate"]["quality"] - arms["baseline"]["quality"]
        if all(arms.values())
        else None,
        "same_prompt": (result.get("baseline") or {}).get("sha256")
        == (result.get("candidate") or {}).get("sha256"),
    }
