"""Trusted per-step ingestion. No historical scan and no browser-supplied verdicts.

Each adapter states its oracle. Retrieval records numeric observations without inventing
per-query pass thresholds. Report's contract is harness completion, not overall quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.evaluation_closure import EvaluationClosure, input_hash
from app.services.evaluation_closure.storage import store_at

BACKEND = Path(__file__).resolve().parents[2]
FIXTURES = BACKEND / "evals/agent_eval_ext/fixtures"


def read(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("invalid_closure_artifact")
    return json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")),
    )


def sources(
    dataset: str, limit: int, *, case_id: str | None = None
) -> tuple[dict[str, dict], dict[str, str]]:
    if case_id is not None:
        from evals.query_selection import validate_report_selection

        validate_report_selection([dataset], limit, case_id)
    paths: list[Path]
    if dataset == "planner-replay-v1":
        manifest = FIXTURES / "replay_suite_v1.json"
        paths = [manifest, *[manifest.parent / p for p in read(manifest)["cases"]]]
        rows = [read(p) for p in paths[1:]]
    elif dataset == "executor-regression-v1":
        paths = [FIXTURES / "executor_suite_v131.json"]
        rows = read(paths[0])["cases"]
    elif dataset == "tool-contract-v1":
        paths = [FIXTURES / "tool_gold_v1.json", FIXTURES / "tool_observed_pass_v1.json"]
        gold = read(paths[0])["cases"]
        observed = {r["case_id"]: r for r in read(paths[1])["cases"]}
        rows = [
            {"case_id": r["case_id"], "gold": r, "observed": observed.get(r["case_id"])}
            for r in gold
        ]
    elif dataset == "judge-calibration":
        paths = [FIXTURES / "judge_calibration.jsonl"]
        rows = [
            json.loads(line)
            for line in paths[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        relative = {
            "report-benchmark": "evals/queries.json",
            "corpus-retrieval": "evals/datasets/retrieval/queries.json",
            "selection-retrieval": "evals/datasets/selection-retrieval/queries.json",
            "grounding-v1": "evals/datasets/grounding/v1/items.json",
        }[dataset]
        paths = [BACKEND / relative]
        obj = read(paths[0])
        rows = obj if isinstance(obj, list) else obj.get("queries", obj.get("items"))
        if dataset in ("corpus-retrieval", "selection-retrieval"):
            paths.extend(sorted((paths[0].parent / "corpus").glob("*")))
        if dataset == "grounding-v1":
            paths.append(paths[0].with_name("labels.json"))
            paths.extend(sorted((paths[0].parent / "documents").glob("*")))
    inventory = {
        p.relative_to(BACKEND).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file()
    }
    if case_id is not None:
        from evals.query_selection import select_queries

        rows = select_queries(rows, case_id=case_id)
    elif limit and dataset in ("report-benchmark", "grounding-v1", "judge-calibration"):
        rows = rows[:limit]
    cases = {}
    for row in rows:
        ident = row.get("case_id", row.get("id"))
        if not isinstance(ident, str) or ident in cases:
            raise ValueError("invalid_source_case_identity")
        cases[ident] = {"case": row, "dataset_sources": inventory}
    return cases, inventory


@dataclass
class Capture:
    store: EvaluationClosure
    run_id: str
    owner: str
    dataset: str
    limit: int
    inputs: dict[str, dict]
    inventory: dict[str, str]
    folder: Path
    case_id: str | None = None


def start(root: Path, folder: Path, request: dict, index: int) -> Capture:
    dataset = request["datasets"][index]
    if (folder / "artifacts" / dataset).exists():
        raise ValueError("closure_requires_new_execution_directory")
    case_id = request.get("case_id")
    case_kwargs = {"case_id": case_id} if case_id is not None else {}
    inputs, inventory = sources(dataset, request["max_cases"], **case_kwargs)
    mode = (
        "synthetic"
        if dataset == "tool-contract-v1"
        else "fake"
        if dataset in ("planner-replay-v1", "executor-regression-v1")
        else "real"
    )
    configured = {
        key: os.getenv(key, "")
        for key in (
            "MODEL_PLANNER",
            "MODEL_EXECUTOR",
            "MODEL_CRITIC",
            "MODEL_SYNTHESIZER",
            "EMBEDDING_MODEL",
        )
    }
    from app.services.evaluation_pipeline_v13 import judge_route

    model = (
        "scripted:" + dataset if mode != "real" else "configured-sha256:" + input_hash(configured)
    )
    judge = judge_route() if dataset in ("judge-calibration", "report-benchmark") else "none"
    store = store_at(root)
    run_id = request["pipeline_id"] + "--" + dataset
    # A future scoring-definition change must not compare as the same closure contract.
    from evals.metrics import METRICS_VERSION
    from evals.retrieval_metrics import RETRIEVAL_METRICS_VERSION

    scorer_version = (
        f"report-parser-{METRICS_VERSION}"
        if dataset in ("report-benchmark", "judge-calibration")
        else f"retrieval-{RETRIEVAL_METRICS_VERSION}"
        if dataset in ("corpus-retrieval", "selection-retrieval")
        else "contract-1"
    )
    provenance = {
        "kind": "new_worker_execution",
        "contract": {
            "report-benchmark": "harness_task_completion_only",
            "corpus-retrieval": "numeric_retrieval_observations_no_pass_threshold",
            "selection-retrieval": "zero_leakage_is_necessary_not_sufficient",
            "grounding-v1": "agreement_with_recorded_reference_labels",
            "judge-calibration": "agreement_with_handwritten_labels",
        }.get(dataset, "controlled_contract_not_research_success"),
        "configuration": configured if mode == "real" else {},
        "model_identity_verified": mode != "real",
        "code_revision_verified": False,
        "dataset_sources": inventory,
    }
    store.register_run(
        run_id=run_id,
        owner_id=request["owner"],
        dataset_id=dataset,
        execution_mode=mode,
        sample_set_hash=input_hash({"sources": inventory, "cases": list(inputs)}),
        metrics_version="closure-contract-v1:" + dataset + "/" + scorer_version,
        model_route=model,
        judge_route=judge or "unconfigured",
        source_pipeline_id=request["pipeline_id"],
        expected_case_ids=list(inputs),
        provenance=provenance,
    )
    return Capture(
        store,
        run_id,
        request["owner"],
        dataset,
        request["max_cases"],
        inputs,
        inventory,
        folder,
        case_id,
    )


def numeric(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        k: v
        for k, v in value.items()
        if k not in ("cost_usd", "api_cost_usd", "avg_cost_usd")
        and isinstance(v, (bool, int, float))
        and (not isinstance(v, float) or math.isfinite(v))
    }


def normalize(capture: Capture) -> list[dict]:
    base = capture.folder / "artifacts" / capture.dataset
    dataset = capture.dataset
    if dataset == "planner-replay-v1":
        path = base / "artifacts/result/replay/suite.json"
        doc = read(path)
        from .replay_suite import _verify_saved_suite

        _verify_saved_suite(path, doc)
        rows = doc["cases"]
    elif dataset == "tool-contract-v1":
        path = base / "artifacts/tool-result.json"
        doc = read(path)
        from .tool_safety import _validate_result

        _validate_result(doc)
        if (
            doc["gold_sha256"]
            != capture.inventory["evals/agent_eval_ext/fixtures/tool_gold_v1.json"]
            or doc["observed_sha256"]
            != capture.inventory["evals/agent_eval_ext/fixtures/tool_observed_pass_v1.json"]
        ):
            raise ValueError("tool_input_fingerprint_mismatch")
        rows = doc["cases"]
    elif dataset == "executor-regression-v1":
        path = base / "executor-result.json"
        doc = read(path)
        from .executor_regression_v131 import _cases, _parse_xml

        rows = _parse_xml(base / "executor-private-junit.xml", _cases())
        if (
            rows != doc["cases"]
            or doc["dataset_sha256"]
            != capture.inventory["evals/agent_eval_ext/fixtures/executor_suite_v131.json"]
        ):
            raise ValueError("executor_observation_mismatch")
    else:
        doc = read(base / "raw_result.json")
        key = (
            "results"
            if dataset == "report-benchmark"
            else "claims"
            if dataset == "grounding-v1"
            else "cases"
            if dataset == "judge-calibration"
            else "queries"
        )
        rows = doc[key]
    if (
        not isinstance(rows, list)
        or {r.get("case_id", r.get("id")) for r in rows} != set(capture.inputs)
        or len(rows) != len(capture.inputs)
    ):
        raise ValueError("closure_case_set_mismatch")
    labels = (
        read(BACKEND / "evals/datasets/grounding/v1/labels.json")["labels"]
        if dataset == "grounding-v1"
        else {}
    )
    result = []
    for row in rows:
        ident = row.get("case_id", row.get("id"))
        verdict = "unmeasured"
        code = None
        latency = None
        metrics = numeric(row.get("metrics", {}))
        trace = None
        details = {}
        if dataset in ("planner-replay-v1", "executor-regression-v1", "tool-contract-v1"):
            passed = row.get("pass") if dataset == "tool-contract-v1" else row.get("passed")
            if type(passed) is bool:
                verdict = "passed" if passed else "failed"
                code = None if passed else "contract_assertion_failed"
            if dataset == "planner-replay-v1":
                if row.get("result_verified") is not True:
                    verdict, code = "unmeasured", None
                else:
                    detail = read(path.parent / row["result_path"])
                    trace = (Path(row["result_path"]).parent / detail["trace_path"]).as_posix()
                # Result validation above checked the actual Trace chain, not only its flag.
        elif dataset == "report-benchmark":
            if type(row.get("completed")) is bool:
                verdict = "passed" if row["completed"] else "failed"
                code = None if row["completed"] else "research_incomplete"
            if (
                type(row.get("latency_s")) in (int, float)
                and math.isfinite(row["latency_s"])
                and row["latency_s"] >= 0
            ):
                latency = round(row["latency_s"] * 1000)
            metrics = numeric(row)
        elif dataset == "judge-calibration":
            gold = capture.inputs[ident]["case"]["expected_supported"]
            predicted = row.get("predicted_supported")
            if type(gold) is bool and type(predicted) is bool:
                verdict = "passed" if gold == predicted else "failed"
                code = None if gold == predicted else "judge_label_disagreement"
        elif dataset == "grounding-v1":
            predicted = row.get("verdict")
            if (
                predicted in {"supported", "not_supported", "contradicted"}
                and doc.get("result", {}).get("status") == "measured"
            ):
                verdict = "passed" if predicted == labels[ident] else "failed"
                code = None if verdict == "passed" else "reference_label_disagreement"
        elif dataset in ("corpus-retrieval", "selection-retrieval"):
            metrics = numeric(row.get("with_cap", row))
            if (
                dataset == "selection-retrieval"
                and type(row.get("leakage")) in (int, float)
                and row["leakage"] > 0
            ):
                verdict, code = "failed", "outside_selection"
        result.append(
            {
                "case_id": ident,
                "status": verdict,
                "input_sha256": input_hash(capture.inputs[ident]),
                "error_code": code,
                "latency_ms": latency,
                "metrics": metrics,
                "trace_ref": trace,
                "details": details,
            }
        )
    return result


def _save_case_observations(capture: Capture, rows: list[dict]) -> None:
    # Reuse adapter verdicts (including reference labels) rather than re-grading in the UI.
    public = [
        {
            "case_id": row["case_id"],
            "status": row["status"],
            "latency_s": row["latency_ms"] / 1000 if row.get("latency_ms") is not None else None,
            "metrics": numeric(row.get("metrics", {})),
        }
        for row in rows
    ]
    try:
        file = capture.folder / "artifacts" / capture.dataset / "case-observations.json"
        with file.open("x", encoding="utf-8") as stream:
            json.dump(
                {"schema_version": 1, "dataset_id": capture.dataset, "cases": public},
                stream,
                ensure_ascii=False,
                allow_nan=False,
            )
    except (OSError, ValueError) as exc:
        import structlog

        structlog.get_logger().warning(
            "case_observation_export_failed", error_type=type(exc).__name__
        )


def finish(capture: Capture, result: dict) -> str:
    _, current = sources(
        capture.dataset,
        capture.limit,
        **({"case_id": capture.case_id} if capture.case_id is not None else {}),
    )
    if current != capture.inventory:
        capture.store.finish_run(capture.owner, capture.run_id, complete=False)
        raise ValueError("dataset_changed_during_execution")
    if result["state"] != "completed":
        # Preserve missing scores. Only a single selected input can be tied to
        # this interrupted execution; a batch timeout cannot identify a bad case.
        for ident, fixture in capture.inputs.items():
            capture.store.ingest_case(
                run_id=capture.run_id,
                owner_id=capture.owner,
                case_id=ident,
                status="unmeasured",
                input_sha256=input_hash(fixture),
                error_code=result.get("reason"),
                error_class="execution_incident" if len(capture.inputs) == 1 else None,
                details={
                    "measurement": "unmeasured",
                    "failure_scope": "single_case_execution"
                    if len(capture.inputs) == 1
                    else "batch_execution",
                    "diagnostic_pipeline_id": capture.run_id.split("--", 1)[0],
                    "root_cause_confirmed": False,
                },
            )
        capture.store.finish_run(capture.owner, capture.run_id, complete=False)
        _save_case_observations(
            capture,
            [{"case_id": ident, "status": "unmeasured", "metrics": {}} for ident in capture.inputs],
        )
        return "incomplete"
    rows = normalize(capture)
    for row in rows:
        capture.store.ingest_case(run_id=capture.run_id, owner_id=capture.owner, **row)
    capture.store.finish_run(capture.owner, capture.run_id)
    _save_case_observations(capture, rows)
    return "recorded"
