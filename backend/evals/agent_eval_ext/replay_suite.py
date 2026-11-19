"""Process-isolated suites over the unchanged v0.5 controlled replay contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from .artifacts import check_output_path, save_artifact
from .replay import ReplayCase
from .trace_inspect import inspect_trace

ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
CASE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}\.json$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_BACKEND = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_snapshot(path):
    manifest = Path(path).resolve()
    data = manifest.read_bytes()
    raw = json.loads(data.decode("utf-8-sig"))
    if (
        not isinstance(raw, dict)
        or type(raw.get("schema_version")) is not int
        or raw["schema_version"] != 1
    ):
        raise ValueError("suite manifest requires schema_version=1")
    if set(raw) != {"schema_version", "cases"}:
        raise ValueError("unknown suite manifest fields")
    names = raw["cases"]
    if not isinstance(names, list) or not 1 <= len(names) <= 100:
        raise ValueError("cases must contain 1..100 filenames")
    seen, result = set(), []
    for name in names:
        if not isinstance(name, str) or not CASE_NAME.fullmatch(name):
            raise ValueError("suite case must be a filename in the manifest directory")
        source = (manifest.parent / name).resolve()
        if source.parent != manifest.parent:
            raise ValueError("suite case resolves outside the manifest directory")
        content = source.read_bytes()
        case = ReplayCase.from_dict(json.loads(content.decode("utf-8-sig")))
        if case.case_id in seen:
            raise ValueError("duplicate case_id")
        seen.add(case.case_id)
        result.append((case, name, source, hashlib.sha256(content).hexdigest(), content))
    return data, result


def load_manifest(path: str | Path) -> list[tuple[str, Path, str]]:
    _, cases = _manifest_snapshot(path)
    return [(case.case_id, source, digest) for case, _, source, digest, _ in cases]


def _save_exclusive(path: Path, payload: dict[str, Any]) -> None:
    save_artifact(payload, path)


def _clean_environment() -> dict[str, str]:
    blocked = {
        "CUSTOM_BASE_URL",
        "EVAL_JUDGE_BASE_URL",
        "DATABASE_URL",
        "REDIS_URL",
        "JWT_SECRET_KEY",
        "RF_JUDGE_BATCH_TRACE_DIR",
        "RF_EVAL_TRACE_DIR",
        "RF_EVAL_TRACE",
        "RF_EVAL_EXECUTION_ROOT",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
    }
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().endswith("_API_KEY") and k.upper() not in blocked
    }
    env.update(
        LLM_MODE="fake",
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        OTEL_SDK_DISABLED="true",
    )
    return env


def _artifact(root: Path, relative: Any) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or PureWindowsPath(relative).drive
        or ".." in relative.replace("\\", "/").split("/")
    ):
        raise ValueError("invalid_artifact_path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("artifact_missing_or_outside_case")
    return path


def _verify_result(case: ReplayCase, digest: str, detail: Any, case_dir: Path) -> dict:
    if not isinstance(detail, dict):
        raise ValueError("invalid_result_shape")
    if (
        detail.get("case_id") != case.case_id
        or detail.get("case_file_sha256") != digest
        or detail.get("replay_type") != "controlled_simulation"
        or detail.get("historical_replay") is not False
        or detail.get("llm_mode") != "fake"
        or detail.get("fault_kind") != case.fault_kind
    ):
        raise ValueError("result_identity_mismatch")
    spec_sha = hashlib.sha256(json.dumps(case.as_dict(), sort_keys=True).encode()).hexdigest()
    if detail.get("case_spec_sha256") != spec_sha:
        raise ValueError("case_spec_mismatch")
    if (
        type(detail.get("schema_version")) is not int
        or detail["schema_version"] != 1
        or type(detail.get("passed")) is not bool
    ):
        raise ValueError("invalid_result_schema_or_verdict")
    assertions = detail.get("assertions")
    required = {"status", "planner_calls", "error_contains", "trace_available", "terminal_trace"}
    if (
        not isinstance(assertions, dict)
        or not required.issubset(assertions)
        or any(v is not None and type(v) is not bool for v in assertions.values())
    ):
        raise ValueError("invalid_result_assertions")
    calls, status = detail.get("planner_calls"), detail.get("status")
    if (
        type(calls) is not int
        or calls < 0
        or status
        not in {
            "failed",
            "awaiting_approval",
            "awaiting_plan",
            "completed",
            "unknown",
            "execution_error",
        }
    ):
        raise ValueError("invalid_result_observations")
    if assertions["status"] is not (status == case.expected_status):
        raise ValueError("status_assertion_mismatch")
    expected_calls = (
        None if case.expected_planner_calls is None else calls == case.expected_planner_calls
    )
    if assertions["planner_calls"] is not expected_calls:
        raise ValueError("count_assertion_mismatch")
    if (case.error_contains is None and assertions["error_contains"] is not None) or (
        case.error_contains is not None and type(assertions["error_contains"]) is not bool
    ):
        raise ValueError("error_assertion_mismatch")
    if detail["passed"] is not all(v is not False for v in assertions.values()):
        raise ValueError("result_verdict_disagrees_with_assertions")
    trace = _artifact(case_dir, detail.get("trace_path"))
    inspection = inspect_trace(trace)
    if (
        not inspection["trace_complete"]
        or inspection["case_id"] != case.case_id
        or inspection["run_id"] != detail.get("run_id")
        or inspection["thread_id"] != detail.get("run_id")
        or inspection != detail.get("trace_inspection")
        or detail.get("trace_status") != "recorded"
    ):
        raise ValueError("trace_identity_or_integrity_mismatch")
    events = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    observed_calls = sum(
        e["kind"] == "llm" and e["status"] != "start" and e["metadata"].get("role") == "planner"
        for e in events
    )
    if observed_calls != calls or assertions["trace_available"] is not True:
        raise ValueError("trace_planner_count_or_availability_mismatch")
    terminal = {
        "failed": "failer",
        "awaiting_approval": "hitl_gate",
        "awaiting_plan": "plan_gate",
        "completed": "finalizer",
    }.get(status)
    has_terminal = bool(terminal and inspection["node_terminals"].get(terminal))
    if assertions["terminal_trace"] is not has_terminal:
        raise ValueError("terminal_trace_mismatch")
    hashes = {detail["trace_path"]: sha256_file(trace)}
    bad_path = detail.get("bad_case_path")
    if status == "failed" and not bad_path:
        raise ValueError("bad_case_missing")
    if bad_path is not None:
        bad = _artifact(case_dir, bad_path)
        record = json.loads(bad.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or any(
            record.get(k) != inspection[k] for k in ("case_id", "run_id", "trace_id")
        ):
            raise ValueError("bad_case_identity_mismatch")
        hashes[bad_path] = sha256_file(bad)
    return hashes


def run_suite(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    timeout_s: int = 180,
    revision: str = "unspecified",
) -> dict[str, Any]:
    if type(timeout_s) is not int or not 1 <= timeout_s <= 3600:
        raise ValueError("timeout_s must be an integer 1..3600")
    if not isinstance(revision, str) or not revision.strip() or len(revision) > 128:
        raise ValueError("revision must be a nonempty <=128 character string")
    manifest_data, cases = _manifest_snapshot(manifest)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    check_output_path(out / "suite.json")
    inputs = out / "inputs"
    inputs.mkdir()
    with (inputs / "manifest.json").open("xb") as stream:
        stream.write(manifest_data)
    for _, name, _, _, content in cases:
        with (inputs / name).open("xb") as stream:
            stream.write(content)
    entries = []
    for case, name, _, digest, _ in cases:
        case_dir = out / case.case_id
        case_dir.mkdir()
        report = case_dir / "result.json"
        command = [
            sys.executable,
            "-m",
            "evals.agent_eval_ext.replay_cli",
            "--case",
            str(inputs / name),
            "--out",
            str(report),
            "--trace-dir",
            str(case_dir / "traces"),
        ]
        process_rc, failure, detail, verified, hashes = None, None, None, False, {}
        try:
            completed = subprocess.run(
                command,
                check=False,
                timeout=timeout_s,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_clean_environment(),
                cwd=_BACKEND,
                shell=False,
            )
            process_rc = completed.returncode
        except subprocess.TimeoutExpired:
            failure = "subprocess_timeout"
        except OSError as exc:
            failure = f"subprocess_launch_failure:{type(exc).__name__}"
        try:
            if case_dir.resolve().parent != out:
                raise ValueError("case_directory_changed")
            if not report.is_file():
                raise ValueError("result_missing")
            _artifact(case_dir, "result.json")
            detail = json.loads(report.read_text(encoding="utf-8"))
            hashes = _verify_result(case, digest, detail, case_dir)
            hashes["result.json"] = sha256_file(report)
            if sha256_file(inputs / name) != digest:
                raise ValueError("case_snapshot_changed")
            verified = True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Preserve a timeout/launch failure; malformed output cannot erase that evidence.
            reason = str(exc) if type(exc) is ValueError else f"invalid_result:{type(exc).__name__}"
            allowed = {
                "invalid_result_shape",
                "result_identity_mismatch",
                "case_spec_mismatch",
                "invalid_result_schema_or_verdict",
                "invalid_result_assertions",
                "invalid_result_observations",
                "status_assertion_mismatch",
                "count_assertion_mismatch",
                "error_assertion_mismatch",
                "result_verdict_disagrees_with_assertions",
                "trace_identity_or_integrity_mismatch",
                "trace_planner_count_or_availability_mismatch",
                "terminal_trace_mismatch",
                "bad_case_missing",
                "bad_case_identity_mismatch",
                "invalid_artifact_path",
                "artifact_missing_or_outside_case",
                "case_snapshot_changed",
                "result_missing",
            }
            failure = failure or (
                reason if reason in allowed else f"invalid_result:{type(exc).__name__}"
            )
        result_passed = (
            detail.get("passed")
            if isinstance(detail, dict) and type(detail.get("passed")) is bool
            else None
        )
        if verified and process_rc in (0, 1) and process_rc != (0 if result_passed else 1):
            failure = failure or "process_result_disagreement"
        entries.append(
            {
                "case_id": case.case_id,
                "case_sha256": digest,
                "fault_kind": case.fault_kind,
                "passed": verified
                and result_passed is True
                and process_rc == 0
                and failure is None,
                "result_passed": result_passed,
                "result_verified": verified,
                "status": detail.get("status") if verified else None,
                "process_exit_code": process_rc,
                "harness_failure": failure,
                "result_path": f"{case.case_id}/result.json" if report.is_file() else None,
                "artifact_sha256": hashes if verified else {},
            }
        )
    summary = {
        "schema_version": 1,
        "kind": "researchflow_replay_suite",
        "suite_id": f"suite-{uuid.uuid4().hex[:12]}",
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "revision_supplied": revision,
        "revision_verified": False,
        "replay_type": "controlled_simulation",
        "historical_replay": False,
        "network_attempts_measured": False,
        "execution": "one_sequential_subprocess_per_case",
        "python_version": sys.version.split()[0],
        "total": len(entries),
        "passed_count": sum(x["passed"] for x in entries),
        "all_passed": all(x["passed"] for x in entries),
        "cases": entries,
    }
    _save_exclusive(out / "suite.json", summary)
    return summary


def _validate_suite(doc):
    if (
        not isinstance(doc, dict)
        or doc.get("kind") != "researchflow_replay_suite"
        or type(doc.get("schema_version")) is not int
        or doc["schema_version"] != 1
    ):
        raise ValueError("input is not a replay suite result")
    if (
        doc.get("replay_type") != "controlled_simulation"
        or doc.get("historical_replay") is not False
    ):
        raise ValueError("unsupported suite execution type")
    rows = doc.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("suite cases must be nonempty")
    if not isinstance(doc.get("manifest_sha256"), str) or not _SHA.fullmatch(
        doc["manifest_sha256"]
    ):
        raise ValueError("missing or invalid manifest fingerprint")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed suite case")
        ident, digest = row.get("case_id"), row.get("case_sha256")
        if not isinstance(ident, str) or not ID.fullmatch(ident) or ident in seen:
            raise ValueError("duplicate or malformed case IDs")
        seen.add(ident)
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ValueError("missing or invalid case fingerprint")
        if type(row.get("passed")) is not bool or type(row.get("result_verified")) is not bool:
            raise ValueError("malformed suite verdicts")
        rc, failure, result_passed = (
            row.get("process_exit_code"),
            row.get("harness_failure"),
            row.get("result_passed"),
        )
        if (
            rc is not None
            and type(rc) is not int
            or failure is not None
            and not isinstance(failure, str)
            or result_passed is not None
            and type(result_passed) is not bool
        ):
            raise ValueError("malformed execution metadata")
        effective = row["result_verified"] and result_passed is True and rc == 0 and failure is None
        if row["passed"] is not effective:
            raise ValueError("suite verdict contradicts execution metadata")
    if (
        type(doc.get("total")) is not int
        or doc["total"] != len(rows)
        or type(doc.get("passed_count")) is not int
        or doc["passed_count"] != sum(x["passed"] for x in rows)
        or type(doc.get("all_passed")) is not bool
        or doc["all_passed"] is not all(x["passed"] for x in rows)
    ):
        raise ValueError("suite aggregate contradicts case verdicts")


def compare_suites(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    _validate_suite(baseline)
    _validate_suite(candidate)
    b, c = ({x["case_id"]: x for x in doc["cases"]} for doc in (baseline, candidate))
    common = {
        "scope": "Case-contract pass/fail metadata only, not live model quality or a controlled code experiment",
        "code_revision_verified": False,
        "artifact_verification_repeated": False,
        "baseline_revision_supplied": baseline.get("revision_supplied"),
        "candidate_revision_supplied": candidate.get("revision_supplied"),
    }
    if set(b) != set(c):
        return {
            **common,
            "comparable": False,
            "reason": "case_set_mismatch",
            "missing_in_candidate": sorted(set(b) - set(c)),
            "added_in_candidate": sorted(set(c) - set(b)),
            "regressions": None,
            "recoveries": None,
            "quality_gate_passed": False,
        }
    changed = [k for k in b if b[k]["case_sha256"] != c[k]["case_sha256"]]
    if changed:
        return {
            **common,
            "comparable": False,
            "reason": "case_bytes_mismatch",
            "changed_case_ids": sorted(changed),
            "regressions": None,
            "recoveries": None,
            "quality_gate_passed": False,
        }
    changes = [
        {
            "case_id": k,
            "baseline_passed": b[k]["passed"],
            "candidate_passed": c[k]["passed"],
            "change": "regressed"
            if b[k]["passed"] and not c[k]["passed"]
            else "recovered"
            if not b[k]["passed"] and c[k]["passed"]
            else "unchanged",
        }
        for k in sorted(b)
    ]
    return {
        **common,
        "comparable": True,
        "manifest_bytes_match": baseline["manifest_sha256"] == candidate["manifest_sha256"],
        "regressions": sum(x["change"] == "regressed" for x in changes),
        "recoveries": sum(x["change"] == "recovered" for x in changes),
        "candidate_passed": sum(x["candidate_passed"] for x in changes),
        "total": len(changes),
        "quality_gate_passed": all(x["candidate_passed"] for x in changes),
        "changes": changes,
    }


def _verify_saved_suite(path: Path, doc: dict) -> None:
    root = path.resolve().parent
    if (root / "inputs").resolve().parent != root:
        raise ValueError("saved inputs escape suite directory")
    manifest_bytes, snapshots = _manifest_snapshot(root / "inputs/manifest.json")
    if hashlib.sha256(manifest_bytes).hexdigest() != doc["manifest_sha256"]:
        raise ValueError("saved manifest changed")
    by_id = {case.case_id: (case, digest) for case, _, _, digest, _ in snapshots}
    if set(by_id) != {row["case_id"] for row in doc["cases"]}:
        raise ValueError("saved case set changed")
    for row in doc["cases"]:
        case, digest = by_id[row["case_id"]]
        if digest != row["case_sha256"]:
            raise ValueError("saved case bytes changed")
        if not row["result_verified"]:
            continue
        case_dir = root / case.case_id
        if case_dir.resolve().parent != root:
            raise ValueError("saved case directory changed")
        if row.get("result_path") != f"{case.case_id}/result.json":
            raise ValueError("saved result link changed")
        result = _artifact(case_dir, "result.json")
        detail = json.loads(result.read_text(encoding="utf-8"))
        hashes = _verify_result(case, digest, detail, case_dir)
        hashes["result.json"] = sha256_file(result)
        if (
            hashes != row.get("artifact_sha256")
            or detail["passed"] is not row["result_passed"]
            or detail["status"] != row["status"]
        ):
            raise ValueError("saved result artifacts changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or compare offline controlled replay suites")
    sub = parser.add_subparsers(dest="command", required=True)
    valid = sub.add_parser("validate")
    valid.add_argument("--manifest", required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--timeout", type=int, default=180)
    run.add_argument("--revision", default="unspecified")
    comp = sub.add_parser("compare")
    comp.add_argument("--baseline", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "cases": [
                        {"case_id": ident, "case_sha256": sha}
                        for ident, _, sha in load_manifest(args.manifest)
                    ]
                },
                indent=2,
            )
        )
        return 0
    if args.command == "run":
        result = run_suite(
            args.manifest, args.out_dir, timeout_s=args.timeout, revision=args.revision
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["all_passed"] else 1
    baseline_path, candidate_path = Path(args.baseline), Path(args.candidate)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    result = compare_suites(baseline, candidate)
    if result["comparable"]:
        try:
            _verify_saved_suite(baseline_path, baseline)
            _verify_saved_suite(candidate_path, candidate)
            result.update(artifact_verification_repeated=True, artifact_integrity_passed=True)
        except OSError, ValueError, KeyError, TypeError:
            result.update(
                comparable=False,
                reason="saved_artifact_validation_failed",
                quality_gate_passed=False,
                regressions=None,
                recoveries=None,
                artifact_verification_repeated=True,
                artifact_integrity_passed=False,
            )
    if args.out:
        _save_exclusive(Path(args.out), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["comparable"] and result["quality_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
