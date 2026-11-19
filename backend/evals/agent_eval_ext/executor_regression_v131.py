"""The seven controlled Executor regression cases, run via a FIXED pytest allowlist.

Unlike an artifact-only demo this executes ResearchFlow's actual Executor code.
Scripted LLM and HTTP mocks are deliberate; PASS is a regression-test verdict,
not empirical real-model tool accuracy. The browser cannot pick test paths.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "fixtures" / "executor_suite_v131.json"
FIXED_TEST_FILE = "tests/task/test_executor_eval_v131_cases.py"
EXPECTED = (
    "search-read-submit",
    "malformed-url",
    "blocked-url",
    "tool-timeout",
    "duplicate-read",
    "duplicate-submit",
    "cancel",
)


def _cases() -> list[dict[str, str]]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = data.get("cases")
    if (
        data.get("schema_version") != 1
        or not isinstance(rows, list)
        or tuple(x.get("case_id") for x in rows) != EXPECTED
    ):
        raise ValueError("executor_manifest_invalid")
    for row in rows:
        test = row.get("test")
        if not isinstance(test, str) or not test.startswith(FIXED_TEST_FILE + "::test_"):
            raise ValueError("executor_test_not_allowlisted")
        if not test.rsplit("::", 1)[-1].replace("_", "").isalnum():
            raise ValueError("executor_test_identifier_invalid")
    if len({x["test"] for x in rows}) != len(rows):
        raise ValueError("duplicate_executor_test")
    test_file = ROOT / FIXED_TEST_FILE
    if test_file.is_symlink() or not test_file.is_file():
        raise ValueError("executor_test_source_missing")
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not all(x["test"].split("::")[-1] in functions for x in rows):
        raise ValueError("executor_test_function_missing")
    return rows


def _parse_xml(path: Path, cases: list[dict]) -> list[dict]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("executor_junit_unavailable")
    root = ET.fromstring(path.read_bytes())
    observed = {}
    for node in root.iter("testcase"):
        name = node.attrib.get("name")
        if not name or name in observed:
            raise ValueError("executor_junit_duplicate_case")
        observed[name] = node
    expected = [row["test"].split("::")[-1] for row in cases]
    if set(observed) != set(expected) or len(observed) != len(expected):
        raise ValueError("executor_junit_case_set_mismatch")
    return [
        {
            "case_id": row["case_id"],
            "passed": not any(
                observed[row["test"].split("::")[-1]].find(x) is not None
                for x in ("failure", "error", "skipped")
            ),
        }
        for row in cases
    ]


def run(output: Path, *, timeout: int = 180) -> dict:
    if not 20 <= timeout <= 600 or output.exists() or output.is_symlink():
        raise ValueError("invalid_executor_output_or_timeout")
    if os.getenv("LLM_MODE") != "fake":
        raise ValueError("executor_regression_requires_fake_environment")
    rows = _cases()
    output.parent.mkdir(parents=True, exist_ok=True)
    junit = output.parent / "executor-private-junit.xml"
    if junit.exists() or junit.is_symlink():
        raise ValueError("executor_junit_output_already_exists")
    env = {
        "PATH": os.getenv("PATH", "/venv/bin:/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": str(ROOT),
        "LLM_MODE": "fake",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "RF_EXECUTOR_CASE_TRACE_DIR": str(output.parent / "traces"),
        "RF_EVAL_TRACE_RUN_ID": os.environ.get("RF_EVAL_TRACE_RUN_ID") or output.parent.name,
    }
    # pytest-asyncio plugin is explicitly loaded; all other auto-loaded plugins
    # are excluded to keep this fixed suite deterministic.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "pytest_asyncio.plugin",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "--junitxml=" + str(junit),
        *[x["test"] for x in rows],
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("executor_test_timeout") from exc
    if proc.returncode not in (0, 1):
        raise RuntimeError("executor_pytest_error")
    verdicts = _parse_xml(junit, rows)
    passed = sum(x["passed"] for x in verdicts)
    data = {
        "schema_version": 1,
        "dataset_id": "executor-regression-v1",
        "evidence_kind": "controlled_executor_regression",
        "real_model_executed": False,
        "paid_api_used": False,
        "dataset_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "case_count": len(verdicts),
        "passed_count": passed,
        "gate_passed": passed == len(verdicts),
        "cases": verdicts,
    }
    with output.open("x", encoding="utf-8") as target:
        json.dump(data, target, ensure_ascii=False, allow_nan=False, indent=2)
        target.write("\n")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.output)
        print(
            json.dumps(
                {"passed_count": result["passed_count"], "case_count": result["case_count"]},
                ensure_ascii=False,
            )
        )
        return 0 if result["gate_passed"] else 1
    except (OSError, ValueError, RuntimeError, ET.ParseError) as exc:
        print("executor_regression_failed: " + type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
