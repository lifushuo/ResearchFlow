"""Exercise workflow shell control flow; Docker is a boundary stub, not an Agent result."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github/workflows/agent-evaluation.yml"


def steps():
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["jobs"][
        "deterministic-eval"
    ]["steps"]


def run_step(tmp_path, name, *, exit_code=0, artifact=True, export_code=0):
    step = next(s for s in steps() if s["name"] == name)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        + """
import json,os,pathlib,subprocess,sys,shutil
args=sys.argv[1:]
volume=pathlib.Path(os.environ['GITHUB_WORKSPACE'])/'volume'
if args[0] in ('rm','volume'): sys.exit(0)
if args[0]=='cp':
    rc=int(os.environ['STUB_EXPORT_EXIT'])
    if rc: sys.exit(rc)
    source=volume/('replay' if '/replay/' in args[1] else 'tools')
    if not source.exists(): sys.exit(1)
    shutil.copytree(source,args[2],dirs_exist_ok=True)
    sys.exit(0)
if args[0]!='run': sys.exit(99)
assert '--read-only' in args and args[args.index('--network')+1]=='none'
mounts=[args[i+1] for i,a in enumerate(args) if a=='--mount']
output=str(volume) if any('target=/app/data' in x for x in mounts) else None
if 'evals.agent_eval_ext.unified_cli' in args:
    if os.environ['STUB_ARTIFACT']=='1':
        out=pathlib.Path(output)/'replay'
        out.mkdir(parents=True)
        (out/'unified.json').write_text(json.dumps({'test_fixture':True}))
    sys.exit(int(os.environ['STUB_EXIT']))
if 'sh' in args and output:
    script=args[-1].replace('/app/data/',output+'/')
    sys.exit(subprocess.run(['sh','-c',script],cwd=os.environ['REAL_BACKEND']).returncode)
sys.exit(int(os.environ['STUB_EXIT']))
""",
        encoding="utf-8",
    )
    (tmp_path / "agent-eval-artifacts").mkdir()
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "GITHUB_WORKSPACE": str(tmp_path),
        "GITHUB_SHA": "unverified-test-label",
        "GITHUB_RUN_ID": "unit",
        "GITHUB_RUN_ATTEMPT": "1",
        "STUB_EXIT": str(exit_code),
        "STUB_EXPORT_EXIT": str(export_code),
        "STUB_ARTIFACT": "1" if artifact else "0",
        "REAL_BACKEND": str(REPO / "backend"),
    }
    # Docker tmpfs is noexec; invoke the boundary stub through the installed interpreter.
    boundary = f'docker() {{ {shlex.quote(sys.executable)} {shlex.quote(str(docker))} "$@"; }}\n'
    return subprocess.run(
        ["bash", "-c", boundary + step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_workflow_yaml_permissions_and_upload_contract():
    doc = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(doc["on"]) == {"pull_request", "workflow_dispatch"}
    assert doc["permissions"] == {"contents": "read"}
    assert len(doc["jobs"]) == 1
    job = doc["jobs"]["deterministic-eval"]
    assert job["name"] == "Agent evaluation offline gate"
    upload = next(s for s in job["steps"] if "upload-artifact@" in s.get("uses", ""))
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"
    assert not any(s.get("continue-on-error") for s in job["steps"])


def test_all_embedded_shell_parses():
    for step in steps():
        if "run" in step:
            subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)


@pytest.mark.parametrize("exit_code", [0, 1, 2, 125])
def test_workflow_propagates_replay_exit_and_keeps_artifacts(tmp_path, exit_code):
    process = run_step(tmp_path, "Run compiled-graph Replay quality gate", exit_code=exit_code)
    assert process.returncode == exit_code
    root = tmp_path / "agent-eval-artifacts"
    assert (root / "replay/unified.json").is_file()
    assert json.loads((root / "ci-status.json").read_text())["run_exit_code"] == exit_code


def test_missing_success_artifact_fails_workflow(tmp_path):
    assert (
        run_step(tmp_path, "Run compiled-graph Replay quality gate", artifact=False).returncode == 1
    )


def test_test_failure_is_not_masked_by_tee(tmp_path):
    assert run_step(tmp_path, "Offline evaluator checks", exit_code=1).returncode == 1


def test_tool_gate_runs_real_scorer_positive_and_negative_controls(tmp_path):
    process = run_step(tmp_path, "Controlled tool contract gate")
    assert process.returncode == 0, process.stderr
    output = tmp_path / "agent-eval-artifacts/tools"
    assert json.loads((output / "pass.json").read_text())["gate_passed"] is True
    assert json.loads((output / "negative-control.json").read_text())["gate_passed"] is False


@pytest.mark.parametrize("run_code,expected", [(0, 1), (2, 2)])
def test_export_failure_blocks_success_without_erasing_original_error(tmp_path, run_code, expected):
    process = run_step(
        tmp_path, "Run compiled-graph Replay quality gate", exit_code=run_code, export_code=1
    )
    assert process.returncode == expected
    state = json.loads((tmp_path / "agent-eval-artifacts/ci-status.json").read_text())
    assert state["export_exit_code"] == 1
    assert state["run_exit_code"] == run_code
    assert state["gate_exit_code"] == expected
