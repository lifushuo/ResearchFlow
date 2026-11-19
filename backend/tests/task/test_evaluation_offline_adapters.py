"""Fixed offline adapters remain usable without the retired single-job queue."""

from pathlib import Path

from evals.agent_eval_ext import evaluation_job_worker as worker


def test_environment_is_explicit_allowlist(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sample-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sample-secret2")
    env = worker._env()
    assert "OPENAI_API_KEY" not in env and "DEEPSEEK_API_KEY" not in env
    assert env["LLM_MODE"] == "fake"
    assert (Path(env["PYTHONPATH"]) / "evals").is_dir()


def test_worker_missing_result_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "_run_command", lambda _command: 1)
    got = worker._execute(tmp_path, {"dataset_id": "planner-replay-v1"})
    assert got == {"state": "failed", "failure_code": "result_missing"}


def test_runner_validation_does_not_invent_pass(tmp_path, monkeypatch):
    def bad_result(_command):
        (tmp_path / "artifacts/tool-result.json").write_text('{"cases_total":4}', encoding="utf8")
        return 0

    monkeypatch.setattr(worker, "_run_command", bad_result)
    got = worker._execute(tmp_path, {"dataset_id": "tool-contract-v1"})
    assert got == {"state": "failed", "failure_code": "result_validation_failed"}
