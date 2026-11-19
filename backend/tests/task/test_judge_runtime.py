from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.agent_eval_ext.judge_runtime import trace_judge_batch, trace_judge_case
from evals.agent_eval_ext.judge_runtime_inspect import summarize


def events(directory: Path) -> list[dict]:
    paths = list(directory.glob("*.jsonl"))
    assert len(paths) == 1
    return [json.loads(x) for x in paths[0].read_text().splitlines()]


def send(*, result=None, text="Claim 1: YES", error=None, count=1):
    trace_judge_batch(
        batch_no=1,
        batch_start=0,
        expected_count=count,
        verdict_by_claim={} if result is None else result,
        response=None if error is not None else SimpleNamespace(),
        text=text,
        error=error,
        elapsed_ms=5.15,
    )


def test_disabled_no_side_effects(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=None):
        send(result={0: True})
    assert not list(tmp_path.iterdir())


def test_complete_yes_and_no_count_as_judged(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(result={0: True, 1: False}, text="Claim 1: YES\nClaim 2: NO", count=2)
    e = events(tmp_path)[1]
    assert (e["status"], e["yes_count"], e["no_count"], e["judged_count"]) == ("complete", 1, 1, 2)
    assert "Claim 1" not in json.dumps(e)


def test_empty_reply(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(text="", count=4)
    assert events(tmp_path)[1]["reason_hint"] == "empty_or_nontext_response"


def test_unparseable_reply(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(text="I agree with everything", count=4)
    assert events(tmp_path)[1]["reason_hint"] == "unparseable_response"


def test_partial_reply(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(result={0: True}, count=4)
    assert (events(tmp_path)[1]["status"], events(tmp_path)[1]["unjudged_count"]) == ("partial", 3)


def test_invoke_exception_metadata_only(tmp_path):
    class RateLimited(Exception):
        status_code = 429

    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(text="", error=RateLimited("SECRET_API_KEY"), count=4)
    e = events(tmp_path)[1]
    assert (e["reason_hint"], e["http_status"], e["exception_type"]) == (
        "invoke_exception",
        429,
        "RateLimited",
    )
    assert "SECRET" not in json.dumps(events(tmp_path))


def test_invalid_and_duplicate_numbers(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(result={0: True}, text="Claim 1: YES\nClaim 1: NO\nClaim 10: YES", count=2)
    e = events(tmp_path)[1]
    assert (e["out_of_range_labels"], e["duplicate_labels"]) == (1, 1)


def test_unique_write_once(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(result={0: True})
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(result={0: True})
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_parallel_context_isolation(tmp_path):
    async def worker(case):
        with trace_judge_case(case_id=case, trace_dir=tmp_path):
            await asyncio.sleep(0)
            send(result={0: True})

    async def run_both():
        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(run_both())
    all_paths = list(tmp_path.glob("*.jsonl"))
    assert len(all_paths) == 2
    assert {json.loads(p.read_text().splitlines()[0])["case_id"] for p in all_paths} == {"a", "b"}


def test_batch_exception_does_not_leak_raw_message(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(text="Claim 1: YES", result={0: True}, error=ValueError("private-prompt"))
    e = events(tmp_path)[1]
    assert e["judged_count"] == 1
    assert "private-prompt" not in json.dumps(e)


def test_recorder_disabled_on_unwritable_parent(tmp_path):
    parent_file = tmp_path / "file.txt"
    parent_file.write_text("x")
    with trace_judge_case(case_id="a", trace_dir=parent_file / "child") as rec:
        assert rec is None
        send(result={0: True})


def test_inspector(tmp_path):
    with trace_judge_case(case_id="test/case", trace_dir=tmp_path):
        send(result={0: True}, count=4)
    path = next(tmp_path.glob("*.jsonl"))
    summary = summarize(path)
    assert (summary["observed_batches"], summary["unjudged_claims"]) == (1, 3)
    assert summary["problem_batches"][0]["reason_hint"] == "partial_response"


def test_does_not_persist_raw_response(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(text="Claim 1: YES\nThis is a sensitive passage 1234", result={0: True})
    assert "sensitive passage" not in next(tmp_path.glob("*.jsonl")).read_text()


def test_cancelled_is_distinct_from_empty_response(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(error=asyncio.CancelledError(), text="", count=4)
    e = events(tmp_path)[1]
    assert (e["status"], e["reason_hint"]) == ("cancelled", "evaluation_cancelled")


@pytest.mark.parametrize("inner", ["disabled", "open_failure"])
def test_nested_disabled_or_failed_scope_does_not_borrow_outer_trace(tmp_path, inner):
    blocker = tmp_path / "blocker"
    blocker.write_text("file")
    with trace_judge_case(case_id="outer", trace_dir=tmp_path):
        with trace_judge_case(
            case_id="inner", trace_dir=None if inner == "disabled" else blocker / "child"
        ):
            send(result={0: True})
        send(result={0: False})
    recorded = events(tmp_path)
    assert [e["event"] for e in recorded] == ["case_start", "batch_finished", "case_end"]
    assert recorded[1]["no_count"] == 1


def test_metadata_property_error_does_not_escape_or_leak(tmp_path, caplog):
    class Error(Exception):
        @property
        def response(self):
            raise RuntimeError("SECRET-METADATA")

    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(error=Error("SECRET-ERROR"), text="")
    assert events(tmp_path)[1]["http_status"] is None
    assert "SECRET" not in caplog.text


@pytest.mark.parametrize("failure", ["write", "flush", "close"])
def test_stream_failure_marks_partial_and_never_leaks(tmp_path, caplog, failure):
    from evals.agent_eval_ext.judge_runtime import judge_trace_artifacts

    class BrokenStream:
        def write(self, _):
            if failure == "write":
                raise OSError("SECRET-WRITE")

        def flush(self):
            if failure == "flush":
                raise OSError("SECRET-FLUSH")

        def close(self):
            if failure == "close":
                raise OSError("SECRET-CLOSE")

    with trace_judge_case(case_id="a", trace_dir=tmp_path) as rec:
        original = rec._handle
        rec._handle = BrokenStream()
        send(result={0: True})
        original.close()
    assert rec.failed
    assert judge_trace_artifacts(rec, requested=True)["judge_batch_trace_status"] == "partial"
    assert "SECRET" not in caplog.text


def test_unknown_payload_fields_and_identity_override_are_ignored(tmp_path):
    with trace_judge_case(case_id="a", trace_dir=tmp_path) as rec:
        rec.write(
            {
                "event": "custom",
                "prompt": "SECRET",
                "headers": {"key": "SECRET"},
                "case_id": "evil",
                "seq": 999,
            }
        )
    rows = events(tmp_path)
    assert "SECRET" not in json.dumps(rows)
    assert [r["seq"] for r in rows] == [1, 2, 3]
    assert {r["case_id"] for r in rows} == {"a"}


@pytest.mark.parametrize("model", ["auto", "auto/best", "vendor/auto"])
def test_router_alias_is_not_a_disclosed_model(tmp_path, model):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        trace_judge_batch(
            batch_no=1,
            batch_start=0,
            expected_count=1,
            verdict_by_claim={0: True},
            response=object(),
            text="Claim 1: YES",
            error=None,
            elapsed_ms=1,
            served_model=model,
        )
    row = events(tmp_path)[1]
    assert row["served_model"] is None and row["disclosed_alias"] == model


@pytest.mark.parametrize("damage", ["identity", "sequence", "counts", "order", "schema"])
def test_inspector_rejects_corrupt_or_mixed_trace(tmp_path, damage):
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(result={0: True})
    path = next(tmp_path.glob("*.jsonl"))
    rows = events(tmp_path)
    if damage == "identity":
        rows[1]["trace_id"] = "other"
    elif damage == "sequence":
        rows[1]["seq"] = 9
    elif damage == "counts":
        rows[1]["judged_count"] = 8
    elif damage == "order":
        rows[1]["event"] = "case_start"
    else:
        rows[1]["schema_version"] = True
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        summarize(path)


def test_empty_and_truncated_traces_are_not_complete(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert summarize(empty)["judge_coverage"] is None
    assert summarize(empty)["expected_claims"] is None
    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(result={0: True})
    path = next(p for p in tmp_path.glob("*.jsonl") if p != empty)
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    info = summarize(path)
    assert info["trace_complete"] is False and info["judge_coverage"] is None
    assert info["observed_judge_coverage"] == 1.0
    assert info["application_level_calls"] is None
    assert info["observed_application_level_calls"] == 1


def test_context_exception_closes_and_restores_recorder(tmp_path):
    with pytest.raises(RuntimeError, match="original"):
        with trace_judge_case(case_id="a", trace_dir=tmp_path):
            raise RuntimeError("original")
    send(result={0: True})
    assert events(tmp_path)[-1]["status"] == "error"


@pytest.mark.parametrize("entry", ["cli", "standalone"])
def test_runtime_inspector_cli_reads_captured_events(entry, tmp_path, monkeypatch, capsys):
    from evals.agent_eval_ext import cli, judge_runtime_inspect

    with trace_judge_case(case_id="a", trace_dir=tmp_path):
        send(result={0: True})
    source = next(tmp_path.glob("*.jsonl"))
    before = source.read_bytes()
    args = (
        ["inspect", "judge-runtime", "--input", str(source)]
        if entry == "cli"
        else ["inspect", str(source)]
    )
    monkeypatch.setattr("sys.argv", args)
    (cli.main if entry == "cli" else judge_runtime_inspect.main)()
    output = json.loads(capsys.readouterr().out)
    assert output["trace_complete"] is True and output["judged_claims"] == 1
    assert source.read_bytes() == before
