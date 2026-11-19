from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from app.services.evaluation_closure import ClosureError, EvaluationClosure, NotFound, input_hash


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = EvaluationClosure(Path(self.tmp.name) / "closure.sqlite3")
        self.fixture = {"question": "structured output smoke test", "revision": 1}
        self.sha = input_hash(self.fixture)
        self.register("runA")
        gate = self.s.quality_gate
        compare = self.s.compare

        def seal(run_id):
            with self.s._db() as db:
                if db.execute(
                    "SELECT COUNT(*) FROM case_results WHERE run_id=?", (run_id,)
                ).fetchone()[0]:
                    self.s.finish_run("alice", run_id)

        def measured_gate(owner, run_id, **kw):
            if owner == "alice":
                seal(run_id)
            return gate(owner, run_id, **kw)

        def measured_compare(owner, a, b):
            seal(a)
            seal(b)
            return compare(owner, a, b)

        self.s.quality_gate = measured_gate
        self.s.compare = measured_compare

    def register(self, run_id="runA", **kw):
        vals = dict(
            run_id=run_id,
            owner_id="alice",
            dataset_id="planner_replay",
            execution_mode="fake",
            sample_set_hash="dataset_v1",
            metrics_version="v1",
            model_route="fake",
            judge_route="none",
        )
        vals.update(kw)
        return self.s.register_run(**vals)

    def fail(self, run_id="runA", case_id="case-1", **kw):
        vals = dict(
            run_id=run_id,
            owner_id="alice",
            case_id=case_id,
            status="failed",
            input_sha256=self.sha,
            error_code="PARSER_INVALID",
            latency_ms=100,
            metrics={"schema_valid": 0.0},
            details={"message": "no valid task list"},
            trace_ref="trace/safe",
        )
        vals.update(kw)
        return self.s.ingest_case(**vals)

    def test_run_is_immutable(self):
        self.register()
        with self.assertRaisesRegex(ClosureError, "signature"):
            self.register(model_route="other")

    def test_case_duplicate_idempotent_and_conflict(self):
        first = self.fail()
        self.assertEqual(first, self.fail())
        self.assertEqual(1, len(self.s.list_bad_cases("alice")))
        with self.assertRaisesRegex(ClosureError, "Immutable case"):
            self.fail(latency_ms=101)

    def test_no_badcase_for_blocked_or_unmeasured(self):
        self.fail(case_id="blocked", status="blocked", error_code=None)
        self.fail(case_id="missed", status="unmeasured", error_code=None)
        self.assertEqual([], self.s.list_bad_cases("alice"))
        g = self.s.quality_gate("alice", "runA")
        self.assertEqual(g["verdict"], "blocked")
        self.assertEqual(g["counts"]["unmeasured"], 1)

    def test_failed_case_dedupe_across_runs(self):
        self.fail()
        self.register("runB")
        self.fail("runB")
        items = self.s.list_bad_cases("alice")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["last_run_id"], "runB")

    def test_owner_isolation(self):
        self.fail()
        bid = self.s.list_bad_cases("alice")[0]["bad_case_id"]
        self.assertEqual(self.s.list_bad_cases("bob"), [])
        for method, args in [
            (self.s.bad_case_detail, ("bob", bid)),
            (self.s.quality_gate, ("bob", "runA")),
        ]:
            with self.assertRaises(NotFound):
                method(*args)

    def test_gate_fail_pass_and_insufficient(self):
        self.assertEqual("blocked", self.s.quality_gate("alice", "runA")["verdict"])
        self.fail()
        self.assertEqual("failed", self.s.quality_gate("alice", "runA")["verdict"])
        self.register("runB")
        self.fail("runB", status="passed", error_code=None, metrics={"schema_valid": 1.0})
        self.assertEqual("passed", self.s.quality_gate("alice", "runB")["verdict"])
        self.assertEqual("blocked", self.s.quality_gate("alice", "runB", min_measured=2)["verdict"])

    def test_gate_metric_missing_blocks(self):
        self.fail()
        g = self.s.quality_gate(
            "alice", "runA", required_metric_minimums={"citation_support_rate": 0.8}
        )
        self.assertEqual(g["verdict"], "blocked")
        self.assertTrue(any("missing" in t for t in g["incomplete_reasons"]))

    def test_comparison_mismatch_is_not_comparable(self):
        self.register("runB", model_route="changed-model")
        r = self.s.compare("alice", "runA", "runB")
        self.assertFalse(r["comparable"])
        self.assertIn("model_route", r["reasons"])

    def test_comparison_matched_detects_regression(self):
        self.fail(status="passed", error_code=None)
        self.register("runB")
        self.fail("runB")
        r = self.s.compare("alice", "runA", "runB")
        self.assertTrue(r["comparable"])
        self.assertEqual(r["regressions"], ["case-1"])

    def test_secrets_redacted_in_results(self):
        self.fail(
            details={"api_key": "secret", "text": "Bearer hello and user@example.com"},
            error_code="bad sk-abcdefghijklmnopqrst",
        )
        result = self.s.bad_case_detail("alice", self.s.list_bad_cases("alice")[0]["bad_case_id"])[
            "last_result"
        ]
        self.assertEqual(result["details_json"]["api_key"], "[REDACTED]")
        self.assertNotIn("hello", str(result))
        self.assertNotIn("example.com", str(result))
        self.assertNotIn("abcdefghijklmnopqrst", str(result))

    def test_concurrent_duplicate_insertion(self):
        errors = []

        def worker():
            try:
                self.fail()
            except Exception as e:
                errors.append(e)

        tasks = [threading.Thread(target=worker) for _ in range(6)]
        for t in tasks:
            t.start()
        for t in tasks:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self.s.list_bad_cases("alice")), 1)


if __name__ == "__main__":
    unittest.main()
