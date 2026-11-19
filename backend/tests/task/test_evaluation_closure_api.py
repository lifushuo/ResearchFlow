"""Optional API tests; use installed FastAPI and httpx/TestClient."""

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from app.api.v1.evaluation_closure import make_router
from app.services.evaluation_closure import EvaluationClosure, input_hash


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = EvaluationClosure(Path(self.tmp.name) / "eval.sqlite3")
        fixture = {"question": "abc"}
        self.store.register_run(
            run_id="p1/planner",
            owner_id="alice",
            dataset_id="planner",
            execution_mode="fake",
            sample_set_hash="v1",
            metrics_version="v1",
            model_route="fake",
        )
        self.store.ingest_case(
            run_id="p1/planner",
            owner_id="alice",
            case_id="a1",
            input_sha256=input_hash(fixture),
            status="failed",
            error_code="schema",
        )
        self.bid = self.store.list_bad_cases("alice")[0]["bad_case_id"]
        self.fixture = fixture

        def session_owner(x_session_user: str | None = Header(default=None)) -> str:
            if x_session_user not in ("alice", "bob"):
                raise HTTPException(401, "Login required")
            # Test-only stub; production MUST read the validated app session.
            return x_session_user

        def action_guard(x_action_token: str | None = Header(default=None)) -> None:
            if x_action_token != "test-only-csrf":
                raise HTTPException(403, "CSRF validation failed")

        app = FastAPI()
        app.include_router(
            make_router(self.store, session_owner, action_guard),
            prefix="/api/v1/evaluation/closure",
        )
        self.client = TestClient(app)
        self.root = "/api/v1/evaluation/closure"

    def test_read_requires_auth(self):
        res = self.client.get(f"{self.root}/bad-cases")
        self.assertEqual(res.status_code, 401)

    def test_cross_owner_read_hidden(self):
        res = self.client.get(
            f"{self.root}/bad-cases/{self.bid}", headers={"x-session-user": "bob"}
        )
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
