"""评测结果与质量门禁存储，仅使用标准库且不访问网络。

数据必须来自可信评测 Worker 的标准化逐题结果，不修改既有评分原文件。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class ClosureError(Exception):
    """预期的输入或状态错误，不泄露其他用户的私有资源。"""


class NotFound(ClosureError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def input_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def now_ms() -> int:
    return int(time.time() * 1000)


SECRET_NAME = re.compile(
    r"(?i)(authorization|api.?key|secret|password|access.?token|refresh.?token|cookie|private.?key)"
)
SECRET_STRING = re.compile(
    r"(?i)(bearer\s+\S+|sk-[a-zA-Z0-9_-]{12,}|(?:api.?key|access.?token|refresh.?token|password|secret|signature)\s*[:=]\s*[^\s&,;]+|(?<!\w)[\w.+-]+@[\w.-]+\.[a-z]{2,}(?!\w))"
)


def sanitize(value: Any, depth: int = 0) -> Any:
    """尽力脱敏；Worker 应在写入前就最小化输入内容。"""
    if depth > 12:
        return "[DEPTH_LIMIT]"
    if isinstance(value, dict):
        return {
            str(k)[:120]: ("[REDACTED]" if SECRET_NAME.search(str(k)) else sanitize(v, depth + 1))
            for k, v in list(value.items())[:150]
        }
    if isinstance(value, list):
        return [sanitize(v, depth + 1) for v in value[:150]]
    if isinstance(value, str):
        return SECRET_STRING.sub("[REDACTED]", value[:12000])
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return "[UNSUPPORTED]"


def nonempty(value: str, field: str, max_len: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_len:
        raise ClosureError(f"Invalid {field}")
    return value.strip()


def load_json(value: str) -> Any:
    return json.loads(value)


def to_dict(row: sqlite3.Row | None, json_columns: tuple[str, ...] = ()) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for key in json_columns:
        if key in d and d[key] is not None:
            d[key] = load_json(d[key])
    return d


DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK(execution_mode IN ('real','fake','synthetic')),
    sample_set_hash TEXT NOT NULL, metrics_version TEXT NOT NULL,
    model_route TEXT NOT NULL, judge_route TEXT NOT NULL,
    created_ms INTEGER NOT NULL, source_pipeline_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_owner ON runs(owner_id,created_ms);
CREATE TABLE IF NOT EXISTS case_results (
    run_id TEXT NOT NULL REFERENCES runs(run_id), case_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('passed','failed','blocked','unmeasured')),
    input_sha256 TEXT NOT NULL, error_code TEXT, error_class TEXT,
    latency_ms INTEGER, metrics_json TEXT NOT NULL, details_json TEXT NOT NULL,
    trace_ref TEXT, finished_ms INTEGER NOT NULL,
    PRIMARY KEY(run_id,case_id)
);
CREATE TABLE IF NOT EXISTS bad_cases (
    bad_case_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
    case_id TEXT NOT NULL, error_key TEXT NOT NULL,
    review_status TEXT NOT NULL CHECK(review_status IN ('new','confirmed','ignored','fixed')),
    first_run_id TEXT NOT NULL, last_run_id TEXT NOT NULL,
    first_seen_ms INTEGER NOT NULL, last_seen_ms INTEGER NOT NULL,
    fixed_by_run_id TEXT,
    UNIQUE(owner_id,dataset_id,case_id,error_key)
);
CREATE INDEX IF NOT EXISTS idx_bad_owner_status ON bad_cases(owner_id,review_status,last_seen_ms);
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, action TEXT NOT NULL,
    target_id TEXT NOT NULL, summary_json TEXT NOT NULL, created_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_owner ON audit_events(owner_id,created_ms);
CREATE TABLE IF NOT EXISTS run_observations (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id),
    expected_json TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'open', provenance_json TEXT NOT NULL
);
"""


class EvaluationClosure:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        if str(db_path) == ":memory:":
            raise ClosureError("Use a persistent path; memory DB is not safe across connections")
        if self.path.is_symlink() or any(p.is_symlink() for p in self.path.parents):
            raise ClosureError("Unsafe closure database path")
        if self.path.exists():
            with sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True) as existing:
                tables = existing.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if tables and existing.execute("PRAGMA application_id").fetchone()[0] != 0x52464543:
                    raise ClosureError("Not a closure database; refusing to modify it")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db(write=True) as db:
            db.execute("PRAGMA application_id=1380336963")
            db.executescript(DDL)

    @contextmanager
    def _db(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except Exception:
            if write:
                conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _audit(db: sqlite3.Connection, owner: str, action: str, target: str, summary: Any) -> None:
        db.execute(
            "INSERT INTO audit_events VALUES(?,?,?,?,?,?)",
            (uuid.uuid4().hex, owner, action, target, canonical_json(sanitize(summary)), now_ms()),
        )

    @staticmethod
    def _owned(
        db: sqlite3.Connection, table: str, id_col: str, object_id: str, owner: str
    ) -> sqlite3.Row:
        if table not in ("runs", "bad_cases", "regressions"):
            raise ValueError("invalid table")
        row = db.execute(
            f"SELECT * FROM {table} WHERE {id_col}=? AND owner_id=?", (object_id, owner)
        ).fetchone()
        if row is None:
            raise NotFound("Resource not found")
        return row

    def register_run(
        self,
        *,
        run_id: str,
        owner_id: str,
        dataset_id: str,
        execution_mode: str,
        sample_set_hash: str,
        metrics_version: str,
        model_route: str,
        judge_route: str = "none",
        source_pipeline_id: str | None = None,
        expected_case_ids: list[str] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """可信 Worker 在逐题写入前登记一次。

        多数据集 Pipeline 每个步骤必须使用独立 run_id，不能只用 Pipeline ID。"""
        values = [
            nonempty(run_id, "run_id", 256),
            nonempty(owner_id, "owner_id"),
            nonempty(dataset_id, "dataset_id"),
            execution_mode,
            nonempty(sample_set_hash, "sample_set_hash"),
            nonempty(metrics_version, "metrics_version"),
            nonempty(model_route, "model_route"),
            nonempty(judge_route, "judge_route"),
        ]
        if execution_mode not in ("real", "fake", "synthetic"):
            raise ClosureError("Invalid execution_mode")
        expected = expected_case_ids or []
        if not isinstance(expected, list) or len(set(expected)) != len(expected):
            raise ClosureError("Invalid expected cases")
        for case_id in expected:
            nonempty(case_id, "case_id", 256)
        expected_j = canonical_json(expected)
        provenance_j = canonical_json(sanitize(provenance or {}))
        with self._db(write=True) as db:
            existing = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing:
                if [
                    existing[k]
                    for k in (
                        "run_id",
                        "owner_id",
                        "dataset_id",
                        "execution_mode",
                        "sample_set_hash",
                        "metrics_version",
                        "model_route",
                        "judge_route",
                    )
                ] != values:
                    raise ClosureError("Immutable run signature conflict")
                observation = db.execute(
                    "SELECT * FROM run_observations WHERE run_id=?", (run_id,)
                ).fetchone()
                if (
                    observation["expected_json"] != expected_j
                    or observation["provenance_json"] != provenance_j
                    or existing["source_pipeline_id"] != source_pipeline_id
                ):
                    raise ClosureError("Immutable run signature conflict")
            else:
                db.execute(
                    "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (*values, now_ms(), source_pipeline_id),
                )
                db.execute(
                    "INSERT INTO run_observations(run_id,expected_json,provenance_json) VALUES(?,?,?)",
                    (run_id, expected_j, provenance_j),
                )
                self._audit(
                    db,
                    owner_id,
                    "run.register",
                    run_id,
                    {"dataset_id": dataset_id, "mode": execution_mode},
                )
            return to_dict(db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())  # type: ignore

    def ingest_case(
        self,
        *,
        run_id: str,
        owner_id: str,
        case_id: str,
        status: str,
        input_sha256: str = "",
        error_code: str | None = None,
        error_class: str | None = None,
        latency_ms: int | None = None,
        metrics: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        trace_ref: str | None = None,
    ) -> dict[str, Any]:
        """仅可信 Worker 可调用，不能直接开放浏览器 POST 写入成绩。

        passed 必须表示真正评测器检查过合格判据，不能由 HTTP 200、模型文字或历史旧分数推断。"""
        nonempty(case_id, "case_id", 256)
        if status not in ("passed", "failed", "blocked", "unmeasured"):
            raise ClosureError("Invalid status")
        if input_sha256 and not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
            raise ClosureError("input_sha256 must be canonical SHA-256 hex")
        if latency_ms is not None and (type(latency_ms) is not int or latency_ms < 0):
            raise ClosureError("Invalid latency_ms")
        metrics_j = canonical_json(sanitize(metrics or {}))
        details_j = canonical_json(sanitize(details or {}))
        error_code = sanitize(error_code)[:160] if error_code else None
        error_class = sanitize(error_class)[:120] if error_class else None
        trace_ref = sanitize(trace_ref)[:512] if trace_ref else None
        now = now_ms()
        with self._db(write=True) as db:
            run = self._owned(db, "runs", "run_id", run_id, owner_id)
            existing = db.execute(
                "SELECT * FROM case_results WHERE run_id=? AND case_id=?", (run_id, case_id)
            ).fetchone()
            proposed = (
                status,
                input_sha256,
                error_code,
                error_class,
                latency_ms,
                metrics_j,
                details_j,
                trace_ref,
            )
            if existing:
                previous = tuple(
                    existing[k]
                    for k in (
                        "status",
                        "input_sha256",
                        "error_code",
                        "error_class",
                        "latency_ms",
                        "metrics_json",
                        "details_json",
                        "trace_ref",
                    )
                )
                if previous != proposed:
                    raise ClosureError("Immutable case result conflict; create a new run")
                return to_dict(existing, ("metrics_json", "details_json"))  # type: ignore
            observation = db.execute(
                "SELECT * FROM run_observations WHERE run_id=?", (run_id,)
            ).fetchone()
            if observation["state"] != "open":
                raise ClosureError("Run is sealed; create a new run")
            expected = load_json(observation["expected_json"])
            if expected and case_id not in expected:
                raise ClosureError("Unexpected case ID")
            db.execute(
                "INSERT INTO case_results VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, case_id, *proposed, now),
            )
            # 单案例中断可进入待排查，但不能编造质量失败分数；UNMEASURED 仍不能通过门禁。
            if status == "failed" or (
                status == "unmeasured" and error_class == "execution_incident"
            ):
                ek = nonempty(error_code or error_class or "unknown_failure", "error_key", 160)
                bc = db.execute(
                    "SELECT * FROM bad_cases WHERE owner_id=? AND dataset_id=? AND case_id=? AND error_key=?",
                    (owner_id, run["dataset_id"], case_id, ek),
                ).fetchone()
                if bc is None:
                    bid = uuid.uuid4().hex
                    db.execute(
                        "INSERT INTO bad_cases VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            bid,
                            owner_id,
                            run["dataset_id"],
                            case_id,
                            ek,
                            "new",
                            run_id,
                            run_id,
                            run["created_ms"],
                            run["created_ms"],
                            None,
                        ),
                    )
                    self._audit(
                        db, owner_id, "badcase.detect", bid, {"case_id": case_id, "error_key": ek}
                    )
                elif (
                    observation["seq"] >= self._run_seq(db, bc["last_run_id"])
                    and run_id != bc["last_run_id"]
                ):
                    next_status = "new" if bc["review_status"] == "fixed" else bc["review_status"]
                    db.execute(
                        "UPDATE bad_cases SET last_run_id=?,last_seen_ms=?,review_status=?,fixed_by_run_id=NULL WHERE bad_case_id=?",
                        (run_id, run["created_ms"], next_status, bc["bad_case_id"]),
                    )
            return to_dict(
                db.execute(
                    "SELECT * FROM case_results WHERE run_id=? AND case_id=?", (run_id, case_id)
                ).fetchone(),
                ("metrics_json", "details_json"),
            )  # type: ignore

    @staticmethod
    def _run_seq(db: sqlite3.Connection, run_id: str) -> int:
        return db.execute("SELECT seq FROM run_observations WHERE run_id=?", (run_id,)).fetchone()[
            0
        ]

    def finish_run(self, owner_id: str, run_id: str, *, complete: bool = True) -> None:
        """Worker 核验完整预期案例集合后，才允许封存。"""
        with self._db(write=True) as db:
            self._owned(db, "runs", "run_id", run_id, owner_id)
            obs = db.execute("SELECT * FROM run_observations WHERE run_id=?", (run_id,)).fetchone()
            expected = set(load_json(obs["expected_json"]))
            actual = {
                r[0]
                for r in db.execute("SELECT case_id FROM case_results WHERE run_id=?", (run_id,))
            }
            state = (
                "finished"
                if complete and actual and (not expected or actual == expected)
                else "incomplete"
            )
            if obs["state"] != "open" and obs["state"] != state:
                raise ClosureError("Immutable run completion conflict")
            if obs["state"] == "open":
                db.execute("UPDATE run_observations SET state=? WHERE run_id=?", (state, run_id))
                self._audit(
                    db, owner_id, "run.finish", run_id, {"state": state, "cases": len(actual)}
                )

    def runs(self, owner_id: str) -> list[dict[str, Any]]:
        with self._db() as db:
            return [
                to_dict(row, ("expected_json", "provenance_json"))
                for row in db.execute(
                    "SELECT r.*,o.state,o.expected_json,o.provenance_json FROM runs r JOIN run_observations o USING(run_id) "
                    "WHERE owner_id=? ORDER BY o.seq DESC LIMIT 100",
                    (owner_id,),
                )
            ]

    def list_bad_cases(
        self, owner_id: str, *, review_status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        if review_status is not None and review_status not in (
            "new",
            "confirmed",
            "ignored",
            "fixed",
        ):
            raise ClosureError("Invalid review_status")
        limit = min(max(limit, 1), 100)
        offset = max(offset, 0)
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM bad_cases WHERE owner_id=? AND (? IS NULL OR review_status=?) "
                "ORDER BY last_seen_ms DESC,bad_case_id LIMIT ? OFFSET ?",
                (owner_id, review_status, review_status, limit, offset),
            ).fetchall()
            return [dict(x) for x in rows]

    def bad_case_detail(self, owner_id: str, bad_case_id: str) -> dict[str, Any]:
        with self._db() as db:
            bc = self._owned(db, "bad_cases", "bad_case_id", bad_case_id, owner_id)
            result = db.execute(
                "SELECT * FROM case_results WHERE run_id=? AND case_id=?",
                (bc["last_run_id"], bc["case_id"]),
            ).fetchone()
            return {
                "bad_case": dict(bc),
                "last_result": to_dict(result, ("metrics_json", "details_json")),
            }

    @staticmethod
    def _p95(values: list[int]) -> int | None:
        if not values:
            return None
        values.sort()
        return values[max(0, math.ceil(len(values) * 0.95) - 1)]  # 使用最近秩分位数。

    def quality_gate(
        self,
        owner_id: str,
        run_id: str,
        *,
        min_measured: int = 1,
        min_pass_rate: float = 0.90,
        max_p95_latency_ms: int | None = None,
        fail_on_blocked: bool = True,
        fail_on_unmeasured: bool = True,
        required_metric_minimums: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if (
            type(min_measured) is not int
            or min_measured < 1
            or isinstance(min_pass_rate, bool)
            or not isinstance(min_pass_rate, (int, float))
            or not math.isfinite(min_pass_rate)
            or not 0 <= min_pass_rate <= 1
            or max_p95_latency_ms is not None
            and (type(max_p95_latency_ms) is not int or max_p95_latency_ms < 0)
            or type(fail_on_blocked) is not bool
            or type(fail_on_unmeasured) is not bool
        ):
            raise ClosureError("Invalid gate thresholds")
        minima = required_metric_minimums or {}
        for key, val in minima.items():
            if (
                not re.fullmatch(r"[A-Za-z0-9_]{1,64}", key)
                or isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
            ):
                raise ClosureError("Invalid required metric threshold")
        with self._db() as db:
            run = self._owned(db, "runs", "run_id", run_id, owner_id)
            rows = db.execute("SELECT * FROM case_results WHERE run_id=?", (run_id,)).fetchall()
            observation = db.execute(
                "SELECT * FROM run_observations WHERE run_id=?", (run_id,)
            ).fetchone()
        counts = {
            s: sum(r["status"] == s for r in rows)
            for s in ("passed", "failed", "blocked", "unmeasured")
        }
        measured = counts["passed"] + counts["failed"]
        rate = counts["passed"] / measured if measured else None
        latencies = [
            r["latency_ms"]
            for r in rows
            if r["status"] in ("passed", "failed") and r["latency_ms"] is not None
        ]
        p95 = self._p95(latencies) if len(latencies) == measured else None
        failures = []
        incomplete = []
        if observation["state"] != "finished":
            incomplete.append("run_not_finished")
        if measured < min_measured:
            incomplete.append(f"measured_cases={measured} < min_measured={min_measured}")
        if fail_on_blocked and counts["blocked"]:
            incomplete.append(f"blocked_cases={counts['blocked']}")
        if fail_on_unmeasured and counts["unmeasured"]:
            incomplete.append(f"unmeasured_cases={counts['unmeasured']}")
        if rate is not None and rate < min_pass_rate:
            failures.append(f"pass_rate={rate:.4f} < {min_pass_rate:.4f}")
        if max_p95_latency_ms is not None:
            if p95 is None:
                incomplete.append("p95_latency_ms unavailable")
            elif p95 > max_p95_latency_ms:
                failures.append(f"p95_latency_ms={p95} > {max_p95_latency_ms}")
        for metric, threshold in minima.items():
            observed = [
                load_json(r["metrics_json"]).get(metric)
                for r in rows
                if r["status"] in ("passed", "failed")
            ]
            valid = [
                v
                for v in observed
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            ]
            if not measured or len(valid) != measured:
                incomplete.append(f"{metric} missing or not numeric in measured cases")
            elif sum(valid) / len(valid) < threshold:
                failures.append(f"{metric} mean={sum(valid) / len(valid):.4f} < {threshold:.4f}")
        result = "blocked" if incomplete else ("failed" if failures else "passed")
        policy = {
            "min_measured": min_measured,
            "min_pass_rate": min_pass_rate,
            "max_p95_latency_ms": max_p95_latency_ms,
            "fail_on_blocked": fail_on_blocked,
            "fail_on_unmeasured": fail_on_unmeasured,
            "required_metric_minimums": minima,
        }
        with self._db(write=True) as db:
            self._audit(
                db,
                owner_id,
                "gate.check",
                run_id,
                {"policy": policy, "policy_sha256": input_hash(policy), "verdict": result},
            )
        return {
            "run_id": run_id,
            "dataset_id": run["dataset_id"],
            "execution_mode": run["execution_mode"],
            "policy_sha256": input_hash(policy),
            "verdict": result,
            "measured": measured,
            "total_recorded": len(rows),
            "counts": counts,
            "pass_rate": rate,
            "p95_latency_ms": p95,
            "failure_reasons": failures,
            "incomplete_reasons": incomplete,
            "note": "p95 is nearest-rank. Missing/unmeasured is never treated as zero or as passing.",
        }

    def compare(self, owner_id: str, baseline_run: str, candidate_run: str) -> dict[str, Any]:
        with self._db() as db:
            a = self._owned(db, "runs", "run_id", baseline_run, owner_id)
            b = self._owned(db, "runs", "run_id", candidate_run, owner_id)
            fields = (
                "dataset_id",
                "execution_mode",
                "sample_set_hash",
                "metrics_version",
                "model_route",
                "judge_route",
            )
            differences = [field for field in fields if a[field] != b[field]]
            states = [
                db.execute("SELECT state FROM run_observations WHERE run_id=?", (rid,)).fetchone()[
                    0
                ]
                for rid in (baseline_run, candidate_run)
            ]
            provenance = [
                load_json(
                    db.execute(
                        "SELECT provenance_json FROM run_observations WHERE run_id=?", (rid,)
                    ).fetchone()[0]
                )
                for rid in (baseline_run, candidate_run)
            ]
            ar = {
                x["case_id"]: x
                for x in db.execute("SELECT * FROM case_results WHERE run_id=?", (baseline_run,))
            }
            br = {
                x["case_id"]: x
                for x in db.execute("SELECT * FROM case_results WHERE run_id=?", (candidate_run,))
            }
        if baseline_run == candidate_run:
            differences.append("same_run")
        if not ar or not br:
            differences.append("empty_case_set")
        if states != ["finished", "finished"]:
            differences.append("run_not_finished")
        if any(p.get("model_identity_verified") is False for p in provenance):
            differences.append("model_identity_unverified")
        if any(r["status"] in ("blocked", "unmeasured") for r in [*ar.values(), *br.values()]):
            differences.append("unmeasured_cases")
        if set(ar) != set(br):
            differences.append("case_id_set")
        if any(not ar[k]["input_sha256"] or not br[k]["input_sha256"] for k in set(ar) & set(br)):
            differences.append("case_input_hash_missing")
        if any(ar[k]["input_sha256"] != br[k]["input_sha256"] for k in set(ar) & set(br)):
            differences.append("case_input_hash")
        if differences:
            return {
                "comparable": False,
                "reasons": differences,
                "note": "Configuration, sample set or inputs differ. Do not attribute metric deltas to code changes.",
            }
        regressions = [
            k for k in sorted(ar) if ar[k]["status"] == "passed" and br[k]["status"] != "passed"
        ]
        improvements = [
            k for k in sorted(ar) if ar[k]["status"] != "passed" and br[k]["status"] == "passed"
        ]
        return {
            "comparable": True,
            "dataset_id": a["dataset_id"],
            "baseline_run_id": baseline_run,
            "candidate_run_id": candidate_run,
            "regressions": regressions,
            "improvements": improvements,
            "unchanged": len(ar) - len(regressions) - len(improvements),
            "caveat": "Only matched configuration and inputs; external web evidence or stochastic model outputs may drift.",
        }

    def record_audit(self, owner_id: str, action: str, target_id: str, summary: Any) -> None:
        """回流等评测中心 UI 动作与 Worker 摄取共用同一审计表，动作与目标必须非空。"""
        with self._db(write=True) as db:
            self._audit(
                db,
                nonempty(owner_id, "owner_id", 120),
                nonempty(action, "action", 80),
                nonempty(target_id, "target_id", 200),
                summary,
            )

    def audit_log(self, owner_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM audit_events WHERE owner_id=? ORDER BY rowid DESC LIMIT ?",
                (owner_id, min(max(limit, 1), 100)),
            ).fetchall()
            return [to_dict(x, ("summary_json",)) for x in rows]  # type: ignore
