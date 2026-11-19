"""Shared-filesystem ownership and durable scoring commits (docs/08, recovery).

An OS lock dies with its process, not with a lease timer. It prevents concurrent
requests but cannot prevent a second charge after a response-before-commit crash.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path


class BatchBusy(RuntimeError):
    pass


class BatchLease:
    def __init__(self, root: Path, batch_id: str):
        self.directory = root / batch_id
        self.batch_id = batch_id
        self.handle = None
        self.attempt_id = uuid.uuid4().hex
        self.state: dict = {}
        self.committed: dict | None = None
        self.uncertain_previous_request = False

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "ownership.lock"
        if path.is_symlink():
            raise ValueError("symlink batch ownership")
        self.handle = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                if not path.stat().st_size:
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise BatchBusy(self.batch_id) from exc
        try:
            commit = self.directory / "committed.json"
            if commit.exists():
                if commit.is_symlink():
                    raise ValueError("symlink batch commit")
                self.committed = json.loads(commit.read_text(encoding="utf8"))
            else:
                for attempt in self.directory.glob("attempt-*.json"):
                    if attempt.is_symlink():
                        raise ValueError("symlink batch attempt")
                    row = json.loads(attempt.read_text(encoding="utf8"))
                    if row.get("request_started") is True and row.get("state") != "committed":
                        self.uncertain_previous_request = True
            return self
        except BaseException:
            self.release()
            raise

    def record(self, state: str, **fields) -> None:
        from .long_run import update_status

        if self.handle is None:
            raise RuntimeError("batch not owned")
        if state in {"request_started", "response_received"}:
            fields["request_started"] = True
        if state == "response_received":
            fields["response_received"] = True
        self.state.update(
            {
                "batch_id": self.batch_id,
                "attempt_id": self.attempt_id,
                "state": state,
                "updated_at": datetime.now(UTC).isoformat(),
                **fields,
            }
        )
        update_status(self.directory / f"attempt-{self.attempt_id}.json", self.state)

    def commit(self, payload: dict) -> None:
        from .long_run import atomic_create

        if self.handle is None or payload.get("status") != "complete":
            raise ValueError("only owned, fully judged batches may commit")
        atomic_create(self.directory / "committed.json", payload)
        self.committed = payload
        self.record("committed")

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None

    def __exit__(self, *_):
        self.release()
