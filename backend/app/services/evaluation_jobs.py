"""统一评测 Worker 共用的有界 JSON 存储与任务准入锁。"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNNABLE = frozenset({"planner-replay-v1", "tool-contract-v1"})
MAX_JSON_BYTES = 256 * 1024


class JobError(ValueError):
    """任务请求无效、无权访问或存储不可用。"""


def now() -> str:
    return datetime.now(UTC).isoformat()


def job_root() -> Path:
    value = os.getenv("RF_EVAL_JOB_ROOT", "").strip()
    if not value:
        raise JobError("runner_not_configured")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise JobError("runner_storage_unavailable")
    return path


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise JobError("invalid_job_record")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, OSError) as exc:
        raise JobError("invalid_job_record") from exc
    if not isinstance(doc, dict):
        raise JobError("invalid_job_record")
    return doc


def _write_new(path: Path, data: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def update_status(path: Path, data: dict[str, Any]) -> None:
    """原子替换状态文件，客户端不会读到只写了一半的 JSON。"""
    temp = path.with_name(".status." + os.urandom(8).hex() + ".tmp")
    try:
        _write_new(temp, data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _admission(root: Path):
    """Linux Compose 使用跨进程准入锁；Windows 分支用于本地测试。"""
    lock_file = root / ".admission.lock"
    with lock_file.open("a+b") as lock:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
