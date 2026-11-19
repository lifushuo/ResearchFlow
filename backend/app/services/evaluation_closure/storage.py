"""评测结果使用共享评测卷中的独立产物数据库，不连接业务表。"""

from pathlib import Path

from app.services.evaluation_closure import EvaluationClosure


def store_at(job_root: Path) -> EvaluationClosure:
    return EvaluationClosure(job_root / "closure-v1.sqlite3")
