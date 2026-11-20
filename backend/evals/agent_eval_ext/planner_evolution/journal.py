"""优化过程的独占审计文件：失败时保留已测结果，不自动重发模型请求。"""

from datetime import UTC, datetime
from pathlib import Path

from evals.agent_eval_ext.long_run import atomic_create


class ExperimentJournal:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sequence = 0

    def __call__(self, kind: str, payload: dict) -> None:
        self.sequence += 1
        atomic_create(
            self.root / f"event-{self.sequence:05d}.json",
            {
                "schema_version": 1,
                "sequence": self.sequence,
                "kind": kind,
                "recorded_at": datetime.now(UTC).isoformat(),
                "payload": payload,
            },
        )
