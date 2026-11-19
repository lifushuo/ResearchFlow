"""Score hand-labeled positive/negative judge controls. No calls to external APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_calibration(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, text in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        if not isinstance(row, dict) or type(row.get("expected_supported")) is not bool:
            raise ValueError(f"line {lineno}: expected_supported must be boolean")
        if any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("case_id", "claim", "evidence")
        ):
            raise ValueError(f"line {lineno}: case_id, claim and evidence required")
        if (
            row.get("predicted_supported") is not None
            and type(row.get("predicted_supported")) is not bool
        ):
            raise ValueError(f"line {lineno}: prediction must be true, false, or null")
        if row.get("split") not in ("dev", "holdout"):
            raise ValueError(f"line {lineno}: split must be dev or holdout")
        rows.append(row)
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate calibration case_id")
    return rows


def score_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = unjudged = 0
    for r in rows:
        pred = r.get("predicted_supported")
        expected = r["expected_supported"]
        if type(expected) is not bool or (pred is not None and type(pred) is not bool):
            raise ValueError(
                "Calibration labels and predictions must be boolean (or null prediction)"
            )
        if pred is None:
            unjudged += 1
        elif pred is True and expected is True:
            tp += 1
        elif pred is False and expected is False:
            tn += 1
        elif pred is True and expected is False:
            fp += 1
        else:
            fn += 1
    n = len(rows)
    judged = n - unjudged
    return {
        "cases": n,
        "judged": judged,
        "unjudged": unjudged,
        "coverage": judged / n if n else None,
        "accuracy_on_judged": (tp + tn) / judged if judged else None,
        "negative_recall": tn / (tn + fp) if tn + fp else None,
        "positive_recall": tp / (tp + fn) if tp + fn else None,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "label": "No calibration accuracy until actual judge predictions exist"
        if not judged
        else "calibrated against handwritten fixture labels",
    }
