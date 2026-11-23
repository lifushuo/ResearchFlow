"""Report human/Judge agreement from a manually reviewed Planner calibration sheet.

Pure offline accounting. Unmeasured fields remain unmeasured, never zero scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(sheet: dict) -> dict:
    if sheet.get("purpose") != "manual_planner_judge_calibration_only":
        raise ValueError("wrong_calibration_sheet_purpose")
    cases = sheet.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("no_calibration_cases")
    tp = tn = fp = fn = skipped = 0
    for case in cases:
        for aspect in case.get("aspects", []):
            human, judge = aspect.get("human_covered"), aspect.get("judge_covered")
            if human is None or judge is None:
                skipped += 1
                continue
            if type(human) is not bool or type(judge) is not bool:
                raise ValueError("calibration_labels_must_be_strict_booleans")
            if human and judge:
                tp += 1
            elif not human and not judge:
                tn += 1
            elif not human and judge:
                fp += 1
            else:
                fn += 1
    count = tp + tn + fp + fn
    if count == 0:
        raise ValueError("no_human_and_judge_pairs_measured")

    # Judge TRUE means 'covered'. FP therefore means it wrongly accepted an
    # uncovered aspect. This is especially risky for optimizing Planner prompts.
    def ratio(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    return {
        "schema_version": 1,
        "metric": "planner_aspect_coverage_agreement",
        "measured_aspects": count,
        "unmeasured_aspects": skipped,
        "true_covered": tp,
        "true_missing": tn,
        "false_covered": fp,
        "false_missing": fn,
        "agreement": ratio(tp + tn, count),
        "coverage_precision": ratio(tp, tp + fp),
        "coverage_recall": ratio(tp, tp + fn),
        "false_covered_rate_on_human_missing": ratio(fp, fp + tn),
        "false_missing_rate_on_human_covered": ratio(fn, fn + tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.sheet.read_bytes()
    sheet = json.loads(raw.decode("utf-8"))
    result = summarize(sheet)
    result["source_sheet_sha256"] = hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Calibration summary saved: {args.out} ({result['measured_aspects']} aspects)")


if __name__ == "__main__":
    main()
