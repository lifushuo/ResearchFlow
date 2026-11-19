"""Read-only report lifecycle projections; trace closure is not scoring completion."""

from pathlib import Path

from .long_run import ProgressStore, ReliabilityError, _read_json


def report_states(base: Path, judges: list[dict]) -> list[dict]:
    root = base / "reliability" / "cases"
    output = []
    if not root.is_dir() or root.is_symlink():
        return output
    try:
        gate = _read_json(base / "raw_result.json").get("release_criteria", {}).get("gate_passed")
    except OSError, ValueError:
        gate = None
    for path in sorted(root.glob("**/progress.json"))[:100]:
        if path.is_symlink():
            continue
        try:
            progress = _read_json(path)
            config = progress["config"]
            case = config["case_id"]
            phase = config.get("phase", "baseline")
            saved = ProgressStore(path.parent, config=config).load(case_id=case)
            total = len(saved.draft["claim_lines"])
            # Restore validation reads canonical committed batches, not response events.
            committed = sum(b["judged_count"] for b in saved.batches.values())
            matches = [
                j
                for j in judges
                if j.get("case_id") == case and j.get("phase", "baseline").startswith(phase)
            ]
            judge = matches[0] if len(matches) == 1 else {}
            judged = judge.get("judged_claims_total")
            if judged is None:
                # 有效持久化批次独立证明判定进度；轨迹缺失不能抹去已校验提交。
                judged = committed
            complete = bool(total and judged == total and committed == total)
            reason = progress.get("reason")
            problems = judge.get("problem_batches") or []
            if not reason and not complete and problems:
                reason = (
                    "评分批次返回空或非文本内容"
                    if all(p.get("reason_hint") == "empty_or_nontext_response" for p in problems)
                    else "评分批次未取得完整有效判定"
                )
            output.append(
                {
                    "case_id": case,
                    "phase": phase,
                    "research": "draft_saved",
                    "judge": "completed" if complete else "partial",
                    "stop_reason": reason,
                    "total_claims": total,
                    "judged_claims": judged,
                    "committed_claims": committed,
                    "committed_batches": len(saved.batches),
                    "resume": "artifacts_validated",
                    "evaluation": "completed" if complete else "incomplete",
                    "quality_gate": "passed"
                    if complete and gate is True
                    else "not_passed"
                    if not complete or gate is False
                    else "pending_quality_checks",
                }
            )
        except OSError, ValueError, KeyError, TypeError, ReliabilityError:
            output.append(
                {
                    "case_id": path.parent.name,
                    "research": "unverified",
                    "judge": "unknown",
                    "resume": "validation_failed",
                    "evaluation": "incomplete",
                    "quality_gate": "not_passed",
                }
            )
    return output
