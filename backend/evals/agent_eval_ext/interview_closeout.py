"""Offline, evidence-scoped interview report for ResearchFlow Evaluation Harness.

Consumes v0.7 normalized evaluation artifacts and optionally the v0.8 controlled
Tool/Safety scores. Never runs an Agent, rescores a Judge, or calls an API.
Only additive files: no modifications to production or prior evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import check_output_path
from .tool_safety import _validate_result as validate_tool_result
from .unified_eval import _replay_cases, compare_unified

SCHEMA_VERSION = 1
VALID_STATUSES = {"measured", "unmeasured", "invalid"}
DOMAINS = ("outcome", "reliability", "retrieval", "grounding")
DOMAIN_LABELS = {
    "outcome": "报告结果 / Outcome",
    "reliability": "故障回放 / Reliability",
    "retrieval": "检索 / Retrieval",
    "grounding": "事实核验 / Grounding",
}
METRIC_LABELS = {
    "completion_rate": "任务完成率",
    "judge_coverage": "Judge 判定覆盖率",
    "citation_support_among_judged": "引用支持率（仅已判定 Claim）",
    "latency_p95_completed_s": "已完成报告 P95 耗时",
    "api_cost_usd": "API 费用（需核实计价）",
    "case_pass_rate": "固定 Replay Case 合约通过率",
    "precision_at_5": "Precision@5",
    "recall_at_5": "Recall@5",
    "false_accept": "False Accept Rate",
    "false_reject": "False Reject Rate",
    "contradicted_recall": "Contradicted Recall",
    "case_coverage": "工具契约覆盖率（合成）",
    "contract_pass_rate": "工具契约通过率（合成）",
    "selection_case_accuracy": "工具选择准确率（合成）",
    "arguments_case_accuracy": "工具参数准确率（合成）",
    "forbidden_attempt_count": "禁止工具调用次数（合成）",
}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard numeric token: {value}")


class _Loaded(dict):
    pass


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if file.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("evaluation input exceeds 20MB limit")
    raw = file.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise ValueError("evaluation artifact must be a JSON object")
    result = _Loaded(data)
    result.byte_sha256 = hashlib.sha256(raw).hexdigest()
    return result


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_metrics(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("metrics must be an object")
    for key, entry in value.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ValueError("invalid metric")
        status = entry.get("status")
        if status not in VALID_STATUSES:
            raise ValueError("unknown metric status")
        n = entry.get("n")
        if n is not None and (type(n) is not int or n < 0):
            raise ValueError("invalid metric denominator")
        metric_value = entry.get("value")
        if status == "unmeasured":
            if metric_value is not None:
                raise ValueError("unmeasured metric cannot have value")
        elif status == "invalid" and metric_value is None:
            pass
        elif type(metric_value) not in (float, int) or not math.isfinite(metric_value):
            raise ValueError("measured or invalid metric must have finite numeric value")
        if entry.get("unit") == "ratio" and metric_value is not None:
            if not 0 <= metric_value <= 1:
                raise ValueError("ratio out of range")


def _validate_unified(value: dict[str, Any]) -> None:
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("kind") != "researchflow_unified_evaluation"
    ):
        raise ValueError("requires v0.7 unified evaluation v1")
    domains = value.get("dimensions")
    if not isinstance(domains, dict) or set(domains) != set(DOMAINS):
        raise ValueError("unified evaluation must contain four known dimensions")
    if value.get("not_a_single_score") is not True:
        raise ValueError("unified evaluation must not claim a single score")
    for name, dim in domains.items():
        if not isinstance(dim, dict) or dim.get("status") not in VALID_STATUSES:
            raise ValueError(f"invalid dimension status: {name}")
        _validate_metrics(dim.get("metrics"))
        if dim["status"] != "measured" and any(
            m["status"] == "measured" for m in dim["metrics"].values()
        ):
            raise ValueError("unmeasured or invalid dimension cannot contain measured scores")
        cases = dim.get("cases")
        if not isinstance(cases, list) or len(cases) > 10000:
            raise ValueError("invalid cases")
        seen = set()
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("invalid case row")
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not _ID.fullmatch(case_id):
                raise ValueError("invalid case id")
            if case_id in seen:
                raise ValueError("duplicate case id")
            seen.add(case_id)
        if name == "reliability" and dim["status"] == "measured":
            _replay_cases(dim)
        if name == "outcome" and dim["status"] == "measured":
            if (
                not cases
                or type(dim.get("sample_count")) is not int
                or dim["sample_count"] != len(cases)
            ):
                raise ValueError("inconsistent report sample count")
            if any(type(c.get("completion")) is not bool for c in cases):
                raise ValueError("invalid report completion")
            rate = dim["metrics"].get("completion_rate", {})
            if rate.get("value") != sum(c["completion"] for c in cases) / len(cases) or rate.get(
                "n"
            ) != len(cases):
                raise ValueError("inconsistent report completion rate")
            counts = dim.get("judge_counts")
            if counts is not None:
                if not isinstance(counts, dict) or any(
                    type(counts.get(k)) is not int or counts[k] < 0
                    for k in ("yes", "no", "unjudged")
                ):
                    raise ValueError("invalid Judge counts")
                for key, numerator, denominator in (
                    ("citation_support_among_judged", counts["yes"], counts["yes"] + counts["no"]),
                    (
                        "judge_coverage",
                        counts["yes"] + counts["no"],
                        sum(counts[k] for k in ("yes", "no", "unjudged")),
                    ),
                ):
                    entry = dim["metrics"].get(key, {})
                    if entry.get("status") == "measured" and (
                        not denominator
                        or entry.get("n") != denominator
                        or not math.isclose(entry["value"], numerator / denominator, abs_tol=1e-12)
                    ):
                        raise ValueError("Judge metric contradicts observed counts")
    if value.get("cross_source_pairing_verified") is not False:
        raise ValueError("v0.7 inputs must preserve unpaired source boundary")


def _validate_tool(value: dict[str, Any]) -> None:
    validate_tool_result(value)
    _validate_metrics(value["metrics"])


def _evidence(path: str | Path, label: str, value: dict) -> dict[str, str]:
    loaded = value if isinstance(value, _Loaded) else _read_json(path)
    if json.dumps(loaded, sort_keys=True) != json.dumps(value, sort_keys=True):
        raise ValueError("supplied data differs from recorded file")
    return {"kind": label, "filename": Path(path).name, "sha256": loaded.byte_sha256}


def build_interview_summary(
    unified: dict[str, Any],
    *,
    unified_path: str | Path,
    tool: dict[str, Any] | None = None,
    tool_path: str | Path | None = None,
    baseline: dict[str, Any] | None = None,
    baseline_path: str | Path | None = None,
    executor: dict | None = None,
    executor_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a summary and conservative triage inbox. No root cause inference."""
    _validate_unified(unified)
    if (tool is None) != (tool_path is None):
        raise ValueError("tool and its file must be supplied together")
    if (baseline is None) != (baseline_path is None):
        raise ValueError("baseline and its file must be supplied together")
    sources = [_evidence(unified_path, "candidate_unified", unified)]
    executor_summary = None
    if (executor is None) != (executor_path is None):
        raise ValueError("Executor result and path must be supplied together")
    if executor is not None:
        from .executor_regression import verify_suite

        details = verify_suite(Path(executor_path), executor)
        sources.append(_evidence(executor_path, "compiled_executor_contracts", executor))
        executor_summary = {
            "total": len(details),
            "passed": sum(d["passed"] for d in details),
            "live_agent_accuracy_measured": False,
            "cases": [
                {"case_id": d["case_id"], "status": d["status"], "passed": d["passed"]}
                for d in details
            ],
        }
    if tool is not None:
        _validate_tool(tool)
        sources.append(_evidence(tool_path, "controlled_tool_fixture", tool))
    comp: dict[str, Any] | None = None
    if baseline is not None:
        _validate_unified(baseline)
        sources.append(_evidence(baseline_path, "baseline_unified", baseline))
        # Existing strict comparator is the source of truth for Replay comparison.
        # It revalidates normalized contract, not original Result/Trace bytes.
        comp = compare_unified(baseline, unified)
    domains = unified["dimensions"]
    triage: list[dict[str, Any]] = []
    for case in domains["outcome"]["cases"]:
        if case.get("completion") is False:
            triage.append(
                {
                    "case_id": case["case_id"],
                    "source_dimension": "outcome",
                    "observed": "report_not_completed",
                    "replay_status": "not_reproduced_from_historical_run",
                    "root_cause": "unknown",
                    "review_status": "needs_manual_triage",
                    "action": "inspect report evidence/trace; construct controlled fixture before replay",
                }
            )
    for case in domains["reliability"]["cases"]:
        if case.get("test_result") == "fail":
            triage.append(
                {
                    "case_id": case["case_id"],
                    "source_dimension": "reliability",
                    "observed": "controlled_replay_contract_failed",
                    "replay_status": "controlled_case_executed",
                    "root_cause": "unknown",
                    "review_status": "needs_manual_triage",
                    "action": "inspect replay result and paired trace; confirm code vs fixture drift",
                }
            )
    # No synthetic Tool failure is represented as a real Agent bad case.
    if executor_summary is not None:
        for case in executor_summary["cases"]:
            if case["passed"] is False:
                triage.append(
                    {
                        "case_id": case["case_id"],
                        "source_dimension": "executor",
                        "observed": "controlled_executor_contract_failed",
                        "replay_status": "controlled_case_executed",
                        "root_cause": "unknown",
                        "review_status": "needs_manual_triage",
                        "action": "inspect Executor assertions and paired trace before changing the frozen contract",
                    }
                )
    triage_doc = {
        "schema_version": 1,
        "kind": "researchflow_badcase_triage_draft",
        "generated_at": datetime.now(UTC).isoformat(),
        "input_sha256": sources[0]["sha256"],
        "automatic_root_cause_claim": False,
        "auto_promoted_to_regression_set": False,
        "draft_count": len(triage),
        "drafts": triage,
        "note": "Manual investigation and frozen Gold required before promotion; never rewrite an old dataset.",
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "researchflow_interview_eval_summary",
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": sources,
        "datasets_not_paired": True,
        "dimensions": domains,
        "dataset": unified.get("dataset"),
        "comparison": comp,
        "comparison_scope": "Replay only; normalized inputs, original Trace chain not re-opened",
        "tool_fixture": (
            {
                "evidence_kind": "controlled_fixture",
                "actual_agent_executed": False,
                "metrics": {
                    key: {"unit": "count" if key == "forbidden_attempt_count" else "ratio", **entry}
                    for key, entry in tool["metrics"].items()
                },
                "cases_total": tool["cases_total"],
                "cases_measured": tool["cases_measured"],
                "gate_passed": tool.get("gate_passed"),
            }
            if tool is not None
            else None
        ),
        "badcase_draft_count": len(triage),
        "executor_regression": executor_summary,
        "evaluation_limitations": [
            "Research and Replay sample universes differ: no weighted total score.",
            "Replay pass means expected exception handling, not task completion.",
            "Judge support is conditional on judged claims; coverage must be reported separately.",
            "Historical scores and static tool fixtures are not a paired model A/B experiment.",
            "v0.8 Tool cases are synthetic, not measured real-Agent Tool Calling accuracy.",
            "Historical Judge metrics are not rescored; version 5 bounds Claim labels locally, unlike legacy versions.",
            "No claims of actual HTTP SDK retry count or priced API cost.",
            "No verified Git commit pairing; this report is offline and cannot block a GitHub PR.",
        ],
    }
    return summary, triage_doc


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _metric_display(entry: dict[str, Any]) -> str:
    status, value = entry["status"], entry["value"]
    if status == "unmeasured":
        return "未测量"
    if value is None:
        return "未测量（无效）"
    if entry.get("unit") == "ratio":
        display = f"{value * 100:.2f}%"
    elif entry.get("unit") == "seconds":
        display = f"{value:.2f} s"
    elif entry.get("unit") == "usd":
        display = f"${value:,.4f}"
    else:
        display = (
            f"{value:,.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
        )
    return display + ("（无效）" if status == "invalid" else "")


def _metrics_table(metrics: dict[str, Any]) -> str:
    rows = []
    for name, entry in metrics.items():
        if not isinstance(entry, dict):
            continue
        label = METRIC_LABELS.get(name, name)
        status = entry.get("status", "unmeasured")
        css = "invalid" if status == "invalid" else "muted" if status == "unmeasured" else ""
        denom = "—" if entry.get("n") is None else str(entry["n"])
        note = entry.get("note", "")
        rows.append(
            f"<tr><td>{_h(label)}</td><td class='{css}'>{_h(_metric_display(entry))}</td>"
            f"<td>{_h(denom)}</td><td class='small'>{_h(note)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>指标</th><th>结果</th><th>n / 分母</th><th>口径说明</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        if rows
        else "<p class='muted'>该维度尚未提供指标。</p>"
    )


def render_html(summary: dict[str, Any], triage: dict[str, Any]) -> str:
    """Self-contained, printable HTML with no scripts or external network assets."""
    domain_cards = []
    for name in DOMAINS:
        dim = summary["dimensions"][name]
        status = dim["status"]
        source = dim.get("source")
        evidence = source.get("basename", "") if isinstance(source, dict) else ""
        judge_note = ""
        if name == "outcome" and isinstance(dim.get("judge_counts"), dict):
            counts = dim["judge_counts"]
            if all(
                type(counts.get(k)) is int and counts[k] >= 0 for k in ("yes", "no", "unjudged")
            ):
                judged = counts["yes"] + counts["no"]
                judge_note = (
                    f"<p class='small'>Judge YES={_h(counts['yes'])}、NO={_h(counts['no'])}、"
                    f"未判定={_h(counts['unjudged'])}；引用支持率分母（已判定）={_h(judged)}</p>"
                )
        domain_cards.append(
            f"<section class='card'><div class='section-top'><h2>{_h(DOMAIN_LABELS[name])}</h2>"
            f"<span class='pill {status}'>{_h(status)}</span></div>"
            f"<p class='small'>样本量：{_h(dim.get('sample_count', '—'))} · 输入：{_h(evidence or '未提供')}</p>"
            f"{judge_note}{_metrics_table(dim['metrics'])}</section>"
        )
    tool = summary.get("tool_fixture")
    if tool:
        tool_content = (
            "<p class='warning'>受控合成契约，不代表真实 Agent 的 Tool Calling 准确率。</p>"
            f"<p class='small'>合成案例：{_h(tool['cases_measured'])} / {_h(tool['cases_total'])}，"
            f"契约门禁：{_h(tool['gate_passed'])}</p>" + _metrics_table(tool["metrics"])
        )
    else:
        tool_content = "<p class='muted'>工具契约结果未提供；本节未测量。</p>"
    comparison = summary.get("comparison")
    executor = summary.get("executor_regression")
    executor_html = ""
    if executor is not None:
        executor_rows = "".join(
            f"<tr><td>{_h(r['case_id'])}</td><td>{_h(r['status'])}</td><td>{_h(r['passed'])}</td></tr>"
            for r in executor["cases"]
        )
        executor_html = (
            "<section class='card'><h2>Executor 编译图契约</h2>"
            "<p class='warning'>真实编译图与工具校验；脚本模型 / Mock HTTP，不是真实模型的工具准确率。</p>"
            f"<p>符合预期：{_h(executor['passed'])} / {_h(executor['total'])}</p>"
            f"<table><thead><tr><th>Case</th><th>执行状态</th><th>符合预期</th></tr></thead><tbody>{executor_rows}</tbody></table></section>"
        )
    if comparison is None:
        comparison_html = "<p class='muted'>未提供 Baseline，不进行版本对比。</p>"
    elif comparison.get("comparable"):
        comparison_html = (
            f"<p>Replay 可比：<strong>YES</strong> · 退化 {_h(comparison['regressions'])} "
            f"· 恢复 {_h(comparison['recoveries'])} · 门禁 {_h(comparison['quality_gate_passed'])}</p>"
            "<p class='small'>仅比较同 Case 文件哈希的确定性 Replay；不代表两次真实模型运行质量对比。</p>"
        )
    else:
        comparison_html = (
            f"<p class='warning'>不可比较：{_h(comparison.get('reason', 'unknown'))}。"
            "未给出 PASS/FAIL 的版本质量结论。</p>"
        )
    rows = []
    for draft in triage["drafts"]:
        rows.append(
            "<tr>"
            f"<td>{_h(draft['case_id'])}</td>"
            f"<td>{_h(draft['source_dimension'])}</td>"
            f"<td>{_h(draft['observed'])}</td>"
            f"<td>{_h(draft['root_cause'])}</td></tr>"
        )
    triage_html = (
        "<table><thead><tr><th>Case</th><th>来源</th><th>现象</th><th>根因</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else "<p class='muted'>本次输入没有符合规则的失败案例。不能据此认定全部系统无故障。</p>"
    )
    source_rows = "".join(
        f"<tr><td>{_h(s['kind'])}</td><td>{_h(s['filename'])}</td>"
        f"<td class='code'>{_h(s['sha256'][:16])}…</td></tr>"
        for s in summary["sources"]
    )
    limitations = "".join(f"<li>{_h(item)}</li>" for item in summary["evaluation_limitations"])
    return f"""<!DOCTYPE html>
<html lang='zh-CN'><head><meta charset='utf-8'>
<meta http-equiv='Content-Security-Policy' content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>ResearchFlow · Agent Evaluation Evidence</title>
<style>
:root {{ color-scheme:light; }}
* {{box-sizing:border-box}}
body {{font:14px/1.65 system-ui,'Microsoft Yahei',sans-serif;color:#16253d;background:#f3f5f9;margin:0}}
main {{max-width:1120px;margin:auto;padding:40px 28px 60px}}
header {{background:#142a44;color:white;padding:38px 42px;border-radius:18px;margin-bottom:24px}}
header h1 {{font-size:29px;line-height:1.2;margin:0 0 9px}}
header p {{margin:0;color:#c1d0e2}}
h2 {{font-size:17px;margin:0}}
h3 {{font-size:14px;margin:0 0 12px}}
.card {{background:white;border:1px solid #e0e5ec;border-radius:14px;padding:22px 26px;margin:0 0 18px}}
.section-top {{display:flex;justify-content:space-between;gap:15px;align-items:center}}
.pill {{padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;background:#eaf2ec;color:#216443}}
.pill.unmeasured {{background:#f0f1f4;color:#59677b}}
.pill.invalid {{background:#ffebe7;color:#a84538}}
.small {{font-size:12px;color:#66768c}}
.muted {{color:#6c7789}}
.invalid,.warning {{color:#a84538}}
table {{border-collapse:collapse;width:100%;margin-top:14px;font-size:13px}}
th,td {{text-align:left;border-bottom:1px solid #e8edf2;padding:9px 12px 9px 0;vertical-align:top}}
th {{background:#f8fafc;color:#52647d;font-weight:600}}
.code {{font:12px ui-monospace,monospace}}
ul {{margin:10px 0 0;padding-left:20px}}
footer {{font-size:12px;color:#687891;padding-top:8px}}
@media print {{body {{background:#fff}} main {{max-width:none;padding:4mm}} header {{border-radius:0}} .card {{break-inside:avoid;border-color:#ddd}}}}
</style></head><body><main>
<header><h1>ResearchFlow · Agent Evaluation</h1>
<p>离线评测证据报告 ｜ Report / Replay / Retrieval / Grounding / Tool fixture</p></header>
<section class='card'><h2>阅读须知</h2>
<p>本报告汇总不同数据集和执行模式的独立证据，<strong>不是统一总分</strong>。历史结果、Fake Replay 与合成 Tool 契约不得相互冒充。</p>
<p class='small'>生成时间：{_h(summary["generated_at"])} ｜ 各输入原始字节 SHA256 已记录。</p></section>
{"".join(domain_cards)}
<section class='card'><h2>工具契约 / Tool & Safety</h2>{tool_content}</section>
{executor_html}
<section class='card'><h2>Replay 版本比较</h2>{comparison_html}</section>
<section class='card'><h2>Bad Case 待处理池 · {_h(triage["draft_count"])} 条</h2>
<p class='small'>仅根据确定的失败症状生成草稿，根因一律 unknown，不自动改写回归数据集。</p>
{triage_html}</section>
<section class='card'><h2>证据来源</h2>
<table><thead><tr><th>类型</th><th>原始文件名</th><th>SHA256 前缀</th></tr></thead><tbody>{source_rows}</tbody></table></section>
<section class='card'><h2>结论边界</h2><ul>{limitations}</ul></section>
<footer>ResearchFlow Agent Eval — 本地离线只读产物，不包含真实模型 API 再评测。</footer>
</main></body></html>"""


def _save_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def generate(
    *,
    unified_path: str | Path,
    html_out: str | Path,
    summary_out: str | Path | None = None,
    triage_out: str | Path | None = None,
    tool_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    executor_path: str | Path | None = None,
) -> dict[str, Any]:
    candidate = _read_json(unified_path)
    tool = _read_json(tool_path) if tool_path is not None else None
    baseline = _read_json(baseline_path) if baseline_path is not None else None
    executor = _read_json(executor_path) if executor_path is not None else None
    summary, triage = build_interview_summary(
        candidate,
        unified_path=unified_path,
        tool=tool,
        tool_path=tool_path,
        baseline=baseline,
        baseline_path=baseline_path,
        executor=executor,
        executor_path=executor_path,
    )
    content = render_html(summary, triage)
    artifacts = [(Path(html_out), content)]
    if summary_out is not None:
        artifacts.append(
            (
                Path(summary_out),
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            )
        )
    if triage_out is not None:
        artifacts.append(
            (
                Path(triage_out),
                json.dumps(triage, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            )
        )
    absolute_paths = [p.resolve() for p, _ in artifacts]
    inputs = {
        Path(p).resolve()
        for p in (unified_path, tool_path, baseline_path, executor_path)
        if p is not None
    }
    if len(absolute_paths) != len(set(absolute_paths)) or inputs.intersection(absolute_paths):
        raise ValueError("outputs must be unique new files and distinct from all inputs")
    if any(path.exists() for path, _ in artifacts):
        raise FileExistsError("cannot overwrite existing evaluation artifact")
    for path, _ in artifacts:
        check_output_path(path)
    created: list[Path] = []
    try:
        for path, text in artifacts:
            # Exclusive creation; never overwrite historical evidence.
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                stream.write(text)
    except OSError, UnicodeError:
        for created_path in created:
            created_path.unlink(missing_ok=True)
        raise
    return {
        "html": str(html_out),
        "summary": str(summary_out) if summary_out else None,
        "triage": str(triage_out) if triage_out else None,
        "triage_count": triage["draft_count"],
        "comparison": summary["comparison"],
        "paid_requests": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ResearchFlow offline interview closeout artifacts"
    )
    parser.add_argument("--unified", required=True, help="v0.7 unified.json, read-only")
    parser.add_argument("--tool", help="optional v0.8 CONTROLLED tool result, read-only")
    parser.add_argument("--baseline", help="optional v0.7 baseline unified.json, read-only")
    parser.add_argument(
        "--executor", help="optional compiled Executor suite with its complete artifact directory"
    )
    parser.add_argument("--html", required=True, help="new standalone HTML path")
    parser.add_argument("--summary", help="new summary JSON path")
    parser.add_argument("--triage", help="new badcase triage draft JSON path")
    args = parser.parse_args(argv)
    try:
        result = generate(
            unified_path=args.unified,
            tool_path=args.tool,
            executor_path=args.executor,
            baseline_path=args.baseline,
            html_out=args.html,
            summary_out=args.summary,
            triage_out=args.triage,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Interview report failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
