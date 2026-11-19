"""按用户隔离的执行证据与待排查视图，独立于评测分数。

执行异常不等于质量判据失败；中断运行仍保持质量未测量，但提供可排查事件。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from app.services.evaluation_catalog import CatalogError, safe_file
from app.services.evaluation_closure.core import SECRET_NAME, SECRET_STRING, input_hash
from app.services.evaluation_jobs import now, update_status
from app.services.evaluation_pipeline_v13 import DATASETS, _read_owned, _rows
from app.services.evaluation_timeline import build_timeline, timestamp
from research_engine.response_diagnostics import safe_response_diagnostic
from research_engine.trace_content import content_snapshot

TOKEN = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
MAX_BYTES = 8 * 1024 * 1024
ANALYSIS_VERSION = "rf-execution-diagnostics/4"
LABELS = {
    "runner_timeout": "评测运行器达到总时限",
    "worker_restarted_result_uncertain": "执行器重启，运行被中断",
    "suite_execution_failed": "评测子进程异常退出",
    "execution_artifacts_already_exist": "执行目录已有产物，已阻止重复运行和覆盖证据",
    "worker_internal_error": "评测执行器内部异常",
    "research_incomplete": "研究流程未完成",
    "case_assertion_failed": "案例合格判据未满足",
    "measurement_invalid": "评测指标无效，不作为有效质量成绩",
    "online_runner_disabled": "在线执行未启用",
    "judge_route_not_configured": "独立评分模型未配置",
    # 协作停止保留具体原因，不压成模糊的子进程失败；runner_timeout 名称保持兼容，以便解析历史超时记录。
    "research_draft_ready": "研究草稿已保存，等待人工审核",
    "research_failed": "研究阶段未完成，草稿未通过校验",
    "judge_partial": "评分部分完成，进度已保存，可断点续评",
    "judge_completed": "评分批次全部完成",
    "judge_failed": "评分阶段未取得任何有效结果",
    "artifact_save_failed": "结果或进度写入失败，未静默丢弃",
}


def token(value):
    return (
        value
        if isinstance(value, str) and TOKEN.fullmatch(value) and not SECRET_STRING.search(value)
        else None
    )


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def numeric_metrics(value, depth=0):
    if depth > 4:
        return None
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        return number(value)
    if isinstance(value, dict):
        return {
            key: numeric_metrics(child, depth + 1)
            for key, child in value.items()
            if isinstance(key, str)
            and re.fullmatch(r"[A-Za-z0-9_@./-]{1,80}", key)
            and not SECRET_NAME.search(key)
            and (child is None or type(child) in (bool, int, float, dict))
        }
    return None


def step_dir(folder: Path, dataset: str) -> Path:
    if dataset not in DATASETS:
        raise KeyError("dataset_not_found")
    path = folder / "artifacts" / dataset
    for parent in (folder, folder / "artifacts", path):
        if parent.is_symlink():
            raise ValueError("unsafe_artifact_path")
    return path


def begin_observation(folder: Path, request: dict, index: int, capture=None) -> None:
    path = step_dir(folder, request["datasets"][index])
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "pipeline_id": request["pipeline_id"],
        "dataset_id": request["datasets"][index],
        "started_at": now(),
        "state": "running",
        "input_signature": input_hash(capture.inputs) if capture is not None else None,
        "case_ids": list(capture.inputs) if capture is not None else [],
    }
    with (path / "execution-start.json").open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)


def finish_observation(folder: Path, dataset: str, result: dict, elapsed: float) -> None:
    path = step_dir(folder, dataset)
    update_status(
        path / "execution-end.json",
        {
            "finished_at": now(),
            "state": result["state"],
            "reason": result.get("reason"),
            "elapsed_seconds": round(elapsed, 3),
        },
    )


def _optional(base: Path, name: str):
    try:
        path = safe_file(base, name)
        # 产物不是任务请求：十题报告可超过任务元数据的 256 KB 上限，仍保留独立大小边界。
        if path.stat().st_size > MAX_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except CatalogError, OSError, ValueError:
        return {}


def _trace_files(base: Path):
    # 证据根目录由 Worker 固定选择，客户端不提供路径。
    if not base.is_dir() or base.is_symlink():
        return []
    found = []
    for path in sorted(base.rglob("*.jsonl")):
        if len(found) >= 64:
            break
        if "traces" not in path.relative_to(base).parts:
            continue
        try:
            found.append(safe_file(base, path.relative_to(base).as_posix()))
        except CatalogError, OSError, ValueError:
            continue
    return found


def traces(base: Path, allowed_cases: list[str], *, journal=False, case_filter=None) -> list[dict]:
    result, remaining = [], 12 * 1024 * 1024
    files = _trace_files(base) if not journal else _journal_files(base)
    for file in files:
        if case_filter and not journal:
            with file.open(encoding="utf-8") as stream:
                try:
                    header = json.loads(stream.readline(MAX_BYTES))
                except ValueError, UnicodeError:
                    continue
            if not isinstance(header, dict) or header.get("case_id") != case_filter:
                continue
        size = file.stat().st_size
        if size > min(MAX_BYTES, remaining):
            result.append({"trace_status": "unavailable", "events": [], "problem": "size_limit"})
            continue
        raw = file.read_bytes()
        remaining -= len(raw)
        events, identity, opened, problem = [], None, set(), None
        seen, begun = set(), {}
        for line in raw.splitlines()[:4000]:
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or event.get("event_version") != 2:
                    raise ValueError("unsupported_event")
                key = (event.get("trace_id"), event.get("run_id"), event.get("case_id"))
                if not all(token(x) for x in key) or (identity and key != identity):
                    raise ValueError("mixed_identity")
                if allowed_cases and key[2] not in allowed_cases:
                    raise ValueError("unexpected_case")
                if type(event.get("seq")) is not int or event["seq"] != len(events) + 1:
                    raise ValueError("broken_sequence")
                identity = key
                span = token(event.get("span_id"))
                status = token(event.get("status"))
                if (
                    status
                    not in {"start", "ok", "error", "business_error", "cancelled", "interrupt"}
                    or not span
                ):
                    raise ValueError("invalid_span_event")
                if status == "start" and span:
                    if span in seen:
                        raise ValueError("duplicate_start")
                    opened.add(span)
                    seen.add(span)
                    begun[span] = event
                elif span:
                    if span not in opened:
                        raise ValueError("unpaired_end")
                    if any(
                        event.get(k) != begun[span].get(k)
                        for k in ["name", "kind", "parent_span_id"]
                    ):
                        raise ValueError("span_identity_changed")
                    opened.remove(span)
                meta = event.get("metadata")
                meta = meta if isinstance(meta, dict) else {}
                safe_meta = {}
                for name in [
                    "role",
                    "tool_name",
                    "configured_model",
                    "served_model",
                    "disclosed_alias",
                    "schema_status",
                    "parsing_error_type",
                    "strategy",
                ]:
                    if token(meta.get(name)):
                        safe_meta[name] = meta[name]
                for name in ["input_tokens", "output_tokens", "http_status", "attempt"]:
                    if type(meta.get(name)) is int and 0 <= meta[name] < 10**12:
                        safe_meta[name] = meta[name]
                events.append(
                    {
                        "event_version": 2,
                        "seq": event["seq"],
                        "case_id": key[2],
                        "timestamp": timestamp(event.get("timestamp")),
                        "span_id": span,
                        "parent_span_id": token(event.get("parent_span_id")),
                        "name": token(event.get("name")),
                        "kind": token(event.get("kind")),
                        "status": status,
                        "duration_ms": number(event.get("duration_ms")),
                        "error_type": token(event.get("error_type")),
                        "metadata": safe_meta,
                        **(
                            {
                                "response_diagnostic": safe_response_diagnostic(
                                    event["response_diagnostic"]
                                )
                            }
                            if "response_diagnostic" in event
                            else {}
                        ),
                        **(
                            {"details": safe_details(event["details"])}
                            if "details" in event
                            else {}
                        ),
                    }
                )
            except ValueError, TypeError, AttributeError, UnicodeError:
                problem = "invalid_or_truncated_event"
                break
        if len(raw.splitlines()) > 4000:
            problem = "event_limit"
        parents = {e["span_id"]: e["parent_span_id"] for e in events}
        for span in parents:
            parent, ancestors = parents[span], {span}
            while parent:
                if parent in ancestors or parent not in parents or len(ancestors) > 128:
                    problem = "invalid_parent_hierarchy"
                    break
                ancestors.add(parent)
                parent = parents[parent]
        result.append(
            {
                "trace_id": identity[0] if identity else None,
                "run_id": identity[1] if identity else None,
                "case_id": identity[2] if identity else None,
                "trace_status": "partial" if problem or opened or not events else "recorded",
                "problem": problem,
                "open_spans": sorted(opened),
                "events": events,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return result


def safe_details(value):
    # 读取时再次脱敏，API 不能信任已存产物中的内容。
    clean = content_snapshot(value, max_chars=8192, max_bytes=20000, max_depth=12)
    return clean["value"] if not clean["truncated"] else clean


def _journal_files(base):
    directory = base / "execution-traces"
    if not directory.is_dir() or directory.is_symlink():
        return []
    result = []
    for file in sorted(directory.glob("*.jsonl"))[:32]:
        try:
            result.append(safe_file(base, file.relative_to(base).as_posix()))
        except CatalogError, OSError, ValueError:
            continue
    return result


def unified_trace(graph, judges, lifecycle):
    events = []
    sources = []
    for prefix, traces_ in [("G", graph), ("J", judges), ("E", lifecycle)]:
        for i, trace in enumerate(traces_):
            ref = f"{prefix}{i}"
            sources.append(
                {
                    "id": ref,
                    "sha256": trace.get("sha256"),
                    "run_id": trace.get("run_id"),
                    "case_id": trace.get("case_id"),
                    "status": trace.get(
                        "trace_status", "recorded" if trace.get("trace_complete") else "partial"
                    ),
                }
            )
            for event in trace.get("events", []):
                events.append(
                    {
                        "evidence_id": f"{ref}:{event['seq']}",
                        "source": prefix,
                        "case_id": trace.get("case_id"),
                        **event,
                    }
                )
    events.sort(key=lambda e: (e.get("timestamp") or "", e["evidence_id"]))
    details = [e for e in events if isinstance(e.get("details"), dict)]
    populated = [e for e in details if e["details"].get("value") not in (None, {}, {"input": None})]
    return {
        "sources": sources,
        "events": events,
        "coverage": {
            "loaded_events": len(events),
            "content_events": len(populated),
            "content_truncated_events": sum(
                bool(e["details"].get("truncated"))
                for e in details
                if isinstance(e["details"], dict)
            ),
            "not_collected": [
                "SDK 内部 HTTP 重试与服务端内部执行未采集",
                "不保存请求头、密钥、二进制内容和完整向量数组",
                "历史运行没有记录的正文无法补回",
            ],
            "note": "统一展示已采集的脱敏记录；事件条数完整不等于所有内容完整。",
        },
    }


def case_observations(base: Path, dataset: str, *, use_canonical=True) -> list[dict]:
    canonical = _optional(base, "case-observations.json") if use_canonical else {}
    if canonical:
        if (
            type(canonical.get("schema_version")) is not int
            or canonical.get("schema_version") != 1
            or canonical.get("dataset_id") != dataset
            or not isinstance(canonical.get("cases"), list)
        ):
            return []
        raw = {row["case_id"]: row for row in case_observations(base, dataset, use_canonical=False)}
        return [
            {
                "case_id": row["case_id"],
                "status": row["status"],
                "latency_s": number(row.get("latency_s")),
                "claim_count": raw.get(row["case_id"], {}).get("claim_count"),
                "metrics": {
                    **raw.get(row["case_id"], {}).get("metrics", {}),
                    **(numeric_metrics(row.get("metrics", {})) or {}),
                },
                "error": raw.get(row["case_id"], {}).get("error"),
            }
            for row in canonical["cases"][:1000]
            if isinstance(row, dict)
            and token(row.get("case_id"))
            and row.get("status") in {"passed", "failed", "blocked", "unmeasured"}
        ]
    filename = {
        "planner-replay-v1": "artifacts/result/replay/suite.json",
        "executor-regression-v1": "executor-result.json",
        "tool-contract-v1": "artifacts/tool-result.json",
    }.get(dataset, "raw_result.json")
    doc = _optional(base, filename)
    rows = doc.get("results", doc.get("claims", doc.get("queries", doc.get("cases", []))))
    if not isinstance(rows, list):
        return []
    output = []
    for row in rows[:1000]:
        if not isinstance(row, dict):
            continue
        ident = token(row.get("id", row.get("case_id")))
        if not ident:
            continue
        passed = (
            row.get("completed")
            if dataset == "report-benchmark"
            else row.get("pass")
            if dataset == "tool-contract-v1"
            else row.get("passed")
        )
        if (
            dataset == "judge-calibration"
            and type(row.get("predicted_supported")) is bool
            and type(row.get("expected_supported")) is bool
        ):
            passed = row["predicted_supported"] == row["expected_supported"]
        latency = number(row.get("latency_s"))
        metric_keys = {
            "precision",
            "recall",
            "ndcg@5",
            "ndcg@10",
            "dedup_rate",
            "span_accuracy",
            "passages",
            "metrics",
            "assertions",
            "with_cap",
            "without_cap",
            "selection_correct",
            "arguments_correct",
            "trajectory_correct",
            "forbidden_attempt_count",
            "observed_count",
            "expected_count",
            "judged",
            "predicted_supported",
            "expected_supported",
            "leakage",
            "precision_at_5",
            "recall_at_5",
            "passage_recall",
        }
        observed_metrics = numeric_metrics({k: v for k, v in row.items() if k in metric_keys})
        output.append(
            {
                "case_id": ident,
                "status": "passed"
                if passed is True
                else "failed"
                if passed is False
                else "unmeasured",
                "latency_s": latency,
                "metrics": observed_metrics,
                "claim_count": len(row["claim_verdicts"])
                if isinstance(row.get("claim_verdicts"), list)
                else None,
                "error": SECRET_STRING.sub("[REDACTED]", row["error"][:1600])
                if isinstance(row.get("error"), str)
                else None,
            }
        )
    return output


def judge_observations(
    base: Path, graph=None, allowed_cases=None, *, case_filter=None
) -> list[dict]:
    directory = base / "judge-batches"
    if not directory.is_dir() or directory.is_symlink():
        return []
    from evals.agent_eval_ext.judge_runtime_inspect import summarize

    rows = []
    for path in sorted(directory.glob("*.jsonl"))[:32]:
        try:
            file = safe_file(directory, path.name)
            if file.stat().st_size > MAX_BYTES:
                continue
            header = json.loads(file.read_text(encoding="utf-8").splitlines()[0])
            if case_filter and header.get("case_id") != case_filter:
                continue
            doc = _optional(base, "raw_result.json")
            arm = (
                doc.get("candidate", {})
                if str(header.get("phase", "")).startswith("candidate")
                else doc
            )
            matches = [
                r
                for r in arm.get("results", [])
                if isinstance(r, dict) and r.get("id") == header.get("case_id")
            ]
            full_claims = matches[0].get("claim_verdicts") if len(matches) == 1 else None
            result = summarize(
                file, total_claims=len(full_claims) if isinstance(full_claims, list) else None
            )
            if not all(token(result.get(k)) for k in ["trace_id", "run_id", "case_id"]):
                raise ValueError("invalid_judge_identity")
            raw_events = [
                json.loads(line)
                for line in file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if allowed_cases and result["case_id"] not in allowed_cases:
                raise ValueError("unexpected_judge_case")
            identities = {
                (t.get("run_id"), t.get("case_id")) for t in (graph or []) if t.get("run_id")
            }
            if identities and (result["run_id"], result["case_id"]) not in identities:
                raise ValueError("unmatched_judge_run")
            safe_events = []
            for event in raw_events:
                row = {
                    "event": event["event"],
                    "seq": event["seq"],
                    "timestamp": timestamp(event.get("timestamp")),
                }
                if "details" in event:
                    row["details"] = safe_details(event["details"])
                if "response_diagnostic" in event:
                    row["response_diagnostic"] = safe_response_diagnostic(
                        event["response_diagnostic"]
                    )
                for key in [
                    "status",
                    "reason_hint",
                    "exception_type",
                    "served_model",
                    "judge_route",
                    "disclosed_alias",
                ]:
                    if token(event.get(key)):
                        row[key] = event[key]
                for key in [
                    "batch_no",
                    "batch_start_zero_based",
                    "expected_count",
                    "judged_count",
                    "unjudged_count",
                    "yes_count",
                    "no_count",
                    "http_status",
                    "application_level_calls",
                    "elapsed_ms",
                    "wait_seconds",
                ]:
                    value = number(event.get(key))
                    if value is not None and value >= 0:
                        row[key] = value
                safe_events.append(row)
            dates = [e["timestamp"] for e in safe_events]
            rows.append(
                {
                    "run_id": result["run_id"],
                    "sha256": result["input_sha256"],
                    "events": safe_events,
                    "timeline_valid": bool(dates) and all(dates) and dates == sorted(dates),
                    **{
                        key: result.get(key)
                        for key in [
                            "case_id",
                            "phase",
                            "trace_complete",
                            "observed_batches",
                            "observed_application_level_calls",
                            "judge_coverage",
                            "observed_judge_coverage",
                            "sdk_retry_attempts",
                            "problem_batches",
                            "total_claims",
                            "total_batches",
                            "successful_batches",
                            "attempted_batches",
                            "new_requested_batches",
                            "responses_received",
                            "successful_requests",
                            "valid_result_batches",
                            "judged_claims_total",
                            "unjudged_claims_total",
                            "full_judge_coverage",
                            "attempted_judge_coverage",
                            "scoring_complete",
                            "batch_measurements",
                            "known_input_tokens",
                            "known_output_tokens",
                            "batches_missing_usage",
                            "wait_observations",
                        ]
                    },
                }
            )
        except CatalogError, OSError, ValueError, TypeError, KeyError:
            rows.append({"trace_complete": False, "problem": "invalid_or_truncated_judge_trace"})
    return rows


def analyze(request: dict, step: dict, base: Path, *, case_id=None) -> dict:
    start = _optional(base, "execution-start.json")
    end = _optional(base, "execution-end.json")
    process = _optional(base, "process.json")
    allowed_cases = start.get("case_ids") or (
        [request["case_id"]] if request.get("case_id") else []
    )
    cases = case_observations(base, step["dataset_id"])
    available_cases = [row["case_id"] for row in cases]
    if case_id and case_id not in {row["case_id"] for row in cases} | set(allowed_cases):
        raise KeyError("case_not_found")
    if case_id:
        cases = [row for row in cases if row["case_id"] == case_id]
    graph = traces(base, allowed_cases, case_filter=case_id)
    events = [e for t in graph for e in t["events"]]
    failures = [
        e
        for e in events
        if e["status"] in {"error", "business_error", "cancelled"}
        or e["metadata"].get("schema_status") == "invalid"
    ]
    active = [
        e
        for t in graph
        for e in t["events"]
        if e["status"] == "start" and e["span_id"] in t.get("open_spans", [])
    ]
    judges = judge_observations(base, graph, allowed_cases, case_filter=case_id)
    from evals.agent_eval_ext.report_state import report_states

    report_status = report_states(base, judges) if step["dataset_id"] == "report-benchmark" else []
    if case_id:
        report_status = [row for row in report_status if row.get("case_id") == case_id]
    lifecycle = traces(base, [], journal=True)
    if base.parent.name == "artifacts":
        lifecycle += traces(
            base.parents[1] / "execution-observations" / base.name, [], journal=True
        )
    lifecycle = [t for t in lifecycle if t.get("run_id") == request["pipeline_id"]]
    metrics = (cases[0].get("metrics") or {}) if case_id and cases else step.get("metrics") or {}
    quality_failed = (
        any(c["status"] == "failed" for c in cases)
        or metrics.get("gate_passed") is False
        or metrics.get("meets_threshold") is False
    )
    reason = step.get("reason") or (
        "measurement_invalid"
        if metrics.get("invalid_metric") is True
        else "research_incomplete"
        if any(c["status"] == "failed" for c in cases)
        else "case_assertion_failed"
        if quality_failed
        else None
    )
    issue = step["state"] in {"failed", "blocked"} or quality_failed
    if case_id and cases:
        issue = cases[0]["status"] in {"failed", "blocked", "unmeasured"}
    state_name = {
        "failed": "执行失败",
        "completed": "执行已结束",
        "blocked": "未就绪",
        "pending": "待执行",
        "running": "运行中",
    }.get(step["state"], step["state"])
    facts = [
        {
            "id": "F1",
            "text": f"运行状态：{state_name}；已记录耗时：{step.get('elapsed_seconds')} 秒。",
        }
    ]
    if reason:
        facts.append(
            {"id": "F2", "text": LABELS.get(reason, "执行或配置检查未满足"), "code": reason}
        )
    facts.append(
        {
            "id": "F3",
            "text": f"取得 {len(graph)} 份图轨迹、{len(events)} 条事件；{len(active)} 个片段尚未结束。",
        }
    )
    timeline = build_timeline(graph, judges, start, end, reason)
    if case_id:
        for row in cases:
            if row.get("error"):
                facts.append({"id": "C1", "text": row["error"]})
    if not issue and step["state"] == "completed":
        timeline.update(
            summary="本轮执行已结束，评测器未登记未通过案例。受控故障可能包含预期错误，请以逐题合格判据为准。",
            phase=None,
            certainty="no_incident",
            evidence_ids=[],
        )
    elif step["state"] in {"pending", "running"}:
        timeline.update(
            summary="任务尚在执行，当前记录不能作为最终失败归因。", certainty="in_progress"
        )
    candidates, actions = [], []
    for event in failures[:12]:
        eid = f"S{len(facts)}"
        facts.append(
            {
                "id": eid,
                "text": f"{event['case_id']} / {event['name']}：{event['status']}，异常类型 {event['error_type'] or '未披露'}。",
                "span_id": event["span_id"],
            }
        )
        error = event["error_type"] or event["metadata"].get("parsing_error_type", "")
        http = event["metadata"].get("http_status")
        if http == 429 or error == "RateLimitError":
            cause, verify = (
                "模型服务限流或配额限制",
                "核对供应商该时段的限流/额度记录；检查并发与请求速率。",
            )
        elif http in {401, 403} or error in {"AuthenticationError", "PermissionDeniedError"}:
            cause, verify = (
                "认证或访问权限被拒绝",
                "核对所用路由、凭据有效性和工具访问策略，不在页面粘贴密钥。",
            )
        elif error in {"ValidationError", "OutputParserException", "JSONDecodeError"}:
            cause, verify = (
                "响应未满足结构或工具参数契约",
                "核对失败节点的 Schema、解析步骤及重试次数；用固定受控响应构建回归案例。",
            )
        elif "Timeout" in error:
            cause, verify = (
                "上游模型或工具请求超时",
                "核对该片段所属模型/工具及其耗时，再检查端点连通性和供应商日志。",
            )
        elif http and http >= 500:
            cause, verify = (
                "上游服务返回错误",
                "核对供应商状态、响应状态码和同一时段其他请求；避免直接重复付费运行。",
            )
        else:
            cause, verify = (
                "节点执行异常或异常向上层传播",
                "从最早的底层失败片段检查触发条件；父节点失败可能只是传播结果。",
            )
        candidates.append(
            {
                "cause": cause,
                "confidence": "有直接异常证据",
                "evidence_ids": [eid],
                "verification": verify,
            }
        )
    if reason == "runner_timeout":
        pending = active[-1]["name"] if active else None
        candidates.insert(
            0,
            {
                "cause": f"运行器总时限已触发；终止时片段为 {pending}"
                if pending
                else "运行器总时限已触发；缺少可定位底层等待的轨迹",
                "confidence": "终止原因已确认，底层诱因待验证",
                "evidence_ids": ["F1", "F2", "F3"],
                "verification": "对照未结束的模型/工具片段检查等待时间、重试和任务量；缺少轨迹时先补采集，再决定是否调整总时限。",
            },
        )
        if not active and not failures:
            candidates.extend(
                [
                    {
                        "cause": "模型请求或网关内部重试可能累计占用了总时限",
                        "confidence": "待验证假设，缺少节点轨迹",
                        "evidence_ids": ["F1", "F2", "F3"],
                        "verification": "核对供应商同一时段的请求耗时、客户端请求时限与内部重试配置；不要把运行器超时直接认作模型超时。",
                    },
                    {
                        "cause": "搜索、网页读取等待或研究任务量可能超过本轮时间预算",
                        "confidence": "待验证假设，缺少工具轨迹",
                        "evidence_ids": ["F1", "F2", "F3"],
                        "verification": "在下一次显式运行中对照工具耗时、任务数和研究深度；用受控慢工具案例验证运行器终止与资源清理。",
                    },
                ]
            )
        actions.extend(
            [
                "检查未结束片段是模型调用、工具请求还是图节点。",
                "将运行器总时限与模型请求时限、检索任务量分别核对。",
            ]
        )
    if issue and not candidates:
        candidates.append(
            {
                "cause": LABELS.get(reason, "执行配置或案例合格判据未满足"),
                "confidence": "基于运行状态或判据结果",
                "evidence_ids": ["F1", "F3"],
                "verification": "检查本次结果与固定输入/预期，补充节点证据后区分业务失败、运行器失败与环境问题。",
            }
        )
    actions.extend([c["verification"] for c in candidates[:4]])
    if issue:
        actions.append("将固定输入和合格判据审核后加入回归；修复后使用新的同契约运行验证。")
    missing = []
    if not graph:
        missing.append(
            "没有图 Trace：无法事后确定具体模型/工具等待；已有状态仍可证明运行器层面的异常。"
        )
    if any(t["trace_status"] != "recorded" for t in graph):
        missing.append("轨迹不完整：保留已落盘事件，不补造结束事件，也不推算未披露用量。")
    if step["dataset_id"] not in {"report-benchmark", "planner-replay-v1"}:
        missing.append(
            "此适配器提供运行级结果；未采集其全部内部模型/工具片段，不能当作完整 Agent 图轨迹。"
        )
    result = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "pipeline_id": request["pipeline_id"],
        "dataset_id": step["dataset_id"],
        "case_id": case_id or request.get("case_id"),
        "state": step["state"],
        "reason": reason,
        "elapsed_seconds": step.get("elapsed_seconds"),
        "has_issue": issue,
        "collection": "worker_observation" if start else "saved_status_analysis",
        "input_signature": start.get("input_signature"),
        "execution_observation": {
            "started_at": start.get("started_at"),
            "finished_at": end.get("finished_at"),
            "exit_code": process.get("exit_code"),
            "timed_out": process.get("timed_out"),
            "stop_mode": process.get("stop_mode"),
            "reliability_status": process.get("reliability_status"),
            "reliability_reason": process.get("reliability_reason"),
        },
        "root_cause_confirmed": False,
        "facts": facts,
        "candidates": candidates,
        "next_actions": list(dict.fromkeys(actions)),
        "missing_evidence": missing,
        "failure_phase": timeline["phase"],
        "timeline": timeline,
        "traces": graph,
        "judge_traces": judges,
        "report_states": report_status,
        "execution_traces": lifecycle,
        "unified_trace": unified_trace(graph, judges, [] if case_id else lifecycle),
        "cases": cases,
        "metrics": metrics,
        "quality_measurement": "unmeasured"
        if step["state"] in {"failed", "blocked"}
        else "evaluator_result",
        "sdk_retry_attempts": None,
    }
    if timeline["phase"]:
        result["candidates"] = (
            [
                {
                    "cause": timeline["summary"],
                    "confidence": "基于跨阶段时间线，等待推断另行标注",
                    "evidence_ids": timeline["evidence_ids"] or ["F1", "F2", "F3"],
                    "verification": "先核对本轮研究、独立评分及批间等待耗时，分别检查阶段预算和请求限时；已继续执行的早期症状不视作终止原因。",
                }
            ]
            if issue
            else []
        )
        result["next_actions"] = (
            [
                "核对研究图、独立评分与整轮任务的时间预算，避免评分或固定等待耗尽剩余时间。"
                if str(timeline["phase"]).startswith("judge")
                else "对照最后执行阶段检查未完成的调用或后处理步骤。",
                "保留本次原始记录；修复后发起新的同输入运行验证，不覆盖历史成绩。",
            ]
            if issue
            else []
        )
    else:
        # 合并重复症状；状态为错误本身不能证明因果关系。
        dedup = {}
        for candidate in result["candidates"]:
            key = candidate["cause"]
            if key not in dedup:
                dedup[key] = candidate
            else:
                dedup[key]["evidence_ids"] = list(
                    dict.fromkeys(dedup[key]["evidence_ids"] + candidate["evidence_ids"])
                )
        result["candidates"] = list(dedup.values())[:3]
    if not issue:
        result["candidates"] = []
        result["next_actions"] = []
    result["available_case_ids"] = available_cases
    if case_id and cases and cases[0].get("error"):
        # 已保存的终止错误是事实，早先恢复的工具异常不能替代它。
        error = cases[0]["error"]
        phase = error.split(":", 1)[0]
        result["failure_phase"] = (
            phase
            if phase in {"planner", "executor", "critic", "synthesizer"}
            else result["failure_phase"]
        )
        result["timeline"].update(
            summary=f"案例 {case_id} 未完成研究；原始错误：{error}",
            phase=result["failure_phase"],
            certainty="observed",
            evidence_ids=["C1"],
        )
        result["next_actions"] = [
            "查看本题终止错误及对应节点的输入、输出和结构校验记录；工具的早期错误仅作为过程症状。",
            "修复后使用同一案例的新运行复测，保留本次评分与原始产物。",
        ]
    if case_id and step["dataset_id"] == "report-benchmark":
        raw = _optional(base, "raw_result.json")
        matching = [
            r for r in raw.get("results", []) if isinstance(r, dict) and r.get("id") == case_id
        ]
        if len(matching) == 1:
            row = matching[0]
            report = row.get("report")
            text = (
                report
                if isinstance(report, str)
                else report.get("draft_report", "")
                if isinstance(report, dict)
                else ""
            )
            verdicts = row.get("claim_verdicts") or []
            result["case_evidence"] = {
                "report_excerpt": SECRET_STRING.sub("[REDACTED]", text[:16000]),
                "report_truncated": len(text) > 16000,
                "claims": [
                    {
                        "text": SECRET_STRING.sub(
                            "[REDACTED]", str(v.get("claim", v.get("text", "")))[:2000]
                        ),
                        "judged": v.get("judged") is True,
                        "supported": v.get("supported")
                        if type(v.get("supported")) is bool
                        else None,
                    }
                    for v in verdicts[:250]
                    if isinstance(v, dict)
                ],
                "claims_truncated": len(verdicts) > 250,
            }
    if issue and reason in {"judge_partial", "judge_failed"}:
        # 部分评分正常结束不等于研究节点失败；已确认的是判定缺口，不能猜空回复的供应商根因。
        partial = [
            (i, j)
            for i, j in enumerate(judges)
            if j.get("run_id") == request["pipeline_id"]
            and j.get("unjudged_claims_total", 0)
            and any(
                s.get("case_id") == j.get("case_id") and s.get("research") == "draft_saved"
                for s in report_status
            )
        ]
        if partial:
            missing = sum(j["unjudged_claims_total"] for _, j in partial)
            refs = [
                f"J{i}:{e['seq']}"
                for i, j in partial
                for e in j["events"]
                if e.get("event") == "batch_finished" and e.get("unjudged_count", 0)
            ]
            summary = f"研究草稿已生成并保存；独立评分仍有 {missing} 条结论未判定，完整评测未完成。"
            problems = [p for _, j in partial for p in j.get("problem_batches", [])]
            if problems and all(
                p.get("reason_hint") == "empty_or_nontext_response" for p in problems
            ):
                summary += "异常评分批次未取得可判定文本；空回复的底层原因尚未确认。"
            result["failure_phase"] = "judge"
            result["timeline"].update(
                phase="judge", certainty="observed", summary=summary, evidence_ids=refs
            )
            verification = "核对未完成批次的回复文本、输出用量、结束原因和实际时间限制；复用已保存草稿及有效提交，不把已恢复的研究工具错误当作终止原因。"
            result["candidates"] = [
                {
                    "cause": summary,
                    "confidence": "已确认评分缺口，底层原因按批次证据核验",
                    "evidence_ids": refs,
                    "verification": verification,
                }
            ]
            result["next_actions"] = [
                verification,
                "任何补评均需新的明确执行授权；保留本次原始记录，不覆盖历史成绩。",
            ]
    result["evidence_fingerprint"] = input_hash(result)
    return result


def read_diagnostics(root: Path, owner: str, pipe: str, dataset: str, case_id=None) -> dict:
    folder, request, status = _read_owned(root, owner, pipe)
    step = next((s for s in status["steps"] if s["dataset_id"] == dataset), None)
    if step is None:
        raise KeyError("dataset_not_found")
    base = step_dir(folder, dataset)
    return analyze(request, step, base, case_id=case_id)


def incident_queue(root: Path, owner: str, pipeline_id=None, dataset_id=None) -> list[dict]:
    groups = {}
    for folder, request, status in _rows(root):
        if request.get("owner") != owner:
            continue
        if pipeline_id and request["pipeline_id"] != pipeline_id:
            continue
        for step in status["steps"]:
            if dataset_id and step["dataset_id"] != dataset_id:
                continue
            if step["state"] not in {"failed", "blocked", "completed"}:
                continue
            diagnostic = analyze(request, step, step_dir(folder, step["dataset_id"]))
            if not diagnostic["has_issue"]:
                continue
            bad_rows = [
                r
                for r in diagnostic["cases"]
                if r["status"] == "failed"
                or (
                    r["status"] in {"blocked", "unmeasured"}
                    and (request.get("case_id") == r["case_id"] or r.get("error"))
                )
            ]
            if bad_rows:
                for row in bad_rows:
                    reason = diagnostic["reason"] or "case_assertion_failed"
                    key = input_hash(
                        [
                            owner,
                            step["dataset_id"],
                            row["case_id"],
                            diagnostic["input_signature"],
                            reason,
                        ]
                    )
                    if key not in groups:
                        groups[key] = {
                            "incident_id": key,
                            "pipeline_id": request["pipeline_id"],
                            "dataset_id": step["dataset_id"],
                            "case_id": row["case_id"],
                            "reason": reason,
                            "occurrences": [],
                            "summary": row.get("error")
                            or LABELS.get(reason, "案例未完成或合格判据未满足"),
                            "collection": diagnostic["collection"],
                        }
                    groups[key]["occurrences"].append(request["pipeline_id"])
                continue
            key = input_hash(
                [
                    owner,
                    step["dataset_id"],
                    [request.get("case_id"), request.get("max_cases")],
                    diagnostic["input_signature"],
                    diagnostic["reason"],
                ]
            )
            if key not in groups:
                groups[key] = {
                    "incident_id": key,
                    "pipeline_id": request["pipeline_id"],
                    "dataset_id": step["dataset_id"],
                    "case_id": request.get("case_id"),
                    "reason": diagnostic["reason"],
                    "occurrences": [],
                    "summary": diagnostic["candidates"][0]["cause"],
                    "collection": diagnostic["collection"],
                }
            groups[key]["occurrences"].append(request["pipeline_id"])
    return list(groups.values())
