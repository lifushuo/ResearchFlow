"""关联真实阶段时间线，不把已恢复的早期症状当作停止原因。"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from statistics import median


def timestamp(value):
    if not isinstance(value, str) or len(value) > 45:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(UTC).isoformat() if parsed.tzinfo else None
    except ValueError, OverflowError:
        return None


def seconds(value):
    clean = timestamp(value)
    return datetime.fromisoformat(clean).timestamp() if clean else None


def interval(start, end):
    a, b = seconds(start), seconds(end)
    return (a, b) if a is not None and b is not None and b >= a else None


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def build_timeline(graph, judges, start, end, reason):
    worker_bounds = interval(start.get("started_at"), end.get("finished_at"))
    spans, opens, symptoms = [], [], {}
    complete_graphs = []
    for index, trace in enumerate(graph):
        begins = {e["span_id"]: e for e in trace["events"] if e["status"] == "start"}
        finished = {e["span_id"]: e for e in trace["events"] if e["status"] != "start"}
        for ident, event in begins.items():
            if ident not in finished:
                opens.append({**event, "evidence_id": f"G{index}:{event['seq']}"})
        for ident, event in finished.items():
            begin = begins.get(ident, {})
            item = {
                **event,
                "started_at": begin.get("timestamp"),
                "evidence_id": f"G{index}:{event['seq']}",
            }
            bounds = interval(item["started_at"], item.get("timestamp"))
            item["bounds"] = bounds
            ancestor, ancestry = ident, set()
            item["root_span_id"] = None
            while ancestor in begins and ancestor not in ancestry:
                ancestry.add(ancestor)
                if begins[ancestor]["kind"] == "graph_node":
                    item["root_span_id"] = ancestor
                ancestor = begins[ancestor].get("parent_span_id")
            spans.append(item)
            if event["status"] not in {"error", "business_error", "cancelled"}:
                continue
            parent, seen, continued = ident, set(), False
            while parent in begins and parent not in seen:
                seen.add(parent)
                closing = finished.get(parent, {})
                if begins[parent].get("kind") == "graph_node" and closing.get("status") == "ok":
                    continued = True
                parent = begins[parent].get("parent_span_id")
            key = (event["case_id"], event["name"], event["error_type"], continued)
            row = symptoms.setdefault(
                key,
                {
                    "case_id": event["case_id"],
                    "name": event["name"],
                    "error_type": event["error_type"],
                    "count": 0,
                    "workflow_continued": continued,
                    "evidence_ids": [],
                    "last_at": None,
                },
            )
            row["count"] += 1
            row["last_at"] = event.get("timestamp")
            if len(row["evidence_ids"]) < 4:
                row["evidence_ids"].append(item["evidence_id"])
        if trace["trace_status"] == "recorded":
            dated = [seconds(e.get("timestamp")) for e in trace["events"]]
            if (
                dated
                and all(t is not None for t in dated)
                and dated == sorted(dated)
                and (
                    not worker_bounds
                    or (min(dated) >= worker_bounds[0] and max(dated) <= worker_bounds[1] + 0.05)
                )
            ):
                complete_graphs.append((min(dated), max(dated)))

    groups = defaultdict(list)
    for s in spans:
        groups[(s["kind"], s["name"])].append(s)
    statistics = []
    for (kind, name), items in sorted(groups.items()):
        measured = [s["duration_ms"] / 1000 for s in items if finite(s["duration_ms"])]
        statistics.append(
            {
                "kind": kind,
                "name": name,
                "calls": len(items),
                "measured_calls": len(measured),
                "sum_s": round(sum(measured), 6) if measured else None,
                "max_s": round(max(measured), 6) if measured else None,
                "errors": sum(
                    s["status"] in {"error", "business_error", "cancelled"} for s in items
                ),
            }
        )
    slowest = sorted(
        [s for s in spans if s["kind"] in {"llm", "tool"} and finite(s["duration_ms"])],
        key=lambda s: s["duration_ms"],
        reverse=True,
    )[:8]
    slowest = [
        {k: s.get(k) for k in ["case_id", "kind", "name", "duration_ms", "status", "evidence_id"]}
        for s in slowest
    ]
    cutoff = seconds(end.get("finished_at"))
    stages = []
    if complete_graphs:
        stages.append(
            {
                "name": "研究图",
                "duration_s": round(union(complete_graphs), 6),
                "basis": "配对完整的图事件时间并集",
            }
        )
    for index, j in enumerate(judges):
        events = j.get("events", [])
        if not events or j.get("timeline_valid") is not True:
            continue
        began = seconds(events[0].get("timestamp"))
        if began is None:
            continue
        closed = events[-1]["event"] == "case_end"
        stop = seconds(events[-1].get("timestamp")) if closed else cutoff
        if (
            stop is None
            or stop < began
            or (cutoff is not None and stop > cutoff + 0.05)
            or (worker_bounds and began < worker_bounds[0])
        ):
            continue
        batches = [e for e in events if e["event"] == "batch_finished"]
        durations = [e["elapsed_ms"] / 1000 for e in batches if finite(e.get("elapsed_ms"))]
        request_s = sum(durations) if len(durations) == len(batches) else None
        gaps, previous = [], began
        valid = True
        for batch in batches:
            at = seconds(batch["timestamp"])
            duration = batch.get("elapsed_ms")
            if (
                at is None
                or not finite(duration)
                or at - duration / 1000 < previous - 0.05
                or at > stop + 0.05
            ):
                valid = False
                break
            gaps.append(max(0, at - duration / 1000 - previous))
            previous = at
        if not valid:
            continue
        waits = [e for e in events if e["event"] == "wait_finished"]
        stage = {
            "name": "独立评分（Judge）",
            "case_id": j.get("case_id"),
            "duration_s": round(stop - began, 6),
            "completed_request_s": round(request_s, 6) if request_s is not None else None,
            "outside_completed_request_s": round(max(0, stop - began - request_s), 6)
            if request_s is not None
            else None,
            "recorded_wait_s": round(sum(e["elapsed_ms"] for e in waits) / 1000, 6)
            if waits
            else None,
            "finished_batches": len(batches),
            "closed": closed,
            "basis": "Judge 时间戳；批次 elapsed_ms 仅包含已完成调用",
            "evidence_id": f"J{index}:1",
        }
        stages.append(stage)
        j["timing"] = {
            **stage,
            "batch_gap_s": [round(g, 6) for g in gaps[1:]],
            "last_batch_to_stop_s": round(stop - previous, 6),
            "interval_note": "已完成请求之外的时间可能包含等待、调度或未完成请求，不能全当作 sleep。",
        }

    # 根据共享时钟选取最终阶段，不能拿第一条历史错误充当终止点。
    phase, summary, certainty, refs = None, None, "insufficient", []
    valid_judges = []
    latest_graph_start = max(
        (
            seconds(e.get("timestamp")) or float("-inf")
            for t in graph
            for e in t["events"]
            if e["status"] == "start"
        ),
        default=float("-inf"),
    )
    for index, j in enumerate(judges):
        if "timing" not in j:
            continue
        events = j["events"]
        began = seconds(events[0]["timestamp"])
        matching_graphs = [
            t
            for t in graph
            if t.get("run_id") == j.get("run_id")
            and t.get("case_id") == j.get("case_id")
            and t["trace_status"] == "recorded"
        ]
        graph_end = max(
            (
                seconds(e.get("timestamp")) or float("-inf")
                for t in matching_graphs
                for e in t["events"]
            ),
            default=float("-inf"),
        )
        if (
            graph_end != float("-inf")
            and began >= graph_end - 0.05
            and began >= latest_graph_start - 0.05
        ):
            valid_judges.append((began, index, j))
    if cutoff is not None and valid_judges and reason == "runner_timeout":
        _, index, j = max(valid_judges, key=lambda item: item[0])
        events = j["events"]
        last = events[-1]
        if last["event"] != "case_end":
            phase = "judge"
            certainty = "observed"
            refs = [f"J{index}:1", f"J{index}:{last['seq']}"]
            summary = "研究图已经结束，整轮任务在独立评分（Judge）阶段达到总时限。"
            if last["event"] == "wait_started":
                phase = "judge_wait"
                summary += "终止时正在已记录的批次等待中。"
            elif last["event"] == "batch_started":
                phase = "judge_request"
                summary += "终止前最后记录为评分请求开始，尚无对应完成事件。"
            elif last["event"] == "batch_finished":
                gaps = j["timing"]["batch_gap_s"]
                tail = j["timing"]["last_batch_to_stop_s"]
                paced = (
                    len(gaps) >= 2
                    and median(gaps) > 1
                    and max(gaps) - min(gaps) < max(0.2, median(gaps) * 0.05)
                )
                if paced and 0 <= tail < median(gaps):
                    phase = "judge_wait_inferred"
                    summary += f"最后一批已返回，此后 {tail:.2f} 秒未记录新请求；结合此前约 {median(gaps):.2f} 秒的稳定批间间隔，更可能处于批次等待。"
                    certainty = "phase_observed_wait_inferred"
                else:
                    summary += "最后一批已返回，之后的请求或等待状态没有单独记录。"
        else:
            phase = "postprocessing"
            certainty = "observed"
            summary = "已记录的研究和评分均已结束；运行器随后达到总时限，应排查后续流程等待、汇总或保存阶段。"
    if summary is None and opens:
        dated = [
            e
            for e in opens
            if seconds(e.get("timestamp")) is not None
            and (cutoff is None or seconds(e["timestamp"]) <= cutoff)
        ]
        active = max(dated, key=lambda e: seconds(e["timestamp"])) if dated else opens[-1]
        phase = active.get("name")
        refs = [active["evidence_id"]]
        summary = "终止前仍有研究图调用未结束，需结合该片段的模型或工具记录检查。"
        certainty = "observed_open_span"
    if summary is None and complete_graphs and reason == "runner_timeout":
        phase = "after_graph"
        summary = "已保存的研究图已结束，但后续阶段缺少可关联记录，暂不能定位具体等待。"
    root_nodes = [s for s in spans if s["kind"] == "graph_node" and s["parent_span_id"] is None]
    if (
        summary is None
        and root_nodes
        and root_nodes[-1]["status"] in {"error", "business_error", "cancelled"}
    ):
        terminal = root_nodes[-1]
        source = (
            next(
                (
                    s
                    for s in reversed(root_nodes[:-1])
                    if s["name"] != "failer" and s["status"] in {"error", "business_error"}
                ),
                terminal,
            )
            if terminal["name"] == "failer"
            else terminal
        )
        schema_errors = [
            s
            for s in spans
            if s.get("root_span_id") == source["span_id"]
            and s["evidence_id"].split(":")[0] == source["evidence_id"].split(":")[0]
            and s.get("error_type")
            in {"ValidationError", "OutputParserException", "JSONDecodeError"}
        ]
        phase = source["name"]
        refs = [source["evidence_id"], terminal["evidence_id"]]
        certainty = "observed_terminal_error"
        summary = "研究图最后一个节点以异常结束，尚未进入后续评分。"
        if schema_errors:
            summary = "研究节点的结构校验未通过，图随后进入失败路由，未进入独立评分。"
            refs.extend(s["evidence_id"] for s in schema_errors[:3])
    if summary is None:
        summary = "根据现有运行状态提供排查建议；尚无足够的跨阶段时间记录定位终止点。"
    attempts = [s for s in spans if s["kind"] == "attempt" and s["name"] == "planner.attempt"]
    executor_rounds = defaultdict(int)
    for s in root_nodes:
        if s["name"] == "executor":
            executor_rounds[s["evidence_id"].split(":")[0]] += 1
    return {
        "summary": summary,
        "phase": phase,
        "certainty": certainty,
        "evidence_ids": refs,
        "worker_wall_s": round(worker_bounds[1] - worker_bounds[0], 6) if worker_bounds else None,
        "stages": stages,
        "operations": statistics,
        "slowest": slowest,
        "symptoms": list(symptoms.values()),
        "retry_observations": {
            "planner_attempts": len(attempts) if attempts else None,
            "extra_executor_rounds": sum(max(0, n - 1) for n in executor_rounds.values())
            if executor_rounds
            else None,
            "sdk_retry_attempts": None,
        },
        "parallel_note": "操作累计耗时可能因并发超过墙钟；父子节点也不能重复相加。",
        "wait_note": "旧记录的稳定批间空档只支持时序推断；新记录的 wait_started 才直接证明等待。",
    }
