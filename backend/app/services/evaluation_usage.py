"""由已校验轨迹提取用量，不改写原始成绩，也不把恢复批次算作新消耗。"""

from research_engine.token_usage import summarize_usage


def model_token_usage(graph, judges, lifecycle=(), *, scripted=False):
    records = []
    for trace in [*graph, *lifecycle]:
        events = [e for e in trace.get("events", []) if e.get("kind") == "llm"]
        closed = {e.get("span_id") for e in events if e.get("status") != "start"}
        for event in events:
            if event.get("status") == "start" and event.get("span_id") in closed:
                continue
            meta = event.get("metadata") or {}
            if meta.get("role") == "judge" and judges:
                continue  # 同一评分请求以批次记录为准，不再重复累计 LLM span。
            records.append(
                {
                    **meta,
                    "status": event.get("status"),
                    "call_id": f"{trace.get('trace_id')}:{event.get('span_id')}",
                }
            )
    for trace in judges:
        for batch in trace.get("batch_measurements") or []:
            records.append(
                {
                    "role": "judge",
                    "llm_mode": trace.get("llm_mode"),
                    "configured_model": trace.get("judge_route"),
                    "served_model": batch.get("served_model"),
                    "input_tokens": batch.get("tokens_input"),
                    "output_tokens": batch.get("tokens_output"),
                    "status": "failed" if batch.get("exception_type") else "completed",
                }
            )
        closed = {
            e.get("batch_no")
            for e in trace.get("events", [])
            if e.get("event") in {"batch_finished", "batch_restored"}
        }
        for event in trace.get("events", []):
            if event.get("event") == "batch_started" and event.get("batch_no") not in closed:
                records.append(
                    {
                        "role": "judge",
                        "llm_mode": trace.get("llm_mode"),
                        "configured_model": trace.get("judge_route"),
                        "status": "unfinished",
                    }
                )
    incomplete = (
        not (graph or lifecycle)
        or any(t.get("trace_status") != "recorded" for t in [*graph, *lifecycle])
        or any(not t.get("trace_complete") for t in judges)
    )
    if scripted:
        # 执行器回归用 real 配置验证 HTTP 安全分支，但模型是脚本；以固定适配器契约为准。
        records = [{**row, "llm_mode": "fake"} for row in records]
    return summarize_usage(records, incomplete=incomplete)
