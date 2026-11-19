"use client";

import { useState } from "react";
import type { UnifiedTrace } from "@/lib/evaluation-diagnostics";

const labels: Record<string, string> = { G: "研究图", J: "独立评分", E: "执行器 / 评测适配器", start: "开始", ok: "完成", error: "异常", business_error: "业务错误", interrupt: "人工关卡中断", cancelled: "已取消", batch_started: "评分请求开始", batch_finished: "评分请求结束", wait_started: "等待开始", wait_finished: "等待结束", case_start: "评分开始", case_end: "评分结束" };

export default function UnifiedExecutionTrace({ trace, caseId = "" }: { trace: UnifiedTrace; caseId?: string }) {
  const [source, setSource] = useState("");
  const [offset, setOffset] = useState(0);
  const [onlyErrors, setOnlyErrors] = useState(false);
  const rows = trace.events.filter((e) => (!source || e.source === source) && (!caseId || e.case_id === caseId || e.source === "E") && (!onlyErrors || ["error", "business_error", "cancelled"].includes(e.status || "")));
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(trace, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = "evaluation-trace.json"; anchor.click(); URL.revokeObjectURL(url);
  }
  return <section aria-label="统一执行轨迹" className="space-y-3 border-t border-border pt-4">
    <h3 className="font-semibold">统一执行轨迹 · 从执行器到结果保存</h3>
    <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={onlyErrors} onChange={(e) => { setOnlyErrors(e.target.checked); setOffset(0); }} />只看异常事件</label>
    <p className="text-xs text-text-muted">过程中的异常可能已恢复；请结合上方已保存的终止错误与后续节点结果判断。</p>
    <p className="text-sm">已加载 {trace.coverage.loaded_events} 条事件，其中 {trace.coverage.content_events} 条包含脱敏上下文，{trace.coverage.content_truncated_events} 条上下文被截断。</p>
    <p className="text-xs text-text-muted">{trace.coverage.note} 新运行可包含问题、提示词、模型回复、工具参数和返回、节点状态及错误片段；请注意其中仍可能包含研究内容，模型辅助分析需另行确认。</p>
    <details className="text-sm"><summary className="cursor-pointer">采集边界和来源文件指纹</summary>{trace.coverage.not_collected.map((s) => <p key={s} className="mt-2">{s}</p>)}<pre className="max-h-60 overflow-auto whitespace-pre-wrap break-all text-xs">{JSON.stringify(trace.sources, null, 2)}</pre></details>
    <div className="flex flex-wrap gap-2 items-center text-sm"><label>轨迹阶段 <select aria-label="统一轨迹阶段" className="rounded border border-border bg-bg-base p-2" value={source} onChange={(e) => { setSource(e.target.value); setOffset(0); }}><option value="">全部阶段</option>{[...new Set(trace.events.map((e) => e.source))].map((s) => <option key={s} value={s}>{labels[s] ?? s}</option>)}</select></label><button onClick={download} className="rounded border border-border px-3 py-2">导出已加载的脱敏轨迹</button></div>
    {rows.length === 0 ? <p className="text-sm">此阶段没有已采集的记录。</p> : <div className="space-y-2">{rows.slice(offset, offset + 40).map((e) => <details key={e.evidence_id} className="rounded border border-border p-3 text-sm"><summary className="cursor-pointer break-all">[{e.evidence_id}] {e.timestamp || "时间未采集"} · {labels[e.source] ?? e.source} · {e.name || labels[e.event || ""] || e.event} · {labels[e.status || ""] || e.status || "已记录"}</summary><p className="my-2 break-all">案例：{e.case_id || "评测步骤"} · 父调用：{e.parent_span_id || "无 / 未记录"} · 耗时：{e.duration_ms ?? e.elapsed_ms ?? "未记录"} ms</p>{e.details ? <><p className="font-medium">输入、输出与错误上下文（已脱敏，截断位置见详情）</p><pre className="max-h-96 overflow-auto whitespace-pre-wrap break-all text-xs">{JSON.stringify(e.details, null, 2)}</pre></> : <p className="text-text-muted">此事件未采集正文；历史内容无法补回。</p>}<details className="mt-2"><summary>事件技术字段</summary><pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all text-xs">{JSON.stringify({ ...e, details: undefined }, null, 2)}</pre></details></details>)}</div>}
    {rows.length > 40 && <div className="flex gap-3 items-center text-sm"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 40))}>上一页轨迹</button><span>{offset + 1}–{Math.min(offset + 40, rows.length)} / {rows.length}</span><button disabled={offset + 40 >= rows.length} onClick={() => setOffset(offset + 40)}>下一页轨迹</button></div>}
  </section>;
}
