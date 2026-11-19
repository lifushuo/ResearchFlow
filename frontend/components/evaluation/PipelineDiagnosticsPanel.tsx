"use client";

import UnifiedExecutionTrace from "./UnifiedExecutionTrace";
import { JudgeProgress, ReportEvaluationState } from "./JudgeProgress";

import { useEffect, useState } from "react";
import { getDiagnostics, type Diagnostic } from "@/lib/evaluation-diagnostics";
import { type DatasetId } from "@/lib/evaluation-pipeline-v13";
import { getDatasetLabel } from "@/lib/evaluation-dataset-zh";
import { evaluationErrorMessage } from "@/lib/evaluation";
import { EvaluationMetrics } from "./EvaluationMetrics";
import { evaluationLabel } from "@/lib/evaluation-display-zh";
import { explainDiagnostic } from "@/lib/evaluation-explanation";
import { caseError } from "./EvaluationCaseTable";

const states: Record<string, string> = { running: "运行中", queued: "排队中", pending: "待执行", completed: "执行已结束", failed: "执行失败", blocked: "未就绪", passed: "通过", unmeasured: "未测量", start: "开始", ok: "完成", error: "异常", business_error: "业务失败", cancelled: "已取消", interrupt: "人工关卡中断", recorded: "配对完整", partial: "不完整", unavailable: "不可用", not_requested: "尚未请求", interrupted: "请求中断，未自动重试", evidence_changed: "证据已变化，未调用模型" };
const css = "rounded-lg border border-border bg-bg-base px-3 py-2 text-sm text-text-primary disabled:opacity-40";

function traceName(value: string | null) {
  if (!value) return "未记录";
  const [prefix, role] = value.split(".");
  const prefixes: Record<string, string> = { llm: "模型调用", schema: "结构校验", structured: "结构化输出" };
  if (role && prefixes[prefix]) return `${evaluationLabel(role)} · ${prefixes[prefix]}`;
  if (value.endsWith(".attempt")) return `${evaluationLabel(prefix)} · 应用层尝试`;
  return ({ attempt: "应用层尝试", structured_output: "结构化输出", schema_validation: "结构校验", judge: "独立评分", judge_request: "独立评分请求", judge_wait: "独立评分批间等待（已记录）", judge_wait_inferred: "独立评分阶段，可能正在批间等待", postprocessing: "评分后的结果处理", after_graph: "研究图之后的处理" } as Record<string, string>)[value] ?? evaluationLabel(value);
}

const duration = (value: number | null | undefined) => value == null ? "未测量" : `${value.toFixed(2)} 秒`;

function batchProblem(row: Record<string, unknown>, index: number) {
  const reasons: Record<string, string> = { empty_or_nontext_response: "未收到可判定的文本", partial_response: "模型只回答了部分结论", unparseable_response: "回复格式无法识别", invoke_exception: "评分请求抛出异常", response_processing_exception: "收到响应但处理失败", evaluation_cancelled: "评分请求被取消" };
  const reason = typeof row.reason_hint === "string" ? reasons[row.reason_hint] : null;
  return `第 ${typeof row.batch_no === "number" ? row.batch_no : index + 1} 批：${reason || "需要核对回复编号或判定映射"}；${typeof row.unjudged_count === "number" ? row.unjudged_count : "数量未披露的"} 条结论未判定。`;
}

export function DiagnosticDetails({ pipe, dataset, initialCase = "" }: { pipe: string; dataset: DatasetId; initialCase?: string }) {
  const [data, setData] = useState<Diagnostic | null>(null);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [caseId, setCaseId] = useState(initialCase);
  const [offset, setOffset] = useState(0);
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = () => (caseId ? getDiagnostics(pipe, dataset, caseId) : getDiagnostics(pipe, dataset)).then((row) => { if (alive) { setData(row); setError(""); if (["running", "pending"].includes(row.state)) timer = setTimeout(() => { void load(); }, 5000); } }).catch((err) => { if (alive) setError(evaluationErrorMessage(err, "诊断证据读取失败")); });
    void load();
    return () => { alive = false; clearTimeout(timer); };
  }, [pipe, dataset, refresh, caseId]);
  if (!data) return <p role={error ? "alert" : "status"} className="text-sm text-text-muted">{error || "正在读取本次运行证据…"}</p>;
  const explanation = explainDiagnostic(data);
  const scopedDuration = caseId ? data.cases.find((row) => row.case_id === caseId)?.latency_s : data.elapsed_seconds;
  const events = data.traces.flatMap((t) => t.events).filter((e) => !caseId || e.case_id === caseId);
  const caseIds = data.available_case_ids ?? [...new Set([...data.cases.map((r) => r.case_id), ...data.traces.map((t) => t.case_id).filter((x): x is string => Boolean(x))])];
  return <div className="space-y-5">
    <label className="block text-sm">选择排查案例<select aria-label="选择排查案例" value={caseId} className={`ml-2 max-w-full ${css}`} onChange={(e) => { setCaseId(e.target.value); setData(null); setOffset(0); }}><option value="">整轮汇总</option>{caseIds.map((id) => <option key={id} value={id}>{id}</option>)}</select></label>
    {caseId && <p className="text-sm text-text-muted">当前仅分析 {caseId}；整轮执行器耗时只作为运行背景，不代表本题耗时。</p>}
    {data.cases.filter((r) => r.error && (!caseId || r.case_id === caseId)).map((r) => <section key={r.case_id} aria-label="已保存的终止错误" className="rounded-lg border border-danger p-4 space-y-2"><h3 className="font-semibold text-danger">{r.case_id} · 研究执行失败</h3><p>{caseError(r.error)}</p><p className="text-sm">研究耗时：{duration(r.latency_s)}。以下错误直接来自本题原始结果，底层诱因仍需结合轨迹核验。</p><pre className="whitespace-pre-wrap break-words text-xs">{r.error}</pre></section>)}
    {error && <p role="alert" className="text-sm text-danger">{error}</p>}
    <div className="rounded-lg border border-border p-4 space-y-2">
      <p className="font-semibold">{getDatasetLabel(dataset)} · {data.report_states?.some((s) => s.research === "draft_saved" && s.judge === "partial") ? "已有研究草稿，评分部分完成" : states[data.state] ?? data.state}</p>
      <button type="button" className={css} onClick={() => setRefresh((x) => x + 1)}>刷新诊断证据</button>
      <p className="font-mono text-xs break-all text-text-muted">{pipe}</p>
      <p className="text-sm">案例：{data.case_id || "本次所选案例集"} · {caseId ? "本题研究耗时" : "整轮耗时"}：{duration(scopedDuration)} · 定位阶段：{data.failure_phase ? traceName(data.failure_phase) : "未采集到具体阶段"}</p>
      <p className="text-xs text-text-muted">{data.collection === "saved_status_analysis" ? "根据已保存状态分析；旧运行缺失的轨迹不能事后恢复。" : "来自本次 Worker 与运行产物。"} {data.quality_measurement === "unmeasured" ? "执行未完成，质量指标保持未测量。" : "评分沿用原评测器及其分母。"}</p>
      <p className="text-xs text-text-muted break-all">运行器开始：{data.execution_observation?.started_at || "原运行未记录"} · 结束：{data.execution_observation?.finished_at || "原运行未记录/尚未结束"} · 子进程退出码：{data.execution_observation?.exit_code ?? "未测量"}</p>
    </div>
    {!!data.report_states?.length && <details className="rounded-lg border border-border p-3"><summary className="cursor-pointer">草稿保存与评分恢复详情</summary><div className="mt-3 space-y-2">{data.report_states.map((s, i) => <ReportEvaluationState key={`${s.case_id}-${s.phase}-${i}`} state={s} />)}</div></details>}
    {data.case_evidence && <section className="space-y-3" aria-label="本题原始报告与判定"><h3 className="font-semibold">本题报告与逐条 Claim 判定</h3>{!data.case_evidence.claims.length && <p>本题未保存 Claim 判定；不能视作评分通过。</p>}{data.case_evidence.claims.filter((c) => !c.judged || c.supported === false).map((c, i) => <div key={i} className="rounded border border-warning p-3"><p className="font-medium">{c.judged ? "已判定：证据不支持" : "未判定"}</p><p className="mt-1 text-sm">{c.text}</p></div>)}<details className="rounded border border-border p-3"><summary className="cursor-pointer">全部已保存 Claim 判定（{data.case_evidence.claims.length} 条）</summary>{data.case_evidence.claims.map((c, i) => <p key={i} className="mt-2 text-sm">{i + 1}. {c.judged ? c.supported ? "支持" : "不支持" : "未判定"} · {c.text}</p>)}{data.case_evidence.claims_truncated && <p>列表已截断，不代表全部输入。</p>}</details><details className="rounded border border-border p-3"><summary className="cursor-pointer">已保存报告正文（脱敏片段）</summary><pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-words text-sm">{data.case_evidence.report_excerpt || "本题没有已保存的报告正文"}</pre>{data.case_evidence.report_truncated && <p>正文超过展示上限，片段已截断。</p>}</details></section>}
    <section aria-label="故障分析" className="space-y-3">
      <h3 className="font-semibold">异常事实与根因分析</h3>
      <div className="rounded-lg border border-accent/40 bg-bg-base p-4 space-y-3">
        <h4 className="font-semibold">{explanation.title}</h4><p className="text-sm leading-relaxed">{explanation.summary}</p>
        {explanation.facts.length > 0 && <ul className="list-disc pl-5 text-sm space-y-2">{explanation.facts.map((fact) => <li key={fact}>{fact}</li>)}</ul>}
        <p className="text-sm text-text-muted">{explanation.boundary}</p>
      </div>
      <details className="rounded-lg border border-border p-3 text-sm"><summary className="cursor-pointer">查看阶段耗时和判断依据（技术详情）</summary>
      {data.timeline && <p className="my-3">{data.timeline.summary}</p>}
      {data.timeline && data.timeline.stages.length > 0 && <div className="space-y-2"><h4 className="font-semibold text-sm">时间花在哪里？</h4>{data.timeline.stages.map((stage, i) => <div key={i} className="rounded-lg border border-border p-3 text-sm"><p className="font-medium">{stage.name}：{duration(stage.duration_s)}</p>{stage.completed_request_s != null && <p>已完成评分请求 {duration(stage.completed_request_s)} · 其余时间 {duration(stage.outside_completed_request_s)} · 已完成 {stage.finished_batches} 批</p>}{stage.recorded_wait_s != null && <p>明确记录的已结束等待：{duration(stage.recorded_wait_s)}</p>}<p className="mt-1 text-xs text-text-muted">{stage.basis}{stage.closed === false ? "；评分记录尚未结束，不能当作完整成绩。" : ""}</p></div>)}<p className="text-xs text-text-muted">其余时间可能包含等待、调度和未完成请求，不能全部算作 sleep。{data.timeline.wait_note}</p></div>}
      {data.timeline && data.timeline.symptoms.length > 0 && <details className="rounded-lg border border-border p-3 text-sm"><summary className="cursor-pointer font-medium">过程中出现过的异常（已合并，不直接当作终止原因）</summary>{data.timeline.symptoms.map((s, i) => <p key={i} className="mt-2">{traceName(s.name)} · {s.count} 次 · {s.workflow_continued ? "所在研究节点随后继续完成" : "需结合后续阶段判断是否阻断"} · {s.error_type || "未披露具体错误类型"}</p>)}</details>}
      <details className="rounded-lg border border-border p-3 text-sm"><summary className="cursor-pointer">查看原始诊断事实与证据编号</summary>{data.facts.map((f) => <p key={f.id} className="mt-2 break-words"><span className="font-mono text-accent">[{f.id}]</span> {f.text}</p>)}</details>
      {!data.has_issue && <p className="text-sm text-text-muted">当前没有登记失败；故障回归通过不等于真实研究成功。</p>}
      {(!data.timeline || data.timeline.certainty === "insufficient") && data.candidates.map((c, i) => <div key={i} className="rounded-lg border border-border bg-bg-base p-3 text-sm space-y-1">
        <p className="font-medium">候选 {i + 1}：{c.cause}</p>
        <p className="text-xs text-text-muted">{c.confidence} · 依据：{c.evidence_ids.join("、")}</p>
        <p>验证方法：{c.verification}</p>
      </div>)}
      {data.missing_evidence.map((x) => <p key={x} className="text-xs text-text-muted">证据边界：{x}</p>)}
      </details>
      {explanation.next.length > 0 && <div className="rounded-lg border border-border p-3"><h4 className="font-medium text-sm">下一步怎么处理？</h4><ol className="mt-2 list-decimal pl-5 space-y-1 text-sm">{explanation.next.map((x) => <li key={x}>{x}</li>)}</ol></div>}
    </section>
    {data.timeline && data.timeline.operations.length > 0 && <section className="space-y-3 border-t border-border pt-4"><h3 className="font-semibold">节点、模型和工具耗时</h3><p className="text-xs text-text-muted">{data.timeline.parallel_note}</p><div className="overflow-auto rounded-lg border border-border"><table className="w-full min-w-[560px] text-left text-sm"><thead><tr>{["操作", "类型", "次数", "累计耗时", "最慢单次", "异常次数"].map((s) => <th key={s} className="p-2">{s}</th>)}</tr></thead><tbody>{data.timeline.operations.filter((r) => ["graph_node", "llm", "tool"].includes(r.kind)).map((r) => <tr key={`${r.kind}/${r.name}`} className="border-t border-border"><td className="p-2">{traceName(r.name)}</td><td className="p-2">{traceName(r.kind)}</td><td className="p-2">{r.calls}</td><td className="p-2">{duration(r.sum_s)}</td><td className="p-2">{duration(r.max_s)}</td><td className="p-2">{r.errors}</td></tr>)}</tbody></table></div><p className="text-sm">规划尝试 {data.timeline.retry_observations.planner_attempts ?? "未测量"} 次 · 额外执行器轮次 {data.timeline.retry_observations.extra_executor_rounds ?? "未测量"} · SDK 内部重试未测量</p>{data.timeline.slowest.length > 0 && <p className="text-sm">最慢单次已记录调用：{traceName(data.timeline.slowest[0].name)}，{duration(data.timeline.slowest[0].duration_ms / 1000)}（{data.timeline.slowest[0].evidence_id}）。</p>}</section>}
    {data.unified_trace && <UnifiedExecutionTrace key={caseId} trace={data.unified_trace} caseId={caseId} />}
    {!data.unified_trace && <section aria-label="本次执行轨迹" className="space-y-3 border-t border-border pt-4">
      <h3 className="font-semibold">本次执行轨迹</h3>
      <p className="text-xs text-text-muted">保留开始、结束、耗时、异常类型和已披露用量；没有结束事件的片段保持未完成。SDK 内部 HTTP 重试次数未测量。</p>
      {data.traces.map((t, i) => <p key={t.trace_id ?? i} className="text-xs break-all">{t.case_id || "轨迹身份未能读取"} · {states[t.trace_status] ?? t.trace_status} · {t.events.length} 条事件 · 未结束片段 {t.open_spans?.length ?? 0}</p>)}
      {data.traces.length === 0 && <p className="text-sm text-text-muted">本次没有图轨迹。上方仍显示已保存的执行事实和排查建议。</p>}
      {caseIds.length > 0 && <label className="block text-sm">轨迹案例筛选<select aria-label="轨迹案例筛选" className={`ml-2 max-w-full ${css}`} value={caseId} onChange={(e) => { setCaseId(e.target.value); setOffset(0); }}><option value="">全部已记录案例</option>{caseIds.map((id) => <option key={id} value={id}>{id}</option>)}</select></label>}
      {events.length > 0 && <>
        <div className="max-h-[560px] overflow-auto rounded-lg border border-border"><table className="w-full min-w-[680px] text-left text-xs"><thead className="bg-bg-base"><tr>{["序号 / 案例", "节点与调用", "状态", "耗时", "异常与用量"].map((x) => <th key={x} className="p-2">{x}</th>)}</tr></thead><tbody>{events.slice(offset, offset + 100).map((e, i) => <tr key={`${e.case_id}-${e.seq}-${i}`} className="border-t border-border align-top"><td className="p-2 break-all">{e.seq} · {e.case_id}</td><td className="p-2 break-all">{traceName(e.name)}<br />{traceName(e.kind)}</td><td className="p-2">{states[e.status || ""] ?? e.status}</td><td className="p-2">{e.duration_ms === null ? "未结束/未测量" : `${e.duration_ms} ms`}</td><td className="p-2 break-all">{e.error_type || "—"}<details><summary className="cursor-pointer">调用元数据</summary><pre className="whitespace-pre-wrap">{JSON.stringify(e.metadata, null, 2)}</pre></details></td></tr>)}</tbody></table></div>
        <div className="flex flex-wrap gap-2 items-center text-xs"><button className={css} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 100))}>上一页事件</button><span>{offset + 1}–{Math.min(offset + 100, events.length)} / {events.length}</span><button className={css} disabled={offset + 100 >= events.length} onClick={() => setOffset(offset + 100)}>下一页事件</button></div>
      </>}
    </section>}
    {data.judge_traces.length > 0 && <section className="space-y-2 border-t border-border pt-4"><h3 className="font-semibold">独立评分批次记录</h3>{data.judge_traces.map((j, i) => <div key={i} className="text-sm border border-border rounded-lg p-3 space-y-2"><JudgeProgress progress={j} />{j.problem_batches?.map((row, n) => <p key={n}>{batchProblem(row, n)}</p>)}<details><summary>批次原始技术字段</summary><pre className="text-xs whitespace-pre-wrap break-all">{JSON.stringify(j.problem_batches ?? [], null, 2)}</pre></details></div>)}</section>}
    {data.cases.length > 0 && <section className="space-y-2 border-t border-border pt-4"><h3 className="font-semibold">本次逐案例结果</h3>{data.cases.map((c) => <div key={c.case_id} className="rounded-lg border border-border p-3 text-sm break-all"><p>{c.case_id} · {states[c.status] ?? c.status} · 耗时 {c.latency_s ?? "未测量"} 秒{c.claim_count !== null ? ` · ${c.claim_count} 条 Claim` : ""}</p>{c.metrics && Object.keys(c.metrics).length > 0 && <details className="mt-2"><summary className="cursor-pointer">逐案例指标与断言</summary><EvaluationMetrics metrics={c.metrics} /></details>}</div>)}</section>}
    {Object.keys(data.metrics).length > 0 && <details className="border-t border-border pt-4"><summary className="cursor-pointer font-semibold">原评测指标与口径</summary><EvaluationMetrics metrics={data.metrics} /></details>}
  </div>;
}
