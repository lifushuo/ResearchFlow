import type { Diagnostic } from "@/lib/evaluation-diagnostics";
import { closureState } from "@/lib/evaluation-closure";

export function caseError(error?: string | null) {
  if (error?.startsWith("planner: could not produce a valid task list")) return "规划器未能生成符合要求的任务列表";
  return error || "未记录具体错误；请查看本题阶段证据";
}

export default function EvaluationCaseTable({ rows, onOpen }: { rows: Diagnostic["cases"]; onOpen: (id: string) => void }) {
  return <div className="overflow-x-auto rounded-lg border border-border"><table className="w-full min-w-[620px] text-left text-sm">
    <thead className="bg-bg-elevated"><tr>{["案例", "原评测判定", "研究耗时", "已记录结论", "失败事实 / 详情"].map((text) => <th key={text} className="p-3">{text}</th>)}</tr></thead>
    <tbody>{rows.map((row) => <tr key={row.case_id} className="border-t border-border align-top">
      <td className="p-3 font-medium break-words">{row.case_id}</td><td className={`p-3 ${row.status === "failed" ? "text-danger" : ""}`}>{closureState(row.status)}</td>
      <td className="p-3">{row.latency_s == null ? "未测量" : `${row.latency_s.toFixed(2)} 秒`}</td><td className="p-3">{row.claim_count ?? "未测量"}</td>
      <td className="max-w-sm space-y-2 p-3">{row.error && <p className="text-danger">{caseError(row.error)}</p>}<button className="rounded border border-border px-3 py-2 hover:bg-bg-elevated" onClick={() => onOpen(row.case_id)}>查看该案例证据与分析</button></td>
    </tr>)}</tbody>
  </table></div>;
}

export function EvaluationSummary({ metrics, cases, onOpen }: { metrics: Record<string, unknown>; cases?: Diagnostic["cases"]; onOpen: (id: string) => void }) {
  const aggregate = metrics.aggregate && typeof metrics.aggregate === "object" ? metrics.aggregate as Record<string, unknown> : metrics;
  const percent = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "未测量";
  const failed = cases?.filter((row) => row.status === "failed") ?? [];
  const cards = [
    ["研究完成率", percent(aggregate.completion_rate ?? metrics.completion_rate)],
    ["全量 Claim 评分覆盖率", percent(aggregate.full_judge_coverage)],
    ["引用支持率（仅已判定）", percent(aggregate.citation_support_rate)],
    ["逐题评测门禁", metrics.meets_threshold === false || metrics.gate_passed === false ? "不通过" : metrics.meets_threshold === true || metrics.gate_passed === true ? "通过" : "未测量"],
  ];
  return <div className="space-y-3"><dl className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-2">{cards.map(([label, value]) => <div key={label} className="min-w-0 rounded-lg border border-border p-3"><dt className="text-sm text-text-muted">{label}</dt><dd className="mt-1 text-xl font-semibold">{value}</dd></div>)}</dl>
    {cases && <p className="text-sm">已记录 {cases.length} 题 · 通过 {cases.filter((row) => row.status === "passed").length} · 失败 {failed.length} · 未完成测量 {cases.filter((row) => ["unmeasured", "blocked"].includes(row.status)).length}</p>}
    {failed.map((row) => <div key={row.case_id} className="rounded-lg border border-danger p-3 text-sm"><p className="font-semibold">需要排查：{row.case_id}</p><p className="mt-1">{caseError(row.error)}</p><button className="mt-2 rounded border border-border px-3 py-2" onClick={() => onOpen(row.case_id)}>定位失败案例</button></div>)}
    {failed.length > 0 && <p className="text-xs text-text-muted">原报告评分器的汇总阈值与逐题门禁分别展示；部分研究完成、已判定结论支持率高，不代表全部案例都通过。</p>}
  </div>;
}
