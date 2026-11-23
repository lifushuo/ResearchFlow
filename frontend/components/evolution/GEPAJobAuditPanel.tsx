"use client";

import { useEffect, useState } from "react";
import { evolutionError, evolutionGEPAAudit, type GEPAAudit, type GEPAObservationEvent, type GEPAComparison, type GEPAScores } from "@/lib/evaluation-evolution";

const short = (sha?: string | null) => sha ? sha.slice(0, 12) : "—";
const percent = (value?: number | null) => value == null ? "未测量" : `${(value * 100).toFixed(1)}%`;
const metrics = [["coverage", "Coverage"], ["specificity", "Specificity"], ["plan_quality", "Plan Quality"], ["redundancy", "Redundancy"], ["quality", "v3 Quality Score"]] as const;
export function QualityComparison({ comparison }: { comparison: GEPAComparison }) {
  return <div aria-label="v3 质量对比"><table className="text-sm"><thead><tr><th>指标</th><th>Baseline</th><th>Candidate</th><th>差异</th></tr></thead><tbody>{metrics.map(([key,label]) => <tr key={key}><td>{label}</td><td>{percent(comparison.baseline?.[key])}</td><td>{percent(comparison.candidate?.[key])}</td><td>{comparison.baseline && comparison.candidate ? percent(comparison.candidate[key]-comparison.baseline[key]) : "未测量"}</td></tr>)}</tbody></table>{comparison.same_prompt && <p>最终保留 Baseline；复用原评分，新提议的验证收益不可据此补造。</p>}</div>;
}
function scoreText(scores?: GEPAScores | null) { return metrics.map(([key,label]) => `${label} ${percent(scores?.[key])}`).join(" · "); }
const names: Record<string, string> = {
  started: "实验开始", observation: "规划与 Judge 评分", reflection_returned: "反思返回",
  final_cases_loaded: "终验案例已加载",
  candidate_frozen: "候选冻结", completed: "实验完成", interrupted: "执行中断",
};

function eventDetail(row: GEPAObservationEvent): string {
  if (row.kind === "observation") {
    return `${row.split ?? "?"} · 样例 ${row.case_key ?? "未记录"} · 候选 ${short(row.prompt_sha256)} · 覆盖率 ${percent(row.coverage)} · 任务 ${row.task_count ?? "?"} 条 · ${scoreText(row.scores)}${row.error_type ? ` · ${row.error_type}` : ""}`;
  }
  if (row.kind === "reflection_returned") return `反思产物指纹 ${short(row.response_sha256)}（原文不展示）`;
  if (row.kind === "candidate_frozen") return `候选 ${short(row.candidate_sha256)} · ${row.accepted_for_holdout ? "候选可送终验（尚未运行）" : "候选未接受"}`;
  if (row.kind === "completed") return `门禁 ${row.holdout_passed == null ? "待终验" : row.holdout_passed ? "通过" : "未通过"}`;
  if (row.kind === "interrupted") return `中断类型 ${row.error_type ?? "未记录"} · ${row.role ?? "角色未记录"} ${row.slot_id ?? ""} · ${row.cause_type ?? "原始异常类型未记录"}`;
  return "已记录";
}

export default function GEPAJobAuditPanel({ jobId, refresh }: { jobId: string; refresh: string }) {
  const [audit, setAudit] = useState<GEPAAudit | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    void evolutionGEPAAudit(jobId).then((data) => { if (active) { setAudit(data); setError(""); } })
      .catch((e) => { if (active) { setAudit(null); setError(evolutionError(e)); } });
    return () => { active = false; };
  }, [jobId, refresh]);

  return (
    <section aria-label="GEPA 实验审计" className="rounded-lg border border-border p-4 space-y-3">
      <h3 className="font-semibold">实验过程与版本证据</h3>
      {error && <p role="alert" className="text-sm">{error}</p>}
      {!audit ? <p className="text-sm text-text-muted">正在读取审计记录……</p> : (
        <>
          <p className="text-xs">{audit.mode === "demo" ? "脚本演示，不能证明模型提升" : "真实模型优化"} · {audit.state} · 实验 {audit.job_id}</p>
          <div className="text-xs space-y-1 break-all">
            <p>数据集：{audit.journal.dataset_id ?? "未记录"} · 指纹：{audit.dataset_sha256 ?? "Demo 未固定正式数据集"}</p>
            <p>Planner：{audit.journal.model_routes?.planner ?? "未记录"} · Judge：{audit.journal.model_routes?.judge ?? "未记录"} · Reflection：{audit.journal.model_routes?.reflection ?? "未记录"}</p>
            <p>结果与版本指纹校验：{audit.result_integrity === "verified" ? "已验证" : audit.result_integrity === "mismatch" ? "校验不一致，请勿用于发布" : "尚未生成完整结果"}</p>
            {audit.result_versions && <p>{audit.task_kind === "planner_final_ab" ? "执行器：Planner A/B（无搜索）" : `优化器：${audit.result_versions.optimizer ?? "未知"} ${audit.result_versions.optimizer_version}`} · 指标：{audit.result_versions.metric_version} · 基线：{short(audit.result_versions.baseline_sha256)} · 候选：{short(audit.result_versions.candidate_sha256)}</p>}
            <p>任务类型：{audit.task_kind === "planner_final_ab" ? "独立 Planner 终验" : "GEPA 搜索"}{audit.source_job_id ? ` · 来源 ${audit.source_job_id}` : ""}</p>
            <p>留出集指纹：{audit.holdout_sha256 ?? (audit.task_kind === "planner_final_ab" ? "模拟终验：未读取正式 Holdout" : "无留出搜索：未读取正式 Holdout")}</p>
            {audit.frozen && <p>冻结版本 {audit.frozen.metric_version} · min_gain {audit.frozen.min_gain} · 规则 {short(audit.frozen.metric_rules_sha256)} · Baseline {short(audit.frozen.baseline_sha256)}</p>}
            {audit.budget && <p aria-label="GEPA 请求预算">{audit.budget.simulated ? "模拟请求账本" : "真实请求账本"} · 上限 {audit.budget.configured.total_cap} · 已使用 {audit.budget.used_total ?? "未测量"} · 剩余 {audit.budget.remaining_total ?? "未测量"} · 未知调用 {audit.budget.unknown?.length ?? "未测量"}</p>}
            <p>{audit.task_kind === "planner_final_ab" ? "终验配置" : "搜索配置"}：最多 {audit.limits.max_evals} 次评测、{audit.limits.proposals} 个提案、Seed {audit.limits.seed}</p>
            <p>冻结请求等待上限：{audit.frozen?.request_timeouts ? `Planner ${audit.frozen.request_timeouts.planner}s · Judge ${audit.frozen.request_timeouts.judge}s · Reflection ${audit.frozen.request_timeouts.reflection}s` : "历史任务未记录，不能补造"}；超时仍保留未知调用并停止。</p>
            <p>已观测评分 {audit.journal.observation_count} 次 · 反思返回 {audit.journal.reflection_count} 次 · 事件总数 {audit.journal.event_count}</p>
            <p>已记账 Planner/Judge Token（不含反思）：输入 {audit.journal.observed_tokens.planner_input != null && audit.journal.observed_tokens.judge_input != null ? audit.journal.observed_tokens.planner_input + audit.journal.observed_tokens.judge_input : "未测量"} · 输出 {audit.journal.observed_tokens.planner_output != null && audit.journal.observed_tokens.judge_output != null ? audit.journal.observed_tokens.planner_output + audit.journal.observed_tokens.judge_output : "未测量"}；Reflection Token 未记录，保持 null。</p>
            <p>{audit.task_kind === "planner_final_ab" && audit.mode === "demo" ? "模拟留出案例是否已执行" : "留出集是否已被读取"}：{audit.journal.holdout_observed ? "是" : "尚未在日志中观测到"}</p>
          </div>
          <p className="text-xs text-text-muted">{audit.task_kind === "planner_final_ab" ? "终验仅比较已冻结的两份 Prompt，不运行搜索或 Reflection；通过仍不代表已发布。" : "GEPA 内部 Pareto 父代选择未被当前日志采集，不会虚构相关搜索步骤。反思正文没有归档，只有内容指纹。"}</p>
          {audit.budget?.unknown_details?.map((call) => <p key={call.slot_id} role="alert" className="text-sm">结果未确认的请求：{call.slot_id} · {call.role} · {call.cause_type ?? "原因未记录"}。后续请求已阻止，不能自动补发或清除旧账本。</p>)}
          {audit.journal.error && <p role="alert" className="text-sm">审计日志校验失败：{audit.journal.error}</p>}
          {audit.journal.candidates.length > 0 && (
            <div className="overflow-x-auto">
              <h4 className="font-medium text-sm mb-2">实际观测到的候选评分（包含重复观测）</h4>
              <table className="min-w-full text-xs text-left"><thead><tr className="border-b border-border">
                <th className="p-2">Prompt SHA</th><th className="p-2">评测次数</th><th className="p-2">有效数</th><th className="p-2">开发集 Coverage</th><th className="p-2">验证集 Coverage</th><th className="p-2">留出集 Coverage</th><th className="p-2">{audit.task_kind === "planner_final_ab" ? "v3 五项指标（终验）" : "v3 五项指标（验证集）"}</th>
              </tr></thead><tbody>{audit.journal.candidates.map((candidate) => <tr className="border-b border-border" key={candidate.sha256}>
                <td className="p-2 font-mono" title={candidate.sha256}>{short(candidate.sha256)}</td><td className="p-2">{candidate.samples}</td><td className="p-2">{candidate.measured}</td>
                <td className="p-2">{percent(candidate.splits.train?.mean_coverage)}</td>
                <td className="p-2">{percent(candidate.splits.validation?.mean_coverage)}</td>
                <td className="p-2">{percent(candidate.splits.holdout?.mean_coverage)}</td><td className="p-2">{scoreText((audit.task_kind === "planner_final_ab" ? candidate.splits.holdout : candidate.splits.validation)?.scores)}</td>
              </tr>)}</tbody></table>
            </div>
          )}
          <details>
            <summary className="cursor-pointer text-sm font-medium">查看执行时间线（{audit.journal.events.length} 条展示）</summary>
            <div className="max-h-96 overflow-y-auto space-y-2 mt-2">
              {audit.journal.events.map((row) => (
                <div key={row.sequence} className="rounded border border-border px-3 py-2 text-xs">
                  <p>#{row.sequence} · {names[row.kind] ?? row.kind} · {row.recorded_at ?? "时间未记录"}</p>
                  <p className="text-text-muted break-words">{eventDetail(row)}</p>
                </div>
              ))}
            </div>
            {audit.journal.events_truncated && <p className="text-xs">仅展示最后 120 条，早期事件仍保留在服务端。</p>}
          </details>
          <p className="text-xs text-text-muted">正式报告 A/B 不由此 Worker 自动启动；完成真实搜索后，需你在评测中心独立验收，才能决定是否发布。</p>
        </>
      )}
    </section>
  );
}
