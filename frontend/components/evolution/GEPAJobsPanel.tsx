"use client";

import { useCallback, useEffect, useState } from "react";
import GEPAJobAuditPanel, { QualityComparison } from "./GEPAJobAuditPanel";
import GEPAFinalPanel from "./GEPAFinalPanel";
import {
  evolutionCancelGEPA,
  evolutionGEPACapabilities,
  evolutionGEPAJobs,
  evolutionGEPAResult,
  evolutionStartGEPA,
  evolutionError,
  type GEPAJob,
  type GEPAResult,
} from "@/lib/evaluation-evolution";

const button = "rounded-lg border border-border px-3 py-2 text-sm disabled:opacity-40 hover:bg-bg-base";
const field = "rounded-lg border border-border bg-bg-base p-2 text-sm";

export default function GEPAJobsPanel() {
  const [jobs, setJobs] = useState<GEPAJob[]>([]);
  const [mode, setMode] = useState<"demo" | "live">("demo");
  const [metricVersion] = useState<"v3">("v3");
  const [maxEvals, setMaxEvals] = useState(8);
  const [preset, setPreset] = useState<"smoke" | "full">("smoke");
  const [proposals, setProposals] = useState(1);
  const [seed, setSeed] = useState(42);
  const [confirmCost, setConfirmCost] = useState(false);
  const [liveAllowed, setLiveAllowed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [result, setResult] = useState<GEPAResult | null>(null);
  const [finalSource, setFinalSource] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const response = await evolutionGEPAJobs();
      setJobs(response.jobs);
    } catch (e) { setError(evolutionError(e)); }
  }, []);

  useEffect(() => {
    // 初始加载走 .then 回调，避免在 effect 体内同步 setState 触发级联渲染。
    let active = true;
    void evolutionGEPAJobs()
      .then((response) => { if (active) setJobs(response.jobs); })
      .catch((e) => { if (active) setError(evolutionError(e)); });
    void evolutionGEPACapabilities()
      .then((x) => { if (active) setLiveAllowed(x.live_enabled); })
      .catch(() => { if (active) setLiveAllowed(false); });
    const timer = window.setInterval(() => { void reload(); }, 3000);
    return () => { active = false; window.clearInterval(timer); };
  }, [reload]);

  useEffect(() => {
    if (!selected) return;
    const selectedRow = jobs.find((j) => j.job_id === selected);
    if (!selectedRow || !["succeeded", "quality_rejected", "pending_final", "no_candidate", "failed", "cancelled"].includes(selectedRow.state)) return;
    let active = true;
    void evolutionGEPAResult(selected).then((x) => { if (active) setResult(x); })
      .catch((e) => { if (active) setError(evolutionError(e)); });
    return () => { active = false; };
  }, [jobs, selected]);

  async function start() {
    if (busy || (mode === "live" && (!liveAllowed || !confirmCost))) return;
    setBusy(true); setError(""); setResult(null);
    try {
      const job = await evolutionStartGEPA({
        request_id: crypto.randomUUID(), mode,
        metric_version: metricVersion,
        dataset_preset: preset,
        max_evals: maxEvals,
        proposals, seed,
        consent: mode === "live" ? "I_CONFIRM_GEPA_MODEL_API_COST" : null,
      });
      setSelected(job.job_id);
      await reload();
    } catch (e) { setError(evolutionError(e)); }
    finally { setBusy(false); }
  }

  async function cancel(jobId: string) {
    setBusy(true); setError("");
    try { await evolutionCancelGEPA(jobId); await reload(); }
    catch (e) { setError(evolutionError(e)); }
    finally { setBusy(false); }
  }

  return (
    <section className="space-y-4" aria-label="GEPA 优化实验">
      <h2 className="text-lg font-semibold">GEPA 优化实验</h2>
      <p className="text-sm text-text-muted">
        免费验收使用 Fake Planner、Judge、Reflection，执行同一份 v3 搜索与请求账本；真实 GEPA 需要管理员显式开启并确认模型费用。
        搜索只使用 Train/Validation，不读取正式 Holdout；完成后待独立终验，不自动发布 Prompt。
      </p>
      <div className="rounded-lg border border-border p-4 space-y-3">
        <label className="flex gap-3 items-center text-sm">
          执行模式
          <select className={field} aria-label="GEPA 执行模式" value={mode} onChange={(e) => { setMode(e.target.value as "demo" | "live"); setConfirmCost(false); }}>
            <option value="demo">免付费 Demo（建议先运行）</option>
            {liveAllowed && <option value="live">真实 GEPA（会产生费用）</option>}
          </select>
        </label>
        <label className="block text-sm">搜索数据
          <select aria-label="GEPA 搜索数据" className={field} value={preset} onChange={(e) => { const value = e.target.value as "smoke" | "full"; setPreset(value); setMaxEvals(value === "smoke" ? 8 : 64); setProposals(value === "smoke" ? 1 : 4); }}>
            <option value="smoke">小规模开发/验证集烟雾测试</option><option value="full">完整 Train/Validation 搜索</option>
          </select>
        </label>
        <label className="block text-sm">评估指标 <select aria-label="GEPA 评估指标" value={metricVersion} disabled className={field}><option value="v3">v3 Quality（Coverage 硬约束）</option></select></label>
        <p className="text-sm">冻结 min_gain=0.05；沿用服务器 Planner / Judge / Reflection 模型配置，提交时冻结路由及 Baseline。</p>
        <p className="text-sm">实际请求上限 {5 * maxEvals + proposals} 次：Planner {4 * maxEvals}、Judge {maxEvals}、Reflection {proposals}。评测次数不等于 API 请求次数；SDK 自动重试关闭。</p>
        <div className="flex flex-wrap gap-4 text-sm">
          <label>评测预算（4–200） <input className={`${field} w-24`} type="number" min={4} max={200} value={maxEvals} onChange={(e) => setMaxEvals(Number(e.target.value))} /></label>
          <label>候选上限（1–8） <input className={`${field} w-20`} type="number" min={1} max={8} value={proposals} onChange={(e) => setProposals(Number(e.target.value))} /></label>
          <label>随机种子 <input className={`${field} w-24`} type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></label>
        </div>
        {mode === "live" && (
          <div className="space-y-2 text-sm">
            <p>真实优化会多次调用 Planner、独立 Judge 和反思模型；评测次数并非费用硬上限。请先检查账户额度、独立 Judge 配置和数据集冻结状态。</p>
            <label className="flex gap-2 items-start">
              <input type="checkbox" checked={confirmCost} onChange={(e) => setConfirmCost(e.target.checked)} />
              我确认启动真实 GEPA 可能产生模型 API 费用。
            </label>
          </div>
        )}
        <button type="button" className={button} onClick={() => void start()}
          disabled={busy || maxEvals < 4 || maxEvals > 200 || proposals < 1 || proposals > 8 || seed < 0 || (mode === "live" && !confirmCost)}>
          {busy ? "提交中…" : mode === "live" ? "确认启动真实 GEPA" : "运行免付费 GEPA Demo"}
        </button>
        {error && <p role="alert" className="text-sm">{error}</p>}
      </div>
      <div className="rounded-lg border border-border p-4 space-y-2">
        <div className="flex justify-between gap-2">
          <h3 className="font-semibold">实验运行记录（每 3 秒自动刷新）</h3>
          <button type="button" className={button} onClick={() => void reload()}>刷新</button>
        </div>
        {jobs.length === 0 && <p className="text-sm text-text-muted">还没有实验记录。</p>}
        {jobs.map((job) => (
          <div key={job.job_id} className="rounded-lg border border-border p-3 space-y-1 text-sm">
            <button type="button" className="underline" onClick={() => { setSelected(job.job_id); setResult(null); }}>
              {job.job_id}
            </button>
            <p>{job.task_kind === "planner_final_ab" ? "独立 Planner 终验 · " : ""}{job.mode === "demo" ? "免费 Fake v3" : "真实模型"} · {job.state === "pending_final" ? "搜索完成待终验" : job.state === "no_candidate" ? "无有效候选" : job.state} · {job.phase}</p>
            <p className="text-xs text-text-muted">预算 {job.max_evals} · 候选 {job.proposals} · 指标 {job.metric_version === "v2" ? "v2 综合" : job.metric_version === "v3" ? "v3 质量分" : "v1 覆盖率"} · 创建 {job.created_at}</p>
            {job.source_job_id && <p>来源实验：{job.source_job_id} · 终验不会修改来源搜索记录。</p>}
            {job.error_type && <p className="text-xs">错误类型：{job.error_type}</p>}
            {["queued", "running", "cancelling"].includes(job.state) &&
              <button type="button" className={button} disabled={busy || job.state === "cancelling"} onClick={() => void cancel(job.job_id)}>取消后续执行</button>}
          </div>
        ))}
      </div>
      {selected && <GEPAJobAuditPanel key={selected} jobId={selected} refresh={jobs.find((j) => j.job_id === selected)?.updated_at ?? ""} />}
      {result && (
        <div className="rounded-lg border border-border p-4 space-y-2" aria-label="GEPA 实验结果">
          <h3 className="font-semibold">实验结果：{result.job_id}</h3>
          {!result.available ? <p className="text-sm">没有完整结果；中断或失败不代表质量不合格。</p> : (
            <>
              <p className="text-sm">{result.evidence_kind === "live_planner" ? "真实 Planner 评测" : result.evidence_kind === "fake_planner" ? "Fake v3 验收（零模型 API 费用，不能证明真实模型改善）" : "脚本演示（不能证明真实模型改善）"}</p>
              {result.task_kind === "planner_final_ab" && <p className="text-sm">{result.evidence_kind === "fake_planner" ? "模拟终验结果（不证明真实质量）" : "独立 Planner 终验结果"} · 来源 {result.source_job_id} · 尚未发布；完整报告 A/B 与人工审批仍需单独完成。</p>}
              <p className="text-sm">{result.outcome === "pending_final" ? "搜索完成待终验" : result.outcome === "no_candidate" ? "搜索完成，无有效改进候选" : "实验结果"} · 留出集门禁：{result.quality_gate_passed == null ? "待终验（未运行）" : result.quality_gate_passed ? "通过" : "未通过"}</p>
              {result.comparison && <QualityComparison comparison={result.comparison} />}
              {result.rejection_reasons?.map((reason) => <p key={reason} className="text-sm">候选未接受原因：{reason}</p>)}
              {result.holdout?.reasons?.map((reason) => <p key={reason} className="text-sm">原因：{reason}</p>)}
              <p className="text-xs">Baseline SHA：{result.baseline_sha256 ?? "无"}</p>
              <p className="text-xs">Candidate SHA：{result.candidate_sha256 ?? "无"}</p>
              {result.outcome === "pending_final" && selected && <button type="button" className={button} onClick={() => setFinalSource(selected)}>准备独立 Planner 终验</button>}
              {result.prompt_diff && <details><summary className="text-sm cursor-pointer">Prompt Diff</summary><pre className="text-xs overflow-auto whitespace-pre-wrap max-h-80">{result.prompt_diff}</pre></details>}
            </>
          )}
        </div>
      )}
      {finalSource && finalSource === selected && <GEPAFinalPanel key={finalSource} sourceJobId={finalSource} onQueued={(job) => { setSelected(job.job_id); setResult(null); setFinalSource(null); void reload(); }} />}
    </section>
  );
}
