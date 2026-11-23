"use client";

import { useEffect, useState } from "react";
import { evolutionFinalPreview, evolutionStartFinal, evolutionError, type GEPAFinalPreview, type GEPAJob } from "@/lib/evaluation-evolution";

const disabledReasons: Record<string, string> = {
  administrator_has_not_enabled_paid_final: "管理员尚未单独开启付费 Planner 终验。",
  final_protocol_unavailable: "冻结留出协议未就绪。",
  formal_holdout_already_reserved: "正式留出集已被终验任务预约，不能重复曝光。",
  live_final_requires_real_source: "模拟搜索不能作为真实终验的来源。",
};

export default function GEPAFinalPanel({ sourceJobId, onQueued }: { sourceJobId: string; onQueued: (job: GEPAJob) => void }) {
  const [preview, setPreview] = useState<GEPAFinalPreview | null>(null);
  const [mode, setMode] = useState<"demo" | "live">("demo");
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [requestId] = useState(() => crypto.randomUUID());
  useEffect(() => {
    let active = true;
    void evolutionFinalPreview(sourceJobId).then((value) => { if (active) setPreview(value); })
      .catch((e) => { if (active) setError(evolutionError(e)); });
    return () => { active = false; };
  }, [sourceJobId]);

  async function start() {
    if (!preview || busy || (mode === "live" && (!preview.live_enabled || !consent))) return;
    setBusy(true); setError("");
    try {
      const job = await evolutionStartFinal(sourceJobId, { request_id: requestId, mode,
        consent: mode === "live" ? "I_CONFIRM_ONE_PLANNER_FINAL_AB_API_COST" : null });
      onQueued(job);
    } catch (e) { setError(evolutionError(e)); }
    finally { setBusy(false); }
  }

  return <section aria-label="独立 Planner 终验" className="rounded-lg border border-border p-4 space-y-3">
    <h3 className="font-semibold">独立 Planner 终验</h3>
    <p className="text-sm">固定当前候选，与原 Baseline 做两臂比较；不会重新搜索、调用 Reflection 或应用生产 Prompt。</p>
    {!preview && !error && <p className="text-sm">正在核对冻结候选和终验配置…</p>}
    {preview && <>
      <p className="text-xs">来源实验：{preview.source_job_id} · Candidate SHA：{preview.candidate_sha256}</p>
      <p className="text-sm">{preview.case_count} 个案例，两臂共 {preview.evaluations} 次评测；请求硬上限 {preview.request_cap}：Planner {preview.quotas.planner}、Judge {preview.quotas.judge}、Reflection {preview.quotas.reflection}。冻结 min_gain={preview.min_gain}。</p>
      <label className="flex gap-3 items-center text-sm">终验执行模式
        <select aria-label="Planner 终验执行模式" className="rounded border border-border bg-bg-base p-2" value={mode} onChange={(e) => { setMode(e.target.value as "demo" | "live"); setConsent(false); }}>
          <option value="demo">免费 Fake 终验流程验收</option>
          <option value="live">真实 Planner 留出终验（会收费）</option>
        </select>
      </label>
      {mode === "demo" ? <p className="text-sm text-text-muted">只使用独立模拟案例，不读取正式 Holdout。模拟成绩不能证明此候选的真实质量。</p> : <>
        <p className="text-sm">正式留出集只允许一次已冻结候选的终验；入队即保留曝光额度，取消或失败不会自动释放。实际费用以模型代理账单为准。</p>
        {!preview.live_enabled && <p role="status" className="text-sm">{disabledReasons[preview.live_disabled_reason ?? ""] ?? "真实终验暂不可启动。"}</p>}
        <label className="flex items-start gap-2 text-sm"><input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />我确认本次独立 Planner 终验可能产生模型 API 费用，并使用一次正式留出曝光。</label>
      </>}
      <button type="button" className="rounded border border-border px-3 py-2 text-sm disabled:opacity-40" disabled={busy || (mode === "live" && (!preview.live_enabled || !consent))} onClick={() => { void start(); }}>
        {busy ? "正在入队…" : mode === "demo" ? "运行免费终验流程验收" : "确认启动一次真实 Planner 终验"}
      </button>
      <p className="text-xs text-text-muted">Planner 终验通过后仍需单独的完整报告 A/B 与人工发布审批；此处不会发布或回滚 Prompt。</p>
    </>}
    {error && <p role="alert" className="text-sm">{error}</p>}
  </section>;
}
