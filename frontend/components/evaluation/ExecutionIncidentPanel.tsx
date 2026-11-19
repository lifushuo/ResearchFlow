"use client";

import { useEffect, useState } from "react";
import { getIncidents, openDiagnostics, type Incident } from "@/lib/evaluation-diagnostics";
import { getDatasetLabel } from "@/lib/evaluation-dataset-zh";
import { evaluationErrorMessage } from "@/lib/evaluation";
import type { DatasetId } from "@/lib/evaluation-pipeline-v13";

const button = "rounded-lg border border-border px-3 py-2 text-xs disabled:opacity-40";

export default function ExecutionIncidentPanel({ pipe, dataset }: { pipe?: string; dataset?: DatasetId | "" }) {
  const [rows, setRows] = useState<Incident[]>([]);
  const [revision, setRevision] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    getIncidents(pipe, dataset).then(({ incidents }) => { if (alive) { setRows(incidents); setError(""); setLoading(false); } }).catch((err) => { if (alive) { setError(evaluationErrorMessage(err, "执行异常待排查池读取失败")); setLoading(false); } });
    return () => { alive = false; };
  }, [revision, pipe, dataset]);
  return <section aria-label="执行异常待排查池" className="space-y-3 rounded-lg border border-border p-4">
    <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="font-semibold">执行异常与未通过案例 · 自动待排查池</h3><button className={button} onClick={() => { setLoading(true); setRevision((x) => x + 1); }}>刷新待排查池</button></div>
    {loading && <p role="status">正在读取本轮坏例与执行证据…</p>}
    <p className="text-xs text-text-muted">超时、中断、未就绪及未通过判据均可查看；同评测集、案例、输入指纹和异常类型去重。执行异常与质量不合格分别记录；通过新运行比较结果核验修复。</p>
    {error && <p role="alert" className="text-sm">{error}</p>}
    {rows.map((row) => <div key={row.incident_id} className="rounded-lg border border-border bg-bg-base p-3 space-y-2 text-sm">
      <p className="font-semibold">{getDatasetLabel(row.dataset_id)} · {row.case_id || "本次案例集"}</p>
      <p>{row.summary}</p><p className="text-xs text-text-muted">独立运行出现 {row.occurrences.length} 次</p>
      {row.collection === "saved_status_analysis" && <p className="text-xs text-text-muted">历史状态分析，未伪造原运行轨迹。</p>}
      <div className="flex flex-wrap gap-2"><button className={button} onClick={() => openDiagnostics(row.pipeline_id, row.dataset_id, row.case_id || undefined)}>查看诊断与轨迹</button></div>
    </div>)}
    {rows.length === 0 && !error && !loading && <p className="text-sm text-text-muted">本轮所选评测集没有已记录的执行异常或未通过案例。</p>}
  </section>;
}
