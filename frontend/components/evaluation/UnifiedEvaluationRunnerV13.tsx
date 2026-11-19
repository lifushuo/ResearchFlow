"use client";

import { useEffect, useMemo, useState } from "react";
import {
  createPipeline, fetchPipeline, fetchPipelineCapabilities, fetchPipelines,
  type Capabilities, type DatasetId, type Pipeline,
} from "@/lib/evaluation-pipeline-v13";
import { evaluationErrorMessage } from "@/lib/evaluation";
import { evaluationDataset, type DatasetCase } from "@/lib/evaluation-datasets";

import { getDatasetLabel, getDatasetShort } from "@/lib/evaluation-dataset-zh";
import { EvaluationMetrics } from "@/components/evaluation/EvaluationMetrics";
import { openDiagnostics } from "@/lib/evaluation-diagnostics";
const reasons: Record<string, string> = {
  ready: "可执行", online_runner_disabled: "在线执行未启用，请检查服务端的真实模型与评分模型配置",
  judge_original_source_invalid: "原始24条校准样本丢失或校验失败",
  executor_suite_missing: "执行器固定测试文件缺失", executor_suite_invalid: "执行器测试集契约校验失败",
  judge_all_unmeasured: "评分模型本次没有有效判定，不计分",
  judge_route_not_configured: "尚未配置独立评分模型",
  runner_timeout: "运行超时", suite_execution_failed: "原评测执行失败",
  execution_artifacts_already_exist: "该任务已有执行产物，已阻止重复调用和覆盖证据",
  worker_restarted_result_uncertain: "执行器中断，本次结果不可判定",
  executor_suite_execution_failed: "执行器回归运行失败", executor_result_validation_failed: "执行器结果校验未通过",
  unsupported_offline_adapter: "该数据集尚无离线执行适配", offline_run_failed: "离线执行失败",
  result_missing: "结果文件缺失", result_validation_failed: "结果校验未通过", case_identity_failed: "案例标识校验未通过",
  unknown_report_case_id: "找不到指定报告案例，未执行", duplicate_report_case_id: "报告数据存在重复案例标识，未执行",
  case_id_requires_report_benchmark_only: "单案例模式只允许选择研究报告评测", case_id_requires_max_cases_1: "单案例模式的案例上限必须为 1",
  unlinked_or_unverified_adapter: "执行适配尚未关联或核验", invalid_tool_result: "工具契约结果无效",
  invalid_replay_result: "回放结果无效", worker_internal_error: "运行器内部错误",
};

const configSourceText: Record<string, string> = {
  env: "复用环境配置文件中的真实模型与评分模型",
  rf_eval_override: "使用评测专用配置",
  disabled: "未启用（脚本模型模式或在线评测已关闭）",
  none: "未配置独立评分模型",
};
const states: Record<string, string> = {
  queued: "排队中", running: "运行中", completed: "已完成",
  completed_with_issues: "已结束（含失败或未就绪）", failed: "失败",
  blocked: "未就绪", pending: "待执行",
};

export default function UnifiedEvaluationRunnerV13({ initial, onCreated, showResults = true }: { initial?: Pick<Pipeline, "datasets" | "max_cases" | "case_id">; onCreated?: (pipeline: Pipeline) => void; showResults?: boolean }) {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [selection, setSelection] = useState<DatasetId[]>(initial?.datasets ?? ["planner-replay-v1"]);
  const [maxCases, setMaxCases] = useState(initial?.max_cases ?? 1);
  const [caseId, setCaseId] = useState(initial?.case_id ?? "");
  const [reportCases, setReportCases] = useState<DatasetCase[]>([]);
  const [reportCaseError, setReportCaseError] = useState("");
  const hasReport = selection.includes("report-benchmark");
  const [checked, setChecked] = useState(false);
  const [loading, setLoading] = useState(false);
  const [jobs, setJobs] = useState<Pipeline[]>([]);
  const [currentId, setCurrentId] = useState("");
  const [detail, setDetail] = useState<Pipeline | null>(null);
  const [error, setError] = useState("");
  // Bumping this counter re-runs the list effect. All setState calls stay inside
  // async callbacks, so the effect body itself never sets state (lint-safe).
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (!hasReport) return;
    let alive = true;
    evaluationDataset("report-benchmark", 0, 100).then(({ dataset }) => {
      if (alive) { setReportCases(dataset.cases); setReportCaseError(""); }
    }).catch((err: unknown) => { if (alive) { setReportCases([]); setReportCaseError(evaluationErrorMessage(err, "无法读取报告案例列表")); } });
    return () => { alive = false; };
  }, [hasReport]);

  useEffect(() => {
    let alive = true;
    fetchPipelineCapabilities().then((value) => {
      if (alive) setCaps(value);
    }).catch((err: unknown) => {
      if (alive) setError(evaluationErrorMessage(err, "评测任务接口不可用"));
    });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    let alive = true;
    fetchPipelines().then(({ pipelines }) => {
      if (!alive) return;
      setJobs(pipelines);
      setCurrentId((old) => old || pipelines[0]?.pipeline_id || "");
    }).catch((err: unknown) => {
      if (alive) setError(evaluationErrorMessage(err, "评测任务接口不可用"));
    });
    return () => { alive = false; };
  }, [refreshKey]);

  useEffect(() => {
    if (!jobs.some((j) => j.state === "queued" || j.state === "running")) return;
    const timer = window.setInterval(() => setRefreshKey((key) => key + 1), 4000);
    return () => window.clearInterval(timer);
  }, [jobs]);

  useEffect(() => {
    if (!currentId) return;
    let alive = true;
    const poll = async () => {
      try {
        const row = await fetchPipeline(currentId);
        if (alive) setDetail(row);
      } catch (err) {
        if (alive) setError(evaluationErrorMessage(err, "无法读取任务结果"));
      }
    };
    void poll();
    const timer = window.setInterval(() => { void poll(); }, 2500);
    return () => { alive = false; window.clearInterval(timer); };
  }, [currentId]);

  const hasOnline = useMemo(() => selection.some((s) => {
    const cap = caps?.datasets.find((c) => c.dataset_id === s);
    return cap?.lane === "online" && cap.ready;
  }), [selection, caps]);
  const active = jobs.some((j) => j.state === "queued" || j.state === "running");
  const toggle = (id: DatasetId, enabled: boolean) => {
    setSelection((old) => enabled ? [...old.filter((x) => x !== id), id] : old.filter((x) => x !== id));
    setChecked(false);
    if (id === "report-benchmark" && !enabled) setCaseId("");
  };
  const start = async () => {
    if (!caps || loading || active || !selection.length || (hasOnline && !checked)) return;
    setLoading(true); setError("");
    try {
      const job = await createPipeline({
        datasets: caps.datasets.map((x) => x.dataset_id).filter((id) => selection.includes(id)),
        max_cases: maxCases,
        ...(caseId ? { case_id: caseId } : {}),
        request_id: crypto.randomUUID(),
        consent: hasOnline ? (maxCases === 0 ? "I_CONFIRM_FULL_DATASET_API_COST" : "I_CONFIRM_REAL_MODEL_API_COST") : null,
      });
      setCurrentId(job.pipeline_id);
      setDetail(job);
      setRefreshKey((key) => key + 1);
      onCreated?.(job);
    } catch (err) {
      setError(evaluationErrorMessage(err, "启动失败"));
    } finally { setLoading(false); }
  };

  return <section className="space-y-4 rounded-xl border border-border bg-bg-elevated p-5">
    <div>
      <h2 className="font-semibold text-text-primary">统一评测运行（v1.3）</h2>
      <p className="mt-1 text-xs text-text-secondary">同一任务编排：数据集 → 专属执行器 → 状态与指标。不同评测口径不合并总准确率。</p>
      {caps && (
        <p className="mt-1 text-xs text-text-secondary">
          在线配置来源：{configSourceText[caps.online_config_source ?? "disabled"]}；独立评分模型：
          {caps.judge_route_configured ? "已配置" : "未配置"}。在线执行需要服务端启用，并确认可能产生的费用。
        </p>
      )}
    </div>
    {caps ? <>
      <div className="grid gap-2 md:grid-cols-2">
        {caps.datasets.map((cap) => <label key={cap.dataset_id} className="flex items-start gap-2 rounded-lg border border-border p-3 text-xs">
          <input type="checkbox" checked={selection.includes(cap.dataset_id)}
            disabled={loading || (!!caseId && cap.dataset_id !== "report-benchmark")}
            onChange={(e) => toggle(cap.dataset_id, e.target.checked)} className="mt-0.5" />
          <span className="space-y-1">
            <strong className="block">{getDatasetLabel(cap.dataset_id)}</strong>
            <span className="block text-text-muted leading-relaxed">{getDatasetShort(cap.dataset_id)}</span>
            <span className={cap.ready ? "text-emerald-700" : "text-amber-700"}>
              {cap.lane === "offline" ? "离线" : "在线 / 向量化服务"} · {reasons[cap.reason] ?? cap.reason}
            </span>
          </span>
        </label>)}
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" disabled={!!caseId || loading} onClick={() => { setSelection(caps.datasets.map((x) => x.dataset_id)); setChecked(false); }}
          className="rounded-lg border border-border px-3 py-2 text-xs">全选（包含未就绪项）</button>
        <button type="button" disabled={!!caseId || loading} onClick={() => { setSelection(caps.datasets.filter((x) => x.ready).map((x) => x.dataset_id)); setChecked(false); }}
          className="rounded-lg border border-border px-3 py-2 text-xs">只选可执行项</button>
        <label className="text-xs">案例上限（用于报告、引用支撑和评分校准；检索始终运行全套）
          <select disabled={!!caseId || loading} value={maxCases} onChange={(e) => { setMaxCases(Number(e.target.value)); setChecked(false); }} className="ml-2 rounded border border-border bg-bg-base p-2">
            {[1, 2, 5, 10, 20].map((num) => <option key={num} value={num}>{num} 条上限</option>)}
            <option value={0}>完整数据集（最高425条，可产生高额费用）</option>
          </select>
        </label>
      </div>
      {hasReport && <div className="space-y-2 rounded-lg border border-border p-3 text-xs text-text-secondary">
        <label className="block">研究报告单案例（可选）
          <select disabled={loading || !reportCases.length} value={caseId} onChange={(e) => {
            setCaseId(e.target.value); setChecked(false);
            if (e.target.value) { setSelection(["report-benchmark"]); setMaxCases(1); }
          }} className="mt-2 block w-full rounded border border-border bg-bg-base p-2">
            <option value="">不指定，按原案例上限顺序运行</option>
            {reportCases.map((item) => <option key={item.case_id} value={item.case_id}>{item.case_id}</option>)}
          </select>
        </label>
        {reportCaseError && <p role="alert">{reportCaseError}；单案例选择暂不可用。</p>}
        {caseId ? <p role="status" className="font-semibold text-text-primary">只运行案例：{caseId}。使用原始题目，仅选择报告评测；案例上限固定为 1。仍会调用真实模型与独立评分模型并可能产生费用。</p>
          : <p>案例来自原始报告数据集。指定后只执行这一题，不同时运行其他数据集；清空选择恢复原模式。</p>}
      </div>}
      {hasOnline && <label className="flex items-start gap-2 rounded-md border border-amber-500/40 p-3 text-xs text-amber-700">
        <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />
        <span>我确认真实模型评测可能产生接口费用，检索评测也需要真实向量化服务。小样本模式最多 20 条；完整模式会运行报告 10 题、评分校准 24 条、引用支撑 425 条等所选数据集全套。两套检索评测始终分别运行全部 12 / 16 题。在线执行使用服务端配置的真实模型与独立评分模型。此处没有强制费用上限，请谨慎选择全量。</span>
      </label>}
      <button type="button" disabled={loading || active || !selection.length || (hasOnline && !checked)}
        onClick={() => void start()} className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-contrast disabled:opacity-50">
        {loading ? "正在创建…" : active ? "有评测正在执行" : "运行选中评测集"}
      </button>
    </> : <p className="text-xs text-text-muted">加载执行能力…</p>}
    {error && <p role="alert" className="text-sm text-rose-700">{error}</p>}
    {showResults && <div className="flex flex-wrap gap-2">
      {jobs.slice(0, 8).map((j) => <button key={j.pipeline_id} type="button" onClick={() => setCurrentId(j.pipeline_id)}
        className={`rounded border px-2 py-1 text-xs ${currentId === j.pipeline_id ? "border-accent" : "border-border"}`}>
        {j.datasets.length} 项 · {states[j.state] ?? j.state}</button>)}
    </div>}
    {showResults && detail && detail.pipeline_id === currentId && <div className="rounded-lg border border-border p-4 space-y-3">
      <div className="text-sm font-semibold">{states[detail.state] ?? detail.state}</div>
      <div className="text-xs font-mono text-text-muted break-all">{detail.pipeline_id}</div>
      {detail.case_id && <p className="text-xs font-semibold text-text-primary">本次只运行报告案例：{detail.case_id}</p>}
      <div className="flex flex-wrap gap-2">{detail.steps.map((step) => <button key={step.dataset_id} type="button" className="rounded border border-border px-3 py-2 text-xs text-accent" onClick={() => openDiagnostics(detail.pipeline_id, step.dataset_id)}>{getDatasetShort(step.dataset_id)}：诊断与轨迹</button>)}</div>
      <div className="space-y-2">
        {detail.steps.map((step) => <details key={step.dataset_id} className="rounded border border-border p-3 text-xs">
          <summary className="cursor-pointer font-medium">{getDatasetLabel(step.dataset_id)} · {states[step.state] ?? step.state}
            {step.case_count !== null ? ` · ${step.case_count} 条` : ""}</summary>
          {step.reason && <p className="mt-2 text-amber-700">{reasons[step.reason] ?? step.reason}</p>}
          <p className="mt-2">已记录耗时：{step.elapsed_seconds === null ? "未测量" : `${step.elapsed_seconds} 秒`}</p>
          <button type="button" className="mt-2 rounded border border-border px-3 py-2 text-accent" onClick={() => openDiagnostics(detail.pipeline_id, step.dataset_id)}>查看本次诊断、轨迹与逐案例结果</button>
          {step.closure_status && <p className="mt-2 text-text-muted">失败回流记录：{({ recorded: "已记录", incomplete: "不完整，不能用于放行", unavailable: "未记录，请检查执行器日志" } as Record<string, string>)[step.closure_status] ?? step.closure_status}</p>}
          {step.metrics && <>
            <EvaluationMetrics metrics={step.metrics} />
            <details className="mt-2"><summary className="cursor-pointer text-text-muted">原始技术数据（排查时查看）</summary>
              <pre className="mt-2 overflow-auto whitespace-pre-wrap break-all text-text-secondary">{JSON.stringify(step.metrics, null, 2)}</pre>
            </details>
          </>}
        </details>)}
      </div>
      {detail.steps.some((s) => s.state === "failed") && <p className="text-sm text-text-secondary">本次有执行异常，已进入「失败回流与门禁 → 执行异常待排查池」。展开对应评测项可查看诊断；未完成的质量指标不计分。</p>}
      <p className="text-xs text-text-muted">本次数据保存在独立评测任务卷；原有历史成绩与评分文件不会覆盖。未就绪或失败不会计作通过。</p>
    </div>}
    <p className="text-xs text-text-muted">旧版离线任务接口仍然保留。评分模型校准使用原始 24 条样本并核对文件指纹；执行器七题使用实际执行逻辑、脚本模型和受控网络响应。是否通过以本次结果为准，不能借用历史成绩。</p>
  </section>;
}
