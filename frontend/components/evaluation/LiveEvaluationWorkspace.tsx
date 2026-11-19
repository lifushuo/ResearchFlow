"use client";

import { useEffect, useState } from "react";
import { fetchPipeline, fetchPipelines, type DatasetId, type Pipeline } from "@/lib/evaluation-pipeline-v13";
import { getDiagnostics, type Diagnostic } from "@/lib/evaluation-diagnostics";
import { closureGet, closurePost, closureState, closureError, type ClosureRun, type Comparison, type Gate } from "@/lib/evaluation-closure";
import { getDatasetLabel } from "@/lib/evaluation-dataset-zh";
import { evaluationErrorMessage } from "@/lib/evaluation";
import { EvaluationMetrics } from "./EvaluationMetrics";
import UnifiedEvaluationRunnerV13 from "./UnifiedEvaluationRunnerV13";
import DatasetRegistryPanel from "./DatasetRegistryPanel";
import ExecutionIncidentPanel from "./ExecutionIncidentPanel";
import { DiagnosticDetails } from "./PipelineDiagnosticsPanel";
import EvaluationCaseTable, { EvaluationSummary } from "./EvaluationCaseTable";

type Tab = "overview" | "runs" | "cases" | "trace" | "badcases" | "compare" | "datasets";
const tabs: [Tab, string][] = [["overview", "总览"], ["runs", "评测运行"], ["cases", "逐题用例"], ["trace", "执行轨迹"], ["badcases", "坏例排查"], ["compare", "版本对比"], ["datasets", "数据集"]];
const button = "rounded-lg border border-border px-3 py-2 text-sm disabled:opacity-40 hover:bg-bg-base";
const select = "min-w-0 w-full rounded-lg border border-border bg-bg-base p-2 text-sm";
const states: Record<string, string> = { queued: "排队中", running: "运行中", completed: "执行结束", completed_with_issues: "执行结束，存在异常", failed: "执行失败", blocked: "未就绪", pending: "待执行" };
const comparisonReasons: Record<string, string> = {
  same_run: "基线与候选必须是两次不同运行", empty_case_set: "缺少逐题结果", run_not_finished: "运行尚未完整结束",
  model_identity_unverified: "实际服务模型身份未验证，不能宣称严格配对实验", unmeasured_cases: "存在未测量或阻塞案例",
  dataset_id: "评测集不同", execution_mode: "执行模式不同", sample_set_hash: "样本集指纹不同", metrics_version: "评分版本不同",
  model_route: "模型配置不同", judge_route: "评分模型配置不同", case_id_set: "案例集合不同", case_input_hash: "案例输入不同", case_input_hash_missing: "缺少输入指纹",
};

function ComparisonPanel({ pipeline, dataset, onRun }: { pipeline: Pipeline | null; dataset: DatasetId | ""; onRun: () => void }) {
  const [runs, setRuns] = useState<ClosureRun[]>([]);
  const [baseline, setBaseline] = useState("");
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [gate, setGate] = useState<Gate | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const candidate = pipeline?.steps.find((s) => s.dataset_id === dataset)?.closure_run_id;
  useEffect(() => {
    let alive = true;
    closureGet<ClosureRun[]>("/runs").then((rows) => { if (alive) setRuns(rows); }).catch((e) => { if (alive) setError(closureError(e)); });
    return () => { alive = false; };
  }, [candidate]);
  async function compare() {
    if (!baseline || !candidate || busy) return;
    setBusy(true); setError(""); setGate(null); setComparison(null);
    try { setComparison(await closureGet<Comparison>(`/compare?baseline_run=${encodeURIComponent(baseline)}&candidate_run=${encodeURIComponent(candidate)}`)); }
    catch (e) { setError(closureError(e)); }
    finally { setBusy(false); }
  }
  async function checkGate() {
    if (!candidate || busy) return;
    setBusy(true); setError("");
    try { setGate(await closurePost<Gate>(`/runs/${encodeURIComponent(candidate)}/gate`, { min_measured: 1, min_pass_rate: 1, fail_on_blocked: true, fail_on_unmeasured: true })); }
    catch (e) { setError(closureError(e)); }
    finally { setBusy(false); }
  }
  const compatible = runs.filter((r) => r.dataset_id === dataset && r.run_id !== candidate);
  return <section className="space-y-4" aria-label="本次基线比较">
    <h2 className="font-semibold text-lg">基线 / 本次候选比较</h2>
    <p className="text-sm text-text-muted">候选来自上方当前运行。比较读取实际 Worker 封存结果，核对数据集、输入指纹、评分和模型配置；检查不会请求模型。</p>
    <p className="text-xs break-all">候选：{candidate || "当前步骤尚无可比较的逐题结果"}</p>
    <label className="block text-sm">同评测集基线<select aria-label="同评测集基线" className={select} value={baseline} onChange={(e) => { setBaseline(e.target.value); setComparison(null); setGate(null); }}><option value="">请选择先前运行</option>{compatible.map((r) => <option key={r.run_id} value={r.run_id}>{closureState(r.execution_mode)} · {closureState(r.state)} · {r.run_id}</option>)}</select></label>
    {compatible.length === 0 && <p className="text-sm">尚无同评测集的另一份结果。先运行一次建立基线，修改后重新执行，再选择比较。</p>}
    <div className="flex flex-wrap gap-2"><button className={button} disabled={!baseline || !candidate || busy} onClick={() => void compare()}>执行基线比较</button><button className={button} disabled={!candidate || busy} onClick={() => void checkGate()}>检查候选质量门禁</button><button className={button} onClick={onRun}>配置新一轮评测</button></div>
    {error && <p role="alert">{error}</p>}
    {comparison && <div role="status" className="rounded-lg border border-border p-4 space-y-2">
      <p className="font-semibold">{comparison.comparable ? `契约可比：退化 ${comparison.regressions?.length ?? 0} 项，恢复 ${comparison.improvements?.length ?? 0} 项` : "未通过可比性检查"}</p>
      {comparison.reasons?.map((r) => <p key={r}>{comparisonReasons[r] ?? r}</p>)}
      {comparison.regressions?.map((id) => <p key={id}>退化：{id}</p>)}{comparison.improvements?.map((id) => <p key={id}>恢复：{id}</p>)}
      <p className="text-xs text-text-muted">契约对比不自动证明代码改善；在线模型与网页证据仍可能随时间变化。</p>
    </div>}
    {gate && <div role="status" className="rounded-lg border border-border p-4"><strong>候选质量门禁：{closureState(gate.verdict)}</strong><p>有效测量 {gate.measured} / 已记录 {gate.total_recorded}；通过率 {gate.pass_rate == null ? "未测量" : `${(gate.pass_rate * 100).toFixed(2)}%`}</p><EvaluationMetrics metrics={gate.counts} /><p className="text-xs text-text-muted">门禁按原评测契约要求完整测量及全部通过；没有质量阈值的指标不会被自动当作通过。</p></div>}
  </section>;
}

export default function LiveEvaluationWorkspace() {
  const [tab, setTab] = useState<Tab>("overview");
  const [jobs, setJobs] = useState<Pipeline[]>([]);
  const [pipe, setPipe] = useState("");
  const [dataset, setDataset] = useState<DatasetId | "">("");
  const [current, setCurrent] = useState<Pipeline | null>(null);
  const [diagnostic, setDiagnostic] = useState<Diagnostic | null>(null);
  const [error, setError] = useState("");
  const [diagnosticError, setDiagnosticError] = useState("");
  const [revision, setRevision] = useState(0);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [traceCase, setTraceCase] = useState("");
  const [preset, setPreset] = useState<Pick<Pipeline, "datasets" | "max_cases" | "case_id"> | undefined>();
  const [launchRevision, setLaunchRevision] = useState(0);
  function choose(row: Pipeline, target?: DatasetId) {
    setPipe(row.pipeline_id); setCurrent(row); setDataset(target || row.datasets[0] || ""); setDiagnostic(null); setDiagnosticError(""); setSearch(""); setStatus(""); setTraceCase("");
  }
  function configure() {
    setPreset(current ? { datasets: [...current.datasets], max_cases: current.max_cases, ...(current.case_id ? { case_id: current.case_id } : {}) } : undefined);
    setLaunchRevision((x) => x + 1); setTab("runs");
  }
  function created(row: Pipeline) { choose(row); setRevision((x) => x + 1); setTab("overview"); }
  function openCases(target: DatasetId) {
    setDataset(target); setDiagnostic(null); setRevision((x) => x + 1); setTab("cases");
  }
  function openCase(id: string) { setTraceCase(id); setTab("trace"); }
  useEffect(() => {
    let alive = true;
    fetchPipelines().then(({ pipelines }) => {
      if (!alive) return;
      setJobs(pipelines); setError("");
      if (!pipe && pipelines[0]) { const first = pipelines[0]; setPipe(first.pipeline_id); setCurrent(first); setDataset(first.datasets[0] || ""); }
    }).catch((e) => { if (alive) setError(evaluationErrorMessage(e, "当前运行加载失败")); });
    return () => { alive = false; };
  }, [revision, pipe]);
  useEffect(() => {
    if (!pipe) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function load() {
      try { const row = await fetchPipeline(pipe); if (!alive) return; setCurrent(row); if (["queued", "running"].includes(row.state)) timer = setTimeout(() => void load(), 3000); }
      catch (e) { if (alive) setError(evaluationErrorMessage(e, "运行状态读取失败")); }
    }
    void load(); return () => { alive = false; clearTimeout(timer); };
  }, [pipe, revision]);
  const step = current?.steps.find((s) => s.dataset_id === dataset);
  useEffect(() => {
    if (!pipe || !dataset) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = () => getDiagnostics(pipe, dataset).then((row) => {
      if (alive) { setDiagnostic(row); setDiagnosticError(""); if (["running", "pending"].includes(row.state)) timer = setTimeout(() => void load(), 4000); }
    }).catch((e) => { if (alive) setDiagnosticError(evaluationErrorMessage(e, "逐题证据读取失败")); });
    void load(); return () => { alive = false; clearTimeout(timer); };
  }, [pipe, dataset, step?.state, revision]);
  useEffect(() => {
    const open = (event: Event) => {
      const value = (event as CustomEvent<{ pipe: string; dataset: DatasetId; caseId?: string }>).detail;
      if (!value || typeof value.pipe !== "string" || typeof value.dataset !== "string") return;
      setPipe(value.pipe); setDataset(value.dataset); setCurrent(null); setDiagnostic(null); setDiagnosticError(""); setTraceCase(value.caseId || ""); setTab("trace"); setRevision((x) => x + 1);
    };
    window.addEventListener("rf-evaluation-diagnostics", open);
    return () => window.removeEventListener("rf-evaluation-diagnostics", open);
  }, []);
  const evidence = diagnostic?.pipeline_id === pipe && diagnostic.dataset_id === dataset ? diagnostic : null;
  const visibleJobs = current && current.pipeline_id === pipe ? [current, ...jobs.filter((j) => j.pipeline_id !== pipe)] : jobs;
  const rows = (evidence?.cases ?? []).filter((r) => r.case_id.toLowerCase().includes(search.toLowerCase()) && (!status || r.status === status));
  return <div className="space-y-5">
    <header className="flex flex-wrap items-start justify-between gap-3"><div><h1 className="font-serif text-3xl font-bold">评测中心</h1><p className="mt-2 text-sm text-text-muted">发起运行 → 逐题测量 → 轨迹分析 → 坏例排查 → 新运行复测 → 质量门禁</p></div><button className={button} onClick={configure}>新建 / 重新配置评测</button></header>
    <div className="rounded-xl border border-border bg-bg-elevated p-4 space-y-3">
      <div className="flex flex-wrap gap-3"><label className="min-w-0 flex-1 text-sm">本次评测运行<select aria-label="本次评测运行" className={select} value={pipe} onChange={(e) => { const row = visibleJobs.find((j) => j.pipeline_id === e.target.value); if (row) choose(row); }}><option value="">请选择运行</option>{visibleJobs.map((j) => <option key={j.pipeline_id} value={j.pipeline_id}>{j.created_at} · {states[j.state] ?? j.state} · {j.pipeline_id}</option>)}</select></label><label className="min-w-0 text-sm">本次评测集<select aria-label="本次评测集" className={select} value={dataset} onChange={(e) => { setDataset(e.target.value as DatasetId); setDiagnostic(null); setDiagnosticError(""); setSearch(""); setStatus(""); setTraceCase(""); }}><option value="">请选择评测集</option>{current?.datasets.map((id) => <option key={id} value={id}>{getDatasetLabel(id)}</option>)}</select></label></div>
      <div className="flex flex-wrap items-center gap-3 text-xs"><button className={button} onClick={() => { setError(""); setRevision((x) => x + 1); }}>刷新本次结果</button><span>{current ? states[current.state] ?? current.state : "尚无运行"}</span>{pipe && <span className="font-mono break-all">{pipe}</span>}</div>
      <p className="text-xs text-text-muted">各分区查看上方选中的当前账号运行。原始结果封存防覆盖；你可以发起新运行、查看证据和比较结果，不需要手工导入历史 JSON。</p>
    </div>
    {error && <p role="alert">{error}</p>}
    <nav aria-label="评测分区" className="flex flex-wrap gap-1 border-b border-border pb-2">{tabs.map(([id, name]) => <button key={id} className={`${button} ${tab === id ? "text-accent bg-bg-elevated font-semibold" : "text-text-secondary"}`} aria-current={tab === id ? "page" : undefined} onClick={() => setTab(id)}>{name}</button>)}</nav>
    {tab === "runs" && <UnifiedEvaluationRunnerV13 key={launchRevision} initial={preset} onCreated={created} showResults={false} />}
    {tab === "datasets" && <DatasetRegistryPanel onCreated={created} />}
    {tab === "overview" && <section className="space-y-4" aria-label="本次运行总览"><h2 className="font-semibold text-lg">本次运行总览</h2>{current ? <>
      <p className="text-sm text-text-muted">只显示本轮选中的评测集，不把未运行的其他维度渲染成空成绩。执行结束与质量通过分别记录。</p>
      <div className={`grid min-w-0 gap-4 ${current.steps.length > 1 ? "xl:grid-cols-2" : "grid-cols-1"}`}>{current.steps.map((s) => <article key={s.dataset_id} className="rounded-xl border border-border bg-bg-elevated p-4 space-y-3"><h3 className="font-semibold">{getDatasetLabel(s.dataset_id)}</h3><p className="text-sm">{states[s.state]} · {s.case_count ?? "尚未记录"} 条案例 · {s.elapsed_seconds == null ? "耗时未测量" : `${s.elapsed_seconds} 秒`}</p>{s.metrics && Object.keys(s.metrics).length > 0 ? s.dataset_id === "report-benchmark" ? <><EvaluationSummary metrics={s.metrics} cases={s.dataset_id === dataset ? evidence?.cases : undefined} onOpen={(id) => { setDataset(s.dataset_id); openCase(id); }} /><details className="text-sm"><summary className="cursor-pointer">原始指标与口径</summary><EvaluationMetrics metrics={s.metrics} /></details></> : <EvaluationMetrics metrics={s.metrics} /> : <p className="text-sm text-text-muted">{s.state === "pending" || s.state === "running" ? "执行器尚未完成测量，结果将自动刷新。" : "本次未产生有效质量测量，请查看执行事实与失败分析。"}</p>}<div className="flex flex-wrap gap-2"><button className={button} onClick={() => { openCases(s.dataset_id); }}>查看本轮逐题结果</button><button className={button} onClick={() => { setDataset(s.dataset_id); setTraceCase(""); setTab("trace"); }}>诊断与轨迹</button></div></article>)}</div>
    </> : <p className="text-sm">尚无本账号的评测任务。点击「新建 / 重新配置评测」运行固定集；离线三套可不使用收费模型执行。</p>}</section>}
    {tab === "cases" && <section className="space-y-3" aria-label="本次逐题结果"><h2 className="font-semibold text-lg">本次逐题结果 · {dataset ? getDatasetLabel(dataset) : "未选择评测集"}</h2><p className="text-xs text-text-muted">来自本轮 Worker 的标准化逐题结果。预期失败的故障回放通过，表示异常处理符合契约。</p><div className="flex flex-wrap gap-2"><input aria-label="按案例 ID 搜索" className={select} placeholder="按案例 ID 搜索" value={search} onChange={(e) => setSearch(e.target.value)} /><select aria-label="案例判定筛选" className={select} value={status} onChange={(e) => setStatus(e.target.value)}><option value="">全部判定</option>{["passed", "failed", "blocked", "unmeasured"].map((s) => <option key={s} value={s}>{closureState(s)}</option>)}</select></div>{diagnosticError && <p role="alert">{diagnosticError}</p>}{rows.length > 0 && <EvaluationCaseTable rows={rows} onOpen={openCase} />}{rows.length === 0 && <p className="text-sm">{evidence ? "没有匹配的逐题结果。若执行中断，请查看运行诊断；没有结果不能计作通过。" : "尚未取得本轮逐题结果，请选择运行或刷新。"}</p>}</section>}
    {tab === "trace" && (pipe && dataset ? <DiagnosticDetails key={`${pipe}/${dataset}/${traceCase}`} pipe={pipe} dataset={dataset} initialCase={traceCase} /> : <p>选择本次运行后查看真实执行证据。</p>)}
    {tab === "badcases" && <section className="space-y-4"><h2 className="font-semibold text-lg">坏例排查与执行异常</h2><p className="text-sm text-text-muted">自动收集并去重当前账号的运行异常与未通过案例。查看原始证据后，通过新运行比较和质量门禁核验。</p><ExecutionIncidentPanel key={`${pipe}/${dataset}/${step?.state}`} pipe={pipe} dataset={dataset} /></section>}
    {tab === "compare" && <ComparisonPanel key={`${pipe}/${dataset}`} pipeline={current} dataset={dataset} onRun={configure} />}
    <footer className="border-t border-border pt-4 text-xs text-text-muted">在线报告、证据与校准调用真实模型，检索使用配置的向量化服务；离线回放和执行器运行真实代码及受控模型，工具契约评估合成调用。不同口径分别展示，不合并成总准确率。付费执行由你确认后启动。</footer>
  </div>;
}
