"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import GEPAJobsPanel from "./GEPAJobsPanel";
import {
  evolutionAnnotations,
  evolutionBackflow,
  evolutionBadCases,
  evolutionBenchmarkCases,
  evolutionDataset,
  evolutionError,
  evolutionExperiments,
  evolutionOverview,
  evolutionPlannerEvidence,
  evolutionPlannerJudge,
  evolutionReviewableCases,
  evolutionReviewableCaseEvidence,
  type ReviewableEvaluationCase,
  type PlannerJudgeSuggestion,
  splitLabel,
  stageLabel,
  type BackflowRequest,
  type BenchmarkAnnotations,
  type BenchmarkCase,
  type BenchmarkCaseSet,
  type EvolutionBadCase,
  type EvolutionDatasetState,
  type EvolutionExperimentRow,
  type EvolutionOverview,
} from "@/lib/evaluation-evolution";
import { evaluationDatasets, type DatasetEntry } from "@/lib/evaluation-datasets";

const button = "rounded-lg border border-border px-3 py-2 text-sm disabled:opacity-40 hover:bg-bg-base";
const input = "min-w-0 w-full rounded-lg border border-border bg-bg-base p-2 text-sm";
const chip = "rounded-full border border-border px-2 py-0.5 text-xs text-text-secondary";
const tabs: [string, string][] = [
  ["flow", "闭环流程"],
  ["backflow", "坏例回流"],
  ["review", "Planner 逐题复核"],
  ["dataset", "开发数据集"],
  ["experiments", "实验记录"],
  ["gepa", "GEPA 优化实验"],
];

const FLOW_STEPS: [string, string, string, string][] = [
  ["① 基线评测", "评测中心 · 真实模型", "在评测中心用 report-benchmark 跑现有版本，得到逐题完成状态与独立 Judge 证据。", "/evaluation"],
  ["② Planner 复核", "对照标注与轨迹", "报告通过或失败的真实已测题均可复核；缺少原始任务证据的案例不能提交。", "/evaluation"],
  ["③ 回流标注", "本页 · 坏例回流", "把 Run ID、必答维度、复核结论存成结构化反馈；归因 Planner 漏项的进入开发集，其余仅存证。", ""],
  ["④ GEPA 优化", "本页 · 需人工启动", "先运行免费 Demo 验证作业队列；真实优化需授权，在开发集/验证集搜索，留出集只用于候选冻结后的验收。", ""],
  ["⑤ A/B 验收", "CLI · 报告回归", "同一批固定报告题跑原版与候选，校验完成率与引用支持率不回退。", ""],
  ["⑥ 人工发布", "CLI + Agent 设置", "审阅者确认后导出 apply/rollback 配置，人工应用到 Agent 设置；下轮评测复测新版本。", ""],
];

function gateBadge(passed: boolean | null, unknownText: string) {
  if (passed === true) return <span className={chip}>留出集门禁通过</span>;
  if (passed === false) return <span className={chip}>留出集门禁未通过</span>;
  return <span className={chip}>{unknownText}</span>;
}

function FlowOverview({ overview, annotations }: { overview: EvolutionOverview | null; annotations: BenchmarkAnnotations | null }) {
  const d = overview?.dataset;
  const e = overview?.experiments;
  const b = overview?.bad_cases;
  return (
    <section className="space-y-4" aria-label="自进化闭环流程">
      <p className="text-sm text-text-muted">
        自进化回答两个问题：Planner 实际漏了什么（真实评测 → 原始任务证据 → 人工复核），优化后的 Prompt 在未见案例上是否更好（候选评测 → A/B 验收）。
        你在评测中心手动启动最终真实模型评测；本页不会自动发起付费调用或发布。
      </p>
      <ol className="grid gap-3 md:grid-cols-2">
        {FLOW_STEPS.map(([title, where, text, href]) => (
          <li key={title} className="rounded-lg border border-border p-4 space-y-1">
            <p className="font-semibold">{title}</p>
            <p className="text-xs text-text-secondary">{where}</p>
            <p className="text-sm text-text-muted">{text}</p>
            {href && <Link className="text-xs underline" href={href}>前往评测中心 →</Link>}
          </li>
        ))}
      </ol>
      {overview && (
        <div className="rounded-lg border border-border p-4 space-y-2" role="status">
          <p className="font-semibold">当前闭环状态</p>
          <div className="flex flex-wrap gap-2">
            <span className={chip}>基准题 {overview.benchmark.case_count} 道，已标注维度 {overview.benchmark.annotated_case_count} 道</span>
            <span className={chip}>待复核基准坏例 {b?.open_benchmark_count ?? 0} 条</span>
            <span className={chip}>已回流坏例 {b?.flowed_count ?? 0} 条</span>
            <span className={chip}>
              开发集 {d ? d.seed_case_count + d.backflow_case_count : "…"} 题
              {d?.splits ? `（训练 ${d.splits.train ?? 0} / 验证 ${d.splits.validation ?? 0} / 留出 ${d.splits.holdout ?? 0}）` : ""}
            </span>
            <span className={chip}>仅存证复核 {d?.review_record_count ?? 0} 条</span>
            {gateBadge(e?.last_holdout_gate_passed ?? null, "尚无 GEPA 留出实验")}
            {gateBadge(e?.last_report_gate_passed ?? null, "尚无报告 A/B 门禁")}
            <span className={chip}>发布记录 {e?.release_count ?? 0} 份</span>
          </div>
          {d?.error && <p role="alert" className="text-sm">开发集合并失败：{d.error}</p>}
          {annotations && (
            <p className="text-xs text-text-muted">维度标注口径：{annotations.note}</p>
          )}
        </div>
      )}
      {!overview && <p className="text-sm text-text-muted">正在读取闭环状态…</p>}
    </section>
  );
}

function aspectListForUI(raw: string): string[] {
  return Array.from(new Set(raw.split("\n").map((v) => v.trim()).filter(Boolean)));
}

type BackflowPrefill = {
  query: string;
  depth: string;
  aspects: string[];
  caseId: string;
  group: string;
};

function BackflowPanel({ onChanged, prefill }: { onChanged: () => void; prefill: BackflowPrefill | null }) {
  const [cases, setCases] = useState<EvolutionBadCase[]>([]);
  const [annotations, setAnnotations] = useState<BenchmarkAnnotations | null>(null);
  const [selected, setSelected] = useState<EvolutionBadCase | null>(null);
  const [manualMode, setManualMode] = useState(prefill !== null);
  const [caseId, setCaseId] = useState(prefill?.caseId ?? "");
  const [group, setGroup] = useState(prefill?.group ?? "");
  const [aspects, setAspects] = useState((prefill?.aspects ?? []).join("\n"));
  const [note, setNote] = useState("");
  const [plannerTasks, setPlannerTasks] = useState("");
  const [evidenceInfo, setEvidenceInfo] = useState("");
  const evidenceRequest = useRef(0);
  const [missingAspects, setMissingAspects] = useState<string[]>([]);
  const [traceRef, setTraceRef] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [stage, setStage] = useState<BackflowRequest["stage"]>("planner");
  const [manualQuery, setManualQuery] = useState(prefill?.query ?? "");
  const manualDepth = prefill?.depth ?? "balanced";
  const [error, setError] = useState("");
  const [done, setDone] = useState("");
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let alive = true;
    evolutionBadCases()
      .then((r) => { if (alive) { setCases(r.bad_cases); setLoaded(true); } })
      .catch((e) => { if (alive) { setError(evolutionError(e)); setLoaded(true); } });
    evolutionAnnotations()
      .then((r) => { if (alive) setAnnotations(r); })
      .catch(() => undefined);
    return () => { alive = false; };
  }, []);

  function pick(row: EvolutionBadCase | null) {
    const requestNumber = ++evidenceRequest.current;
    setSelected(row);
    setDone("");
    setError("");
    setPlannerTasks(""); setMissingAspects([]); setTraceRef(""); setConfirmed(false); setEvidenceInfo("");
    if (!row) return;
    setCaseId(`planner-gap-${row.case_id.slice(0, 24).replace(/[^a-zA-Z0-9._-]/g, "-").toLowerCase()}`);
    const benchmarkGroup = annotations?.aspects[row.case_id] ? row.case_id.replace(/[^a-zA-Z0-9._-]/g, "-").toLowerCase() : "";
    setGroup(benchmarkGroup || "report-benchmark-gap");
    setAspects((row.suggested_aspects ?? []).join("\n"));
    setTraceRef("");
    setEvidenceInfo("正在从本次评测归档读取 Planner 原始任务…");
    // 读取接口校验账号、坏例、运行身份与归档指纹；这里不自动勾选漏项或确认框。
    void evolutionPlannerEvidence(row.bad_case_id).then((evidence) => {
      if (requestNumber !== evidenceRequest.current) return;
      if (evidence.status !== "recorded") {
        setEvidenceInfo("该运行尚无可验证的 Planner 任务归档。旧评测需重新运行，不能凭空补造。");
        return;
      }
      setPlannerTasks(evidence.tasks.map((t) => t.query).join("\n"));
      setTraceRef(evidence.evidence_reference ?? "");
      setEvidenceInfo(`已取得真实任务 ${evidence.tasks.length} 条；仍需逐项人工判断漏项。`);
    }).catch(() => {
      if (requestNumber === evidenceRequest.current) {
        setEvidenceInfo("任务归档读取失败，请检查当前评测 Run 与账号权限。");
      }
    });
  }

  async function submit() {
    if (busy) return;
    setBusy(true); setError(""); setDone("");
    const aspectList = aspects.split("\n").map((x) => x.trim()).filter(Boolean);
    const evidence = stage === "planner" ? {
      planner_tasks: plannerTasks.split("\n").map((v) => v.trim()).filter(Boolean),
      missing_aspects: missingAspects,
      evidence_reference: traceRef.trim(),
      review_confirmed: confirmed,
    } : {};
    const body: BackflowRequest = manualMode
      ? { case_id: caseId.trim(), group: group.trim(), aspects: aspectList, review_note: note, stage, manual_query: manualQuery, manual_depth: manualDepth as BackflowRequest["manual_depth"], ...evidence }
      : { case_id: caseId.trim(), group: group.trim(), aspects: aspectList, review_note: note, stage, bad_case_id: selected?.bad_case_id, ...evidence };
    try {
      const record = await evolutionBackflow(body);
      setDone(stage === "planner"
        ? `已进入开发集：${record.case.case_id}（训练集，${record.case.required_aspects.length} 个必答维度）`
        : `已存证：${record.case.case_id}（${stageLabel[record.review.stage]}）`);
      onChanged();
    } catch (e) {
      setError(evolutionError(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-4" aria-label="坏例回流">
      <p className="text-sm text-text-muted">
        选择评测中心 report-benchmark 的坏例，对照必答维度标注与执行轨迹确认问题环节后回流。
        只有提供真实 Planner 任务列表、具体漏项及证据编号，并完成手动复核，才允许进入训练集；检索或报告问题仅存证。
        系统不会伪造 Planner 任务，最终优化评测仍需你手动执行。
      </p>
      <div className="rounded-lg border border-border p-4 space-y-2">
        <p className="font-semibold">基准坏例（report-benchmark）</p>
        {!loaded && <p className="text-sm text-text-muted">正在读取坏例…</p>}
        {loaded && cases.length === 0 && <p className="text-sm">当前账号没有待复核的基准坏例。先到评测中心用真实模型跑一轮报告评测。</p>}
        <ul className="space-y-1">
          {cases.map((row) => (
            <li key={row.bad_case_id}>
              <button
                type="button"
                className={`${button} w-full text-left ${selected?.bad_case_id === row.bad_case_id ? "text-accent bg-bg-elevated font-semibold" : "text-text-secondary"}`}
                onClick={() => pick(row)}
              >
                <span className="font-medium">{row.case_id}</span>
                <span className="text-xs text-text-secondary"> · {row.error_key} · {row.review_status === "new" ? "未复核" : row.review_status}</span>
                <span className="block text-xs text-text-muted">Run {row.last_run_id}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
      <div className="rounded-lg border border-border p-4 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <p className="font-semibold">回流标注</p>
          <label className="text-xs flex items-center gap-1">
            <input type="checkbox" checked={manualMode} onChange={(e) => { setManualMode(e.target.checked); pick(null); }} />
            手动回流（复核已有研究记录）
          </label>
        </div>
        {manualMode && (
          <label className="block text-sm">研究问题<textarea aria-label="研究问题" className={input} rows={2} value={manualQuery} onChange={(e) => setManualQuery(e.target.value)} placeholder="复核过的研究问题原文" /></label>
        )}
        {!manualMode && !selected && <p className="text-sm text-text-muted">先选择一条坏例。</p>}
        <label className="block text-sm">案例 ID<input aria-label="案例 ID" className={input} value={caseId} onChange={(e) => setCaseId(e.target.value)} placeholder="如 planner-gap-01" /></label>
        <label className="block text-sm">分组（同组题目会留在同一 split）<input aria-label="分组" className={input} value={group} onChange={(e) => setGroup(e.target.value)} placeholder="如 model-routing" /></label>
        <label className="block text-sm">必答维度（每行一条；已按基准标注预填，请对照轨迹确认或修改）<textarea aria-label="必答维度" className={input} rows={4} value={aspects} onChange={(e) => { const next = e.target.value; setAspects(next); setMissingAspects((prev) => prev.filter((x) => aspectListForUI(next).includes(x))); }} placeholder={"参考解定义\n测试硬件\n时间预算"} /></label>
        {selected && annotations?.aspects[selected.case_id] && (
          <p className="text-xs text-text-muted">标注依据：{annotations.aspects[selected.case_id].rationale}</p>
        )}
        <label className="block text-sm">归因阶段<select aria-label="归因阶段" className={input} value={stage} onChange={(e) => setStage(e.target.value as BackflowRequest["stage"])}>
          {Object.entries(stageLabel).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select></label>
        <label className="block text-sm">复核结论（必填；引用 Run ID 与轨迹证据）<textarea aria-label="复核结论" className={input} rows={3} value={note} onChange={(e) => setNote(e.target.value)} placeholder="例如：Run pipe-1--report-benchmark 的任务列表只覆盖解质量与参考解，未安排硬件与时间预算的检索。" /></label>
        {stage === "planner" && (
          <fieldset className="rounded-lg border border-border p-3 space-y-3">
            <legend className="font-semibold text-sm">Planner 漏项证据（必填）</legend>
            <p className="text-xs text-text-muted">在评测中心打开原 Run 的 Planner agent_log，复制真实 tasks；只勾选确实没有对应检索任务的必答维度。这里存储的是人工复核证据，并非系统自动验证的 Trace。</p>
            {evidenceInfo && <p role="status" className="text-xs text-text-muted">{evidenceInfo}</p>}
            <label className="block text-sm">实际 Planner 任务（每行一个，原样复制）
              <textarea aria-label="实际 Planner 任务" className={input} rows={5} value={plannerTasks} readOnly={traceRef.startsWith("auto:")} onChange={(e) => setPlannerTasks(e.target.value)} placeholder={"检索方法 A 的测试实例和规模\n核对方法 B 的参考解定义"} />
            </label>
            <div className="space-y-1" role="group" aria-label="未覆盖的必答维度">
              <p className="text-sm">未被上述任务覆盖的维度（至少一项）</p>
              {aspectListForUI(aspects).map((aspect) => (
                <label key={aspect} className="flex items-center gap-2 text-sm">
                  <input type="checkbox" aria-label={`漏项：${aspect}`} checked={missingAspects.includes(aspect)} onChange={(e) => setMissingAspects((v) => e.target.checked ? [...v, aspect] : v.filter((x) => x !== aspect))} />
                  {aspect}
                </label>
              ))}
            </div>
            <label className="block text-sm">Planner Trace 证据编号（Run ID / 事件号 / 文件路径）
              <input aria-label="Planner Trace 证据编号" className={input} value={traceRef} onChange={(e) => setTraceRef(e.target.value)} placeholder="如 pipe-123/planner.agent_log#7；坏例必须包含其 Run ID" />
            </label>
            <label className="flex items-start gap-2 text-sm">
              <input type="checkbox" aria-label="已核对 Planner 任务证据" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />
              <span>已逐项核对真实 Planner 任务与必答维度，确认不是检索或报告阶段漏项。</span>
            </label>
            <Link href="/evaluation" className="text-xs underline">前往评测中心核对任务 Trace →</Link>
          </fieldset>
        )}
        <div className="flex flex-wrap gap-2">
          <button className={button} disabled={busy || (!manualMode && !selected) || (stage === "planner" && (!confirmed || !traceRef.trim() || !plannerTasks.trim() || missingAspects.length === 0))} onClick={() => void submit()}>提交回流</button>
        </div>
        {error && <p role="alert" className="text-sm">{error}</p>}
        {done && <p role="status" className="text-sm">{done}</p>}
      </div>
    </section>
  );
}

function PlannerCaseReviewPanel({ onChanged }: { onChanged: () => void }) {
  const [cases, setCases] = useState<ReviewableEvaluationCase[]>([]);
  const [selected, setSelected] = useState<ReviewableEvaluationCase | null>(null);
  const [evidence, setEvidence] = useState<Awaited<ReturnType<typeof evolutionReviewableCaseEvidence>> | null>(null);
  const [missing, setMissing] = useState<string[]>([]);
  const [judgeConsent, setJudgeConsent] = useState(false);
  const [judgeBusy, setJudgeBusy] = useState(false);
  const [judgeSuggestion, setJudgeSuggestion] = useState<PlannerJudgeSuggestion | null>(null);
  const [judgeError, setJudgeError] = useState("");
  const [note, setNote] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [trainGroups, setTrainGroups] = useState<string[]>([]);
  const [trainGroup, setTrainGroup] = useState("");
  const requestSeq = useRef(0);

  useEffect(() => {
    let active = true;
    void evolutionReviewableCases().then((r) => { if (active) setCases(r.cases); })
      .catch((e) => { if (active) setError(evolutionError(e)); })
      .finally(() => { if (active) setLoaded(true); });
    void evolutionDataset().then((data) => {
      if (active) setTrainGroups(Array.from(new Set((data.dataset?.cases ?? [])
        .filter((c) => c.split === "train" && !c.source_bad_case_id)
        .map((c) => c.group))).sort());
    }).catch(() => undefined);
    return () => { active = false; };
  }, []);

  function pick(row: ReviewableEvaluationCase) {
    const current = ++requestSeq.current;
    setSelected(row); setEvidence(null); setMissing([]); setTrainGroup(""); setNote(""); setConfirmed(false);
    setJudgeConsent(false); setJudgeSuggestion(null); setJudgeError(""); setJudgeBusy(false);
    setError(""); setDone("");
    void evolutionReviewableCaseEvidence(row.run_id, row.case_id)
      .then((ev) => { if (current === requestSeq.current) setEvidence(ev); })
      .catch((e) => { if (current === requestSeq.current) setError(evolutionError(e)); });
  }

  async function analyzeWithJudge() {
    if (!selected || evidence?.status !== "recorded" || !evidence.evidence_reference || !judgeConsent || judgeBusy) return;
    const current = requestSeq.current;
    setJudgeBusy(true); setJudgeError(""); setJudgeSuggestion(null);
    try {
      const result = await evolutionPlannerJudge(selected.run_id, selected.case_id, evidence.evidence_reference);
      if (current === requestSeq.current) setJudgeSuggestion(result);
    } catch (e) {
      if (current === requestSeq.current) setJudgeError(evolutionError(e));
    } finally {
      if (current === requestSeq.current) setJudgeBusy(false);
    }
  }

  async function submit() {
    if (!selected || evidence?.status !== "recorded" || !evidence.evidence_reference || !confirmed || note.trim().length < 2 || (missing.length > 0 && !trainGroup) || busy) return;
    setBusy(true); setError(""); setDone("");
    const stage = missing.length ? "planner" : "planner_complete";
    try {
      await evolutionBackflow({
        case_id: `planner-review-${selected.case_id.slice(0, 28).replace(/[^a-zA-Z0-9._-]/g, "-")}-${selected.run_id.slice(5, 13)}`,
        group: missing.length ? trainGroup : "planner-complete-reviews",
        aspects: selected.required_aspects,
        review_note: note.trim(), stage,
        evaluation_run_id: selected.run_id,
        evaluation_case_id: selected.case_id,
        planner_tasks: evidence.tasks.map((task) => task.query),
        missing_aspects: missing,
        evidence_reference: evidence.evidence_reference,
        review_confirmed: true,
      });
      setDone(missing.length ? "已确认规划漏项，进入开发集（请检查数据集分组约束）。" : "已标注规划完整，仅存审核记录，不进入 GEPA 开发集。");
      onChanged();
    } catch (e) { setError(evolutionError(e)); }
    finally { setBusy(false); }
  }

  return (
    <section className="space-y-4" aria-label="Planner 逐题复核">
      <p className="text-sm text-text-muted">报告通过也可以检查 Planner 规划是否完整。只展示当前账号的真实已测案例，只有确认遗漏才进入开发集。不会启动模型或改变报告评分。</p>
      <div className="rounded-lg border border-border p-4 space-y-2">
        <p className="font-semibold">最近的真实报告评测案例</p>
        {!loaded && <p className="text-sm text-text-muted">正在读取案例…</p>}
        {loaded && cases.length === 0 && <p className="text-sm">没有可复核的已测案例。</p>}
        {cases.map((row) => (
          <button key={`${row.run_id}:${row.case_id}`} className={`${button} block w-full text-left`} type="button" onClick={() => pick(row)}>
            <span className="font-semibold">{row.case_id}</span> · {row.result_status === "passed" ? "报告通过" : row.result_status}
            <span className="block text-xs text-text-muted">Run {row.run_id}</span>
          </button>
        ))}
      </div>
      {selected && (
        <div className="rounded-lg border border-border p-4 space-y-3">
          <p className="font-semibold">当前复核：{selected.case_id}</p>
          <p className="text-sm">{selected.query}</p>
          {!evidence && <p role="status" className="text-sm">正在读取已归档任务…</p>}
          {evidence?.status === "unavailable" && <p role="alert" className="text-sm">无可验证的 Planner 任务归档；不能提交复核。</p>}
          {evidence?.status === "recorded" && (
            <>
              <p className="text-sm">已取证 {evidence.tasks.length} 条任务；请核对哪些必答维度没有对应的检索任务。</p>
              <ol className="list-decimal pl-5 space-y-1 text-sm">{evidence.tasks.map((task) => <li key={task.id}>{task.query}</li>)}</ol>
              <div className="rounded-lg border border-border p-3 space-y-2" aria-label="Planner Judge 辅助建议">
                <p className="font-semibold text-sm">独立 Judge 辅助分析（可选）</p>
                <p className="text-xs text-text-muted">只分析这次已归档的原始任务，不重新运行 Planner。一次点击可能产生真实模型费用。Judge 的建议不会自动勾选漏项或写入训练集。</p>
                <label className="flex items-start gap-2 text-sm">
                  <input type="checkbox" aria-label="同意 Judge API 费用" checked={judgeConsent} onChange={(e) => setJudgeConsent(e.target.checked)} />
                  我确认本次独立 Judge 调用可能产生 API 费用。
                </label>
                <button type="button" className={button} disabled={!judgeConsent || judgeBusy || busy || !!judgeSuggestion} onClick={() => void analyzeWithJudge()}>
                  {judgeBusy ? "Judge 分析中…" : "一键分析疑似漏项"}
                </button>
                {judgeError && <p role="alert" className="text-sm">{judgeError}</p>}
                {judgeSuggestion && (
                  <div className="space-y-2" role="region" aria-label="Judge 逐维度建议">
                    <p className="text-sm">模型建议覆盖率 {(judgeSuggestion.coverage * 100).toFixed(1)}%；Judge：{judgeSuggestion.served_judge ?? judgeSuggestion.judge_route}。这是辅助意见，请自行逐项确认。</p>
                    {judgeSuggestion.aspects.map((item) => (
                      <div key={item.aspect_index} className="rounded-md border border-border p-2 text-sm">
                        <p className="font-semibold">{item.covered ? "建议已覆盖" : "建议疑似漏项"}：{item.aspect}</p>
                        <p>对应任务：{item.task_ids.length ? item.task_ids.join("、") : "无"}</p>
                        <p className="text-text-muted">理由：{item.reason}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <fieldset className="rounded-lg border border-border p-3 space-y-2">
                <legend className="text-sm font-semibold">逐项确认未覆盖维度（全部未勾选表示规划完整）</legend>
                {selected.required_aspects.map((aspect) => (
                  <label key={aspect} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={missing.includes(aspect)} onChange={(e) => setMissing((prev) => e.target.checked ? [...prev, aspect] : prev.filter((x) => x !== aspect))} />
                    {aspect}
                  </label>
                ))}
              </fieldset>
              {missing.length > 0 && (
                <label className="block text-sm">选择已有开发集主题组（防止验证/留出集泄漏）
                  <select className={input} aria-label="开发集主题组" value={trainGroup} onChange={(e) => setTrainGroup(e.target.value)}>
                    <option value="">请选择与研究主题相同的开发集组</option>
                    {trainGroups.map((group) => <option key={group} value={group}>{group}</option>)}
                  </select>
                </label>
              )}
              <label className="block text-sm">人工复核理由<textarea className={input} aria-label="逐题复核理由" rows={3} value={note} onChange={(e) => setNote(e.target.value)} placeholder="说明各遗漏维度与任务证据的关系；如果规划完整，请解释判断依据。" /></label>
              <label className="flex items-start gap-2 text-sm"><input type="checkbox" aria-label="确认逐项复核" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />我已对照原始任务逐项复核。{missing.length ? "确认存在 Planner 漏项" : "确认 Planner 规划完整"}。</label>
              <button type="button" className={button} disabled={busy || !confirmed || note.trim().length < 2 || (missing.length > 0 && !trainGroup)} onClick={() => void submit()}>{missing.length ? "确认漏项并回流开发集" : "记录规划完整（不回流训练）"}</button>
            </>
          )}
          {error && <p role="alert" className="text-sm">{error}</p>}
          {done && <p role="status" className="text-sm">{done}</p>}
        </div>
      )}
      {!selected && error && <p role="alert" className="text-sm">{error}</p>}
    </section>
  );
}

function DatasetPanel({
  state,
  revision,
  onReload,
  registry,
  benchmark,
  onReflow,
}: {
  state: EvolutionDatasetState | null;
  revision: number;
  onReload: () => void;
  registry: DatasetEntry[];
  benchmark: BenchmarkCaseSet | null;
  onReflow: (prefill: BackflowPrefill) => void;
}) {
  const [filter, setFilter] = useState<string>("all");
  const [registryOpen, setRegistryOpen] = useState(false);
  const cases = state?.dataset?.cases;
  const visible = cases?.filter((c) => filter === "all" || c.split === filter) ?? [];
  const reflowCase = (row: BenchmarkCase) => {
    const aspects = row.required_aspects ?? [];
    onReflow({
      query: row.query,
      depth: row.depth,
      aspects,
      caseId: `planner-gap-${row.case_id.slice(0, 24).replace(/[^a-zA-Z0-9._-]/g, "-").toLowerCase()}`,
      group: row.case_id.replace(/[^a-zA-Z0-9._-]/g, "-").toLowerCase(),
    });
  };
  return (
    <section className="space-y-4" aria-label="开发数据集">
      <p className="text-sm text-text-muted">
        先在此查看评测中心当前数据集，从 report-benchmark 选题并预填回流；回流后的训练集与下方自进化开发集合并，交给 GEPA 优化。
        GEPA 只接触训练与验证 split；留出集在候选冻结后独立评分。
      </p>
      <div className="rounded-lg border border-border p-4 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="font-semibold">评测中心数据集注册表（{registry.length || "…"} 套，只读）</p>
          <button className={button} onClick={() => setRegistryOpen((v) => !v)}>
            {registryOpen ? "收起" : "展开数据集"}
          </button>
        </div>
        <p className="text-xs text-text-muted">与评测中心「数据集」页同源；数量来源与执行方式各不相同，不可合成为总准确率。</p>
        {registryOpen && (
          <ul className="space-y-1">
            {registry.map((row) => (
              <li key={row.id} className="flex flex-wrap items-center gap-2 rounded-lg border border-border p-2 text-sm">
                <span className="font-medium">{row.title}</span>
                <span className={chip}>{row.id}</span>
                <span className={chip}>{row.declared_count} 题</span>
                <span className={chip}>{row.count_basis === "source_verified" ? "源已核对" : row.count_basis === "registry_declared" ? "清单声明" : "仅上报"}</span>
                {row.id === "report-benchmark" && <span className="chip text-accent">可选题回流</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="rounded-lg border border-border p-4 space-y-2">
        <p className="font-semibold">从评测中心题目预填回流</p>
        <p className="text-xs text-text-muted">
          report-benchmark 的题目与人工维度标注来自评测中心；选择后跳到「坏例回流」并预填问题、分组与必答维度，复核结论仍需你对照轨迹填写。
        </p>
        {!benchmark && <p className="text-sm text-text-muted">正在读取基准题目…</p>}
        <ul className="space-y-1">
          {(benchmark?.cases ?? []).map((row) => (
            <li key={row.case_id} className="flex flex-wrap items-center gap-2 rounded-lg border border-border p-2 text-sm">
              <span className="font-medium">{row.case_id}</span>
              <span className="text-xs text-text-muted">{row.query}</span>
              <button className={button} onClick={() => reflowCase(row)}>预填回流</button>
            </li>
          ))}
        </ul>
      </div>
      {state?.error && <p role="alert" className="text-sm">合并失败：{state.error}（请在坏例回流中修正分组）</p>}
      {state && (
        <div className="flex flex-wrap gap-2" role="status">
          <span className={chip}>种子 {state.seed_case_count} 题</span>
          <span className={chip}>回流训练例 {state.backflow_case_count} 题</span>
          <span className={chip}>仅存证复核 {state.review_record_count} 条</span>
          {(state.unreadable_record_count ?? 0) > 0 && <span className={chip}>不可读记录 {state.unreadable_record_count}</span>}
          {state.dataset_sha256 && <span className={chip}>数据指纹 {state.dataset_sha256.slice(0, 12)}…</span>}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <select aria-label="按 split 筛选" className={`${input} max-w-48`} value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">全部题目</option>
          {Object.entries(splitLabel).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
        <button className={button} onClick={onReload}>重新加载</button>
        <span className="text-xs text-text-muted">共 {visible.length} 题（修订 {revision}）</span>
      </div>
      {!state && <p className="text-sm text-text-muted">正在读取开发集…</p>}
      <ul className="space-y-2">
        {visible.map((c) => (
          <li key={c.case_id} className="rounded-lg border border-border p-3 space-y-1">
            <p className="text-sm">
              <span className="font-medium">{c.case_id}</span>
              <span className={chip + " ml-2"}>{splitLabel[c.split]}</span>
              <span className={chip + " ml-1"}>{c.depth}</span>
              <span className={chip + " ml-1"}>{c.group}</span>
              {c.source_bad_case_id && <span className={chip + " ml-1"}>来自坏例 {c.source_bad_case_id.slice(0, 8)}…</span>}
            </p>
            <p className="text-sm text-text-muted">{c.query}</p>
            <div className="flex flex-wrap gap-1">{c.required_aspects.map((a) => <span key={a} className={chip}>{a}</span>)}</div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function ExperimentsPanel({ rows, loaded }: { rows: EvolutionExperimentRow[]; loaded: boolean }) {
  return (
    <section className="space-y-4" aria-label="实验记录">
      <p className="text-sm text-text-muted">
        本页扫描评测卷 planner-evolution/experiments/ 下的实验目录，只读取摘要投影，不展示逐题观测正文。
        把 CLI 的 --out-dir 指到该目录即可在这里看到门禁结论与发布记录。
      </p>
      {!loaded && <p className="text-sm text-text-muted">正在读取实验证据…</p>}
      {loaded && rows.length === 0 && <p className="text-sm">尚无实验。完成一次 <code>optimize --demo</code>（免密钥验证链路）后即可在此查看。</p>}
      <ul className="space-y-3">
        {rows.map((row) => (
          <li key={row.name} className="rounded-lg border border-border p-4 space-y-2">
            <p className="font-semibold">{row.name}</p>
            {row.experiment && (
              <div className="space-y-1 text-sm">
                <div className="flex flex-wrap gap-2">
                  {gateBadge(row.experiment.holdout_gate?.passed ?? null, "无留出门禁")}
                  <span className={chip}>{row.experiment.evidence_kind === "live_planner" ? "真实 Planner" : "脚本演示"}</span>
                  <span className={chip}>{row.experiment.optimizer?.name} {row.experiment.optimizer?.version}</span>
                  <span className={chip}>{row.experiment.case_count} 题</span>
                  {row.experiment.has_prompt_diff && <span className={chip}>含 Prompt 差异</span>}
                </div>
                <p className="text-xs text-text-muted">{row.experiment.created_at} · 指标 {row.experiment.metric_version}</p>
                {typeof row.experiment.holdout_gate?.delta === "number" && (
                  <p className="text-sm">
                    留出集覆盖率 {((row.experiment.holdout_gate.baseline_coverage ?? 0) * 100).toFixed(2)}% → {((row.experiment.holdout_gate.candidate_coverage ?? 0) * 100).toFixed(2)}%
                    （提升 {(row.experiment.holdout_gate.delta * 100).toFixed(2)} 个百分点）
                  </p>
                )}
                {row.experiment.holdout_gate?.reasons?.map((r) => <p key={r} className="text-xs">原因：{r}</p>)}
              </div>
            )}
            {row.report_gate && (
              <div className="space-y-1 text-sm">
                <p className="font-medium">报告 A/B 门禁：{row.report_gate.passed ? "通过" : "未通过"}</p>
                <p className="text-xs text-text-muted">{row.report_gate.cases} 题 · Judge {row.report_gate.served_judge ?? "未披露"}</p>
              </div>
            )}
            {row.release && (
              <div className="space-y-1 text-sm">
                <p className="font-medium">发布记录 {row.release.version}</p>
                <p className="text-xs text-text-muted">{row.release.approved_by} 审阅于 {row.release.approved_at}</p>
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

function CommandsPanel() {
  const commands: [string, string][] = [
    ["导出活跃数据集（与本页同源）", "python -m evals.agent_eval_ext.planner_evolution dataset --cases-dir <评测卷>\\planner-evolution\\cases --out .\\planner-active.json"],
    ["链路验证（免密钥，零模型调用）", "python -m evals.agent_eval_ext.planner_evolution optimize --demo --out-dir <评测卷>\\planner-evolution\\experiments\\planner-demo-01"],
  ];
  return (
    <section className="space-y-3" aria-label="运行命令">
      <p className="text-sm text-text-muted">
        命令在 backend 目录执行，Python 3.14 环境。&lt;评测卷&gt; 是服务端 RF_EVAL_JOB_ROOT 配置的评测目录；把 --out-dir 指到它的 planner-evolution/experiments/ 下，本页即可读取证据。
        当前 v3 搜索请在“GEPA 优化实验”分区提交；任务会冻结配置并建立请求账本，真实启动仍需你确认费用。以下仅保留数据导出和旧版免费链路命令。
      </p>
      <ul className="space-y-2">
        {commands.map(([label, command]) => (
          <li key={label} className="rounded-lg border border-border p-3 space-y-1">
            <p className="text-sm font-medium">{label}</p>
            <pre className="overflow-x-auto text-xs"><code>{command}</code></pre>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function EvolutionWorkspace() {
  const [tab, setTab] = useState("flow");
  const [overview, setOverview] = useState<EvolutionOverview | null>(null);
  const [annotations, setAnnotations] = useState<BenchmarkAnnotations | null>(null);
  const [datasetState, setDatasetState] = useState<EvolutionDatasetState | null>(null);
  const [experiments, setExperiments] = useState<EvolutionExperimentRow[]>([]);
  const [experimentsLoaded, setExperimentsLoaded] = useState(false);
  const [datasetRevision, setDatasetRevision] = useState(0);
  const [registry, setRegistry] = useState<DatasetEntry[]>([]);
  const [benchmark, setBenchmark] = useState<BenchmarkCaseSet | null>(null);
  const [prefill, setPrefill] = useState<BackflowPrefill | null>(null);
  const [error, setError] = useState("");

  const reloadAll = useCallback(() => {
    let alive = true;
    evolutionOverview().then((r) => { if (alive) { setOverview(r); setError(""); } }).catch((e) => { if (alive) setError(evolutionError(e)); });
    evolutionDataset().then((r) => { if (alive) setDatasetState(r); }).catch(() => undefined);
    evolutionAnnotations().then((r) => { if (alive) setAnnotations(r); }).catch(() => undefined);
    evolutionBenchmarkCases().then((r) => { if (alive) setBenchmark(r); }).catch(() => undefined);
    evaluationDatasets().then((r) => { if (alive) setRegistry(r.datasets); }).catch(() => undefined);
    evolutionExperiments().then((r) => { if (alive) { setExperiments(r); setExperimentsLoaded(true); } }).catch(() => { if (alive) setExperimentsLoaded(true); });
    return () => { alive = false; };
  }, []);

  useEffect(() => reloadAll(), [reloadAll]);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="font-serif text-3xl font-bold">自进化</h1>
          <p className="mt-2 text-sm text-text-muted">评测中心 → 坏例复核回流 → GEPA 优化 → 留出集与报告 A/B → 人工发布 → 下轮评测复测</p>
        </div>
        <Link className={button} href="/evaluation">去评测中心运行基线</Link>
      </header>
      {error && <p role="alert" className="text-sm">{error}</p>}
      <nav className="flex flex-wrap gap-2" aria-label="自进化分区">
        {tabs.map(([value, label]) => (
          <button
            key={value}
            type="button"
            className={`${button} ${tab === value ? "text-accent bg-bg-elevated font-semibold" : "text-text-secondary"}`}
            aria-current={tab === value ? "page" : undefined}
            onClick={() => setTab(value)}
          >
            {label}
          </button>
        ))}
      </nav>
      {tab === "flow" && <FlowOverview overview={overview} annotations={annotations} />}
      {tab === "review" && <PlannerCaseReviewPanel onChanged={reloadAll} />}
      {tab === "backflow" && <BackflowPanel key={prefill?.caseId ?? "blank"} onChanged={reloadAll} prefill={prefill} />}
      {tab === "dataset" && (
        <DatasetPanel
          state={datasetState}
          revision={datasetRevision}
          onReload={() => { setDatasetRevision((x) => x + 1); reloadAll(); }}
          registry={registry}
          benchmark={benchmark}
          onReflow={(value) => { setPrefill(value); setTab("backflow"); }}
        />
      )}
      {tab === "gepa" && <GEPAJobsPanel />}
      {tab === "experiments" && (
        <div className="space-y-8">
          <ExperimentsPanel rows={experiments} loaded={experimentsLoaded} />
          <CommandsPanel />
        </div>
      )}
    </div>
  );
}
