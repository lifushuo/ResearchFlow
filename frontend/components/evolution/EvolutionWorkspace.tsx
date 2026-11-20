"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  evolutionAnnotations,
  evolutionBackflow,
  evolutionBadCases,
  evolutionBenchmarkCases,
  evolutionDataset,
  evolutionError,
  evolutionExperiments,
  evolutionOverview,
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
  ["dataset", "开发数据集"],
  ["experiments", "实验记录"],
];

const FLOW_STEPS: [string, string, string, string][] = [
  ["① 基线评测", "评测中心 · 真实模型", "在评测中心用 report-benchmark 跑现有版本，得到逐题完成状态与独立 Judge 证据。", "/evaluation"],
  ["② 坏例复核", "对照标注与轨迹", "坏例排查面板列出失败题；复核人对照必答维度标注与执行轨迹，确认问题在哪个环节。", "/evaluation"],
  ["③ 回流标注", "本页 · 坏例回流", "把 Run ID、必答维度、复核结论存成结构化反馈；归因 Planner 漏项的进入开发集，其余仅存证。", ""],
  ["④ GEPA 优化", "CLI · 真实模型", "用活跃数据集训练/验证 GEPA 搜索，选出候选 Prompt；留出集独立评分，不通过就停在证据上。", ""],
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
        自进化回答两个问题：现有 Planner 漏了什么（基线评测 → 坏例回流），改完 Prompt 换一道新题还有效吗（候选评测 → A/B 验收）。
        评测产物只在本页展示与回流；付费优化与发布始终在 CLI 人工执行。
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
    setSelected(row);
    setDone("");
    setError("");
    if (!row) return;
    setCaseId(`planner-gap-${row.case_id.slice(0, 24).replace(/[^a-zA-Z0-9._-]/g, "-").toLowerCase()}`);
    const benchmarkGroup = annotations?.aspects[row.case_id] ? row.case_id.replace(/[^a-zA-Z0-9._-]/g, "-").toLowerCase() : "";
    setGroup(benchmarkGroup || "report-benchmark-gap");
    setAspects((row.suggested_aspects ?? []).join("\n"));
  }

  async function submit() {
    if (busy) return;
    setBusy(true); setError(""); setDone("");
    const aspectList = aspects.split("\n").map((x) => x.trim()).filter(Boolean);
    const body: BackflowRequest = manualMode
      ? { case_id: caseId.trim(), group: group.trim(), aspects: aspectList, review_note: note, stage, manual_query: manualQuery, manual_depth: manualDepth as BackflowRequest["manual_depth"] }
      : { case_id: caseId.trim(), group: group.trim(), aspects: aspectList, review_note: note, stage, bad_case_id: selected?.bad_case_id };
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
        归因 Planner 漏项的题目进入训练集；检索或报告环节的问题仅存证，不进入优化数据集。
        评测轨迹默认只含元数据，Planner 实际任务列表不会被补造。
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
        <label className="block text-sm">必答维度（每行一条；已按基准标注预填，请对照轨迹确认或修改）<textarea aria-label="必答维度" className={input} rows={4} value={aspects} onChange={(e) => setAspects(e.target.value)} placeholder={"参考解定义\n测试硬件\n时间预算"} /></label>
        {selected && annotations?.aspects[selected.case_id] && (
          <p className="text-xs text-text-muted">标注依据：{annotations.aspects[selected.case_id].rationale}</p>
        )}
        <label className="block text-sm">归因阶段<select aria-label="归因阶段" className={input} value={stage} onChange={(e) => setStage(e.target.value as BackflowRequest["stage"])}>
          {Object.entries(stageLabel).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select></label>
        <label className="block text-sm">复核结论（必填；引用 Run ID 与轨迹证据）<textarea aria-label="复核结论" className={input} rows={3} value={note} onChange={(e) => setNote(e.target.value)} placeholder="例如：Run pipe-1--report-benchmark 的任务列表只覆盖解质量与参考解，未安排硬件与时间预算的检索。" /></label>
        <div className="flex flex-wrap gap-2">
          <button className={button} disabled={busy || (!manualMode && !selected)} onClick={() => void submit()}>提交回流</button>
        </div>
        {error && <p role="alert" className="text-sm">{error}</p>}
        {done && <p role="status" className="text-sm">{done}</p>}
      </div>
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
    ["链路验证（免密钥，零模型调用）", "python -m evals.agent_eval_ext.planner_evolution optimize --demo --dataset .\\planner-active.json --out-dir <评测卷>\\planner-evolution\\experiments\\planner-demo-01"],
    ["GEPA 优化（真实模型，可能产生费用）", "python -m evals.agent_eval_ext.planner_evolution optimize --live --dataset .\\planner-active.json --out-dir <评测卷>\\planner-evolution\\experiments\\planner-01 --max-evals 80 --proposals 4"],
    ["报告 A/B 回归（真实模型）", "python -m evals.agent_eval_ext.planner_evolution report --live --experiment <评测卷>\\planner-evolution\\experiments\\planner-01\\experiment.json --out-dir <评测卷>\\planner-evolution\\experiments\\planner-01-report"],
    ["导出发布与回滚配置（人工审阅）", "python -m evals.agent_eval_ext.planner_evolution release --experiment <评测卷>\\planner-evolution\\experiments\\planner-01\\experiment.json --report <评测卷>\\planner-evolution\\experiments\\planner-01-report\\report-ab.json --out-dir <评测卷>\\planner-evolution\\experiments\\planner-01-release --approved-by '审阅人'"],
  ];
  return (
    <section className="space-y-3" aria-label="运行命令">
      <p className="text-sm text-text-muted">
        命令在 backend 目录执行，Python 3.14 环境。&lt;评测卷&gt; 是服务端 RF_EVAL_JOB_ROOT 配置的评测目录；把 --out-dir 指到它的 planner-evolution/experiments/ 下，本页即可读取证据。
        优化与报告 A/B 调用真实模型，费用发生在命令行；本页不会自动发起付费调用，也不会自动发布。
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
      {tab === "experiments" && (
        <div className="space-y-8">
          <ExperimentsPanel rows={experiments} loaded={experimentsLoaded} />
          <CommandsPanel />
        </div>
      )}
    </div>
  );
}
