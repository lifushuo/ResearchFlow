"use client";

import UnifiedEvaluationRunnerV13 from "@/components/evaluation/UnifiedEvaluationRunnerV13";
import type { Pipeline } from "@/lib/evaluation-pipeline-v13";
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  evaluationDataset, evaluationDatasets,
  type DatasetDetail, type DatasetEntry,
} from "@/lib/evaluation-datasets";
import { evaluationErrorMessage } from "@/lib/evaluation";
import { getDatasetLabel, getDatasetShort, getDatasetDescription, getDatasetMetrics, getDatasetEnglish } from "@/lib/evaluation-dataset-zh";
import { evaluationLabel, evaluationNote } from "@/lib/evaluation-display-zh";

const pageSize = 50;

// 原始字段值继续驱动判断与来源保真，这里只做展示层本地化；未知值回显原文。
const originLabels: Record<string, string> = { upstream: "原项目", custom: "自建" };
const categoryLabels: Record<string, string> = {
  outcome: "报告质量", retrieval: "检索", grounding: "引用支撑", judge: "评分模型校准",
  reliability: "回放可靠性", executor: "执行器", tool_contract: "工具契约",
};
const splitLabels: Record<string, string> = {
  benchmark: "基准测试", regression: "回归", dev_holdout: "开发/留出集", fixture: "合成夹具",
};
const executionLabels: Record<string, string> = {
  optional_paid: "可选付费运行", offline: "离线", controlled_free: "受控免费",
  controlled_fixture: "受控夹具", archived_only: "仅归档",
};
const statusLabels: Record<string, string> = {
  available: "源文件可用", unlinked: "未关联", count_mismatch: "数量不一致",
  invalid_or_missing: "源文件缺失或无效", incomplete: "支撑文件不完整", unavailable: "不可用",
};
const countBasisLabels: Record<string, string> = {
  source_verified: "源文件核对", registry_declared: "清单声明",
  reported_only: "仅报告记录", unmeasured: "未测量",
};

function label(map: Record<string, string>, value: string): string {
  return map[value] ?? value.replaceAll("_", " ");
}

function Pill({ children }: { children: ReactNode }) {
  return <span className="rounded-full border border-border px-2 py-0.5 text-xs text-text-muted">{children}</span>;
}

function Status({ dataset }: { dataset: DatasetEntry }) {
  if (dataset.source_status === "available") {
    return <span className="text-emerald-700 dark:text-emerald-400">源文件可用</span>;
  }
  return <span className="text-amber-700 dark:text-amber-400">{label(statusLabels, dataset.source_status)}</span>;
}

export default function DatasetRegistryPanel({ onCreated }: { onCreated?: (pipeline: Pipeline) => void }) {
  const sequence = useRef(0);
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    evaluationDatasets().then(({ datasets: entries }) => {
      if (active) setDatasets(entries);
    }).catch((err: unknown) => {
      if (active) setError(evaluationErrorMessage(err, "无法读取数据集清单"));
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  const choose = (id: string, offset = 0) => {
    const request = ++sequence.current;
    setSelected(id);
    setDetail(null);
    setDetailError(null);
    if (!id) return;
    setDetailLoading(true);
    evaluationDataset(id, offset, pageSize).then(({ dataset }) => {
      if (request === sequence.current) setDetail(dataset);
    }).catch((err: unknown) => {
      if (request === sequence.current) setDetailError(evaluationErrorMessage(err, "无法加载数据集详情"));
    }).finally(() => { if (request === sequence.current) setDetailLoading(false); });
  };

  const selectedRow = datasets.find((d) => d.id === selected);
  return <div className="space-y-5">
    <UnifiedEvaluationRunnerV13 onCreated={onCreated} showResults={!onCreated} />
    <section className="rounded-xl border border-border bg-bg-elevated p-5 space-y-2">
      <h2 className="font-semibold text-text-primary">统一数据集清单</h2>
      <p className="text-sm text-text-muted">原项目与自建评测集的统一登记入口。本页只读取静态登记文件，不运行模型、不移动原始文件，也不把不同测试合并成一个分数。</p>
      <p className="text-xs text-amber-700 dark:text-amber-400">各数据集彼此独立：引用支撑逐条检查结论与证据，工具契约使用合成调用记录，规划器与执行器回归使用受控模型。案例数量不能相加为综合智能体测评题数。</p>
      {loading && <p className="text-sm text-text-muted">正在加载已登记数据集…</p>}
      {error && <p role="alert" className="text-sm text-rose-700">{error}</p>}
      {!loading && !error && <p className="text-xs text-text-muted">已登记 {datasets.length} 个数据集 · 源文件可用时核对案例数与 SHA256 · 只读</p>}
    </section>
    {datasets.length > 0 && <div className="grid gap-3 lg:grid-cols-2">
      {datasets.map((row) => <button type="button" key={row.id} onClick={() => choose(row.id)}
        aria-pressed={selected === row.id}
        className={`text-left rounded-xl border p-4 bg-bg-elevated space-y-2 transition-colors hover:border-accent ${selected === row.id ? "border-accent" : "border-border"}`}>
        <div className="flex flex-wrap items-start justify-between gap-2">
          <strong className="text-sm text-text-primary">{getDatasetLabel(row.id, row.title)}</strong>
          <Pill>{label(originLabels, row.origin)}</Pill>
        </div>
        <p className="text-xs leading-relaxed text-text-secondary">{getDatasetShort(row.id, row.description)}</p>
        <p className="text-xs text-text-muted">{label(categoryLabels, row.category)} · {label(splitLabels, row.split)} · {label(executionLabels, row.execution)}</p>
        <div className="text-xs flex flex-wrap gap-2 items-center">
          <Status dataset={row} />
          <span className="text-text-secondary">
            {row.verified_count === null ? `申报 ${row.declared_count} 条 · 未经源文件核对` : `${row.verified_count} 条案例 · 已核对源文件`}
          </span>
        </div>
      </button>)}
    </div>}
    {selectedRow && <section className="rounded-xl border border-border bg-bg-elevated p-5 space-y-4">
      <div>
        <h3 className="font-semibold text-text-primary">{getDatasetLabel(selectedRow.id, selectedRow.title)}</h3>
        <p className="mt-1 text-sm text-text-secondary">{getDatasetDescription(selectedRow.id, selectedRow.description)}</p>
        <p className="mt-2 text-xs text-text-muted">评分口径：{getDatasetMetrics(selectedRow.id, selectedRow.score_family)}</p>
        <p className="mt-1 text-xs text-text-muted">数量依据：{label(countBasisLabels, selectedRow.count_basis)}；这些是数据集案例，不是已完成的智能体运行。</p>
        <details className="mt-2 text-xs text-text-muted">
          <summary className="cursor-pointer">技术标识（排查时查看）</summary>
          <p className="mt-1 break-all">原始名称：{getDatasetEnglish(selectedRow.id)}；标识：{selectedRow.id}</p>
          <p className="mt-1 break-all">源文件：{selectedRow.source ?? "原始样本未关联"}</p>
          {selectedRow.source_sha256 && <p className="mt-1 break-all">文件指纹（SHA256）：{selectedRow.source_sha256}</p>}
        </details>
        {selectedRow.warnings.map((warning, i) => <p key={i} className="mt-1 text-xs text-amber-700">⚠ {evaluationNote(warning)}</p>)}
      </div>
      {detailLoading && <p className="text-xs text-text-muted">正在读取案例索引…</p>}
      {detailError && <p role="alert" className="text-sm text-rose-700">{detailError}</p>}
      {detail && detail.id === selected && <>
        <div className="overflow-x-auto"><table className="w-full text-left text-sm">
          <thead className="border-b border-border text-xs text-text-muted"><tr>
            <th className="py-2 pr-3">序号</th><th className="py-2 pr-3">案例标识</th>
            <th className="py-2">案例类型 / 问题原文</th>
          </tr></thead>
          <tbody>{detail.cases.map((item) => <tr key={item.case_id} className="border-b border-border/60">
            <td className="py-2 pr-3 text-text-muted">{item.index}</td>
            <td className="py-2 pr-3 font-mono text-xs">{item.case_id}</td>
            <td className="py-2 text-xs text-text-secondary">
              {item.question_preview ?? evaluationLabel(item.kind ?? item.match_mode ?? item.domain ?? "仅元数据；不展示原始提示词或工具参数")}
            </td>
          </tr>)}</tbody>
        </table></div>
        {detail.cases.length === 0 && <p className="text-xs text-text-muted">案例清单不可用，或尚未关联到已核验的源文件。</p>}
        <div className="flex items-center gap-3 text-xs text-text-muted">
          <span>显示第 {detail.cases.length ? detail.offset + 1 : 0}–{detail.offset + detail.cases.length} 条，共 {detail.verified_count ?? "未核验"} 条</span>
          <button type="button" disabled={detailLoading || detail.offset === 0} onClick={() => choose(selected, Math.max(0, detail.offset - pageSize))}
            className="rounded border border-border px-3 py-1 disabled:opacity-40">上一页</button>
          <button type="button" disabled={detailLoading || !detail.has_more} onClick={() => choose(selected, detail.offset + pageSize)}
            className="rounded border border-border px-3 py-1 disabled:opacity-40">下一页</button>
        </div>
      </>}
    </section>}
  </div>;
}
