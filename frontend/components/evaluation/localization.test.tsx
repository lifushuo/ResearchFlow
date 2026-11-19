import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DATASET_ZH, getDatasetLabel, getDatasetEnglish, getDatasetMetrics } from "@/lib/evaluation-dataset-zh";
import { evaluationLabel, evaluationNote } from "@/lib/evaluation-display-zh";
import type { Capabilities, Pipeline } from "@/lib/evaluation-pipeline-v13";

import DatasetRegistryPanel from "./DatasetRegistryPanel";
import UnifiedEvaluationRunnerV13 from "./UnifiedEvaluationRunnerV13";
import { EvaluationMetrics } from "./EvaluationMetrics";

const api = vi.hoisted(() => ({
  caps: vi.fn(), list: vi.fn(), detail: vi.fn(), create: vi.fn(), datasets: vi.fn(), dataset: vi.fn(),
}));
vi.mock("@/lib/evaluation-pipeline-v13", () => ({
  fetchPipelineCapabilities: api.caps, fetchPipelines: api.list, fetchPipeline: api.detail, createPipeline: api.create,
}));
vi.mock("@/lib/evaluation-datasets", () => ({ evaluationDatasets: api.datasets, evaluationDataset: api.dataset }));

const ids = Object.keys(DATASET_ZH) as Array<keyof typeof DATASET_ZH>;
const caps: Capabilities = {
  schema_version: 1, online_enabled: true, paid_confirm_token_required: true, not_a_single_score: true,
  datasets: ids.map((dataset_id) => ({ dataset_id, ready: true, reason: "ready", lane: dataset_id === "report-benchmark" ? "online" : "offline" })),
};
const job: Pipeline = {
  pipeline_id: "original-job-id", datasets: ["planner-replay-v1"], max_cases: 1, state: "completed",
  steps: [], created_at: "", updated_at: "", paid_execution_authorized: false, not_a_single_score: true,
};

beforeEach(() => {
  vi.clearAllMocks();
  api.caps.mockResolvedValue(caps);
  api.list.mockResolvedValue({ pipelines: [] });
  api.detail.mockResolvedValue(job);
  api.create.mockResolvedValue(job);
  api.datasets.mockResolvedValue({ datasets: ids.map((id) => ({
    id, title: getDatasetEnglish(id), origin: "upstream", category: "retrieval", split: "benchmark",
    execution: "offline", source_status: "available", verified_count: 2, declared_count: 2,
    description: "Original English description", score_family: "Original metric", count_basis: "source_verified",
    source: "unchanged/source.json", source_sha256: "a".repeat(64), warnings: [],
  })) });
  api.dataset.mockResolvedValue({ dataset: {
    id: "corpus-retrieval", cases: [{ index: 1, case_id: "original-case-id", question_preview: "Original evidence question" }],
    offset: 0, verified_count: 1, has_more: false,
  } });
});

describe("中文展示与原始评测契约", () => {
  it.each(ids)("%s 有独立中文名称、说明与评分口径", (id) => {
    expect(getDatasetLabel(id)).toMatch(/[\u4e00-\u9fff]/);
    expect(DATASET_ZH[id].description).toMatch(/[\u4e00-\u9fff]/);
    expect(getDatasetMetrics(id)).not.toContain("Schema");
    expect(getDatasetEnglish(id)).not.toEqual(getDatasetLabel(id));
  });

  it("未知标识保持原文，不能继承对象属性作为名称", () => {
    expect(getDatasetLabel("future-dataset", "未来数据集")).toBe("未来数据集");
    expect(getDatasetLabel("__proto__")).toBe("__proto__");
    expect(evaluationLabel("unknown_backend_code")).toBe("unknown_backend_code");
  });

  it("中文指标不把未测量、无效与零混淆", () => {
    render(<EvaluationMetrics metrics={{ gate_passed: false, accuracy: 0, coverage: null, status: "invalid" }} />);
    expect(screen.getByText("门禁通过").nextElementSibling).toHaveTextContent("否");
    expect(screen.getByText("准确率").nextElementSibling).toHaveTextContent("0");
    expect(screen.getByText("判定覆盖率").nextElementSibling).toHaveTextContent("未测量");
    expect(screen.getByText("测量状态").nextElementSibling).toHaveTextContent("无效");
    expect(evaluationLabel("precision_at_5")).toBe("前 5 项精确率");
    expect(evaluationNote("Original pricing not verified; a reported zero is not free API usage")).toContain("不代表");
  });

  it("八张中文卡片查询原 ID，保留样本原文和源文件技术标识", async () => {
    render(<DatasetRegistryPanel />);
    for (const id of ids) {
      expect(await screen.findByRole("button", { name: new RegExp(getDatasetLabel(id)) })).toBeInTheDocument();
    }
    fireEvent.click(screen.getByRole("button", { name: new RegExp(getDatasetLabel("corpus-retrieval")) }));
    await waitFor(() => expect(api.dataset).toHaveBeenCalledWith("corpus-retrieval", 0, 50));
    expect(await screen.findByText("Original evidence question")).toBeInTheDocument();
    expect(screen.getByText("original-case-id")).toBeInTheDocument();
    const disclosure = screen.getByText("技术标识（排查时查看）").closest("details")!;
    expect(disclosure).not.toHaveAttribute("open");
    expect(within(disclosure).getByText(/unchanged\/source.json/)).toBeInTheDocument();
    expect(api.create).not.toHaveBeenCalled();
  });

  it("中文离线入口仍发送原始数据集 ID，不附带付费授权", async () => {
    render(<UnifiedEvaluationRunnerV13 />);
    fireEvent.click(await screen.findByRole("button", { name: "运行选中评测集" }));
    await waitFor(() => expect(api.create).toHaveBeenCalledWith({
      datasets: ["planner-replay-v1"], max_cases: 1, request_id: expect.any(String), consent: null,
    }));
  });

  it("中文在线入口继续要求显式费用确认，授权值保持原样", async () => {
    render(<UnifiedEvaluationRunnerV13 />);
    const report = await screen.findByRole("checkbox", { name: /研究报告质量评测/ });
    fireEvent.click(report);
    const run = screen.getByRole("button", { name: "运行选中评测集" });
    expect(run).toBeDisabled();
    expect(api.create).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("checkbox", { name: /我确认真实模型评测可能产生接口费用/ }));
    fireEvent.click(run);
    await waitFor(() => expect(api.create).toHaveBeenCalledWith(expect.objectContaining({
      datasets: ["report-benchmark", "planner-replay-v1"], consent: "I_CONFIRM_REAL_MODEL_API_COST",
    })));
  });
});
