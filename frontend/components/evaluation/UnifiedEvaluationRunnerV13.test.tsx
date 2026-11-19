import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";

import { createPipeline, fetchPipeline, fetchPipelineCapabilities, fetchPipelines, type Capabilities, type Pipeline } from "@/lib/evaluation-pipeline-v13";
import { evaluationDataset } from "@/lib/evaluation-datasets";
import UnifiedEvaluationRunnerV13 from "./UnifiedEvaluationRunnerV13";

vi.mock("@/lib/evaluation-pipeline-v13", () => ({ createPipeline: vi.fn(), fetchPipeline: vi.fn(), fetchPipelineCapabilities: vi.fn(), fetchPipelines: vi.fn() }));
vi.mock("@/lib/evaluation-datasets", () => ({ evaluationDataset: vi.fn() }));
const target = "quantum-error-correction";
const caps: Capabilities = {
  schema_version: 1, online_enabled: true, paid_confirm_token_required: true, not_a_single_score: true,
  datasets: [
    { dataset_id: "report-benchmark", lane: "online", ready: true, reason: "ready" },
    { dataset_id: "planner-replay-v1", lane: "offline", ready: true, reason: "ready" },
    { dataset_id: "tool-contract-v1", lane: "offline", ready: true, reason: "ready" },
  ],
};
const job: Pipeline = { pipeline_id: "pipe-fixture", datasets: ["report-benchmark"], max_cases: 1, case_id: target, state: "completed", steps: [], created_at: "", updated_at: "", paid_execution_authorized: true, not_a_single_score: true };
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(fetchPipelineCapabilities).mockResolvedValue(caps);
  vi.mocked(fetchPipelines).mockResolvedValue({ pipelines: [] });
  vi.mocked(fetchPipeline).mockResolvedValue(job);
  vi.mocked(createPipeline).mockResolvedValue(job);
  vi.mocked(evaluationDataset).mockResolvedValue({ schema_version: 1, registry_sha256: "fixture", read_only: true, dataset: { cases: [{ case_id: "llm-memory", index: 1 }, { case_id: target, index: 6 }] } } as Awaited<ReturnType<typeof evaluationDataset>>);
});

async function reportSelect() {
  fireEvent.click(await screen.findByRole("checkbox", { name: /研究报告质量评测/ }));
  const input = await screen.findByRole("combobox", { name: "研究报告单案例（可选）" });
  await waitFor(() => expect(input).toBeEnabled());
  return input;
}

it("仅加载原案例选项，打开配置不提交任何任务", async () => {
  render(<UnifiedEvaluationRunnerV13 />);
  const select = await reportSelect();
  expect(select).toHaveValue("");
  expect(screen.getByRole("option", { name: target })).toBeInTheDocument();
  expect(evaluationDataset).toHaveBeenCalledWith("report-benchmark", 0, 100);
  expect(createPipeline).not.toHaveBeenCalled();
});

it("单案例强制只选报告和一题，并仍要求费用确认", async () => {
  render(<UnifiedEvaluationRunnerV13 />);
  fireEvent.change(await reportSelect(), { target: { value: target } });
  expect(screen.getByText(/只运行案例：quantum-error-correction/)).toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: /工具调用契约评分测试/ })).toBeDisabled();
  expect(screen.getByRole("combobox", { name: /案例上限/ })).toHaveValue("1");
  expect(screen.getByRole("combobox", { name: /案例上限/ })).toBeDisabled();
  expect(screen.getByRole("button", { name: "运行选中评测集" })).toBeDisabled();
  fireEvent.click(screen.getByRole("checkbox", { name: /我确认真实模型评测可能/ }));
  fireEvent.click(screen.getByRole("button", { name: "运行选中评测集" }));
  await waitFor(() => expect(createPipeline).toHaveBeenCalledOnce());
  expect(vi.mocked(createPipeline).mock.calls[0][0]).toEqual(expect.objectContaining({ datasets: ["report-benchmark"], max_cases: 1, case_id: target, consent: "I_CONFIRM_REAL_MODEL_API_COST" }));
  expect(await screen.findByText(`本次只运行报告案例：${target}`)).toBeInTheDocument();
});

it("清空单案例选择恢复原前缀模式，旧请求不带 case_id", async () => {
  vi.mocked(createPipeline).mockResolvedValue({ ...job, case_id: undefined });
  render(<UnifiedEvaluationRunnerV13 />);
  const select = await reportSelect();
  fireEvent.change(select, { target: { value: target } });
  fireEvent.change(select, { target: { value: "" } });
  const limit = screen.getByRole("combobox", { name: /案例上限/ });
  expect(limit).toBeEnabled();
  fireEvent.change(limit, { target: { value: "2" } });
  fireEvent.click(screen.getByRole("checkbox", { name: /我确认真实模型评测可能/ }));
  fireEvent.click(screen.getByRole("button", { name: "运行选中评测集" }));
  await waitFor(() => expect(createPipeline).toHaveBeenCalledOnce());
  const payload = vi.mocked(createPipeline).mock.calls[0][0];
  expect(payload.max_cases).toBe(2);
  expect(payload).not.toHaveProperty("case_id");
});

it("读取案例失败不提供伪造选项，也不自动提交", async () => {
  vi.mocked(evaluationDataset).mockRejectedValue(new Error("Invalid dataset registry or pagination"));
  render(<UnifiedEvaluationRunnerV13 />);
  fireEvent.click(await screen.findByRole("checkbox", { name: /研究报告质量评测/ }));
  expect(await screen.findByRole("alert")).toHaveTextContent("单案例选择暂不可用");
  expect(screen.getByRole("combobox", { name: "研究报告单案例（可选）" })).toBeDisabled();
  expect(createPipeline).not.toHaveBeenCalled();
});
