import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { DiagnosticDetails } from "./PipelineDiagnosticsPanel";
import ExecutionIncidentPanel from "./ExecutionIncidentPanel";
import { getDiagnostics, getIncidents, openDiagnostics, type Diagnostic } from "@/lib/evaluation-diagnostics";

vi.mock("@/lib/evaluation-diagnostics", () => ({ getDiagnostics: vi.fn(), getIncidents: vi.fn(), openDiagnostics: vi.fn() }));
const data: Diagnostic = {
  pipeline_id: "pipe-fixture", dataset_id: "report-benchmark", case_id: "case", state: "failed", reason: "runner_timeout", elapsed_seconds: 420, has_issue: true, root_cause_confirmed: false,
  execution_observation: { started_at: null, finished_at: null, exit_code: 124, timed_out: true },
  collection: "saved_status_analysis", failure_phase: "planner", quality_measurement: "unmeasured", evidence_fingerprint: "a".repeat(64),
  facts: [{ id: "F1", text: "运行器总时限已触发" }], candidates: [{ cause: "模型调用仍在等待", confidence: "底层诱因待验证", evidence_ids: ["F1"], verification: "检查未结束片段" }], next_actions: ["补充节点证据"], missing_evidence: ["轨迹不完整"],
  traces: [{ trace_id: "trace", case_id: "case", trace_status: "partial", open_spans: ["span"], events: [{ seq: 1, case_id: "case", span_id: "span", name: "planner", kind: "llm", status: "start", duration_ms: null, error_type: null, metadata: {} }] }], judge_traces: [], cases: [], metrics: {},
};
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getDiagnostics).mockResolvedValue(data);
  vi.mocked(getIncidents).mockResolvedValue({ incidents: [{ incident_id: "id", pipeline_id: data.pipeline_id, dataset_id: data.dataset_id, case_id: data.case_id, reason: "runner_timeout", occurrences: [data.pipeline_id], summary: "运行器达到总时限", collection: "saved_status_analysis" }] });
});

it("保留事实与轨迹，不提供付费归因或审核入口", async () => {
  render(<DiagnosticDetails pipe={data.pipeline_id} dataset={data.dataset_id} />);
  expect(await screen.findByRole("heading", { name: "异常事实与根因分析" })).toBeVisible();
  expect(screen.getByText(/整轮耗时：420.00 秒/)).toBeVisible();
  expect(screen.getByText(/执行未完成，质量指标保持未测量/)).toBeVisible();
  expect(screen.getByText(/候选 1：模型调用仍在等待/)).toBeInTheDocument();
  expect(screen.queryByText("请求模型辅助分析")).not.toBeInTheDocument();
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
});

it("缺失轨迹继续明确披露，不伪造事件", async () => {
  vi.mocked(getDiagnostics).mockResolvedValue({ ...data, traces: [] });
  render(<DiagnosticDetails pipe={data.pipeline_id} dataset={data.dataset_id} />);
  expect(await screen.findByText(/本次没有图轨迹/)).toBeVisible();
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
});

it("单题耗时取本题结果，不把 420 秒整轮背景当作本题耗时", async () => {
  vi.mocked(getDiagnostics).mockResolvedValue({ ...data, cases: [{ case_id: "case", status: "failed", latency_s: 9.15, claim_count: null }] });
  render(<DiagnosticDetails pipe={data.pipeline_id} dataset={data.dataset_id} initialCase="case" />);
  expect(await screen.findByText(/本题研究耗时：9.15 秒/)).toBeVisible();
  expect(screen.queryByText(/本题研究耗时：420/)).not.toBeInTheDocument();
});

it("只读待排查池仍能跳到原运行证据，不提供忽略或审核操作", async () => {
  render(<ExecutionIncidentPanel />);
  expect(await screen.findByText("运行器达到总时限")).toBeVisible();
  expect(screen.getByText(/独立运行出现 1 次/)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "查看诊断与轨迹" }));
  expect(openDiagnostics).toHaveBeenCalledWith(data.pipeline_id, data.dataset_id, data.case_id);
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  expect(screen.queryByText("接手排查")).not.toBeInTheDocument();
});
