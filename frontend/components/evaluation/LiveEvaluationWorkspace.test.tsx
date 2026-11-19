import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import LiveEvaluationWorkspace from "./LiveEvaluationWorkspace";
import { DATASET_ZH, getDatasetLabel } from "@/lib/evaluation-dataset-zh";
import type { DatasetId, Pipeline } from "@/lib/evaluation-pipeline-v13";
import type { Diagnostic } from "@/lib/evaluation-diagnostics";

const api = vi.hoisted(() => ({ list: vi.fn(), detail: vi.fn(), diagnostic: vi.fn(), caps: vi.fn(), create: vi.fn(), get: vi.fn(), post: vi.fn(), incidents: vi.fn(), archive: vi.fn() }));
vi.mock("@/lib/evaluation-pipeline-v13", () => ({ fetchPipelines: api.list, fetchPipeline: api.detail, fetchPipelineCapabilities: api.caps, createPipeline: api.create }));
vi.mock("@/lib/evaluation-diagnostics", async (original) => ({ ...await original<object>(), getDiagnostics: api.diagnostic, getIncidents: api.incidents }));
vi.mock("@/lib/evaluation-closure", async (original) => ({ ...await original<object>(), closureGet: api.get, closurePost: api.post }));
vi.mock("@/lib/evaluation", async (original) => ({ ...await original<object>(), evaluationRuns: api.archive }));
const ids = Object.keys(DATASET_ZH) as DatasetId[];
function pipeline(id = "pipe-new", datasets: DatasetId[] = ["planner-replay-v1"]): Pipeline {
  return { pipeline_id: id, datasets, max_cases: 1, created_at: id, updated_at: id, paid_execution_authorized: false, not_a_single_score: true, state: "completed", steps: datasets.map((dataset_id) => ({ dataset_id, state: "completed", reason: null, lane: "offline", case_count: 1, elapsed_seconds: 3, metrics: { accuracy: 0.75 }, closure_run_id: `${id}--${dataset_id}` })) };
}
function diagnostic(pipe: string, dataset: DatasetId): Diagnostic {
  return { pipeline_id: pipe, dataset_id: dataset, case_id: null, state: "completed", reason: null, elapsed_seconds: 3, root_cause_confirmed: false, execution_observation: { started_at: null, finished_at: null, timed_out: false, exit_code: 0 }, has_issue: false, collection: "worker_observation", failure_phase: null, quality_measurement: "evaluator_result", evidence_fingerprint: "hash", facts: [], candidates: [], next_actions: [], missing_evidence: [], traces: [], judge_traces: [], cases: [{ case_id: `${pipe}-case`, status: "passed", latency_s: 2, claim_count: null, metrics: { accuracy: 0.75 } }], metrics: { accuracy: 0.75 } };
}
beforeEach(() => {
  vi.clearAllMocks();
  api.list.mockResolvedValue({ pipelines: [pipeline(), pipeline("pipe-old")] });
  api.detail.mockImplementation(async (id) => pipeline(id));
  api.diagnostic.mockImplementation(async (id, dataset) => diagnostic(id, dataset));
  api.incidents.mockResolvedValue({ incidents: [] });
  api.get.mockImplementation(async (path) => path === "/runs" ? [{ run_id: "pipe-old--planner-replay-v1", dataset_id: "planner-replay-v1", execution_mode: "fake", state: "finished" }] : { comparable: true, regressions: ["regressed-case"], improvements: ["recovered-case"] });
  api.post.mockResolvedValue({ verdict: "blocked", measured: 0, total_recorded: 1, pass_rate: null, counts: { passed: 0, failed: 0, blocked: 0, unmeasured: 1 } });
  api.caps.mockResolvedValue({ datasets: ids.map((dataset_id) => ({ dataset_id, lane: dataset_id === "report-benchmark" ? "online" : "offline", ready: true, reason: "ready" })), online_enabled: true });
  api.create.mockResolvedValue(pipeline("pipe-created"));
  api.archive.mockResolvedValue({ runs: [] });
});

it("工作台读取真实运行，不请求历史目录，也不自动创建任务", async () => {
  render(<LiveEvaluationWorkspace />);
  expect(await screen.findByRole("heading", { name: "本次运行总览" })).toBeVisible();
  await waitFor(() => expect(api.list).toHaveBeenCalled());
  expect(api.archive).not.toHaveBeenCalled(); expect(api.create).not.toHaveBeenCalled(); expect(api.post).not.toHaveBeenCalled();
});

it("从总览进入同一个数据集会重新读取逐题结果，不会清空后永久空白", async () => {
  render(<LiveEvaluationWorkspace />);
  await waitFor(() => expect(api.diagnostic).toHaveBeenCalled());
  const count = api.diagnostic.mock.calls.length;
  fireEvent.click(screen.getByRole("button", { name: "查看本轮逐题结果" }));
  expect(await screen.findByText("pipe-new-case")).toBeVisible();
  await waitFor(() => expect(api.diagnostic.mock.calls.length).toBeGreaterThan(count));
});

it("从总览打开诊断再切回逐题页，保留同一轮结果", async () => {
  render(<LiveEvaluationWorkspace />);
  await waitFor(() => expect(api.diagnostic).toHaveBeenCalled());
  fireEvent.click(screen.getByRole("button", { name: "诊断与轨迹" }));
  await screen.findByRole("heading", { name: "异常事实与根因分析" });
  fireEvent.click(screen.getByRole("button", { name: "逐题用例" }));
  expect(await screen.findByText("pipe-new-case")).toBeVisible();
});

it("坏例读取绑定本轮选择，读取完成前不宣称无坏例", async () => {
  api.incidents.mockReturnValue(new Promise(() => {}));
  render(<LiveEvaluationWorkspace />);
  await screen.findByRole("heading", { name: getDatasetLabel("planner-replay-v1") });
  fireEvent.click(screen.getByRole("button", { name: "坏例排查" }));
  expect(await screen.findByText("正在读取本轮坏例与执行证据…")).toBeVisible();
  expect(screen.queryByText(/没有已记录的执行异常/)).not.toBeInTheDocument();
  expect(api.incidents).toHaveBeenCalledWith("pipe-new", "planner-replay-v1");
});

it.each(ids)("%s 的新运行贯通总览及逐题原指标", async (dataset) => {
  const row = pipeline("pipe-measured", [dataset]); api.list.mockResolvedValue({ pipelines: [row] }); api.detail.mockResolvedValue(row);
  render(<LiveEvaluationWorkspace />);
  expect(await screen.findByRole("heading", { name: getDatasetLabel(dataset) })).toBeVisible();
  if (dataset !== "report-benchmark") expect(screen.getByText("0.75")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "逐题用例" }));
  expect(await screen.findByText("pipe-measured-case")).toBeVisible();
  expect(api.diagnostic).toHaveBeenCalledWith("pipe-measured", dataset);
});

it("只显示本轮所选维度，不用空报告卡片盖住有效 Replay 成绩", async () => {
  render(<LiveEvaluationWorkspace />);
  expect(await screen.findByRole("heading", { name: getDatasetLabel("planner-replay-v1") })).toBeVisible();
  expect(screen.queryByRole("heading", { name: getDatasetLabel("report-benchmark") })).not.toBeInTheDocument();
});

it("运行下拉框采用已刷新的实际状态，不残留排队标签", async () => {
  api.list.mockResolvedValue({ pipelines: [{ ...pipeline(), state: "queued" }] });
  render(<LiveEvaluationWorkspace />);
  expect(await screen.findByRole("option", { name: /执行结束.*pipe-new/ })).toBeVisible();
  expect(screen.queryByRole("option", { name: /排队中.*pipe-new/ })).not.toBeInTheDocument();
});

it("切换运行清除旧逐题结果，迟到的旧请求不能覆盖新结果", async () => {
  let resolveOld!: (d: Diagnostic) => void;
  api.diagnostic.mockImplementation((pipe, dataset) => pipe === "pipe-new" ? new Promise<Diagnostic>((r) => { resolveOld = r; }) : Promise.resolve(diagnostic(pipe, dataset)));
  render(<LiveEvaluationWorkspace />);
  await waitFor(() => expect(api.diagnostic).toHaveBeenCalledWith("pipe-new", "planner-replay-v1"));
  fireEvent.change(screen.getByRole("combobox", { name: "本次评测运行" }), { target: { value: "pipe-old" } });
  fireEvent.click(screen.getByRole("button", { name: "逐题用例" }));
  expect(await screen.findByText("pipe-old-case")).toBeVisible();
  resolveOld(diagnostic("pipe-new", "planner-replay-v1"));
  await waitFor(() => expect(screen.queryByText("pipe-new-case")).not.toBeInTheDocument());
});

it("切换数据集后读取同一运行该数据集的结果", async () => {
  const row = pipeline("pipe-multi", ["planner-replay-v1", "tool-contract-v1"]); api.list.mockResolvedValue({ pipelines: [row] }); api.detail.mockResolvedValue(row);
  render(<LiveEvaluationWorkspace />);
  await screen.findByRole("heading", { name: getDatasetLabel("planner-replay-v1") });
  fireEvent.change(screen.getByRole("combobox", { name: "本次评测集" }), { target: { value: "tool-contract-v1" } });
  await waitFor(() => expect(api.diagnostic).toHaveBeenCalledWith("pipe-multi", "tool-contract-v1"));
});

it("逐题状态筛选保留未测量，实际案例可以进入同运行分析", async () => {
  api.diagnostic.mockImplementation(async (pipe, ds) => ({ ...diagnostic(pipe, ds), cases: [{ case_id: "interrupted-case", status: "unmeasured", latency_s: null, claim_count: null }] }));
  render(<LiveEvaluationWorkspace />);
  fireEvent.click(screen.getByRole("button", { name: "逐题用例" }));
  await screen.findByText("interrupted-case");
  fireEvent.change(screen.getByRole("combobox", { name: "案例判定筛选" }), { target: { value: "passed" } });
  expect(screen.queryByText("interrupted-case")).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "案例判定筛选" }), { target: { value: "unmeasured" } });
  fireEvent.click(screen.getByRole("button", { name: "查看该案例证据与分析" }));
  expect(await screen.findByRole("heading", { name: "异常事实与根因分析" })).toBeVisible();
});

it("同账号实时坏例池替代空历史草稿，并可进入审核复测", async () => {
  render(<LiveEvaluationWorkspace />);
  await screen.findByRole("heading", { name: getDatasetLabel("planner-replay-v1") });
  fireEvent.click(screen.getByRole("button", { name: "坏例排查" }));
  expect(await screen.findByRole("heading", { name: /自动待排查池/ })).toBeVisible();
  expect(api.incidents).toHaveBeenCalled(); expect(api.archive).not.toHaveBeenCalled();
});

it("实际两次运行执行基线比较并展示退化与恢复案例", async () => {
  render(<LiveEvaluationWorkspace />);
  await screen.findByRole("heading", { name: getDatasetLabel("planner-replay-v1") });
  fireEvent.click(screen.getByRole("button", { name: "版本对比" }));
  fireEvent.change(await screen.findByRole("combobox", { name: "同评测集基线" }), { target: { value: "pipe-old--planner-replay-v1" } });
  fireEvent.click(screen.getByRole("button", { name: "执行基线比较" }));
  expect(await screen.findByText("退化：regressed-case")).toBeVisible();
  expect(screen.getByText("恢复：recovered-case")).toBeVisible();
  expect(api.get).toHaveBeenCalledWith("/compare?baseline_run=pipe-old--planner-replay-v1&candidate_run=pipe-new--planner-replay-v1");
  expect(api.post).not.toHaveBeenCalled();
});

it("比较不可用给出具体原因，不把未验证的模型配对说成质量提升", async () => {
  api.get.mockImplementation(async (path) => path === "/runs" ? [{ run_id: "baseline", dataset_id: "planner-replay-v1", execution_mode: "real", state: "finished" }] : { comparable: false, reasons: ["model_identity_unverified", "unmeasured_cases"] });
  render(<LiveEvaluationWorkspace />);
  await screen.findByRole("heading", { name: getDatasetLabel("planner-replay-v1") }); fireEvent.click(screen.getByRole("button", { name: "版本对比" }));
  fireEvent.change(await screen.findByRole("combobox", { name: "同评测集基线" }), { target: { value: "baseline" } });
  fireEvent.click(screen.getByRole("button", { name: "执行基线比较" }));
  expect(await screen.findByText(/实际服务模型身份未验证/)).toBeVisible();
  expect(screen.getByText("存在未测量或阻塞案例")).toBeVisible();
});

it("门禁必须显式操作，未测量保持 null 并阻止通过", async () => {
  render(<LiveEvaluationWorkspace />); await screen.findByRole("heading", { name: getDatasetLabel("planner-replay-v1") });
  fireEvent.click(screen.getByRole("button", { name: "版本对比" })); expect(api.post).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "检查候选质量门禁" }));
  expect(await screen.findByText("候选质量门禁：阻塞")).toBeVisible();
  expect(screen.getByText(/有效测量 0.*通过率 未测量/)).toBeVisible(); expect(api.create).not.toHaveBeenCalled();
});

it("无历史目录也可创建真实离线任务并立即切换到新任务总览", async () => {
  api.list.mockResolvedValue({ pipelines: [] }); api.detail.mockImplementation(async (id) => pipeline(id));
  render(<LiveEvaluationWorkspace />);
  fireEvent.click(screen.getByRole("button", { name: "新建 / 重新配置评测" }));
  fireEvent.click(await screen.findByRole("button", { name: "运行选中评测集" }));
  await waitFor(() => expect(api.create).toHaveBeenCalledOnce());
  expect(api.create.mock.calls[0][0]).toEqual(expect.objectContaining({ datasets: ["planner-replay-v1"], consent: null }));
  await waitFor(() => expect(screen.getByRole("combobox", { name: "本次评测运行" })).toHaveValue("pipe-created"));
  expect(await screen.findByRole("heading", { name: "本次运行总览" })).toBeVisible();
});
