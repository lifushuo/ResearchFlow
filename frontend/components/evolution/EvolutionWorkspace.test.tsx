import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import EvolutionWorkspace from "./EvolutionWorkspace";
import { type EvolutionBadCase, type EvolutionOverview } from "@/lib/evaluation-evolution";

const api = vi.hoisted(() => ({
  overview: vi.fn(),
  dataset: vi.fn(),
  annotations: vi.fn(),
  badCases: vi.fn(),
  experiments: vi.fn(),
  backflow: vi.fn(),
}));

vi.mock("@/lib/evaluation-evolution", () => ({
  evolutionOverview: api.overview,
  evolutionDataset: api.dataset,
  evolutionAnnotations: api.annotations,
  evolutionBadCases: api.badCases,
  evolutionExperiments: api.experiments,
  evolutionBackflow: api.backflow,
  evolutionError: (error: unknown) => `错误：${String(error)}`,
  stageLabel: { planner: "Planner 漏项（进开发集）", retrieval: "检索环节问题（仅存证）", report: "报告环节问题（仅存证）" },
  splitLabel: { train: "训练集", validation: "验证集", holdout: "留出集" },
}));

const overview: EvolutionOverview = {
  dataset: {
    seed_case_count: 30,
    backflow_case_count: 1,
    review_record_count: 1,
    group_count: 16,
    splits: { train: 13, validation: 9, holdout: 9 },
    dataset_sha256: "ab".repeat(32),
    error: null,
  },
  benchmark: { case_count: 10, annotated_case_count: 10 },
  bad_cases: { open_benchmark_count: 2, flowed_count: 1 },
  experiments: { directory_count: 1, last_holdout_gate_passed: true, last_report_gate_passed: null, release_count: 0 },
};

const badCase: EvolutionBadCase = {
  bad_case_id: "bc-1",
  case_id: "llm-memory",
  error_key: "research_incomplete",
  review_status: "new",
  first_run_id: "pipe-1--report-benchmark",
  last_run_id: "pipe-1--report-benchmark",
  last_seen_ms: 1,
  query: "What are the leading approaches to long-term memory in LLM agents?",
  depth: "balanced",
  suggested_aspects: ["memory write and retrieval mechanism", "measured effectiveness evidence"],
};

beforeEach(() => {
  vi.clearAllMocks();
  api.overview.mockResolvedValue(overview);
  api.dataset.mockResolvedValue({
    seed_case_count: 30,
    backflow_case_count: 1,
    review_record_count: 1,
    dataset_sha256: "ab".repeat(32),
    error: null,
    splits: { train: 13, validation: 9, holdout: 9 },
    dataset: {
      schema_version: 1,
      dataset_id: "planner-evolution-reviewed-v1",
      provenance: "reviewed",
      cases: [
        {
          case_id: "tsp-baselines",
          group: "routing-optimization",
          split: "train",
          query: "比较学习型 TSP 求解器与 LKH。",
          depth: "balanced",
          required_aspects: ["相同实例规模的解质量", "参考解定义"],
          source_bad_case_id: null,
        },
        {
          case_id: "planner-gap-01",
          group: "agent-memory",
          split: "train",
          query: "What are the leading approaches to long-term memory in LLM agents?",
          depth: "balanced",
          required_aspects: ["memory write and retrieval mechanism"],
          source_bad_case_id: "bc-1",
        },
      ],
    },
  });
  api.annotations.mockResolvedValue({
    schema_version: 1,
    dataset_id: "report-benchmark",
    source: "evals/queries.json",
    note: "标注口径",
    aspects: { "llm-memory": { required_aspects: badCase.suggested_aspects!, rationale: "四个可核查维度" } },
  });
  api.badCases.mockResolvedValue({ bad_cases: [badCase], annotated_cases: 10 });
  api.experiments.mockResolvedValue([
    {
      name: "planner-demo-01",
      experiment: {
        created_at: "2026-11-20T00:00:00+00:00",
        evidence_kind: "scripted_demo",
        metric_version: "planner-task-coverage/1",
        optimizer: { name: "GEPA", version: "0.1.4" },
        limits: null,
        dataset_sha256: "cd".repeat(32),
        case_count: 12,
        routes: { judge: "custom:judge" },
        judge_settings: null,
        validation_gain: 0.1,
        holdout_gate: { passed: true, delta: 0.1, baseline_coverage: 0.5, candidate_coverage: 0.6, reasons: [] },
        candidate: "ef".repeat(32),
        has_prompt_diff: true,
      },
      report_gate: null,
      release: null,
    },
  ]);
  api.backflow.mockResolvedValue({
    case: { case_id: "planner-gap-llm-memory", group: "agent-memory", split: "train", query: badCase.query, depth: "balanced", required_aspects: badCase.suggested_aspects!, source_bad_case_id: "bc-1" },
    review: { stage: "planner", review_note: "n", reviewed_at: "2026-11-20T00:00:00+00:00", source: { type: "bad_case", bad_case_id: "bc-1", benchmark_case_id: "llm-memory", run_id: "pipe-1--report-benchmark", error_key: "research_incomplete" }, benchmark_aspects: badCase.suggested_aspects, task_evidence_status: "not_collected" },
  });
});

it("闭环流程页读取总览并展示分阶段状态，不自动发起回流", async () => {
  render(<EvolutionWorkspace />);
  expect(await screen.findByRole("heading", { name: "自进化" })).toBeVisible();
  await waitFor(() => expect(api.overview).toHaveBeenCalled());
  expect(await screen.findByText("待复核基准坏例 2 条")).toBeVisible();
  expect(api.backflow).not.toHaveBeenCalled();
});

it("选择坏例后预填维度标注，提交回流进入训练集并刷新状态", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "坏例回流" }));
  const pick = await screen.findByRole("button", { name: /llm-memory/ });
  fireEvent.click(pick);
  const aspects = await screen.findByLabelText("必答维度（每行一条；已按基准标注预填，请对照轨迹确认或修改）");
  expect((aspects as HTMLTextAreaElement).value).toContain("memory write and retrieval mechanism");
  fireEvent.change(screen.getByLabelText(/复核结论/), { target: { value: "任务列表缺少记忆写入路径的检索。" } });
  fireEvent.click(screen.getByRole("button", { name: "提交回流" }));
  await waitFor(() => expect(api.backflow).toHaveBeenCalledTimes(1));
  expect(api.backflow.mock.calls[0][0]).toMatchObject({ stage: "planner", bad_case_id: "bc-1", group: "llm-memory" });
  expect(await screen.findByText(/已进入开发集/)).toBeVisible();
  await waitFor(() => expect(api.overview).toHaveBeenCalledTimes(2));
});

it("手动回流携带研究问题且不引用坏例", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "坏例回流" }));
  fireEvent.click(await screen.findByRole("checkbox"));
  fireEvent.change(await screen.findByLabelText("研究问题"), { target: { value: "比较两种共识协议的广域网延迟与一致性代价。" } });
  fireEvent.change(screen.getByLabelText("案例 ID"), { target: { value: "manual-01" } });
  fireEvent.change(screen.getByLabelText("分组（同组题目会留在同一 split）"), { target: { value: "distributed-systems" } });
  fireEvent.change(screen.getByLabelText("必答维度（每行一条；已按基准标注预填，请对照轨迹确认或修改）"), { target: { value: "延迟量级\n一致性代价" } });
  fireEvent.change(screen.getByLabelText(/复核结论/), { target: { value: "来自人工复核的既有研究记录。" } });
  fireEvent.click(screen.getByRole("button", { name: "提交回流" }));
  await waitFor(() => expect(api.backflow).toHaveBeenCalledTimes(1));
  const body = api.backflow.mock.calls[0][0];
  expect(body).toMatchObject({ manual_query: expect.stringContaining("共识协议"), stage: "planner" });
  expect(body.bad_case_id).toBeUndefined();
});

it("回流失败展示可读原因，不静默成功", async () => {
  api.backflow.mockRejectedValue(new Error("该坏例已回流: bc-1"));
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "坏例回流" }));
  fireEvent.click(await screen.findByRole("button", { name: /llm-memory/ }));
  fireEvent.change(await screen.findByLabelText(/复核结论/), { target: { value: "重复回流尝试。" } });
  fireEvent.click(screen.getByRole("button", { name: "提交回流" }));
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("错误：");
  expect(alert).toHaveTextContent("该坏例已回流");
});

it("开发数据集按 split 展示种子与回流题目", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "开发数据集" }));
  expect(await screen.findByText("种子 30 题")).toBeVisible();
  expect(await screen.findByText("tsp-baselines")).toBeVisible();
  expect(screen.getByText(/来自坏例 bc-1/)).toBeVisible();
  fireEvent.change(screen.getByLabelText("按 split 筛选"), { target: { value: "validation" } });
  expect(screen.queryByText("tsp-baselines")).toBeNull();
});

it("实验记录展示留出集门禁结论与证据来源", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "实验记录" }));
  expect(await screen.findByText("planner-demo-01")).toBeVisible();
  expect(await screen.findByText("留出集门禁通过")).toBeVisible();
  expect(screen.getByText(/提升 10.00 个百分点/)).toBeVisible();
  expect(screen.getByText(/脚本演示/)).toBeVisible();
  expect(screen.getByText(/planner_evolution dataset/)).toBeVisible();
});
