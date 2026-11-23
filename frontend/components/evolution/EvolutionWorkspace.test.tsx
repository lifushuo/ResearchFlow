import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import EvolutionWorkspace from "./EvolutionWorkspace";
import { type EvolutionBadCase, type EvolutionOverview } from "@/lib/evaluation-evolution";

const api = vi.hoisted(() => ({
  overview: vi.fn(),
  dataset: vi.fn(),
  annotations: vi.fn(),
  badCases: vi.fn(),
  benchmarkCases: vi.fn(),
  experiments: vi.fn(),
  backflow: vi.fn(),
  evidence: vi.fn(),
  reviewable: vi.fn(),
  reviewableEvidence: vi.fn(),
  plannerJudge: vi.fn(),
}));

vi.mock("@/lib/evaluation-evolution", () => ({
  evolutionOverview: api.overview,
  evolutionDataset: api.dataset,
  evolutionAnnotations: api.annotations,
  evolutionBadCases: api.badCases,
  evolutionBenchmarkCases: api.benchmarkCases,
  evolutionExperiments: api.experiments,
  evolutionBackflow: api.backflow,
  evolutionPlannerEvidence: api.evidence,
  evolutionReviewableCases: api.reviewable,
  evolutionReviewableCaseEvidence: api.reviewableEvidence,
  evolutionPlannerJudge: api.plannerJudge,
  evolutionError: (error: unknown) => `错误：${String(error)}`,
  stageLabel: { planner: "Planner 漏项（进开发集）", retrieval: "检索环节问题（仅存证）", report: "报告环节问题（仅存证）" },
  splitLabel: { train: "训练集", validation: "验证集", holdout: "留出集" },
}));

vi.mock("@/lib/evaluation-datasets", () => ({
  evaluationDatasets: vi.fn().mockResolvedValue({
    schema_version: 1,
    registry_sha256: "ab".repeat(32),
    read_only: true,
    datasets: [
      {
        id: "report-benchmark",
        title: "报告基准评测",
        origin: "upstream",
        category: "report",
        split: "dev_holdout",
        collection: "cases",
        source: "evals/queries.json",
        declared_count: 10,
        verified_count: 10,
        count_basis: "source_verified",
        source_status: "available",
        source_sha256: "cd".repeat(32),
        supporting_file_count: 1,
        supporting_files_valid: true,
        execution: "online",
        score_family: "report",
        description: "十道报告题",
        warnings: [],
      },
    ],
  }),
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

function fillPlannerEvidence() {
  fireEvent.change(screen.getByLabelText("实际 Planner 任务"), {
    target: { value: "Search for memory write architecture details" },
  });
  fireEvent.click(screen.getByLabelText("漏项：measured effectiveness evidence"));
  fireEvent.change(screen.getByLabelText("Planner Trace 证据编号"), {
    target: { value: "pipe-1--report-benchmark/planner.agent_log#3" },
  });
  fireEvent.click(screen.getByLabelText("已核对 Planner 任务证据"));
}

beforeEach(() => {
  vi.clearAllMocks();
  api.overview.mockResolvedValue(overview);
  api.evidence.mockResolvedValue({ status: "unavailable", run_id: badCase.last_run_id, tasks: [] });
  api.reviewable.mockResolvedValue({ cases: [{
    run_id: "pipe-aaaaaaaaaaaaaaaaaaaaaaaa--report-benchmark", case_id: "llm-memory",
    result_status: "passed", query: "研究 LLM 长程记忆机制",
    depth: "balanced", required_aspects: ["记忆写入路径", "检索延迟量级"],
  }] });
  api.reviewableEvidence.mockResolvedValue({
    status: "recorded", case_id: "llm-memory",
    run_id: "pipe-aaaaaaaaaaaaaaaaaaaaaaaa--report-benchmark",
    tasks: [{ id: "1", query: "检索 LLM 记忆写入方案与实现路径" }],
    evidence_reference: "auto:pipe-aaaaaaaaaaaaaaaaaaaaaaaa--report-benchmark:abcd",
  });
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
        {
          case_id: "memory-write-survey",
          group: "agent-memory",
          split: "train",
          query: "Survey long-term memory write and retrieval pipelines for LLM agents.",
          depth: "balanced",
          required_aspects: ["memory write and retrieval mechanism"],
          source_bad_case_id: null,
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
  api.benchmarkCases.mockResolvedValue({
    dataset_id: "report-benchmark",
    case_count: 10,
    annotated_case_count: 10,
    cases: [
      {
        case_id: "postgres-vs-mysql",
        query: "Compare transactional semantics of Postgres and MySQL.",
        depth: "balanced",
        required_aspects: ["transaction semantics differences", "replication characteristics"],
        rationale: "选型对比维度",
      },
    ],
  });
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
    review: { stage: "planner", review_note: "n", reviewed_at: "2026-11-20T00:00:00+00:00", source: { type: "bad_case", bad_case_id: "bc-1", benchmark_case_id: "llm-memory", run_id: "pipe-1--report-benchmark", error_key: "research_incomplete" }, benchmark_aspects: badCase.suggested_aspects, task_evidence_status: "reviewer_supplied", planner_tasks: ["Search for memory write architecture details"], missing_aspects: ["measured effectiveness evidence"], evidence_reference: "pipe-1--report-benchmark/planner.agent_log#3", evidence_sha256: "a".repeat(64), review_confirmed: true },
  });
});

it("闭环流程页读取总览并展示分阶段状态，不自动发起回流", async () => {
  render(<EvolutionWorkspace />);
  expect(await screen.findByRole("heading", { name: "自进化" })).toBeVisible();
  await waitFor(() => expect(api.overview).toHaveBeenCalled());
  expect(await screen.findByText("待复核基准坏例 2 条")).toBeVisible();
  expect(api.backflow).not.toHaveBeenCalled();
});

it("没有 Planner 原始任务与明确人工复核时禁止提交训练坏例", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "坏例回流" }));
  fireEvent.click(await screen.findByRole("button", { name: /llm-memory/ }));
  // 等自动取证的异步响应落定，避免 act(...) 外的状态更新警告掩盖真实回归。
  await screen.findByText(/尚无可验证的 Planner 任务归档/);
  fireEvent.change(screen.getByLabelText(/复核结论/), { target: { value: "报告存在遗漏但未取证。" } });
  expect(screen.getByRole("button", { name: "提交回流" })).toBeDisabled();
  expect(api.backflow).not.toHaveBeenCalled();
  fillPlannerEvidence();
  expect(screen.getByRole("button", { name: "提交回流" })).toBeEnabled();
});

it("选择坏例后预填维度标注，提交回流进入训练集并刷新状态", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "坏例回流" }));
  const pick = await screen.findByRole("button", { name: /llm-memory/ });
  fireEvent.click(pick);
  const aspects = await screen.findByLabelText("必答维度（每行一条；已按基准标注预填，请对照轨迹确认或修改）");
  expect((aspects as HTMLTextAreaElement).value).toContain("memory write and retrieval mechanism");
  fireEvent.change(screen.getByLabelText(/复核结论/), { target: { value: "任务列表缺少记忆写入路径的检索。" } });
  fillPlannerEvidence();
  fireEvent.click(screen.getByRole("button", { name: "提交回流" }));
  await waitFor(() => expect(api.backflow).toHaveBeenCalledTimes(1));
  expect(api.backflow.mock.calls[0][0]).toMatchObject({ stage: "planner", bad_case_id: "bc-1", group: "llm-memory", review_confirmed: true, missing_aspects: ["measured effectiveness evidence"] });
  expect(await screen.findByText(/已进入开发集/)).toBeVisible();
  await waitFor(() => expect(api.overview).toHaveBeenCalledTimes(2));
});

it("手动回流携带研究问题且不引用坏例", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "坏例回流" }));
  fireEvent.click(await screen.findByLabelText(/手动回流/));
  fireEvent.change(await screen.findByLabelText("研究问题"), { target: { value: "比较两种共识协议的广域网延迟与一致性代价。" } });
  fireEvent.change(screen.getByLabelText("案例 ID"), { target: { value: "manual-01" } });
  fireEvent.change(screen.getByLabelText("分组（同组题目会留在同一 split）"), { target: { value: "distributed-systems" } });
  fireEvent.change(screen.getByLabelText("必答维度（每行一条；已按基准标注预填，请对照轨迹确认或修改）"), { target: { value: "延迟量级\n一致性代价" } });
  fireEvent.change(screen.getByLabelText(/复核结论/), { target: { value: "来自人工复核的既有研究记录。" } });
  fireEvent.change(screen.getByLabelText("实际 Planner 任务"), { target: { value: "检索共识协议的广域网延迟统计数据" } });
  fireEvent.click(screen.getByLabelText("漏项：一致性代价"));
  fireEvent.change(screen.getByLabelText("Planner Trace 证据编号"), { target: { value: "manual-run-01/planner.agent_log#1" } });
  fireEvent.click(screen.getByLabelText("已核对 Planner 任务证据"));
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
  fillPlannerEvidence();
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

it("开发数据集列出评测中心注册表并从基准题预填回流", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "开发数据集" }));
  expect(await screen.findByText(/评测中心数据集注册表/)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "展开数据集" }));
  expect(screen.getByText("报告基准评测")).toBeVisible();
  expect(screen.getByText("可选题回流")).toBeVisible();
  fireEvent.click(await screen.findByRole("button", { name: "预填回流" }));
  const query = await screen.findByLabelText("研究问题");
  expect((query as HTMLTextAreaElement).value).toContain("Postgres");
  const aspects = screen.getByLabelText("必答维度（每行一条；已按基准标注预填，请对照轨迹确认或修改）");
  expect((aspects as HTMLTextAreaElement).value).toContain("transaction semantics differences");
  expect(api.backflow).not.toHaveBeenCalled();
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

it("自动取证后任务只读、漏项仍须人工确认", async () => {
  api.evidence.mockResolvedValue({
    status: "recorded",
    run_id: badCase.last_run_id,
    case_id: badCase.case_id,
    tasks: [{ id: "1", query: "Search for memory write architecture details" }],
    evidence_reference: `auto:${badCase.last_run_id}:${"e".repeat(64)}`,
  });
  render(<EvolutionWorkspace />);
  fireEvent.click(await screen.findByRole("button", { name: "坏例回流" }));
  fireEvent.click(await screen.findByRole("button", { name: /llm-memory/ }));
  expect(await screen.findByText(/已取得真实任务 1 条/)).toBeVisible();
  expect((screen.getByLabelText("实际 Planner 任务") as HTMLTextAreaElement).readOnly).toBe(true);
  expect((screen.getByLabelText("实际 Planner 任务") as HTMLTextAreaElement).value).toContain("memory write");
  expect(screen.getByRole("button", { name: "提交回流" })).toBeDisabled();
  fireEvent.click(screen.getByLabelText("漏项：measured effectiveness evidence"));
  fireEvent.click(screen.getByLabelText("已核对 Planner 任务证据"));
  fireEvent.change(screen.getByLabelText(/复核结论/), { target: { value: "未安排效果评测" } });
  fireEvent.click(screen.getByRole("button", { name: "提交回流" }));
  await waitFor(() => expect(api.backflow).toHaveBeenCalledWith(expect.objectContaining({
    evidence_reference: `auto:${badCase.last_run_id}:${"e".repeat(64)}`,
    planner_tasks: ["Search for memory write architecture details"],
    missing_aspects: ["measured effectiveness evidence"],
  })));
});


it("通过的报告也能进入 Planner 逐题复核，完整的规划不会进入训练集", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(screen.getByRole("button", { name: "Planner 逐题复核" }));
  const target = await screen.findByRole("button", { name: /llm-memory.*报告通过/ });
  fireEvent.click(target);
  expect(await screen.findByText("检索 LLM 记忆写入方案与实现路径")).toBeVisible();
  fireEvent.change(screen.getByLabelText("逐题复核理由"), { target: { value: "经逐项比对，当前研究计划已经覆盖全部必答维度。" } });
  fireEvent.click(screen.getByLabelText("确认逐项复核"));
  fireEvent.click(screen.getByRole("button", { name: "记录规划完整（不回流训练）" }));
  await waitFor(() => expect(api.backflow).toHaveBeenCalledTimes(1));
  expect(api.backflow.mock.calls[0][0]).toMatchObject({
    stage: "planner_complete", missing_aspects: [], review_confirmed: true,
    evaluation_case_id: "llm-memory",
  });
});

it("通过的报告也能在人工确认漏项后生成结构化 Planner 训练反馈", async () => {
  render(<EvolutionWorkspace />);
  fireEvent.click(screen.getByRole("button", { name: "Planner 逐题复核" }));
  fireEvent.click(await screen.findByRole("button", { name: /llm-memory.*报告通过/ }));
  expect(await screen.findByText("检索 LLM 记忆写入方案与实现路径")).toBeVisible();
  fireEvent.click(screen.getByText("检索延迟量级"));
  await screen.findByRole("option", { name: "agent-memory" });
  fireEvent.change(screen.getByLabelText("开发集主题组"), { target: { value: "agent-memory" } });
  fireEvent.change(screen.getByLabelText("逐题复核理由"), { target: { value: "只有记忆路径，没有检索延迟测试任务。" } });
  fireEvent.click(screen.getByLabelText("确认逐项复核"));
  fireEvent.click(screen.getByRole("button", { name: "确认漏项并回流开发集" }));
  await waitFor(() => expect(api.backflow).toHaveBeenCalledTimes(1));
  expect(api.backflow.mock.calls[0][0]).toMatchObject({
    stage: "planner", missing_aspects: ["检索延迟量级"],
    evaluation_case_id: "llm-memory", review_confirmed: true,
  });
});

it("Judge 需要人工确认费用，输出建议但不替用户勾选或提交", async () => {
  api.plannerJudge.mockResolvedValue({
    status: "suggested", run_id: "pipe-aaaaaaaaaaaaaaaaaaaaaaaa--report-benchmark",
    case_id: "llm-memory", evidence_reference: "auto:test", question: "LLM 长程记忆",
    judge_route: "custom:qwen4-max", served_judge: "qwen4-max", judge_tokens: { input: 150, output: 80 },
    coverage: 0.5, automatic_label: false,
    aspects: [
      { aspect: "记忆写入路径", aspect_index: 1, covered: true, task_ids: [1], reason: "任务 1 调查写入与检索" },
      { aspect: "检索延迟量级", aspect_index: 2, covered: false, task_ids: [], reason: "没有量化延迟任务" },
    ],
  });
  render(<EvolutionWorkspace />);
  fireEvent.click(screen.getByRole("button", { name: "Planner 逐题复核" }));
  fireEvent.click(await screen.findByRole("button", { name: /llm-memory.*报告通过/ }));
  expect(await screen.findByText("检索 LLM 记忆写入方案与实现路径")).toBeVisible();
  const button = screen.getByRole("button", { name: "一键分析疑似漏项" });
  expect(button).toBeDisabled();
  expect(api.plannerJudge).not.toHaveBeenCalled();
  fireEvent.click(screen.getByLabelText("同意 Judge API 费用"));
  fireEvent.click(button);
  await waitFor(() => expect(api.plannerJudge).toHaveBeenCalledTimes(1));
  expect(await screen.findByText("建议疑似漏项：检索延迟量级")).toBeVisible();
  expect(screen.getByText("理由：没有量化延迟任务")).toBeVisible();
  expect(screen.getByRole("region", { name: "Judge 逐维度建议" })).toBeVisible();
  // 模型的建议不修改人工 checkbox，也不会自动调用回流 API。
  const checkbox = screen.getByRole("checkbox", { name: "检索延迟量级" });
  expect(checkbox).not.toBeChecked();
  expect(api.backflow).not.toHaveBeenCalled();
});

it("Judge 请求失败只能显示错误，不能回流或自动认定漏项", async () => {
  api.plannerJudge.mockRejectedValue(new Error("Judge 服务不可用"));
  render(<EvolutionWorkspace />);
  fireEvent.click(screen.getByRole("button", { name: "Planner 逐题复核" }));
  fireEvent.click(await screen.findByRole("button", { name: /llm-memory.*报告通过/ }));
  expect(await screen.findByText("检索 LLM 记忆写入方案与实现路径")).toBeVisible();
  fireEvent.click(screen.getByLabelText("同意 Judge API 费用"));
  fireEvent.click(screen.getByRole("button", { name: "一键分析疑似漏项" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Judge 服务不可用");
  expect(api.backflow).not.toHaveBeenCalled();
});
