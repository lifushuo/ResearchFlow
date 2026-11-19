import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { StartResearchForm } from "./StartResearchForm";

/** 发起研究会产生费用，表单因此既要支持提交也要明确拒绝无效操作：空或过短的问题、没有所属项目、首个请求尚未完成时再次提交。每次拒绝都应说明原因，不能只留下一个像是坏掉的禁用按钮。 */

const mutate = vi.fn();
let pending = false;
let activeId: string | undefined = "p1";
let projectsLoading = false;

vi.mock("@/hooks/runs", () => ({
  useStartV2Research: () => ({
    mutate,
    isPending: pending,
    isError: false,
    error: null,
  }),
}));

vi.mock("@/hooks/queries", () => ({
  useCorpusStatus: () => ({ data: undefined }),
  // 表单读取目录，展示研究实际使用的模型；
  // 测试将五个角色解析到同一路由，符合选择器以单一后端表示当前模型的形状。
  useModelCatalog: () => ({
    data: {
      available_providers: ["custom", "ollama"],
      models: [],
      effective_routing: {
        planner: "custom:auto/best-fast",
        executor: "custom:auto/best-fast",
        critic: "custom:auto/best-fast",
        synthesizer: "custom:auto/best-fast",
        chat: "custom:auto/best-fast",
      },
    },
    isLoading: false,
    isError: false,
  }),
  useCustomEndpointStatus: () => ({ data: undefined, isLoading: false }),
  useLocalLLMStatus: () => ({ data: undefined, isLoading: false }),
}));

vi.mock("@/components/ActiveProject", () => ({
  useActiveProject: () => ({
    activeId,
    active: activeId ? { id: activeId, name: "Agent Memory" } : undefined,
    isLoading: projectsLoading,
    projects: [],
    setActiveId: vi.fn(),
  }),
}));

function view(selection?: { folder_ids: string[]; doc_keys: string[] }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <StartResearchForm onStarted={vi.fn()} selection={selection} />
    </QueryClientProvider>,
  );
}

const LONG = "What does the evidence say about agent memory?";

beforeEach(() => {
  mutate.mockClear();
  pending = false;
  activeId = "p1";
  projectsLoading = false;
});

describe("StartResearchForm", () => {
  it("refuses a too-short question and says how short it is", () => {
    view();
    fireEvent.change(screen.getByLabelText("研究问题"), { target: { value: "why" } });
    expect(screen.getByText("至少 10 个字符")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始研究" })).toBeDisabled();
  });

  it("refuses to submit with no project, and names that as the reason", () => {
    activeId = undefined;
    view();
    fireEvent.change(screen.getByLabelText("研究问题"), { target: { value: LONG } });
    expect(screen.getByRole("button", { name: "开始研究" })).toBeDisabled();
    expect(screen.getByText(/请先创建项目/)).toBeInTheDocument();
  });

  it("says it is still loading rather than blaming the user for a missing project", () => {
    activeId = undefined;
    projectsLoading = true;
    view();
    expect(screen.getByText("正在加载项目…")).toBeInTheDocument();
    expect(screen.queryByText(/请先创建项目/)).not.toBeInTheDocument();
  });

  it("posts the question with the run's settings", () => {
    view();
    fireEvent.change(screen.getByLabelText("研究问题"), { target: { value: LONG } });
    fireEvent.click(screen.getByRole("button", { name: "开始研究" }));
    expect(mutate).toHaveBeenCalledWith(
      {
        project_id: "p1",
        question: LONG,
        depth: "balanced",
        corpus_mode: false,
        // 研究表单默认启用计划审核，与接口默认值相反，
        // 详见 AGENTS.md 中三个 skip_plan_gate 默认值的区别；
        // 因此请求体必须显式发送该字段。
        skip_plan_gate: false,
      },
      expect.anything(),
    );
  });

  it("sends a selection from the Corpus page, locked to corpus-only", () => {
    view({ folder_ids: ["f1"], doc_keys: ["k1", "k2"] });
    expect(screen.getByRole("checkbox", { name: /仅使用选中的文件/ })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("研究问题"), { target: { value: LONG } });
    fireEvent.click(screen.getByRole("button", { name: "开始研究" }));
    expect(mutate).toHaveBeenCalledWith(
      expect.objectContaining({
        corpus_mode: true,
        corpus_selection: { folder_ids: ["f1"], doc_keys: ["k1", "k2"] },
      }),
      expect.anything(),
    );
    expect(mutate.mock.calls[0][0]).not.toHaveProperty("offline");
  });

  // AC-11（RFC V3.1 §6.3）：仅选中文件限制证据范围，不限制模型运行地点。
  // 此处均为云路由，卡片必须说明摘录会送达提供方，
  // 不能借用离线开关关于本地运行的保证。
  it("says a selection on cloud routes still sends excerpts to the provider", () => {
    view({ folder_ids: [], doc_keys: ["k1"] });
    const card = screen
      .getByRole("checkbox", { name: /仅使用选中的文件/ })
      .closest("label")!.textContent!;
    expect(card).toMatch(/文件的摘录会发送给已配置的模型提供方/);
    expect(card).not.toMatch(/offline|on this computer|private|never leave|stays? local/i);
    expect(screen.getByRole("checkbox", { name: /离线/ })).not.toBeChecked();
  });

  it("asks for Offline only when the switch is on, and that implies corpus-only", () => {
    view();
    fireEvent.click(screen.getByRole("button", { name: /选项/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /离线/ }));
    fireEvent.change(screen.getByLabelText("研究问题"), { target: { value: LONG } });
    fireEvent.click(screen.getByRole("button", { name: "开始研究" }));
    expect(mutate).toHaveBeenCalledWith(
      expect.objectContaining({ offline: true, corpus_mode: true }),
      expect.anything(),
    );
  });

  it("sends the advanced options a person actually chose", () => {
    view();
    fireEvent.change(screen.getByLabelText("研究问题"), { target: { value: LONG } });
    fireEvent.click(screen.getByRole("radio", { name: "全面" }));
    fireEvent.click(screen.getByRole("button", { name: /选项/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /仅研究已上传语料/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /搜索前先审核研究计划/ }));
    fireEvent.click(screen.getByRole("button", { name: "开始研究" }));
    expect(mutate).toHaveBeenCalledWith(
      expect.objectContaining({
        depth: "comprehensive",
        corpus_mode: true,
        // 用户取消勾选后，本次研究跳过计划关卡。
        skip_plan_gate: true,
      }),
      expect.anything(),
    );
  });

  it("names departures from the default on the closed disclosure, and only those", () => {
    view();
    fireEvent.click(screen.getByRole("button", { name: /选项/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /仅研究已上传语料/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /搜索前先审核研究计划/ }));
    fireEvent.click(screen.getByRole("button", { name: /选项/ }));
    expect(screen.getByText(/仅限语料库 · 跳过计划审核/)).toBeInTheDocument();
  });

  it("says the run will stop at both gates when the defaults are untouched", () => {
    view();
    expect(screen.getByText(/先暂停等待计划审核/)).toBeInTheDocument();
  });

  it("cannot be submitted twice while the first request is in flight", () => {
    pending = true;
    view();
    fireEvent.change(screen.getByLabelText("研究问题"), { target: { value: LONG } });
    const button = screen.getByRole("button", { name: /启动中/ });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    fireEvent.submit(button.closest("form")!);
    expect(mutate).not.toHaveBeenCalled();
  });

  it("says what is happening the moment the run is being opened", () => {
    pending = true;
    view();
    expect(screen.getByText(/正在打开研究工作区/)).toBeInTheDocument();
  });
});
