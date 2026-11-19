import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";

import HistoryPage from "./page";

const mocks = vi.hoisted(() => ({ runs: vi.fn() }));
vi.mock("@/components/ActiveProject", () => ({
  useActiveProject: () => ({ activeId: "original-project", active: { name: "Original project" }, projects: [] }),
}));
vi.mock("@/hooks/runs", () => ({ useRuns: mocks.runs }));
vi.mock("@/components/history/SessionHistory", () => ({ SessionHistory: () => null }));
vi.mock("@/components/runs/RunCard", () => ({ RunCard: ({ run }: { run: { question: string } }) => <span>{run.question}</span> }));

beforeEach(() => mocks.runs.mockReturnValue({ data: [
  { id: "1", question: "Original evidence question", status: "COMPLETED", has_artifact: false },
  { id: "2", question: "已核验的研究", status: "COMPLETED", has_artifact: true },
], isLoading: false, isError: false }));

it("中文成果筛选依据原 has_artifact 字段，不把 completed 等同于已有成果", () => {
  render(<HistoryPage />);
  expect(screen.getByRole("heading", { name: "历史记录" })).toBeInTheDocument();
  expect(screen.getByText("Original evidence question")).toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "研究成果" }), { target: { value: "artifact" } });
  expect(screen.queryByText("Original evidence question")).not.toBeInTheDocument();
  expect(screen.getByText("已核验的研究")).toBeInTheDocument();
  expect(within(screen.getByRole("combobox", { name: "研究成果" })).getByRole("option", { name: "已有可验证成果" })).toHaveValue("artifact");
});

it("中文归档与范围选择继续向查询传递原始值", () => {
  render(<HistoryPage />);
  fireEvent.click(screen.getByRole("button", { name: "已归档" }));
  expect(mocks.runs).toHaveBeenLastCalledWith("original-project", true);
  fireEvent.change(screen.getByRole("combobox", { name: "项目范围" }), { target: { value: "all" } });
  expect(mocks.runs).toHaveBeenLastCalledWith(null, true);
});
