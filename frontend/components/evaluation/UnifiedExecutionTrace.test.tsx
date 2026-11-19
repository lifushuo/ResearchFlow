import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import UnifiedExecutionTrace from "./UnifiedExecutionTrace";
import type { UnifiedTrace } from "@/lib/evaluation-diagnostics";

const trace: UnifiedTrace = {
  sources: [{ id: "G0", sha256: "source-hash", run_id: "pipe-test", case_id: "case", status: "recorded" }],
  coverage: { loaded_events: 3, content_events: 2, content_truncated_events: 1, not_collected: ["旧运行正文没有记录"], note: "统一展示已采集的脱敏记录。" },
  events: [
    { evidence_id: "G0:1", source: "G", name: "llm.planner", status: "ok", details: { value: { input: "用户的问题", output: "任务拆解" } } },
    { evidence_id: "J0:2", source: "J", event: "batch_finished", status: "error" },
    { evidence_id: "E0:3", source: "E", name: "result.save", status: "error", details: { value: { error: "PermissionError" }, truncated: true } },
  ],
};

describe("统一执行轨迹", () => {
  it("同时展示实际阶段并允许展开脱敏正文，不把缺失正文说成完整", () => {
    render(<UnifiedExecutionTrace trace={trace} />);
    expect(screen.getByText(/其中 2 条包含脱敏上下文，1 条上下文被截断/)).toBeVisible();
    fireEvent.click(screen.getByText(/\[G0:1\]/));
    expect(screen.getByText(/用户的问题/)).toBeVisible();
    fireEvent.click(screen.getByText(/\[J0:2\]/));
    expect(screen.getByText("此事件未采集正文；历史内容无法补回。")).toBeVisible();
    fireEvent.change(screen.getByRole("combobox", { name: "统一轨迹阶段" }), { target: { value: "E" } });
    expect(screen.getByText(/\[E0:3\]/)).toBeVisible();
    expect(screen.queryByText(/\[G0:1\]/)).not.toBeInTheDocument();
  });

  it("按页查看全部已加载事件，切换阶段回到第一页", () => {
    render(<UnifiedExecutionTrace trace={{ ...trace, events: Array.from({ length: 85 }, (_, i) => ({ evidence_id: `G0:${i}`, source: "G", name: `call-${i}` })) }} />);
    expect(screen.getByText(/call-39 ·/)).toBeVisible();
    expect(screen.queryByText(/call-40 ·/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "下一页轨迹" }));
    expect(screen.getByText(/call-40 ·/)).toBeVisible();
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "G" } });
    expect(screen.getByText(/call-0 ·/)).toBeVisible();
  });

  it("导出当前脱敏轨迹，不发起网络或模型请求", () => {
    const create = vi.fn(() => "blob:fixture");
    const revoke = vi.fn();
    vi.stubGlobal("URL", { createObjectURL: create, revokeObjectURL: revoke });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    try {
      render(<UnifiedExecutionTrace trace={trace} />);
      fireEvent.click(screen.getByRole("button", { name: "导出已加载的脱敏轨迹" }));
      expect(create).toHaveBeenCalledOnce(); expect(click).toHaveBeenCalledOnce();
      expect(revoke).toHaveBeenCalledWith("blob:fixture");
    } finally { click.mockRestore(); vi.unstubAllGlobals(); }
  });
});
