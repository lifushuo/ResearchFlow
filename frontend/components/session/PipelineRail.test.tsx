import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PipelineRail } from "./PipelineRail";
import type { AgentEvent, SessionStatus } from "@/lib/types";

/** 状态栏在规划器与执行器间新增计划审核节点（internal/07 Phase 4）。测试关注截图不易发现的问题：后续编号全部后移，跳过关卡的旧会话不能在完成流程中间留下永久待办节点。 */

const event = (agent: AgentEvent["agent"]): AgentEvent => ({ type: "agent_log", agent });

function renderRail(status: SessionStatus, events: AgentEvent[] = []) {
  return render(<PipelineRail events={events} status={status} />);
}

function nodes() {
  return within(screen.getByRole("list", { name: "流水线进度" })).getAllByRole("listitem");
}

describe("PipelineRail with the design gate", () => {
  it("shows six steps, with Plan review between Planner and Executor", () => {
    renderRail("RUNNING", [event("planner")]);
    const labels = screen
      .getAllByText(/^(规划器|待审核计划|执行器|评审器|综合器|报告审核)$/)
      .map((el) => el.textContent);
    expect(labels).toEqual([
      "规划器",
      "待审核计划",
      "执行器",
      "评审器",
      "综合器",
      "报告审核",
    ]);
  });

  it("numbers the nodes 1..6 with no duplicate and no gap", () => {
    // 所有节点尚未开始，因此显示编号而不是对勾。
    renderRail("PENDING");
    for (const n of ["1", "2", "3", "4", "5", "6"]) {
      expect(screen.getByText(n)).toBeInTheDocument();
    }
  });

  it("marks Plan review active and everything downstream pending at the gate", () => {
    renderRail("AWAITING_PLAN", [event("planner")]);
    const [planner, , executor] = nodes();

    // 到达计划关卡意味着规划器已经完成。
    expect(within(planner).getByText("已完成")).toBeInTheDocument();
    // 无论历史事件到过哪个智能体，关卡后的步骤此时都尚未执行。
    expect(within(executor).getByText("待开始")).toBeInTheDocument();
    expect(screen.getByText("等待你审核研究计划")).toBeInTheDocument();
  });

  it("does not leave Plan review pending on a run that skipped the gate", () => {
    // 防止已经完成的会话（包括功能引入前创建的会话）
    // 在流程中间出现永久未完成的计划审核节点，
    // 让用户误以为研究卡住，而不是根本未使用该功能。
    renderRail("COMPLETED", [event("synthesizer")]);
    expect(screen.queryByText("等待你审核研究计划")).not.toBeInTheDocument();
    // 六个对勾表示全部智能体及两个审核步骤都已结束。
    expect(screen.getAllByText("✓")).toHaveLength(6);
  });

  it("keeps the draft gate as the active step at AWAITING_APPROVAL", () => {
    renderRail("AWAITING_APPROVAL", [event("synthesizer")]);
    expect(screen.getByText("等待你批准报告")).toBeInTheDocument();
    expect(screen.queryByText("等待你审核研究计划")).not.toBeInTheDocument();
  });
});
