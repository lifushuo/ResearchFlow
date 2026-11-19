import { describe, expect, it } from "vitest";
import { explainDiagnostic } from "./evaluation-explanation";
import type { Diagnostic } from "./evaluation-diagnostics";

const base = {
  dataset_id: "report-benchmark", state: "failed", has_issue: true, reason: "runner_timeout", elapsed_seconds: 420.189,
  failure_phase: "judge_request", next_actions: [], candidates: [], judge_traces: [{ events: [{ event: "batch_started", batch_no: 12, seq: 46 }] }],
  execution_observation: { started_at: null, finished_at: null, exit_code: 124, timed_out: true },
  timeline: { stages: [{ name: "研究图", duration_s: 152.296201 }, { name: "独立评分", duration_s: 265.617586, recorded_wait_s: 165.150832, completed_request_s: 94.442298, finished_batches: 11 }] },
} as unknown as Diagnostic;

describe("通俗归因说明", () => {
  it("根据实际阶段显示研究结束、评分未完成和已记录等待，不把早期工具错误当根因", () => {
    const text = explainDiagnostic(base);
    expect(text.title).toBe("评测达到总时限，被系统停止");
    expect(text.summary).toContain("研究流程已经结束");
    expect(text.facts).toContain("其中，明确记录的批次间等待占 165.15 秒。");
    expect(text.facts.some((x) => x.includes("第 12 批"))).toBe(true);
    expect(text.boundary).toContain("不能据此认定模型额度不足");
  });
  it("没有明确等待事件时不编造等待时长", () => {
    const row = { ...base, failure_phase: "judge_wait_inferred", timeline: { ...base.timeline!, stages: [{ name: "独立评分", duration_s: 100, finished_batches: 2, basis: "legacy" }] } };
    const text = explainDiagnostic(row);
    expect(text.facts.some((x) => x.includes("等待占"))).toBe(false);
    expect(text.boundary).toContain("推断");
  });
  it("区分整轮总时限与最后一次评分才开始的时间", () => {
    const row = { ...base, execution_observation: { ...base.execution_observation, finished_at: "2026-11-15T00:00:06Z" }, judge_traces: [{ trace_complete: false, events: [{ seq: 1, event: "batch_started", batch_no: 12, timestamp: "2026-11-15T00:00:00Z" }] }] };
    expect(explainDiagnostic(row).facts.some((x) => x.includes("6.00 秒后") && x.includes("不等于单次模型请求超时"))).toBe(true);
  });
  it("多案例按时间定位最后的评分，不采用文件列表的先后顺序", () => {
    const row = { ...base, judge_traces: [
      { trace_complete: false, events: [{ seq: 5, event: "batch_started", batch_no: 12, timestamp: "2026-11-15T00:00:10Z" }] },
      { trace_complete: true, events: [{ seq: 2, event: "batch_started", batch_no: 2, timestamp: "2026-11-15T00:00:00Z" }] },
    ] };
    expect(explainDiagnostic(row).facts.some((x) => x.includes("第 12 批"))).toBe(true);
  });
  it("其他数据集、规划失败和正常运行不套用报告评分超时结论", () => {
    expect(explainDiagnostic({ ...base, dataset_id: "judge-calibration" }).summary).not.toContain("研究流程已经结束");
    expect(explainDiagnostic({ ...base, failure_phase: "planner" }).summary).toContain("研究任务规划");
    expect(explainDiagnostic({ ...base, has_issue: false }).title).toBe("本轮没有登记执行故障");
  });
});
