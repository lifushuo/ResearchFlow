import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { JudgeProgress, ReportEvaluationState } from "./JudgeProgress";

describe("评分口径", () => {
  it("完整记录不等于完整评分，两个覆盖率各有分母", () => {
    render(<JudgeProgress progress={{ case_id: "case", trace_complete: true, successful_batches: 14, attempted_batches: 15, total_batches: 19, judged_claims_total: 56, total_claims: 74, unjudged_claims_total: 18, full_judge_coverage: 56 / 74, attempted_judge_coverage: 56 / 60, new_requested_batches: 15, successful_requests: 14, valid_result_batches: 14, scoring_complete: false }} />);
    expect(screen.getByText(/成功批次 14 \/ 19 · 已尝试批次 15 \/ 19/)).toBeVisible();
    expect(screen.getByText(/全量 Claim 覆盖率 75.68% · 已尝试范围覆盖率 93.33%/)).toBeVisible();
    expect(screen.getByText(/全量判定进度 56 \/ 74 · 未判定 18/)).toBeVisible();
    expect(screen.getByText(/全量评分尚未确认完成/)).toBeVisible();
    expect(screen.queryByText(/已完成批次 15/)).not.toBeInTheDocument();
  });
  it("旧记录不使用已尝试数量冒充全量", () => {
    render(<JudgeProgress progress={{ trace_complete: true, observed_batches: 15, observed_judge_coverage: 56 / 60 }} />);
    expect(screen.getByText(/全量 Claim 覆盖率 未测量 · 已尝试范围覆盖率 93.33%/)).toBeVisible();
    expect(screen.getByText(/历史记录缺少全量输入/)).toBeVisible();
  });
  it("保存、评分、恢复与门禁分别显示", () => {
    render(<ReportEvaluationState state={{ case_id: "case", research: "draft_saved", judge: "partial", stop_reason: "judge_deadline:request", total_claims: 74, judged_claims: 56, committed_claims: 56, committed_batches: 14, resume: "artifacts_validated", evaluation: "incomplete", quality_gate: "not_passed" }} />);
    expect(screen.getByText("研究草稿：已生成并保存")).toBeVisible();
    expect(screen.getByText(/Judge 评分：部分完成 · 评分截止原因：阶段预算耗尽/)).toBeVisible();
    expect(screen.getByText(/最终完整评测：未完成 · 质量门禁：不放行/)).toBeVisible();
    expect(screen.getByText(/保存产物只读校验通过/)).toBeVisible();
  });
});
