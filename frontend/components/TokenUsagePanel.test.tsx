import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TokenUsagePanel } from "./TokenUsagePanel";
import type { TokenUsage } from "@/lib/token-usage";

const usage: TokenUsage = {
  schema_version: 1, scope: "observed_application_calls", calls: 3, missing_usage_calls: 1,
  collection_incomplete: false, known_input_tokens: 1200, known_output_tokens: 80, known_total_tokens: 1280,
  models: [
    { kind: "research", llm_mode: "real", served_model: "deepseek-chat", configured_model: "custom:deepseek-chat", roles: ["planner", "executor"], calls: 2, failed_calls: 0, missing_usage_calls: 0, known_input_tokens: 1200, known_output_tokens: 80, known_total_tokens: 1280 },
    { kind: "judge", llm_mode: "real", served_model: "qwen4-max", configured_model: "custom:qwen4-max", roles: ["judge"], calls: 1, failed_calls: 1, missing_usage_calls: 1, known_input_tokens: null, known_output_tokens: null, known_total_tokens: null },
  ],
};

describe("per model Token usage", () => {
  it("shows model, purpose, directions, totals and unknown usage separately", () => {
    render(<TokenUsagePanel usage={usage} />);
    expect(screen.getByText("deepseek-chat")).toBeInTheDocument();
    const judge = screen.getByText("qwen4-max").closest("tr")!;
    expect(within(judge).getAllByText("未披露")).toHaveLength(3);
    expect(screen.getByText(/上述数字仅为已披露部分/)).toBeInTheDocument();
    expect(screen.getByText("1,280", { selector: "strong" })).toBeInTheDocument();
  });

  it("does not infer old per-model counts from an absent ledger", () => {
    render(<TokenUsagePanel />);
    expect(screen.getByText(/历史总 Token 不能倒推/)).toBeInTheDocument();
  });

  it("labels scripted usage without claiming real provider calls", () => {
    const fake = { ...usage, models: [{ ...usage.models[0], llm_mode: "fake" as const }] };
    render(<TokenUsagePanel usage={fake} />);
    expect(screen.getByText("脚本模型（未调用供应商）")).toBeInTheDocument();
  });
});
