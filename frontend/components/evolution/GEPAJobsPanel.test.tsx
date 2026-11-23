import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import GEPAJobsPanel from "./GEPAJobsPanel";

const api = vi.hoisted(() => ({
  list: vi.fn(), capabilities: vi.fn(), start: vi.fn(), cancel: vi.fn(), result: vi.fn(), audit: vi.fn(),
}));

vi.mock("@/lib/evaluation-evolution", () => ({
  evolutionGEPAJobs: api.list,
  evolutionGEPACapabilities: api.capabilities,
  evolutionStartGEPA: api.start,
  evolutionCancelGEPA: api.cancel,
  evolutionGEPAResult: api.result,
  evolutionGEPAAudit: api.audit,
  evolutionError: (error: unknown) => String(error),
}));

beforeEach(() => {
  vi.clearAllMocks();
  api.list.mockResolvedValue({ jobs: [] });
  api.capabilities.mockResolvedValue({ demo_enabled: true, live_enabled: false });
  api.start.mockResolvedValue({ job_id: "gepa-" + "a".repeat(24) });
  api.result.mockResolvedValue({ available: false });
  api.audit.mockResolvedValue({ job_id: "gepa-" + "a".repeat(24), mode: "demo", state: "running", phase: "starting", dataset_sha256: null, holdout_sha256: null, limits: { max_evals: 64, proposals: 4, seed: 42 }, journal: { available: false, error: null, event_count: 0, events: [], candidates: [], observed_tokens: {}, observation_count: 0, reflection_count: 0, holdout_observed: false, events_truncated: false }, holdout_policy: "one_web_live_exposure_per_frozen_holdout", full_report_ab: "manual" });
});

it("默认只显示免付费 Demo，点击才发起作业", async () => {
  render(<GEPAJobsPanel />);
  expect(screen.getByText("GEPA 优化实验")).toBeTruthy();
  await waitFor(() => expect(api.capabilities).toHaveBeenCalledTimes(1));
  expect(api.start).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "运行免付费 GEPA Demo" }));
  await waitFor(() => expect(api.start).toHaveBeenCalledTimes(1));
  expect(api.start.mock.calls[0][0]).toMatchObject({ mode: "demo", metric_version: "v3", consent: null });
});

it("真实模式必须勾选费用确认才能提交", async () => {
  api.capabilities.mockResolvedValue({ demo_enabled: true, live_enabled: true });
  render(<GEPAJobsPanel />);
  await waitFor(() => expect(api.capabilities).toHaveBeenCalled());
  fireEvent.change(screen.getByLabelText("GEPA 执行模式"), { target: { value: "live" } });
  const start = screen.getByRole("button", { name: "确认启动真实 GEPA" }) as HTMLButtonElement;
  expect(start.disabled).toBe(true);
  expect(api.start).not.toHaveBeenCalled();
  // 指标选择默认 v2 综合质量；启动体携带所选指标。
  expect((screen.getByLabelText("GEPA 评估指标") as HTMLSelectElement).value).toBe("v3");
  fireEvent.click(screen.getByLabelText(/我确认启动真实 GEPA/));
  expect(start.disabled).toBe(false);
  fireEvent.click(start);
  await waitFor(() => expect(api.start).toHaveBeenCalledTimes(1));
  expect(api.start.mock.calls[0][0]).toMatchObject({ mode: "live", metric_version: "v3" });
});
