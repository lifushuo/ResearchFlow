import { render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import GEPAJobAuditPanel from "./GEPAJobAuditPanel";

it("连接异常显示已分类的未知槽位，不显示为成功待终验", async () => {
  api.audit.mockResolvedValue({job_id:"gepa-failed",mode:"live",state:"failed",phase:"finished",limits:{max_evals:64,proposals:4,seed:42},
    budget:{configured:{total_cap:324},used_total:40,remaining_total:284,unknown:["planner-000023"],unknown_details:[{slot_id:"planner-000023",role:"planner",cause_type:"OpenAIConnectionError"}]},
    journal:{available:true,error:null,event_count:1,events_truncated:false,observation_count:16,reflection_count:1,observed_tokens:{},candidates:[],events:[{sequence:1,kind:"interrupted",error_type:"RequestOutcomeUnknown",slot_id:"planner-000023",role:"planner",cause_type:"OpenAIConnectionError"}]}});
  render(<GEPAJobAuditPanel jobId="gepa-failed" refresh="error" />);
  const alert=await screen.findByRole("alert");
  expect(alert.textContent).toContain("planner-000023");
  expect(alert.textContent).toContain("OpenAIConnectionError");
  expect(screen.queryByText("门禁 待终验")).toBeNull();
});

const api = vi.hoisted(() => ({ audit: vi.fn() }));
vi.mock("@/lib/evaluation-evolution", () => ({
  evolutionGEPAAudit: api.audit,
  evolutionError: (e: unknown) => String(e),
}));

it("只展示日志真实观测，不虚构 Pareto 父代选择", async () => {
  api.audit.mockResolvedValue({
    job_id: "gepa-" + "a".repeat(24), mode: "demo", state: "succeeded", phase: "finished",
    dataset_sha256: null, holdout_sha256: null,
    frozen: {metric_version:"v3",metric_rules_sha256:"abc",baseline_sha256:"def",min_gain:0.05,routes:{pipeline:{planner:"fake:planner"},judge:"fake:judge",reflection:"fake:reflection"},request_timeouts:{planner:90,judge:240,reflection:90}},
    limits: { max_evals: 64, proposals: 4, seed: 42 },
    journal: {
      available: true, error: null, event_count: 3, events_truncated: false,
      observed_tokens: { planner_input: 120, planner_output: 15, judge_input: 30, judge_output: 7 },
      observation_count: 1, reflection_count: 1, holdout_observed: false,
      candidates: [{ sha256: "a".repeat(64), samples: 1, measured: 1, errors: 0,
        splits: { train: { count: 1, measured: 1, mean_coverage: 0.75 } } }],
      events: [{ sequence: 1, kind: "observation", split: "train", prompt_sha256: "a".repeat(64), coverage: 0.75, case_key: "test-hash", task_count: 3 },
        { sequence: 2, kind: "reflection_returned", response_sha256: "b".repeat(64) },
        { sequence: 3, kind: "candidate_frozen", candidate_sha256: "c".repeat(64), accepted_for_holdout: true }],
    },
    holdout_policy: "one_web_live_exposure_per_frozen_holdout", full_report_ab: "manual",
  });
  render(<GEPAJobAuditPanel jobId={"gepa-" + "a".repeat(24)} refresh="v1" />);
  await waitFor(() => expect(screen.getByText(/已观测评分 1 次/)).toBeTruthy());
  expect(screen.getByText(/Pareto 父代选择未被当前日志采集/)).toBeTruthy();
  expect(screen.getByText(/输入 150 · 输出 22/)).toBeTruthy();
  expect(screen.getByText("75.0%")).toBeTruthy();
  expect(screen.getByText(/Planner 90s · Judge 240s · Reflection 90s/)).toBeTruthy();
  expect(api.audit).toHaveBeenCalledTimes(1);
});
