import { readFileSync } from "node:fs";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import GEPAJobsPanel from "./GEPAJobsPanel";
import { QualityComparison } from "./GEPAJobAuditPanel";

const api=vi.hoisted(() => ({ list:vi.fn(),capabilities:vi.fn(),start:vi.fn(),result:vi.fn(),audit:vi.fn(),cancel:vi.fn() }));
vi.mock("@/lib/evaluation-evolution",()=>({ evolutionGEPAJobs:api.list,evolutionGEPACapabilities:api.capabilities,evolutionStartGEPA:api.start,evolutionGEPAResult:api.result,evolutionGEPAAudit:api.audit,evolutionCancelGEPA:api.cancel,evolutionError:(e:unknown)=>String(e) }));

beforeEach(()=>{vi.clearAllMocks();api.capabilities.mockResolvedValue({live_enabled:false});api.list.mockResolvedValue({jobs:[]});});

// In the free acceptance command this consumes the ACTUAL subprocess/API result.
// Ordinary unit-suite runs without that artifact keep only the portable contract test.
it.skipIf(!process.env.GEPA_ACCEPTANCE_ARTIFACT)("显示实际 API→Worker→v3 CLI 产物的五项指标、请求账本和待终验",async()=>{
  const artifact=JSON.parse(readFileSync(process.env.GEPA_ACCEPTANCE_ARTIFACT!,"utf8"));
  api.list.mockResolvedValue({jobs:[artifact.job]});
  api.result.mockResolvedValue(artifact.result);
  api.audit.mockResolvedValue(artifact.audit);
  render(<GEPAJobsPanel />);
  fireEvent.click(await screen.findByRole("button",{name:artifact.job.job_id}));
  await screen.findByLabelText("v3 质量对比");
  expect(screen.getByText(/Fake v3 验收（零模型 API 费用/)).toBeTruthy();
  expect(screen.queryByText("真实 Planner 评测")).toBeNull();
  for (const metric of ["Coverage","Specificity","Plan Quality","Redundancy","v3 Quality Score"]){expect(screen.getByText(metric)).toBeTruthy();}
  expect(screen.getAllByText(/搜索完成待终验/).length).toBeGreaterThan(0);
  const budget=screen.getByLabelText("GEPA 请求预算");
  expect(budget.textContent).toContain(`已使用 ${artifact.audit.budget.used_total}`);
  expect(budget.textContent).toContain(`剩余 ${artifact.audit.budget.remaining_total}`);
  expect(screen.getByText(/Reflection Token 未记录/)).toBeTruthy();
  expect(screen.getByText(`Candidate SHA：${artifact.result.candidate_sha256}`)).toBeTruthy();
  expect(screen.getByText(/Planner：fake:planner/)).toBeTruthy();
  expect(api.start).not.toHaveBeenCalled();
});

it("旧记录缺失的质量子分显示未测量，不补零",()=>{
  render(<QualityComparison comparison={{baseline:null,candidate:null,quality_delta:null,same_prompt:false}} />);
  expect(screen.getAllByText("未测量")).toHaveLength(15);
});

it("默认固定v3，烟雾预算与API请求上限分开显示",async()=>{
  render(<GEPAJobsPanel />);
  await waitFor(()=>expect(api.list).toHaveBeenCalled());
  expect((screen.getByLabelText("GEPA 评估指标") as HTMLSelectElement).value).toBe("v3");
  expect(screen.getByText(/实际请求上限 41 次/)).toBeTruthy();
  expect(screen.getByText(/冻结 min_gain=0.05/)).toBeTruthy();
  expect(api.start).not.toHaveBeenCalled();
});
