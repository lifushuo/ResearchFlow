import { readFileSync } from "node:fs";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import GEPAFinalPanel from "./GEPAFinalPanel";
import GEPAJobAuditPanel, { QualityComparison } from "./GEPAJobAuditPanel";

const api=vi.hoisted(()=>({preview:vi.fn(),start:vi.fn(),audit:vi.fn()}));
vi.mock("@/lib/evaluation-evolution",()=>({evolutionFinalPreview:api.preview,evolutionStartFinal:api.start,evolutionGEPAAudit:api.audit,evolutionError:(e:unknown)=>String(e)}));
const preview={source_job_id:"gepa-source",candidate_sha256:"candidate-sha",baseline_sha256:"baseline-sha",fake_enabled:true,live_enabled:false,live_disabled_reason:"administrator_has_not_enabled_paid_final",case_count:9,evaluations:18,request_cap:90,quotas:{planner:72,judge:18,reflection:0},min_gain:0.05};
beforeEach(()=>{vi.clearAllMocks();api.preview.mockResolvedValue(preview);api.start.mockResolvedValue({job_id:"gepa-final",task_kind:"planner_final_ab"});});

it("默认免费，预览不启动终验，点击后才创建任务",async()=>{
  const queued=vi.fn();render(<GEPAFinalPanel sourceJobId="gepa-source" onQueued={queued} />);
  const button=await screen.findByRole("button",{name:"运行免费终验流程验收"});
  expect(api.start).not.toHaveBeenCalled();
  expect(screen.getByText(/请求硬上限 90/)).toBeTruthy();
  expect(screen.getByText(/不读取正式 Holdout/)).toBeTruthy();
  fireEvent.click(button);
  await waitFor(()=>expect(api.start).toHaveBeenCalledTimes(1));
  expect(api.start.mock.calls[0][0]).toBe("gepa-source");
  expect(api.start.mock.calls[0][1]).toMatchObject({mode:"demo",consent:null});
  expect(queued).toHaveBeenCalledTimes(1);
});

it("管理员未开启时，即使确认费用也不允许真实启动",async()=>{
  render(<GEPAFinalPanel sourceJobId="gepa-source" onQueued={vi.fn()} />);
  fireEvent.change(await screen.findByLabelText("Planner 终验执行模式"),{target:{value:"live"}});
  fireEvent.click(screen.getByLabelText(/我确认本次独立 Planner 终验/));
  expect((screen.getByRole("button",{name:"确认启动一次真实 Planner 终验"}) as HTMLButtonElement).disabled).toBe(true);
  expect(api.start).not.toHaveBeenCalled();
});

it("付费启动需要独立同意，模式切换清除同意",async()=>{
  api.preview.mockResolvedValue({...preview,live_enabled:true,live_disabled_reason:null});
  render(<GEPAFinalPanel sourceJobId="gepa-source" onQueued={vi.fn()} />);
  const mode=await screen.findByLabelText("Planner 终验执行模式");
  fireEvent.change(mode,{target:{value:"live"}});
  expect((screen.getByRole("button",{name:"确认启动一次真实 Planner 终验"}) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByLabelText(/我确认本次独立 Planner 终验/));
  fireEvent.change(mode,{target:{value:"demo"}});
  fireEvent.change(mode,{target:{value:"live"}});
  expect((screen.getByLabelText(/我确认本次独立 Planner 终验/) as HTMLInputElement).checked).toBe(false);
  fireEvent.click(screen.getByLabelText(/我确认本次独立 Planner 终验/));
  fireEvent.click(screen.getByRole("button",{name:"确认启动一次真实 Planner 终验"}));
  await waitFor(()=>expect(api.start).toHaveBeenCalledTimes(1));
  expect(api.start.mock.calls[0][1]).toMatchObject({mode:"live",consent:"I_CONFIRM_ONE_PLANNER_FINAL_AB_API_COST"});
});

it.skipIf(!process.env.GEPA_FINAL_ACCEPTANCE_ARTIFACT)("展示实际免费终验产物的Holdout五项指标、36次请求和未发布状态",async()=>{
  const artifact=JSON.parse(readFileSync(process.env.GEPA_FINAL_ACCEPTANCE_ARTIFACT!,"utf8"));
  api.audit.mockResolvedValue(artifact.audit);
  render(<><GEPAJobAuditPanel jobId={artifact.job.job_id} refresh="final" /><QualityComparison comparison={artifact.result.comparison} /></>);
  expect(await screen.findByText("v3 五项指标（终验）")).toBeTruthy();
  expect(screen.getByLabelText("GEPA 请求预算").textContent).toContain("已使用 36");
  expect(screen.getByText(/模拟终验：未读取正式 Holdout/)).toBeTruthy();
  expect(screen.getByText(/不运行搜索或 Reflection；通过仍不代表已发布/)).toBeTruthy();
  for(const label of ["Coverage","Specificity","Plan Quality","Redundancy","v3 Quality Score"]) expect(screen.getByText(label)).toBeTruthy();
});
