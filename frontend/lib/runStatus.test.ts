import { describe, expect, it } from "vitest";

import { V2_STATUS_META, RUN_STATUS_ORDER, runStatusMeta } from "./runStatus";
import type { RunStatus } from "./types";

/** 研究任务徽标、页头状态句、历史筛选及卡片共用一套状态定义。Record<RunStatus, …> 保证映射包含全部状态；测试进一步保证有序列表也同步包含，避免新增 AWAITING_PLAN 后只改徽标而漏掉筛选。 */

const ALL: RunStatus[] = [
  "PENDING",
  "RUNNING",
  "AWAITING_PLAN",
  "AWAITING_REVIEW",
  "COMPLETED",
  "FAILED",
  "CANCELLED",
];

describe("Run status vocabulary", () => {
  it("covers every status the backend can report, in the order given", () => {
    expect([...RUN_STATUS_ORDER].sort()).toEqual([...ALL].sort());
  });

  it("gives every status both a badge label and a full sentence", () => {
    for (const status of ALL) {
      expect(V2_STATUS_META[status].label).toMatch(/\S/);
      expect(V2_STATUS_META[status].sentence).toMatch(/\S/);
    }
  });

  it("marks exactly the two gates as waiting on a person", () => {
    const needsYou = ALL.filter((s) => V2_STATUS_META[s].needsYou);
    expect(needsYou.sort()).toEqual(["AWAITING_PLAN", "AWAITING_REVIEW"]);
  });

  it("marks exactly the two statuses where the server still has something to say", () => {
    const live = ALL.filter((s) => V2_STATUS_META[s].live);
    expect(live.sort()).toEqual(["PENDING", "RUNNING"]);
  });

  it("never claims a cancelled or failed run produced an artifact", () => {
    expect(V2_STATUS_META.CANCELLED.sentence).not.toMatch(/成果文件/);
    expect(V2_STATUS_META.FAILED.sentence).not.toMatch(/成果文件/);
    expect(V2_STATUS_META.COMPLETED.sentence).toMatch(/成果文件/);
  });

  it("survives a status it has not been taught, without inventing one", () => {
    const meta = runStatusMeta("SOMETHING_NEW");
    expect(meta.label).toBe("something new");
    expect(meta.sentence).toMatch(/尚不认识/);
    // 不能把未知状态静默映射成研究实际并未处于的状态。
    expect(meta.label).not.toBe(V2_STATUS_META.COMPLETED.label);
    expect(meta.needsYou).toBeUndefined();
  });
});
