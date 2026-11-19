import { describe, expect, it } from "vitest";

import { STATUS_META, STATUS_ORDER, statusMeta } from "./status";
import type { SessionStatus } from "./types";

/** 此回归测试防止新增 AWAITING_PLAN 时只更新徽标而漏掉历史筛选，导致待审核会话可显示却无法筛选；这是用户最需要找到的状态。 */

describe("status vocabulary", () => {
  it("covers every status the app can render, in pipeline order", () => {
    // 联合类型新增状态但映射未补全时，STATUS_META 的类型检查会失败。
    // 此处反向检查有序列表是否包含映射中的全部状态。
    expect(new Set(STATUS_ORDER)).toEqual(new Set(Object.keys(STATUS_META)));
    expect(STATUS_ORDER).toHaveLength(Object.keys(STATUS_META).length);
  });

  it("gives every status a distinct human label", () => {
    const labels = STATUS_ORDER.map((s) => statusMeta(s).label);
    expect(new Set(labels).size).toBe(labels.length);
  });

  it("marks both human gates as waiting on the user, and nothing else", () => {
    const waiting = STATUS_ORDER.filter((s) => statusMeta(s).needsYou);
    expect(waiting).toEqual(["AWAITING_PLAN", "AWAITING_APPROVAL"]);
  });

  it("falls back rather than throwing on a status from a newer backend", () => {
    // 客户端可能落后于服务端，未知状态也必须显示中性兜底文案，
    // 不能使历史页崩溃。
    expect(statusMeta("SOMETHING_NEW" as SessionStatus).label).toBe("排队中");
  });
});
