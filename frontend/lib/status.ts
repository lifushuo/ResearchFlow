import type { SessionStatus } from "./types";

/**
 * 列表、卡片和详情共用一套状态文案（internal/07 Phase 7）。
 *
 * 此前，`StatusBadge` 的 CONFIG 与历史页的 FILTERS 各维护一份状态。
 * 新增 `AWAITING_PLAN` 时只更新了徽标，导致等待计划审核的会话能显示却无法筛选；
 * 而这恰好是用户浏览列表时最需要找到的会话。
 *
 * `Record<SessionStatus, …>` 要求新增联合类型成员时同步补全此处，否则类型检查失败。
 * 下游均从本对象派生状态信息，避免重复定义。
 */
export interface StatusMeta {
  /** CSS 设计令牌名称（不含 `--`）。状态由文字说明，颜色仅作辅助。 */
  token: string;
  label: string;
  pulse?: boolean;
  /** 此状态是否在等待用户操作。 */
  needsYou?: boolean;
}

export const STATUS_META: Record<SessionStatus, StatusMeta> = {
  PENDING: { token: "text-muted", label: "排队中" },
  RUNNING: { token: "info", label: "运行中", pulse: true },
  // 两个人工审核关卡都在等待用户，紧急程度相同，因此统一使用琥珀色；
  // 具体需要审核什么由文字说明，避免不同颜色暗示并不存在的优先级差异。
  AWAITING_PLAN: { token: "warning", label: "待审核计划", needsYou: true },
  AWAITING_APPROVAL: { token: "warning", label: "待审核报告", needsYou: true },
  COMPLETED: { token: "success", label: "已完成" },
  FAILED: { token: "danger", label: "失败" },
};

/** 按研究流程中的流转顺序排列全部状态。 */
export const STATUS_ORDER: SessionStatus[] = [
  "PENDING",
  "RUNNING",
  "AWAITING_PLAN",
  "AWAITING_APPROVAL",
  "COMPLETED",
  "FAILED",
];

export function statusMeta(status: SessionStatus): StatusMeta {
  return STATUS_META[status] ?? STATUS_META.PENDING;
}
