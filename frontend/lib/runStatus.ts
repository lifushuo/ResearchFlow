import type { RunStatus } from "./types";

/** 列表、卡片、页头和筛选器共用研究任务的状态文案（internal/07 Phase 7）。会话状态另见 lib/status.ts：研究任务新增 AWAITING_REVIEW、CANCELLED，不含 AWAITING_APPROVAL，不能合并成双方均无法完整产生的状态集合。Record<RunStatus, …> 让缺少状态定义成为类型错误。sentence 用于页头解释，label 用于紧凑徽标；两种长度分别定义，避免表格过长或页头含糊。 */
export interface V2StatusMeta {
  /** CSS 设计令牌名称，不含 --；文字说明状态，颜色仅作辅助。 */
  token: string;
  label: string;
  /** 用于研究任务页头的一句状态解释。 */
  sentence: string;
  /** 是否在等待用户操作。 */
  needsYou?: boolean;
  /** 服务端是否还会继续报告，用于控制闪烁标记与实时订阅。 */
  live?: boolean;
}

export const V2_STATUS_META: Record<RunStatus, V2StatusMeta> = {
  PENDING: {
    token: "text-muted",
    label: "排队中",
    sentence: "排队中，等待工作进程开始处理。",
    live: true,
  },
  RUNNING: {
    token: "info",
    label: "运行中",
    sentence: "研究正在进行，此页会随流水线进展自动更新。",
    live: true,
  },
  // 两个人工关卡都在等待用户，紧急程度相同，因此均使用琥珀色，
  // 具体需要作出哪种决定由文字标签说明。
  AWAITING_PLAN: {
    token: "warning",
    label: "待审核计划",
    sentence: "搜索尚未开始：请先审核并批准研究计划。",
    needsYou: true,
  },
  AWAITING_REVIEW: {
    token: "warning",
    label: "待审核报告",
    sentence: "研究已暂停：请审核报告草稿。",
    needsYou: true,
  },
  COMPLETED: {
    token: "success",
    label: "已批准",
    sentence: "已批准，本次研究已有可验证的成果文件。",
  },
  CANCELLED: {
    token: "text-muted",
    label: "已停止",
    sentence: "已请求停止；已发出的工作会执行到下一个检查点。",
  },
  FAILED: {
    token: "danger",
    label: "失败",
    sentence: "本次研究在生成报告前失败。",
  },
};

/** 按研究流转顺序列出全部状态。 */
export const RUN_STATUS_ORDER: RunStatus[] = [
  "PENDING",
  "RUNNING",
  "AWAITING_PLAN",
  "AWAITING_REVIEW",
  "COMPLETED",
  "CANCELLED",
  "FAILED",
];

/** 客户端不认识的状态也不能导致页面崩溃。滚动发布时服务端可能先新增状态；此时显示原始协议值及未知状态说明，不能猜测为相近状态并造成误报。 */
export function runStatusMeta(status: string): V2StatusMeta {
  return (
    V2_STATUS_META[status as RunStatus] ?? {
      token: "text-muted",
      label: status.replace(/_/g, " ").toLowerCase(),
      sentence: `此页尚不认识本次研究返回的状态（${status}）。`,
    }
  );
}
