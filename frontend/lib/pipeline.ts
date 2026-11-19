import type { AgentEvent, AgentName } from "./types";

/** 流水线结构及智能体事件流的派生状态（docs/04，docs/35）。 */

export const AGENTS: AgentName[] = ["planner", "executor", "critic", "synthesizer"];

export const AGENT_LABELS: Record<AgentName, string> = {
  planner: "规划器",
  executor: "执行器",
  critic: "评审器",
  synthesizer: "综合器",
};

/** 各智能体强调色的 CSS 设计令牌名称，不包含 --。 */
export const AGENT_TOKEN: Record<AgentName, string> = {
  planner: "agent-planner",
  executor: "agent-executor",
  critic: "agent-critic",
  synthesizer: "agent-synthesizer",
};

/** 从 provider:model 路由中提取模型 ID，供紧凑展示（docs/04，真实模型归属披露）。角色路由尚未解析时返回 —，不能猜测默认模型，遵循未测量不等于零的原则。 */
export function routeModelLabel(route: string | undefined | null): string {
  if (!route) return "—";
  const i = route.indexOf(":");
  return i === -1 ? route : route.slice(i + 1);
}

/** 返回已观察到的最靠后智能体的顺序索引；尚无事件时返回 -1。 */
export function latestAgentOrder(events: AgentEvent[]): number {
  let order = -1;
  for (const e of events) {
    if (e.agent) {
      const i = AGENTS.indexOf(e.agent);
      if (i > order) order = i;
    }
  }
  return order;
}

/** 根据规划器与执行器事件计算任务进度，重写后仍可正确计算。 */
export function taskProgress(events: AgentEvent[]): { done: number; total: number } {
  let total = 0;
  const gathered = new Set<string>();
  for (const e of events) {
    const detail = (e.detail ?? {}) as Record<string, unknown>;
    if (e.agent === "planner" && Array.isArray(detail.tasks)) {
      total = detail.tasks.length;
    }
    if (e.agent === "executor" && typeof e.message === "string" && e.message.startsWith("Gathered")) {
      gathered.add(detail.task_id !== undefined ? String(detail.task_id) : String(gathered.size));
    }
  }
  const done = total ? Math.min(gathered.size, total) : gathered.size;
  return { done, total };
}
