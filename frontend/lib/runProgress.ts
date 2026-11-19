import { AGENTS, latestAgentOrder, taskProgress } from "./pipeline";
import type { AgentEvent, RunStatus } from "./types";

/** 根据后端实际报告的状态推导研究正在做什么。
  * 此处不能显示百分比：后端只提供离散事件和研究状态，没有完成度，进度条会虚构分母，
  * 违背可验证性承诺。阶段列表只标记已观察到的步骤；任务数量来自规划器和执行器事件，
  * 规划器尚未报告总数时不显示分母。
  * 纯函数与组件分离，便于在不渲染 DOM 的情况下测试中途失败和从排队开始观察的情况。 */

export type StageState =
  /** 尚未到达。 */
  | "pending"
  /** 正在进行。 */
  | "active"
  /** 已观察到完成。 */
  | "done"
  /** 正在等待用户操作。 */
  | "waiting"
  /** 研究结束时尚未执行此步骤。 */
  | "stopped";

export interface Stage {
  id: string;
  agentName: string;
  actionLabel: string;
  label: string;
  state: StageState;
  /** 本阶段已知的事实，或 null；不能猜测。 */
  detail: string | null;
}

/** 按流水线顺序列出四个引擎智能体，并给出便于理解的名称。 */
const AGENT_STAGES: { id: string; agentName: string; actionLabel: string; label: string }[] = [
  {
    id: "planner",
    agentName: "规划器",
    actionLabel: "规划研究",
    label: "Planner: Planning the research",
  },
  {
    id: "executor",
    agentName: "执行器",
    actionLabel: "搜索来源并收集证据",
    label: "Executor: Searching sources and gathering evidence",
  },
  {
    id: "critic",
    agentName: "评审器",
    actionLabel: "核查证据",
    label: "Critic: Checking the evidence",
  },
  {
    id: "synthesizer",
    agentName: "综合器",
    actionLabel: "撰写报告草稿",
    label: "Synthesizer: Drafting the report",
  },
];

export interface ProgressInput {
  status: RunStatus | string;
  events: AgentEvent[];
  /** 此研究是否启用了计划审核关卡；跳过关卡的研究不应显示这个步骤。 */
  planGate: boolean;
  /** 已有 PLAN 审核记录时为 true。 */
  planDecided: boolean;
}

/** 生成研究阶段清单。与 PipelineRail 一样，关卡暂停或终态对相邻智能体状态具有最高解释权，只有运行中间态才参考事件流；否则回放积压事件可能让停在计划关卡的研究误显示执行器正在工作。 */
export function deriveStages({ status, events, planGate, planDecided }: ProgressInput): Stage[] {
  const order = latestAgentOrder(events);
  const running = status === "PENDING" || status === "RUNNING";
  const atPlanGate = status === "AWAITING_PLAN";
  const atReview = status === "AWAITING_REVIEW";
  const completed = status === "COMPLETED";
  const ended = status === "FAILED" || status === "CANCELLED";
  const agentsDone = atReview || completed;

  const agentState = (i: number): StageState => {
    if (agentsDone) return "done";
    // 停在计划关卡时，规划器已完成，后续步骤尚未运行，
    // 无论回放的历史事件中最靠后的智能体是什么。
    if (atPlanGate) return i === 0 ? "done" : "pending";
    if (ended) {
      if (order < 0) return "stopped";
      if (i < order) return "done";
      return "stopped";
    }
    if (order < 0) return i === 0 && running ? "active" : "pending";
    if (i < order) return "done";
    if (i === order) return running ? "active" : "done";
    return "pending";
  };

  const { done, total } = taskProgress(events);

  const stages: Stage[] = [];
  AGENT_STAGES.forEach((stage, i) => {
    const state = agentState(i);
    let detail: string | null = null;
    // 只有执行器显示数量，且必须先由规划器给出任务总数；
    // 单独显示 done 会让用户误以为知道距离完成还有多少。
    if (stage.id === "executor" && total > 0 && (state === "active" || state === "done")) {
      detail = `已收集 ${Math.min(done, total)} / ${total} 项研究任务的证据`;
    }
    stages.push({ ...stage, state, detail });

    // 计划关卡位于规划器与执行器之间，且仅在实际启用时绘制；
    // 不能在跳过关卡的研究中展示一个从未存在的步骤。
    if (i === 0 && planGate) {
      stages.push({
        id: "plan-gate",
        agentName: "待审核计划",
        actionLabel: "请审核研究计划",
        label: "计划审核：等待你审核",
        state: atPlanGate
          ? "waiting"
          : planDecided || agentsDone || order >= 1
            ? "done"
            : ended
              ? "stopped"
              : "pending",
        detail: atPlanGate ? "批准计划后开始搜索" : null,
      });
    }
  });

  stages.push({
    id: "review",
    agentName: "最终审核",
    actionLabel: "请审核研究报告",
    label: "最终审核：等待你审核",
    state: completed ? "done" : atReview ? "waiting" : ended ? "stopped" : "pending",
    detail: atReview ? "批准草稿以生成可验证的成果文件" : null,
  });

  return stages;
}

/** 在派生逻辑旁检查其使用的智能体顺序与引擎一致。 */
export const AGENT_STAGE_COUNT = AGENTS.length;
