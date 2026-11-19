import { AGENT_LABELS, AGENT_TOKEN, AGENTS, latestAgentOrder, routeModelLabel } from "@/lib/pipeline";
import type { AgentEvent, SessionStatus } from "@/lib/types";

type NodeState = "pending" | "active" | "done";
const NODE_STATE_LABEL: Record<NodeState, string> = {
  pending: "待开始",
  active: "进行中",
  done: "已完成",
};

/** 实时流水线状态栏（docs/25）：规划器→计划审核→执行器→评审器→综合器→报告审核。两个审核节点由 status 推导，而非智能体事件；它们是 LangGraph interrupt 检查点，不增加 AgentName。仍须展示人工步骤，避免综合器完成时误导用户认为研究已经结束。两个关卡共用人工审核色，位置、编号和文字共同区分，不依赖颜色，也不暗示不同紧急程度。 */
const REVIEW_COLOR = "var(--agent-hitl)";

function nodeStyle(state: NodeState, color: string) {
  return state === "pending"
    ? {
        borderColor: "var(--border)",
        color: "var(--text-muted)",
        backgroundColor: "var(--bg-elevated)",
      }
    : {
        borderColor: color,
        color,
        backgroundColor: `color-mix(in srgb, ${color} 10%, var(--bg-surface))`,
        animation: state === "active" ? "pulse-ring 1.4s ease-in-out infinite" : undefined,
      };
}

/** 人工关卡使用与智能体相同的视觉结构，但不显示模型行，因为关卡不会调用模型，破折号会被误读成路由尚未解析。 */
function GateNode({
  state,
  index,
  label,
  activeHint,
}: {
  state: NodeState;
  index: number;
  label: string;
  activeHint: string;
}) {
  return (
    <div className="flex min-w-0 flex-1 flex-col items-center gap-1.5">
      <span
        className="flex h-7 w-7 items-center justify-center border font-mono text-xs font-semibold"
        style={nodeStyle(state, REVIEW_COLOR)}
        aria-hidden
      >
        {state === "done" ? "✓" : index}
      </span>
      <span
        className="truncate font-mono text-[0.6875rem] uppercase tracking-wider font-medium"
        style={{ color: state === "pending" ? "var(--text-muted)" : REVIEW_COLOR }}
      >
        {label}
      </span>
      <span className="sr-only">{state === "active" ? activeHint : NODE_STATE_LABEL[state]}</span>
    </div>
  );
}

export function PipelineRail({
  events,
  status,
  modelRouting,
}: {
  events: AgentEvent[];
  status: SessionStatus;
  /** 已解析的角色路由（docs/04）；缺失或 null 时显示 —，不猜测模型，遵循未测量不等于零的原则。 */
  modelRouting?: Record<string, string> | null;
}) {
  const order = latestAgentOrder(events);
  const atPlanGate = status === "AWAITING_PLAN";
  const agentsDone = status === "AWAITING_APPROVAL" || status === "COMPLETED";
  const running = status === "PENDING" || status === "RUNNING";

  // 任何下游步骤开始后，计划关卡即视为已通过；
  // 跳过关卡或在该功能引入前创建的会话也应如此显示，
  // 不能让已经结束的流程中间仍有一个永远待办的节点，
  // 误导用户以为研究卡住。
  //
  // 不能只根据 atPlanGate 推断；否则初始状态为 PENDING、
  // 规划器尚未输出任何内容时，就会错误显示关卡已通过。
  const planGateState: NodeState = atPlanGate
    ? "active"
    : agentsDone || order >= 1
      ? "done"
      : "pending";

  const reviewState: NodeState =
    status === "COMPLETED" ? "done" : status === "AWAITING_APPROVAL" ? "active" : "pending";

  const nodeState = (i: number): NodeState => {
    if (agentsDone) return "done";
    // 停在计划关卡时，规划器已完成，后续步骤尚未运行，
    // 不受事件流中最靠后的智能体影响。
    if (atPlanGate) return i === 0 ? "done" : "pending";
    if (order < 0) return i === 0 && running ? "active" : "pending";
    if (i < order) return "done";
    if (i === order) return running ? "active" : "done";
    return "pending";
  };

  return (
    <ol className="flex items-center gap-1" aria-label="流水线进度">
      {AGENTS.map((agent, i) => {
        const s = nodeState(i);
        const color = `var(--${AGENT_TOKEN[agent]})`;
        // 计划关卡紧接规划器，因此后续节点编号都向后移动一位；
        // 编号由顺序推导而非硬编码，新增阶段时
        // 不会出现两个节点显示同一编号。
        const number = i === 0 ? 1 : i + 2;
        return (
          <li key={agent} className="flex flex-1 items-center gap-1">
            <div className="flex min-w-0 flex-1 flex-col items-center gap-1.5">
              <span
                className="flex h-7 w-7 items-center justify-center border font-mono text-xs font-semibold"
                style={nodeStyle(s, color)}
                aria-hidden
              >
                {s === "done" ? "✓" : number}
              </span>
              <span
                className="truncate font-mono text-[0.6875rem] uppercase tracking-wider font-medium"
                style={{ color: s === "pending" ? "var(--text-muted)" : color }}
              >
                {AGENT_LABELS[agent]}
              </span>
              <span
                className="max-w-full truncate font-mono text-[length:var(--text-micro)] text-text-muted"
                title={modelRouting?.[agent] || "模型尚未解析"}
              >
                {routeModelLabel(modelRouting?.[agent])}
              </span>
              <span className="sr-only">{NODE_STATE_LABEL[s]}</span>
            </div>
            <span
              className="mb-5 h-px flex-1"
              style={{
                backgroundColor:
                  i < order || agentsDone ? color : atPlanGate && i === 0 ? color : "var(--border)",
              }}
              aria-hidden
            />
            {/* 计划审核位于规划器与执行器之间，嵌入规划器的列表项以保持两侧连线连续。 */}
            {i === 0 && (
              <>
                <GateNode
                  state={planGateState}
                  index={2}
                  label="待审核计划"
                  activeHint="等待你审核研究计划"
                />
                <span
                  className="mb-5 h-px flex-1"
                  style={{
                    backgroundColor: planGateState === "done" ? REVIEW_COLOR : "var(--border)",
                  }}
                  aria-hidden
                />
              </>
            )}
          </li>
        );
      })}

      {/* 报告审核沿用相同视觉结构，位于流程末尾。 */}
      <li className="flex flex-1 items-center gap-1">
        <GateNode
          state={reviewState}
          index={AGENTS.length + 2}
          label="报告审核"
          activeHint="等待你批准报告"
        />
      </li>
    </ol>
  );
}
