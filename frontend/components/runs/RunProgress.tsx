"use client";

import { useMemo, useState } from "react";

import { IconList, IconSearch, IconThought } from "@/components/icons";
import { AGENT_TOKEN } from "@/lib/pipeline";
import type { AgentEvent, RunGraph } from "@/lib/types";
import { deriveStages, type Stage, type StageState } from "@/lib/runProgress";

/** 展示研究当前状态、智能体活动及可展开的说明。不显示百分比、进度条或基于用时的推测，因为后端只报告离散事件和状态，不提供完成度。状态推导在 lib/runProgress.ts 中独立测试，此组件只负责绘制。颜色只作辅助，每阶段同时提供文字与读屏说明。 */

const STAGE_COLOR: Record<string, string> = {
  planner: `var(--${AGENT_TOKEN.planner})`,
  executor: `var(--${AGENT_TOKEN.executor})`,
  critic: `var(--${AGENT_TOKEN.critic})`,
  synthesizer: `var(--${AGENT_TOKEN.synthesizer})`,
  "plan-gate": "var(--agent-hitl)",
  review: "var(--agent-hitl)",
};

const STATE_WORD: Record<StageState, string> = {
  pending: "尚未开始",
  active: "进行中",
  done: "已完成",
  waiting: "等待你处理",
  stopped: "未执行",
};

function Marker({ state, color }: { state: StageState; color: string }) {
  if (state === "done") {
    return (
      <span aria-hidden style={{ color }} className="font-mono text-sm leading-none">
        ✓
      </span>
    );
  }
  if (state === "active") {
    return (
      <span
        aria-hidden
        className="spinner"
        style={{ color, width: "0.75rem", height: "0.75rem" }}
      />
    );
  }
  if (state === "waiting") {
    return (
      <span
        aria-hidden
        className="status-marker"
        style={{ backgroundColor: "var(--warning)" }}
      />
    );
  }
  // 未执行与尚未开始是不同事实，须使用不同标记；
  // 前者画为短横线，避免都显示为空方框。
  if (state === "stopped") {
    return (
      <span aria-hidden className="font-mono text-sm leading-none text-text-muted">
        –
      </span>
    );
  }
  return (
    <span
      aria-hidden
      className="status-marker"
      style={{ backgroundColor: "transparent", border: "1px solid var(--border)" }}
    />
  );
}

function StageDetailDrawer({
  stage,
  events,
}: {
  stage: Stage;
  events: AgentEvent[];
}) {
  const agentEvents = useMemo(
    () => events.filter((e) => e.agent === stage.id),
    [events, stage.id],
  );

  // 从智能体事件提取活动说明。
  const thoughts = useMemo(() => {
    const list: { thought: string; message?: string | null; ts?: string | null }[] = [];
    for (const e of agentEvents) {
      const detail = (e.detail ?? {}) as Record<string, unknown>;
      const thought = typeof detail.thought === "string" ? detail.thought.trim() : null;
      if (thought) {
        list.push({ thought, message: e.message, ts: e.ts });
      }
    }
    return list;
  }, [agentEvents]);

  // 提取最近的工具调用及操作。
  const toolCalls = useMemo(() => {
    const calls: { tool: string; args?: unknown; observation?: string | null; message?: string | null }[] = [];
    for (const e of agentEvents) {
      const detail = (e.detail ?? {}) as Record<string, unknown>;
      if (typeof detail.tool === "string") {
        calls.push({
          tool: detail.tool,
          args: detail.args,
          observation: typeof detail.observation === "string" ? detail.observation : null,
          message: e.message,
        });
      }
    }
    return calls;
  }, [agentEvents]);

  const latestThought = thoughts.length > 0 ? thoughts[thoughts.length - 1] : null;
  const recentEvents = agentEvents.slice(-4);

  if (agentEvents.length === 0 && stage.state === "pending") {
    return (
      <div className="mt-2.5 border border-dashed border-border/60 bg-bg-surface/50 p-3 text-xs text-text-muted">
        此智能体尚未开始，运行后将在这里显示活动及说明。
      </div>
    );
  }

  if (agentEvents.length === 0 && stage.state === "active") {
    return (
      <div className="mt-2.5 border border-border bg-bg-surface/60 p-3 text-xs text-text-secondary">
        <div className="flex items-center gap-2">
          <span className="spinner" style={{ width: 10, height: 10 }} />
          <span>{stage.agentName} 正在初始化并处理任务…</span>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-2.5 space-y-2.5 border border-border bg-bg-surface/80 p-3 text-xs">
      {/* 最近的智能体活动说明。 */}
      {latestThought && (
        <div className="border-l-2 border-accent bg-bg-elevated/70 p-2.5">
          <div className="mb-1 flex items-center gap-1.5 font-mono text-[length:var(--text-micro)] font-semibold uppercase tracking-wider text-accent">
            <IconThought className="h-3.5 w-3.5" />
            <span>智能体思考与推理</span>
          </div>
          <p className="whitespace-pre-wrap font-sans text-xs leading-relaxed text-text-primary">
            {latestThought.thought}
          </p>
        </div>
      )}

      {/* 最近的搜索、页面抓取等工具操作。 */}
      {toolCalls.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5 font-mono text-[length:var(--text-micro)] font-semibold uppercase tracking-wider text-text-muted">
            <IconSearch className="h-3 w-3" />
            <span>工具与搜索请求（{toolCalls.length})</span>
          </div>
          <div className="space-y-1">
            {toolCalls.slice(-3).map((call, idx) => (
              <div
                key={idx}
                className="flex items-start gap-2 border border-border/50 bg-bg-base px-2.5 py-1.5 font-mono text-[0.6875rem] text-text-secondary"
              >
                <span className="font-semibold text-accent">{call.tool}:</span>
                <span className="truncate">
                  {typeof call.args === "object" && call.args !== null
                    ? JSON.stringify(call.args)
                    : String(call.args ?? call.message ?? "")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 活动时间线。 */}
      {recentEvents.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5 font-mono text-[length:var(--text-micro)] font-semibold uppercase tracking-wider text-text-muted">
            <IconList className="h-3 w-3" />
            <span>最近的智能体活动</span>
          </div>
          <ul className="space-y-1">
            {recentEvents.map((e, idx) => (
              <li
                key={idx}
                className="flex items-center justify-between gap-2 border-b border-border/20 py-0.5 text-xs text-text-secondary last:border-b-0"
              >
                <span>{e.message || "智能体状态已更新"}</span>
                {e.ts && (
                  <span className="font-mono text-[0.6875rem] text-text-muted">
                    {new Date(e.ts).toLocaleTimeString()}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function StageRow({
  stage,
  events,
  isExpanded,
  onToggle,
}: {
  stage: Stage;
  events: AgentEvent[];
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const color = STAGE_COLOR[stage.id] ?? "var(--text-muted)";
  const dim = stage.state === "pending" || stage.state === "stopped";
  const isAgent = ["planner", "executor", "critic", "synthesizer"].includes(stage.id);

  return (
    <li className="py-2">
      <div className="flex items-start justify-between gap-2.5">
        <div className="flex items-start gap-2.5 min-w-0 flex-1">
          <span className="mt-0.5 flex h-3.5 w-3.5 shrink-0 items-center justify-center">
            <Marker state={stage.state} color={color} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-1.5 text-[0.8125rem]">
              <span
                className="font-mono text-[length:var(--text-micro)] font-bold uppercase tracking-wider px-1.5 py-0.5"
                style={{
                  color: color,
                  backgroundColor: `color-mix(in srgb, ${color} 10%, var(--bg-surface))`,
                  border: `1px solid color-mix(in srgb, ${color} 25%, var(--border))`,
                }}
              >
                {stage.agentName}
              </span>
              <span
                className={`${dim ? "text-text-muted" : "text-text-primary"} ${
                  stage.state === "active" || stage.state === "waiting" ? "font-semibold" : ""
                }`}
              >
                {stage.actionLabel}
              </span>
            </div>
            {stage.detail && (
              <span className="block text-xs text-text-secondary mt-0.5">{stage.detail}</span>
            )}
            <span className="sr-only">{STATE_WORD[stage.state]}</span>
          </div>
        </div>

        {/* 智能体说明与活动的展开开关。 */}
        {isAgent && (
          <button
            type="button"
            onClick={onToggle}
            aria-expanded={isExpanded}
            aria-label={`${isExpanded ? "收起" : "展开"}${stage.agentName}的活动与说明`}
            className="btn btn-ghost shrink-0 px-2 py-0.5 font-mono text-[length:var(--text-micro)] text-text-muted hover:text-text-primary"
          >
            <span className="flex items-center gap-1">
              <span>{isExpanded ? "收起说明" : "说明与活动"}</span>
              <span aria-hidden className="text-[0.625rem]">{isExpanded ? "▲" : "▼"}</span>
            </span>
          </button>
        )}
      </div>

      {/* 展开的详情抽屉。 */}
      {isAgent && isExpanded && (
        <StageDetailDrawer stage={stage} events={events} />
      )}
    </li>
  );
}

export function RunProgress({
  graph,
  events,
  degraded,
}: {
  graph: RunGraph;
  events: AgentEvent[];
  /** 实时流中断时必须明确提示，不能让页面看起来像停滞。 */
  degraded: boolean;
}) {
  const stages = useMemo(
    () =>
      deriveStages({
        status: graph.run.status,
        events,
        planGate: !graph.run.skip_plan_gate,
        planDecided: graph.reviews.some((r) => r.gate === "PLAN"),
      }),
    [graph.run.status, graph.run.skip_plan_gate, graph.reviews, events],
  );

  // 按阶段 ID 保存展开状态。
  const [expandedStages, setExpandedStages] = useState<Record<string, boolean>>({});
  const [expandAll, setExpandAll] = useState(false);

  const toggleStage = (stageId: string) => {
    setExpandedStages((prev) => ({
      ...prev,
      [stageId]: !prev[stageId],
    }));
  };

  const toggleAll = () => {
    const nextState = !expandAll;
    setExpandAll(nextState);
    const updated: Record<string, boolean> = {};
    for (const s of stages) {
      updated[s.id] = nextState;
    }
    setExpandedStages(updated);
  };

  // 原样显示流水线最后报告的信息，
  // 避免前端改述自身无法观察的研究进程。
  const lastMessage = [...events].reverse().find((e) => e.message)?.message ?? null;

  return (
    <section aria-labelledby="run-progress" className="card space-y-3">
      <div className="flex items-center justify-between border-b border-border pb-2.5">
        <h2
          id="run-progress"
          className="font-mono text-[length:var(--text-micro)] font-semibold uppercase tracking-wider text-text-muted"
        >
          研究进展
        </h2>
        <button
          type="button"
          onClick={toggleAll}
          className="btn btn-ghost px-2 py-0.5 font-mono text-[length:var(--text-micro)] text-text-secondary hover:text-text-primary"
        >
          <span className="flex items-center gap-1.5">
            <IconThought className="h-3 w-3 text-accent" />
            <span>{expandAll ? "收起全部智能体说明" : "展开全部智能体说明"}</span>
            <span aria-hidden className="text-[0.625rem]">{expandAll ? "▲" : "▼"}</span>
          </span>
        </button>
      </div>

      <ol className="divide-y divide-border/20">
        {stages.map((s) => (
          <StageRow
            key={s.id}
            stage={s}
            events={events}
            isExpanded={Boolean(expandedStages[s.id])}
            onToggle={() => toggleStage(s.id)}
          />
        ))}
      </ol>

      {lastMessage && (
        <div className="bg-bg-elevated p-2.5 text-xs text-text-secondary">
          <span className="font-mono text-[length:var(--text-micro)] font-semibold uppercase tracking-wider text-text-muted">
            Latest:{" "}
          </span>
          {lastMessage}
        </div>
      )}

      {degraded && (
        <p
          className="mt-3 border p-2 text-xs"
          style={{
            color: "var(--warning)",
            backgroundColor: "var(--warning-soft)",
            borderColor: "var(--warning-line)",
          }}
        >
          实时连接已中断，将自动重连并回放遗漏事件；数据不会丢失，页面也没有卡住。
        </p>
      )}
    </section>
  );
}
