"use client";

import { useEffect, useState } from "react";

import { formatCost, formatDuration } from "@/lib/format";
import { taskProgress } from "@/lib/pipeline";
import type { AgentEvent, SessionDetail } from "@/lib/types";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="font-mono text-[0.6875rem] font-semibold uppercase tracking-wider text-text-muted">{label}</span>
      <span className="font-mono text-sm font-medium text-text-primary tabular-nums mt-0.5">{value}</span>
    </div>
  );
}

export function StatusBar({
  session,
  events,
  running,
}: {
  session: SessionDetail;
  events: AgentEvent[];
  running: boolean;
}) {
  const start = new Date(session.created_at).getTime();
  // 使用 created_at 作为确定的初始时间，保持服务端与客户端首次渲染一致；
  // 时钟只在异步定时回调中推进，不在副作用函数体内同步更新，
  // 以保持渲染纯粹，避免级联渲染检查报错。
  const [now, setNow] = useState(start);

  useEffect(() => {
    if (!running) return;
    const kick = setTimeout(() => setNow(Date.now()), 0);
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      clearTimeout(kick);
      clearInterval(id);
    };
  }, [running]);

  const elapsed =
    session.elapsed_seconds !== null && session.elapsed_seconds !== undefined
      ? session.elapsed_seconds
      : Math.max(0, (now - start) / 1000);

  const { done, total } = taskProgress(events);

  return (
    <div className="card flex flex-wrap items-center gap-x-8 gap-y-3 py-3">
      <Stat label="已用时间" value={formatDuration(elapsed)} />
      <Stat label="费用" value={formatCost(session.total_cost_usd)} />
      <Stat label="任务" value={total ? `${done}/${total}` : done ? String(done) : "—"} />
    </div>
  );
}
