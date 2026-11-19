"use client";

import Link from "next/link";
import { useState } from "react";
import toast from "react-hot-toast";

import { RelativeTime } from "@/components/RelativeTime";
import { useArchiveRun, useDeleteRun } from "@/hooks/runs";
import { formatCost } from "@/lib/format";
import { runStatusMeta } from "@/lib/runStatus";
import type { RunSummary } from "@/lib/types";

/** 研究列表中的一条记录，由研究页、历史页和项目概览共用，避免同一研究被各页描述为不同状态。按浏览顺序说明研究问题、是否需用户处理、是否已有核验成果及时间；只有等待用户决定的状态需要重点强调。 */
export function RunCard({ run, showProject }: { run: RunSummary; showProject?: string | null }) {
  const meta = runStatusMeta(run.status);
  const archived = Boolean(run.archived_at);
  const archive = useArchiveRun();
  const del = useDeleteRun();
  const [confirming, setConfirming] = useState(false);
  const busy = archive.isPending || del.isPending;

  const stop = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const onArchive = async (e: React.MouseEvent) => {
    stop(e);
    try {
      await archive.mutateAsync({ id: run.id, archived: !archived });
      toast.success(archived ? "已恢复到历史记录" : "已归档");
    } catch {
      toast.error(archived ? "无法恢复此次研究。" : "无法归档此次研究。");
    }
  };

  const onDelete = async (e: React.MouseEvent) => {
    stop(e);
    try {
      await del.mutateAsync(run.id);
      toast.success("研究记录已删除");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "无法删除此次研究。");
      setConfirming(false);
    }
  };


  // 左侧边框使用状态色作辅助提示。
  let borderLeftColor = "transparent";
  if (meta.needsYou) {
    borderLeftColor = "var(--warning)";
  } else if (run.status === "FAILED" || run.status === "CANCELLED") {
    borderLeftColor = "var(--danger)";
  } else if (run.status === "COMPLETED") {
    borderLeftColor = "var(--success)";
  } else if (run.status === "RUNNING" || run.status === "PENDING") {
    borderLeftColor = "var(--info)";
  }

  return (
    <Link
      href={`/research/run?id=${run.id}`}
      className="card card-interactive flex flex-col justify-between group p-4 hover:border-text-secondary transition-all h-full"
      style={{
        borderLeft: `3px solid ${borderLeftColor}`,
        paddingLeft: "calc(var(--card-pad, 1rem) - 2px)",
      }}
    >
      <div>
        {/* 顶部上下文与状态行。 */}
        <div className="flex items-center justify-between gap-3 border-b border-border/40 pb-2 mb-2.5">
          <div className="flex flex-wrap items-center gap-2 font-mono text-[length:var(--text-micro)] text-text-muted">
            <RelativeTime iso={run.created_at} />
            <span>·</span>
            <span>{({ fast: "快速", balanced: "均衡", comprehensive: "全面" } as Record<string, string>)[run.depth] ?? run.depth}</span>
            <span>·</span>
            <span className="font-semibold text-text-secondary">{formatCost(run.cost_usd)}</span>
            {showProject && (
              <>
                <span>·</span>
                <span className="text-text-secondary">{showProject}</span>
              </>
            )}
            {run.demo && (
              <span
                className="border border-warning-line bg-warning-soft px-1.5 py-0.2 font-semibold text-warning"
                title="脚本模型与固定示例来源。"
              >
                演示
              </span>
            )}
          </div>

          {/* 状态徽标。 */}
          <span
            className="badge shrink-0 border-border bg-bg-elevated font-mono text-[length:var(--text-micro)] font-medium text-text-primary"
            title={meta.sentence}
          >
            <span
              aria-hidden
              className="status-marker"
              style={{ backgroundColor: `var(--${meta.token})` }}
            />
            {meta.label}
          </span>
        </div>

        {/* 研究问题。 */}
        <p className="min-w-0 line-clamp-2 break-words text-[0.9375rem] font-medium leading-snug text-text-primary group-hover:text-accent transition-colors">
          {run.question}
        </p>
      </div>

      {/* 底部运行数据与成果状态。 */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-border/30 pt-2.5 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          {run.has_artifact ? (
            <span className="inline-flex items-center gap-1 font-mono text-[length:var(--text-micro)] font-semibold text-success bg-success-soft border border-success-line px-2 py-0.5">
              ✓ 已验证成果
            </span>
          ) : (
            <span
              className="font-mono text-[length:var(--text-micro)] text-text-muted bg-bg-elevated px-2 py-0.5"
              title="报告经人工批准后才会生成成果文件。"
            >
              尚无成果文件
            </span>
          )}
        </div>

        {/* 引用解析率标签。 */}
        {run.citation_resolution_rate !== null && (
          <span
            className={`font-mono text-[length:var(--text-micro)] px-2 py-0.5 ${
              run.citation_resolution_rate < 1
                ? "border border-warning-line bg-warning-soft text-warning font-semibold"
                : "bg-bg-elevated text-text-secondary"
            }`}
            title="本报告中可解析到真实来源的引用标记比例。"
          >
            {Math.round(run.citation_resolution_rate * 100)}% 的引用可解析
          </span>
        )}

        {/* 鼠标环境下悬停或聚焦时显示操作；触屏环境下始终显示。 */}
        <span className="ml-auto flex items-center gap-1.5 opacity-100 transition-opacity sm:opacity-0 sm:group-focus-within:opacity-100 sm:group-hover:opacity-100 font-sans">
          {confirming ? (
            <>
              <span className="text-[0.6875rem] text-danger font-medium">永久删除？</span>
              <button
                type="button"
                onClick={onDelete}
                disabled={busy}
                className="px-1.5 py-0.5 text-[0.6875rem] font-medium text-danger border border-danger/30 hover:bg-danger/10"
              >
                {del.isPending ? "删除中…" : "是"}
              </button>
              <button
                type="button"
                onClick={(e) => {
                  stop(e);
                  setConfirming(false);
                }}
                className="px-1.5 py-0.5 text-[0.6875rem] text-text-muted border border-border hover:text-text-secondary"
              >
                否
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={onArchive}
                disabled={busy}
                className="px-1.5 py-0.5 text-[0.6875rem] text-text-muted border border-transparent hover:border-border hover:text-text-secondary"
                title={archived ? "恢复到历史记录" : "从历史记录中移出"}
              >
                {archived ? "恢复" : "归档"}
              </button>
              <button
                type="button"
                onClick={(e) => {
                  stop(e);
                  setConfirming(true);
                }}
                disabled={busy}
                className="px-1.5 py-0.5 text-[0.6875rem] text-text-muted border border-transparent hover:border-danger/30 hover:text-danger"
                title="永久删除"
              >
                删除
              </button>
            </>
          )}
        </span>
      </div>
    </Link>
  );
}
