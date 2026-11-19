"use client";

import Link from "next/link";
import { useState } from "react";
import toast from "react-hot-toast";

import { useArchiveSession, useDeleteSession } from "@/hooks/queries";
import { sessionHref } from "@/lib/desktop";
import { researchDepthLabel, formatCost } from "@/lib/format";
import type { SessionSummary } from "@/lib/types";

import { RelativeTime } from "./RelativeTime";
import { DemoBadge } from "./DemoBadge";
import { StatusBadge } from "./StatusBadge";

/**
 * A session row with its own archive/delete affordances.
 *
 * The whole card is a link, so the actions are real buttons layered on top and must
 * stop propagation — otherwise "Delete" would also navigate into the session being
 * deleted. Delete asks for confirmation inline rather than via window.confirm: it is
 * irreversible (report, logs, chat, and the graph checkpoints all go), and an inline
 * two-step keeps the warning honest about that.
 */
export function SessionCard({ session }: { session: SessionSummary }) {
  const archived = Boolean(session.archived_at);
  const archive = useArchiveSession();
  const del = useDeleteSession();
  const [confirming, setConfirming] = useState(false);
  const busy = archive.isPending || del.isPending;

  const stop = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const onArchive = async (e: React.MouseEvent) => {
    stop(e);
    try {
      await archive.mutateAsync({ id: session.session_id, archived: !archived });
      toast.success(archived ? "已恢复到历史记录" : "已归档");
    } catch {
      toast.error(archived ? "无法恢复研究记录。" : "无法归档研究记录。");
    }
  };

  const onDelete = async (e: React.MouseEvent) => {
    stop(e);
    try {
      await del.mutateAsync(session.session_id);
      toast.success("研究记录已删除");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "无法删除研究记录。");
      setConfirming(false);
    }
  };

  return (
    <Link
      href={sessionHref(session.session_id)}
      className="card card-interactive group relative block p-5"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-1.5">
          <StatusBadge status={session.status} />
          {/* Beside the status, not down with depth and cost: in a grid of cards this is
              the one attribute that decides whether the rest means anything. */}
          {session.demo && <DemoBadge />}
        </span>
        <span className="shrink-0 text-xs text-text-muted">
          <RelativeTime iso={session.created_at} />
        </span>
      </div>

      <p className="mt-3 line-clamp-2 font-serif text-[1rem] font-medium leading-snug text-text-primary transition-colors group-hover:text-accent">
        {session.prompt}
      </p>

      <div className="mt-3.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs text-text-muted">
        <span className="capitalize">{researchDepthLabel(session.research_depth)}</span>
        <span aria-hidden>·</span>
        <span className="tabular-nums">{formatCost(session.total_cost_usd)}</span>
        {/* The product's central claim, per row. Rendered only when it was actually
            measured — `null` means the report made no citable claims, and printing
            "0% verified" for that would be the opposite finding. */}
        {session.citation_resolution_rate !== null &&
          session.citation_resolution_rate !== undefined && (
            <>
              <span aria-hidden>·</span>
              <span
                className="tabular-nums"
                title="报告中可定位到真实来源的引用标记占比"
                style={{
                  color:
                    session.citation_resolution_rate === 1
                      ? "var(--success)"
                      : "var(--warning)",
                }}
              >
                {Math.round(session.citation_resolution_rate * 100)}% 引用可定位
              </span>
            </>
          )}
        {session.rework_count > 0 && (
          <>
            <span aria-hidden>·</span>
            <span>
              返工 {session.rework_count} 次
            </span>
          </>
        )}

        {/* Actions. Visible on hover/focus at pointer sizes, always visible on touch. */}
        <span className="ml-auto flex items-center gap-1.5 opacity-100 transition-opacity sm:opacity-0 sm:group-focus-within:opacity-100 sm:group-hover:opacity-100 font-sans">
          {confirming ? (
            <>
              <span className="text-[0.6875rem] text-danger">确定永久删除？</span>
              <button
                type="button"
                onClick={onDelete}
                disabled={busy}
                className="px-1.5 py-0.5 text-[0.6875rem] font-medium text-danger border border-danger/30 hover:bg-danger/10"
              >
                {del.isPending ? "删除中…" : "确认删除"}
              </button>
              <button
                type="button"
                onClick={(e) => {
                  stop(e);
                  setConfirming(false);
                }}
                className="px-1.5 py-0.5 text-[0.6875rem] text-text-muted border border-border hover:text-text-secondary"
              >
                取消
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={onArchive}
                disabled={busy}
                className="px-1.5 py-0.5 text-[0.6875rem] text-text-muted border border-transparent hover:border-border hover:text-text-secondary"
                title={archived ? "恢复到历史记录" : "移出历史记录"}
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
