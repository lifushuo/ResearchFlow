"use client";

import { useState } from "react";

import { useActiveProject } from "@/components/ActiveProject";
import { SessionCard } from "@/components/SessionCard";
import { useSessions } from "@/hooks/queries";
import { researchDepthLabel } from "@/lib/format";
import { STATUS_ORDER, statusMeta } from "@/lib/status";
import type { SessionStatus } from "@/lib/types";

const LIMIT = 20;

/**
 * Derived from the shared vocabulary, never restated (internal/07 Phase 7). This list
 * was hand-written and had already fallen behind: `AWAITING_PLAN` was missing, so a run
 * parked at the design gate — the one a user is most likely scanning for — could be seen
 * and not filtered for.
 */
const FILTERS: { value: SessionStatus | "ALL"; label: string }[] = [
  { value: "ALL", label: "全部" },
  ...STATUS_ORDER.map((value) => ({ value, label: statusMeta(value).label })),
];

/** Depth is on every session row, so it filters client-side like status does. */
const DEPTHS = ["fast", "balanced", "comprehensive"] as const;

/**
 * Verified-citation rate bands. The product's central claim, made scannable.
 *
 * "Unmeasured" is its own band rather than being swept into the lowest one: a report
 * with no citable claims and a report whose every marker is broken are opposite
 * findings, and `citation_resolution_rate` is `null` for the first and `0` for the
 * second. A band of "under 80%" that quietly included nulls would be the
 * unmeasured-as-zero bug wearing a filter.
 */
const CITATION_BANDS = {
  ALL: { label: "不限", match: () => true },
  PERFECT: { label: "100% 引用可定位", match: (r: number | null) => r === 1 },
  PARTIAL: { label: "引用定位率低于 100%", match: (r: number | null) => r !== null && r < 1 },
  UNMEASURED: { label: "未测量", match: (r: number | null) => r === null },
} as const;

export function SessionHistory() {
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState<SessionStatus | "ALL">("ALL");
  const [depth, setDepth] = useState<(typeof DEPTHS)[number] | "ALL">("ALL");
  const [band, setBand] = useState<keyof typeof CITATION_BANDS>("ALL");
  const [model, setModel] = useState<string>("ALL");
  // Archived is a separate destination, not another status filter — a session is
  // archived *or* active, and mixing them would defeat the point of getting one
  // out of the way. Switching views resets paging.
  const [showArchived, setShowArchived] = useState(false);
  const { activeId, active } = useActiveProject();
  const { data, isLoading, isError, refetch, isFetching } = useSessions(
    page,
    LIMIT,
    showArchived,
    activeId
  );

  const rows = data?.sessions ?? [];
  // Every distinct route any listed run dialled, so the picker only offers models that
  // actually appear on this page rather than the whole catalog.
  const models = Array.from(
    new Set(rows.flatMap((s) => Object.values(s.model_routing ?? {}))),
  ).sort();

  const visible = rows.filter(
    (s) =>
      (filter === "ALL" || s.status === filter) &&
      (depth === "ALL" || s.research_depth === depth) &&
      CITATION_BANDS[band].match(s.citation_resolution_rate) &&
      (model === "ALL" || Object.values(s.model_routing ?? {}).includes(model)),
  );
  const totalPages = data ? Math.max(1, Math.ceil(data.total / LIMIT)) : 1;

  // Nothing to show and nothing archived to reveal: this whole section is noise on an
  // account that has only ever used runs, which is every account created since, so it
  // renders nothing at all rather than an empty state explaining an absence.
  if (!isLoading && !isError && rows.length === 0 && !showArchived) return null;

  return (
    <section aria-labelledby="session-history" className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2
            id="session-history"
            className="font-serif text-lg font-bold tracking-tight text-text-primary"
          >
            {showArchived ? "已归档的旧版研究" : "旧版研究记录"}
          </h2>
          <p className="mt-0.5 max-w-2xl text-xs leading-relaxed text-text-muted">
            旧版研究记录{active && <>，所属项目：{active.name}</>}。仍可阅读、追问和导出；新研究显示在上方列表。
          </p>
        </div>
        {/* Two independent axes, laid out as two controls: a single row of tabs would
            imply that picking a depth clears the status, which it does not. */}
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1.5 font-mono text-xs text-text-muted">
            <span>研究深度</span>
            <select
              value={depth}
              onChange={(e) => setDepth(e.target.value as typeof depth)}
              className="border border-border bg-bg-surface px-2 py-1 font-mono text-xs text-text-primary"
            >
              <option value="ALL">不限</option>
              {DEPTHS.map((d) => (
                <option key={d} value={d}>
                  {researchDepthLabel(d)}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-1.5 font-mono text-xs text-text-muted">
            <span>引用定位率</span>
            <select
              value={band}
              onChange={(e) => setBand(e.target.value as keyof typeof CITATION_BANDS)}
              className="border border-border bg-bg-surface px-2 py-1 font-mono text-xs text-text-primary"
            >
              {Object.entries(CITATION_BANDS).map(([key, b]) => (
                <option key={key} value={key}>
                  {b.label}
                </option>
              ))}
            </select>
          </label>

          {models.length > 1 && (
            <label className="flex items-center gap-1.5 font-mono text-xs text-text-muted">
              <span>模型</span>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="max-w-[12rem] border border-border bg-bg-surface px-2 py-1 font-mono text-xs text-text-primary"
              >
                <option value="ALL">不限</option>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>
          )}

          <div className="flex flex-wrap gap-1" role="tablist" aria-label="按状态筛选">
            {FILTERS.map((f) => (
              <button
                key={f.value}
                type="button"
                role="tab"
                aria-selected={filter === f.value}
                onClick={() => setFilter(f.value)}
                className={`px-3 py-1 font-mono text-xs font-medium border transition-colors ${
                  filter === f.value
                    ? "bg-accent text-accent-contrast border-accent"
                    : "bg-bg-surface text-text-muted border-border hover:text-text-primary"
                }`}
              >
                {f.label}
              </button>
            ))}
            <button
              type="button"
              onClick={() => {
                setShowArchived((v) => !v);
                setPage(1);
              }}
              aria-pressed={showArchived}
              className={`ml-1 px-3 py-1 font-mono text-xs font-medium border transition-colors ${
                showArchived
                  ? "bg-accent text-accent-contrast border-accent"
                  : "bg-bg-surface text-text-muted border-border hover:text-text-primary"
              }`}
            >
              {showArchived ? "← 返回历史记录" : "已归档"}
            </button>
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="card h-28 animate-pulse" aria-hidden />
          ))}
        </div>
      ) : isError ? (
        <div className="card text-sm text-text-muted">
          无法加载旧版研究记录。{" "}
          <button onClick={() => refetch()} className="text-accent hover:underline">
            重试
          </button>
        </div>
      ) : rows.length === 0 ? (
        <div className="card text-center">
          {showArchived ? (
            <p className="text-sm text-text-secondary">
              暂无归档记录。归档会将研究移出历史列表，但不会删除它。
            </p>
          ) : (
            <>
              <p className="text-sm text-text-secondary">
                暂无旧版研究记录，所属项目：{active ? `“${active.name}”` : "当前项目"}。
              </p>
            </>
          )}
        </div>
      ) : visible.length === 0 ? (
        <div className="card text-center text-sm text-text-muted">
          本页没有符合筛选条件的旧版研究记录。
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2" aria-busy={isFetching}>
          {visible.map((s) => (
            <SessionCard key={s.session_id} session={s} />
          ))}
        </div>
      )}

      {data && totalPages > 1 && (
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            type="button"
            className="btn btn-secondary"
            disabled={page <= 1 || isFetching}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            ← 上一页
          </button>
          <span className="text-sm text-text-muted tabular-nums">
            第 {page} 页，共 {totalPages} 页
          </span>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={page >= totalPages || isFetching}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          >
            下一页 →
          </button>
        </div>
      )}
    </section>
  );
}
