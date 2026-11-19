"use client";

import { useMemo, useState } from "react";

import { useActiveProject } from "@/components/ActiveProject";
import { SessionHistory } from "@/components/history/SessionHistory";
import { EmptyState } from "@/components/ui/EmptyState";
import { RunCard } from "@/components/runs/RunCard";
import { useRuns } from "@/hooks/runs";
import { RUN_STATUS_ORDER, runStatusMeta } from "@/lib/runStatus";
import type { RunStatus } from "@/lib/types";

/**
 * Everything that has been researched.
 *
 * This page used to list sessions and nothing else, while the runs it exists to record
 * were reachable only from a list at the bottom of the Research page. A user who clicked
 * "History" after doing research in this product saw none of it.
 *
 * The filters are derived from data every row already carries, so none of them can lie
 * about what they select. Two are worth spelling out:
 *
 * - **Artifact** is not a synonym for "finished". A run can be COMPLETED and carry no
 *   artifact, and the difference is exactly the product's claim, so it filters separately.
 * - **Status options are the statuses actually present on this page**, not the whole
 *   vocabulary, so the strip never offers a filter that selects nothing.
 */

type Scope = "project" | "all";
type Verified = "any" | "artifact" | "none";

export default function HistoryPage() {
  const { activeId, active, projects } = useActiveProject();
  const [scope, setScope] = useState<Scope>("project");
  const [status, setStatus] = useState<RunStatus | "ALL">("ALL");
  const [verified, setVerified] = useState<Verified>("any");
  const [query, setQuery] = useState("");
  const [showArchived, setShowArchived] = useState(false);

  const { data: runs, isLoading, isError, refetch } = useRuns(
    scope === "project" ? (activeId ?? null) : null,
    showArchived,
  );

  const projectNames = useMemo(
    () => new Map(projects.map((p) => [p.id, p.name])),
    [projects],
  );

  const present = useMemo(() => {
    const seen = new Set(runs?.map((r) => r.status) ?? []);
    return RUN_STATUS_ORDER.filter((s) => seen.has(s));
  }, [runs]);

  const needle = query.trim().toLowerCase();
  const visible = (runs ?? []).filter(
    (r) =>
      (status === "ALL" || r.status === status) &&
      (verified === "any" ||
        (verified === "artifact" ? r.has_artifact : !r.has_artifact)) &&
      (needle === "" || r.question.toLowerCase().includes(needle)),
  );

  const filtering = status !== "ALL" || verified !== "any" || needle !== "";

  return (
    <div className="space-y-8">
      <section aria-labelledby="history-heading" className="space-y-4">
        <div>
          <h1
            id="history-heading"
            className="font-serif text-2xl font-bold tracking-tight text-text-primary"
          >
            {showArchived ? "已归档的研究" : "历史记录"}
          </h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-text-muted">
            {showArchived
              ? "已归档的研究任务。可以恢复到历史记录，也可以永久删除。"
              : "查看每项研究的进度、待审批事项，以及是否已生成可验证的成果。"}
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[length:var(--text-micro)] uppercase tracking-wider text-text-muted">
              搜索研究问题
            </span>
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="输入问题关键词…"
              className="input-base h-8 w-56 py-1 text-xs"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="font-mono text-[length:var(--text-micro)] uppercase tracking-wider text-text-muted">
              项目范围
            </span>
            <select
              value={scope}
              onChange={(e) => setScope(e.target.value as Scope)}
              className="input-base h-8 w-44 py-1 text-xs"
            >
              <option value="project">{active ? active.name : "当前项目"}</option>
              <option value="all">全部项目</option>
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="font-mono text-[length:var(--text-micro)] uppercase tracking-wider text-text-muted">
              研究成果
            </span>
            <select
              value={verified}
              onChange={(e) => setVerified(e.target.value as Verified)}
              className="input-base h-8 w-44 py-1 text-xs"
            >
              <option value="any">不限</option>
              <option value="artifact">已有可验证成果</option>
              <option value="none">暂无成果</option>
            </select>
          </label>

          {present.length > 1 && (
            <div className="flex flex-wrap gap-1" role="group" aria-label="按状态筛选">
              <button
                type="button"
                aria-pressed={status === "ALL"}
                onClick={() => setStatus("ALL")}
                className={`border px-3 py-1 font-mono text-xs font-medium transition-colors ${
                  status === "ALL"
                    ? "border-accent bg-accent text-accent-contrast"
                    : "border-border bg-bg-surface text-text-muted hover:text-text-primary"
                }`}
              >
                全部
              </button>
              {present.map((s) => (
                <button
                  key={s}
                  type="button"
                  aria-pressed={status === s}
                  onClick={() => setStatus(s)}
                  className={`border px-3 py-1 font-mono text-xs font-medium transition-colors ${
                    status === s
                      ? "border-accent bg-accent text-accent-contrast"
                      : "border-border bg-bg-surface text-text-muted hover:text-text-primary"
                  }`}
                >
                  {runStatusMeta(s).label}
                </button>
              ))}
            </div>
          )}

          <button
            type="button"
            onClick={() => setShowArchived((v) => !v)}
            aria-pressed={showArchived}
            className={`border px-3 py-1 font-mono text-xs font-medium transition-colors ${
              showArchived
                ? "border-accent bg-accent text-accent-contrast"
                : "border-border bg-bg-surface text-text-muted hover:text-text-primary"
            }`}
          >
            {showArchived ? "← 返回历史记录" : "已归档"}
          </button>
        </div>

        {isLoading ? (
          <div className="grid gap-3 sm:grid-cols-2">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="card h-28 animate-pulse" aria-hidden />
            ))}
            <span className="sr-only">正在加载研究历史…</span>
          </div>
        ) : isError ? (
          <EmptyState
            title="无法加载历史记录"
            description="请求失败，当前页面无法读取记录，请重试。"
            action={
              <button type="button" onClick={() => refetch()} className="btn btn-secondary">
                重试
              </button>
            }
          />
        ) : (runs ?? []).length === 0 ? (
          <EmptyState
            title={showArchived ? "暂无归档记录" : "尚无研究记录"}
            description={
              showArchived
                ? "归档会将研究任务移出历史列表，但不会删除它。"
                : scope === "project"
                  ? "当前项目尚无研究任务。发起研究后，记录会显示在这里。"
                  : "所有项目中暂时都没有研究记录。"
            }
          />
        ) : visible.length === 0 ? (
          <EmptyState
            title="没有符合筛选条件的记录"
            description={`当前筛选条件隐藏了 ${runs!.length} 项研究任务。`}
            action={
              <button
                type="button"
                onClick={() => {
                  setStatus("ALL");
                  setVerified("any");
                  setQuery("");
                }}
                className="btn btn-secondary"
              >
                清除筛选
              </button>
            }
          />
        ) : (
          <>
            <p className="font-mono text-xs text-text-muted" aria-live="polite">
              {filtering
                ? `显示 ${visible.length} 项，共 ${runs!.length} 项研究任务`
                : `共 ${visible.length} 项研究任务`}
            </p>
            <ul className="grid gap-3 sm:grid-cols-2">
              {visible.map((r) => (
                <li key={r.id} className="h-full">
                  <RunCard
                    run={r}
                    showProject={
                      scope === "all" ? (projectNames.get(r.project_id) ?? null) : null
                    }
                  />
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      {/* Research recorded as sessions, kept readable rather than hidden — and rendered
          as nothing at all on an account that has none. */}
      <SessionHistory />
    </div>
  );
}
