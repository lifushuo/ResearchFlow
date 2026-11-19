"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";

import { useActiveProject } from "@/components/ActiveProject";
import { FirstRunNotice } from "@/components/FirstRunNotice";
import { EmptyState } from "@/components/ui/EmptyState";
import { RunCard } from "@/components/runs/RunCard";
import { StartResearchForm } from "@/components/runs/StartResearchForm";
import { useRuns } from "@/hooks/runs";

/** 研究页兼顾提出问题和查看项目已有成果。将发起入口与近期研究放在一起，让用户先看到已建立的知识。项目范围统一来自选择器上下文，不增加第二个控件或路由参数，保持单一信息来源（frontend/AGENTS.md）。 */
export default function ResearchPage() {
  // useSearchParams 会触发客户端渲染，静态导出需要边界；
  // 兜底不能还是读取查询参数的同一组件，
  // 否则 build:desktop 仍会在导出时失败，尽管普通 build 通过。
  // 这是 frontend/AGENTS.md 说明的多构建目标陷阱：
  // 一个分支只有在对应目标的构建中才真正受到检查。
  return (
    <Suspense fallback={<ResearchSkeleton />}>
      <ResearchPageInner />
    </Suspense>
  );
}

function ResearchSkeleton() {
  return (
    <div className="space-y-8">
      <div className="h-8 w-48 animate-pulse bg-bg-elevated" aria-hidden />
      <div className="card h-64 animate-pulse" aria-hidden />
      <span className="sr-only">正在加载研究…</span>
    </div>
  );
}

function ResearchPageInner() {
  const router = useRouter();
  // 从失败研究的再次提问链接初始化，仅在创建表单状态时读取一次。
  const params = useSearchParams();
  const seeded = params?.get("q") ?? "";
  // 语料库页的文件选择通过 URL 中的 ID 传入，不存入浏览器存储；
  // 服务端在创建研究时解析并冻结范围。
  const list = (key: string) => (params?.get(key) ?? "").split(",").filter(Boolean);
  const selection = { folder_ids: list("folders"), doc_keys: list("files") };
  // 项目选择器加载期间 activeId 为 undefined，此时暂缓项目范围内的请求。
  const { activeId: projectId, active, isLoading: projectsLoading } = useActiveProject();
  const { data: runs, isLoading, isError, refetch } = useRuns(projectId ?? null);

  const waiting =
    runs?.filter((r) => r.status === "AWAITING_REVIEW" || r.status === "AWAITING_PLAN") ?? [];
  // 已在上方展示的研究不在下方重复，
  // 避免同一屏中的两条记录被误认为两次研究。
  const promoted = new Set(waiting.map((r) => r.id));
  const rest = runs?.filter((r) => !promoted.has(r.id)) ?? [];
  const recent = rest.slice(0, 8);

  return (
    <div className="space-y-8">
      {/* Above the form, not inside it: someone with no model configured needs to see the
          next step before they type a question they cannot run. */}
      <FirstRunNotice />

      <section aria-labelledby="new-research">
        <h1
          id="new-research"
          className="mb-1 font-serif text-2xl font-bold tracking-tight text-text-primary"
        >
          研究
        </h1>
        <p className="mb-5 max-w-2xl text-sm leading-relaxed text-text-muted">
          提出问题，获得可逐项追溯证据、来源和审核决定的报告；最终生成冻结的成果文件，任何人都可独立核查，无需信任本应用。
        </p>

        <StartResearchForm
          initialQuestion={seeded}
          selection={selection}
          onStarted={(runId) => router.push(`/research/run?id=${runId}`)}
        />
      </section>

      {waiting.length > 0 && (
        <section aria-labelledby="waiting-on-you">
          <h2
            id="waiting-on-you"
            className="mb-3 font-serif text-lg font-bold tracking-tight text-text-primary"
          >
            等待你处理
          </h2>
          <ul className="grid gap-3.5 sm:grid-cols-2">
            {waiting.map((r) => (
              <li key={r.id} className="h-full">
                <RunCard run={r} />
              </li>
            ))}
          </ul>
        </section>
      )}

      <section aria-labelledby="recent-research">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h2
            id="recent-research"
            className="font-serif text-lg font-bold tracking-tight text-text-primary"
          >
            本项目的历史研究
          </h2>
          <div className="flex items-baseline gap-3">
            {active && (
              <span className="font-mono text-xs text-text-muted">所属项目：{active.name}</span>
            )}
            {rest.length > recent.length && (
              <Link href="/history" className="font-mono text-xs text-accent hover:underline">
                全部研究 →
              </Link>
            )}
          </div>
        </div>

        {projectsLoading || isLoading ? (
          <div className="grid gap-3.5 sm:grid-cols-2">
            {[0, 1].map((i) => (
              <div key={i} className="card h-28 animate-pulse" aria-hidden />
            ))}
            <span className="sr-only">正在加载本项目的研究…</span>
          </div>
        ) : isError ? (
          <EmptyState
            title="无法加载本项目的研究"
            description="列表请求失败，研究记录仍然存在，只是此页未能读取。"
            action={
              <button type="button" onClick={() => refetch()} className="btn btn-secondary">
                重试
              </button>
            }
          />
        ) : recent.length === 0 ? (
          <EmptyState
            title={waiting.length > 0 ? "暂时没有其他研究" : "尚无研究"}
            description={
              waiting.length > 0
                ? "本项目的全部研究都在上方等待你的决定。"
                : "在上方提出第一个问题。每次研究都会在这里显示审核状态及是否生成可验证成果。"
            }
          />
        ) : (
          <ul className="grid gap-3.5 sm:grid-cols-2">
            {recent.map((r) => (
              <li key={r.id} className="h-full">
                <RunCard run={r} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
