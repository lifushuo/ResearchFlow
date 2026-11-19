"use client";

import Link from "next/link";
import { useState } from "react";

import { useActiveProject } from "@/components/ActiveProject";
import {
  defaultKind,
  isUnpricedRoute,
  PROVIDER_LABEL,
  ProviderPicker,
  routeForKind,
  routingFor,
  type ProviderKind,
} from "@/components/research/ProviderPicker";
import {
  useCorpusStatus,
  useCustomEndpointStatus,
  useLocalLLMStatus,
  useModelCatalog,
} from "@/hooks/queries";
import { useStartV2Research } from "@/hooks/runs";
import { ApiError } from "@/lib/api";
import type { ResearchDepth } from "@/lib/types";

/** 问题输入是本页核心，较少变化的选项放入折叠区，关闭时仍说明已选状态。表单只提交 POST /runs 接受的 project_id、question、depth、corpus_mode、skip_plan_gate 和 model_routing，不提供接口不接受的 demo。模型类型需在此明确显示，因为已保存偏好会覆盖部署配置；展示解析后的路由可避免用户不知情地沿用旧选择。 */

/** API 接受 1–2000 字符；本表单将最小值设为 10，提前提示过短问题，避免生成无用计划。 */
const MIN_QUERY = 10;
const MAX_QUERY = 2000;

/** 深度说明明确列出工具轮数，而非空泛承诺。网页抓取并发且限时十秒，主要等待来自模型响应；不同深度必须实际改变每任务的轮数，不能只改变提示词中的名称。 */
const DEPTHS: { value: ResearchDepth; label: string; hint: string }[] = [
  {
    value: "fast",
    label: "快速",
    hint: "每个任务最多 3 轮、3 页，耗时最短、成本最低。",
  },
  {
    value: "balanced",
    label: "均衡",
    hint: "每个任务最多 5 轮、5 页，适合大多数问题。",
  },
  {
    value: "comprehensive",
    label: "全面",
    hint: "每个任务最多 8 轮、8 页，研究最深入，等待也最长。",
  },
];

const SAMPLE_QUERY =
  "大模型智能体实现长期记忆的主要方法有哪些，各有什么取舍？";

export function StartResearchForm({
  onStarted,
  initialQuestion = "",
  selection,
}: {
  onStarted: (runId: string) => void;
  /** 从 ?q= 初始化，让失败研究中的再次提问只需点击后编辑。 */
  initialQuestion?: string;
  /** 来自语料库页的文件选择（?folders=、?files=），研究范围锁定为这些文件。 */
  selection?: { folder_ids: string[]; doc_keys: string[] };
}) {
  const { activeId, active, isLoading: projectsLoading } = useActiveProject();
  const [question, setQuestion] = useState(initialQuestion);
  const [depth, setDepth] = useState<ResearchDepth>("balanced");
  const hasSelection = !!selection && selection.folder_ids.length + selection.doc_keys.length > 0;
  const [corpusModeChoice, setCorpusMode] = useState(false);
  // 选择文件就意味着只研究语料库，因此开关启用并锁定。
  const corpusMode = hasSelection || corpusModeChoice;
  const [offline, setOffline] = useState(false);
  // 默认启用计划审核，每次请求都显式发送。
  //
  // POST /runs 默认 skip_plan_gate=true，以兼容
  // 尚未更新请求体的旧脚本；研究表单则面向另一类使用场景，
  // 默认必须经过计划审核，让用户先看见将搜索什么，
  // 再决定是否开始花费。
  // 首次研究的此步骤记录在 docs/getting-started/20-quick-start.md。
  // 因此即使是默认选择，也必须显式放入请求体，
  // 否则会继承脚本默认值，意外跳过审核关卡。
  const [planGate, setPlanGate] = useState(true);
  // 选择文件时默认展开，让用户看见研究的范围限制。
  const [showAdvanced, setShowAdvanced] = useState(hasSelection);
  // 探测完成前为 null，避免默认按钮在用户光标下改变；
  // 用户自己的 picked 选择一旦存在，始终优先。
  const [picked, setPicked] = useState<ProviderKind | null>(null);

  const start = useStartV2Research();
  const corpus = useCorpusStatus(corpusMode ? activeId : null);
  const catalog = useModelCatalog();
  const customEndpoint = useCustomEndpointStatus();
  const localLLM = useLocalLLMStatus();

  // 依次选择实际可达的自定义端点、本地模型和 API；
  // 这里只是在开始前解析并显示的默认选择，
  // 不会在研究中途替换提供方。
  const kind = picked ?? defaultKind(catalog.data, customEndpoint.data, localLLM.data);
  const chosenRoute = kind
    ? routeForKind(kind, catalog.data, customEndpoint.data, localLLM.data)
    : null;

  const trimmed = question.trim();
  const tooShort = trimmed.length > 0 && trimmed.length < MIN_QUERY;
  const noProject = !projectsLoading && !activeId;
  const canSubmit =
    trimmed.length >= MIN_QUERY && trimmed.length <= MAX_QUERY && Boolean(activeId) && !start.isPending;

  // 折叠区关闭时只提示偏离默认值的选项，
  // 避免默默隐藏非默认状态，也避免给普通默认配置
  // 加上容易被误解为警告的徽标。
  const overrides = [
    corpusMode ? "仅限语料库" : null,
    planGate ? null : "跳过计划审核",
    // 选定模型若偏离默认值，关闭折叠区后也必须可见，
    // 使路由选择始终明确。
    kind && kind !== "custom" ? PROVIDER_LABEL[kind] : null,
  ].filter(Boolean);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    run();
  };

  const run = () => {
    // canSubmit 包含 isPending 检查，同时防止回车与双击
    // 在首次请求尚未返回时再次提交。
    if (!canSubmit || !activeId) return;
    start.mutate(
      {
        project_id: activeId,
        question: trimmed,
        depth,
        corpus_mode: corpusMode || offline,
        ...(hasSelection ? { corpus_selection: selection } : {}),
        ...(offline ? { offline: true } : {}),
        skip_plan_gate: !planGate,
        // 未选模型时省略字段，不能发送 null；字段缺失才表示
        // 按已保存的设置解析，空映射则会因角色不全
        // 而无法通过校验。
        // 解析到实际路由后就显式发送，记录研究启动时使用的模型，
        // 而非在执行时从可能已改变的设置中
        // 重新解析；仅在完全没有配置时省略。
        ...(chosenRoute ? { model_routing: routingFor(chosenRoute) } : {}),
      },
      { onSuccess: (r) => onStarted(r.run_id) },
    );
  };

  return (
    <form onSubmit={submit} className="card space-y-5 border border-border bg-bg-surface p-5 sm:p-6" aria-describedby="start-context">
      {/* 研究问题输入区。 */}
      <div>
        <div className="mb-2 flex items-center justify-between gap-2">
          <label htmlFor="question" className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
            研究问题
          </label>
          {question.length === 0 && (
            <button
              type="button"
              onClick={() => setQuestion(SAMPLE_QUERY)}
              className="font-mono text-xs text-accent hover:underline"
            >
              使用示例问题 ↗
            </button>
          )}
        </div>
        <textarea
          id="question"
          rows={4}
          value={question}
          onChange={(e) => setQuestion(e.target.value.slice(0, MAX_QUERY))}
          placeholder={`例如：${SAMPLE_QUERY}`}
          className="textarea-base w-full resize-y border border-border bg-bg-surface p-3.5 font-serif text-base leading-relaxed transition-colors focus:border-accent"
          aria-describedby="query-counter"
          aria-invalid={tooShort || undefined}
          disabled={start.isPending}
        />
        <div id="query-counter" className="mt-1.5 flex justify-between font-mono text-xs">
          <span style={{ color: tooShort ? "var(--warning)" : "var(--text-muted)" }}>
            {tooShort ? `至少 ${MIN_QUERY} 个字符` : " "}
          </span>
          <span className="tabular-nums text-text-muted">
            {trimmed.length} / {MAX_QUERY}
          </span>
        </div>
      </div>

      {/* 研究深度与选项控制栏。 */}
      <div className="flex flex-col gap-3 border border-border bg-bg-elevated/40 p-3.5 sm:flex-row sm:items-center sm:justify-between sm:p-4">
        <div className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:gap-3">
          {/* 在 fieldset 中使用原生单选按钮，支持方向键和读屏器。 */}
          <fieldset className="flex items-center gap-2.5">
            <legend className="sr-only">研究深度</legend>
            <span
              aria-hidden
              className="font-mono text-[length:var(--text-micro)] font-semibold uppercase tracking-wider text-text-muted"
            >
              深度
            </span>
            <div className="inline-flex border border-border bg-bg-surface">
              {DEPTHS.map((d) => {
                const selected = depth === d.value;
                return (
                  <label
                    key={d.value}
                    className="cursor-pointer border-r border-border px-3 py-1.5 text-xs font-medium transition-colors last:border-r-0 hover:bg-bg-elevated"
                    style={{
                      backgroundColor: selected ? "var(--accent)" : "transparent",
                      color: selected ? "var(--accent-contrast)" : "var(--text-secondary)",
                    }}
                  >
                    <input
                      type="radio"
                      name="depth"
                      value={d.value}
                      checked={selected}
                      onChange={() => setDepth(d.value)}
                      className="sr-only"
                      disabled={start.isPending}
                    />
                    {d.label}
                  </label>
                );
              })}
            </div>
          </fieldset>
          <span className="text-xs text-text-muted">
            {DEPTHS.find((d) => d.value === depth)?.hint}
          </span>
        </div>

        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          aria-expanded={showAdvanced}
          aria-controls="run-options"
          className="inline-flex items-center gap-1.5 self-start border border-transparent px-2.5 py-1 font-mono text-xs text-text-muted transition-colors hover:border-border hover:bg-bg-surface/80 hover:text-text-primary sm:self-auto"
        >
          <span aria-hidden className="inline-block w-2 text-center text-text-muted">
            {showAdvanced ? "▾" : "▸"}
          </span>
          <span className="font-semibold uppercase tracking-wider">选项</span>
          {!showAdvanced && overrides.length > 0 && (
            <span className="text-[length:var(--text-micro)] font-semibold text-accent">
              · {overrides.join(" · ")}
            </span>
          )}
        </button>
      </div>

      {/* 高级选项面板。 */}
      {showAdvanced && (
        <div id="run-options" className="space-y-3.5 border border-border bg-bg-elevated/30 p-4">
          <div className="border-b border-border/60 pb-3">
            <ProviderPicker
              value={kind ?? "custom"}
              onChange={setPicked}
              disabled={start.isPending}
            />
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex cursor-pointer items-start gap-3 border border-border bg-bg-surface p-3 transition-colors hover:border-text-secondary">
              <input
                type="checkbox"
                checked={corpusMode}
                onChange={(e) => setCorpusMode(e.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 border-border accent-[var(--accent)]"
                disabled={start.isPending || hasSelection}
              />
              <span>
                <span className="block text-sm font-medium text-text-primary">
                  {hasSelection ? "仅使用选中的文件" : "仅研究已上传语料"}
                </span>
                <span className="block text-xs leading-relaxed text-text-muted">
                  {hasSelection
                    ? `证据仅来自你选择的 ${selection!.doc_keys.length} 个文件和 ${selection!.folder_ids.length} 个文件夹。`
                    : "不搜索网页，证据仅来自本项目的文档。"}
                  你的问题和这些文件的摘录会发送给已配置的模型提供方。
                </span>
                {hasSelection && (
                  <Link href="/corpus" className="mt-1 block text-xs font-medium underline">
                    更改选择
                  </Link>
                )}
                {corpusMode && corpus.data && corpus.data.documents === 0 && (
                  <span className="mt-1 block text-xs font-medium" style={{ color: "var(--warning)" }}>
                    本项目尚无文档。{" "}
                    <Link href="/corpus" className="underline font-semibold">
                      上传文档
                    </Link>{" "}
                    ，否则研究将没有可读取的材料。
                  </span>
                )}
                {corpusMode && corpus.data && corpus.data.documents > 0 && (
                  <span className="mt-1 block text-xs text-text-secondary">
                    {corpus.data.documents} 份文档可用。
                  </span>
                )}
              </span>
            </label>

            <label className="flex cursor-pointer items-start gap-3 border border-border bg-bg-surface p-3 transition-colors hover:border-text-secondary">
              <input
                type="checkbox"
                checked={offline}
                onChange={(e) => setOffline(e.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 border-border accent-[var(--accent)]"
                disabled={start.isPending}
              />
              <span>
                <span className="block text-sm font-medium text-text-primary">离线</span>
                <span className="block text-xs leading-relaxed text-text-muted">
                  所有模型调用都指向本机地址；若任何模型配置为云端提供方，将拒绝启动。请注意，本地路由器若转发到云服务，仍会将文本发送到外部。
                </span>
              </span>
            </label>

            <label className="flex cursor-pointer items-start gap-3 border border-border bg-bg-surface p-3 transition-colors hover:border-text-secondary">
              <input
                type="checkbox"
                checked={planGate}
                onChange={(e) => setPlanGate(e.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 border-border accent-[var(--accent)]"
                disabled={start.isPending}
              />
              <span>
                <span className="block text-sm font-medium text-text-primary">
                  搜索前先审核研究计划
                </span>
                <span className="block text-xs leading-relaxed text-text-muted">
                  规划器完成后暂停，让你查看并排除不需要的子主题。批准后才开始后续搜索花费；关闭后将直接开始搜索。
                </span>
              </span>
            </label>
          </div>
        </div>
      )}

      {/* 操作与状态栏。 */}
      <div className="flex flex-col gap-4 border-t border-border pt-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex max-w-xl flex-col gap-1.5">
          <p id="start-context" className="text-xs font-medium leading-snug text-text-secondary">
            {projectsLoading
              ? "正在加载项目…"
              : noProject
                ? "请先创建项目——每次研究都必须属于一个项目。"
                : start.isPending
                  ? "正在打开研究工作区。"
                  : planGate
                    ? `保存到 ${active?.name}。研究会先暂停等待计划审核，生成报告后再次等待你审核。`
                    : `保存到 ${active?.name}。最终确定前会暂停等待你审核。`}
          </p>
          <p className="flex flex-wrap items-center gap-1.5 font-mono text-[length:var(--text-micro)] text-text-muted">
            {kind === null ? (
              "模型：正在解析…"
            ) : (
              <>
                <span className="text-text-muted">模型：</span>
                <span className="border border-border bg-bg-elevated px-1.5 py-0.5 font-medium text-text-primary">
                  {chosenRoute ?? `${PROVIDER_LABEL[kind]}——尚未配置`}
                </span>
                {isUnpricedRoute(chosenRoute) && (
                  <span className="text-text-muted">
                    · 无法计量该提供方的费用，请在提供方设置消费上限。
                  </span>
                )}
              </>
            )}
          </p>
        </div>

        <button type="submit" disabled={!canSubmit} className="btn btn-primary self-start px-5 py-2.5 font-medium sm:self-auto">
          {start.isPending && <span className="spinner" />}
          {start.isPending ? "启动中…" : "开始研究"}
        </button>
      </div>

      {start.isError && (
        <div
          className="border p-3 text-xs leading-relaxed"
          role="alert"
          style={{
            color: "var(--danger)",
            backgroundColor: "var(--danger-soft)",
            borderColor: "var(--danger-line)",
          }}
        >
          <p className="font-medium">无法启动研究。</p>
          <p className="mt-1">
            {start.error instanceof ApiError
              ? start.error.message
              : "请求未能到达服务端，请检查网络连接。"}
          </p>
          <button type="button" className="btn btn-secondary mt-2" disabled={!canSubmit} onClick={run}>
            重试
          </button>
        </div>
      )}
    </form>
  );
}
