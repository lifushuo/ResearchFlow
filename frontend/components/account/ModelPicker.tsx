"use client";

import { useMemo, useState } from "react";
import toast from "react-hot-toast";

import {
  useCustomEndpointStatus,
  useModelCatalog,
  useResetModelRouting,
  useSetModelRouting,
} from "@/hooks/queries";
import { ROLE_COPY } from "@/lib/agentRoles";
import type { ModelInfo, ModelRouting } from "@/lib/types";

import { Section } from "./Section";

/** 按角色选择模型（docs/12 M8）。预设提供一步可用的默认方案，逐角色选择藏在自定义抽屉中。不同角色使用不同模型关乎质量：执行器多次调用工具，重速度和覆盖；综合器撰写用户最终阅读的成果，重输出质量。文案应解释取舍，而非只给出参数。 */

const PRESET_COPY: Record<string, string> = {
  fast: "速度最快、费用最低，适合初步界定问题。",
  balanced: "关键环节使用强模型，常规环节使用快速模型。",
  best: "优先保证质量，费用也最高。",
};

const PROVIDER_NAMES: Record<string, string> = {
  custom: "自定义端点（兼容 OpenAI）",
  google: "Google Gemini",
  anthropic: "Anthropic Claude",
  openai: "OpenAI",
  openrouter: "OpenRouter",
  ollama: "本地模型（Ollama）",
};

function formatPrice(model: ModelInfo): string {
  if (model.provider === "custom") return "自定义端点";
  if (model.input_per_mtok === null || model.output_per_mtok === null) return "价格未配置";
  if (model.input_per_mtok === 0 && model.output_per_mtok === 0) return "免费——在本地运行";
  return `$${model.input_per_mtok}/$${model.output_per_mtok} per 1M`;
}

/** 用每次研究的大致相对费用帮助比较速度与质量，避免只靠模糊印象选择。 */
function relativeCost(routing: ModelRouting, byRoute: Map<string, ModelInfo>): number {
  return Object.values(routing).reduce((sum, route) => {
    const m = byRoute.get(route);
    return sum + (m?.output_per_mtok ?? 0);
  }, 0);
}

export function ModelPicker() {
  const { data: catalog, isLoading } = useModelCatalog();
  const customStatus = useCustomEndpointStatus();
  const setRouting = useSetModelRouting();
  const resetRouting = useResetModelRouting();

  const [customizing, setCustomizing] = useState(false);
  const [draft, setDraft] = useState<ModelRouting | null>(null);

  const current = draft ?? catalog?.effective_routing ?? null;

  const customModels = useMemo(() => {
    const models = (customStatus.data?.models ?? []).filter(
      (m) => !m.toLowerCase().includes("deepseek-r1"),
    );
    const reachable = Boolean(customStatus.data?.reachable);
    const list: ModelInfo[] = models.map((m) => ({
      route: `custom:${m}`,
      provider: "custom",
      model_id: m,
      display_name: m,
      input_per_mtok: null,
      output_per_mtok: null,
      context_window: null,
      max_output_tokens: null,
      supports_tools: true,
      supports_structured_output: true,
      notes: "由你配置的兼容 OpenAI 的自定义端点提供服务。",
      available: reachable,
    }));

    // 即使端点未列出，也保留当前生效的 custom: 路由。
    if (current) {
      for (const route of Object.values(current)) {
        if (route.startsWith("custom:") && !list.some((m) => m.route === route)) {
          const id = route.slice("custom:".length);
          list.push({
            route,
            provider: "custom",
            model_id: id,
            display_name: id,
            input_per_mtok: null,
            output_per_mtok: null,
            context_window: null,
            max_output_tokens: null,
            supports_tools: true,
            supports_structured_output: true,
            notes: "由你配置的兼容 OpenAI 的自定义端点提供服务。",
            available: reachable,
          });
        }
      }
    }

    return list;
  }, [customStatus.data, current]);

  const allModels = useMemo(
    () => [...customModels, ...(catalog?.models ?? [])],
    [customModels, catalog?.models],
  );

  const byRoute = useMemo(
    () => new Map(allModels.map((m) => [m.route, m])),
    [allModels],
  );

  const groupedModels = useMemo(() => {
    const groups: Record<string, ModelInfo[]> = {};
    for (const m of allModels) {
      const p = m.provider;
      if (!groups[p]) groups[p] = [];
      groups[p].push(m);
    }
    return groups;
  }, [allModels]);

  if (isLoading || !catalog || !current) {
    return (
      <Section title="模型" description="选择流水线各阶段使用的模型。">
        <div className="h-24 animate-pulse border border-border bg-bg-surface" />
      </Section>
    );
  }

  // 只为用户当前可访问的提供方展示预设。
  const presetProviders = Object.keys(catalog.presets).filter((p) =>
    catalog.available_providers.includes(p),
  );

  const matchesPreset = (provider: string, name: string) =>
    JSON.stringify(catalog.presets[provider]?.[name]) === JSON.stringify(current);

  async function applyRouting(next: ModelRouting) {
    setDraft(next);
    try {
      await setRouting.mutateAsync(next);
      toast.success("模型已更新，将用于下一次研究。");
    } catch (e) {
      setDraft(null);
      toast.error(e instanceof Error ? e.message : "无法保存模型选择。");
    }
  }

  async function useDeploymentDefault() {
    setDraft(null);
    try {
      await resetRouting.mutateAsync();
      toast.success("已恢复此部署的默认模型。");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "无法重置。");
    }
  }

  const busy = setRouting.isPending || resetRouting.isPending;

  return (
    <Section
      title="模型"
      description="各阶段可以使用不同模型：常规处理选择快速低价模型，报告写作选择更强模型。"
      footer={
        <>
          <span className="text-xs text-text-muted">
            {catalog.user_routing
              ? "正在使用你的选择。"
              : "正在使用此部署的默认模型。"}
          </span>
          <div className="flex items-center gap-2">
            {catalog.user_routing && (
              <button
                type="button"
                onClick={useDeploymentDefault}
                disabled={busy}
                className="btn btn-ghost"
              >
                使用默认配置
              </button>
            )}
            <button
              type="button"
              onClick={() => setCustomizing((v) => !v)}
              className="btn btn-ghost"
              aria-expanded={customizing}
            >
              {customizing ? "收起角色配置" : "自定义"}
            </button>
          </div>
        </>
      }
    >
      {/* 模型预设。 */}
      {presetProviders.length === 0 ? (
        <div className="flex items-center gap-2.5 border border-border/70 bg-bg-surface/50 p-4 font-mono text-xs text-text-secondary">
          <span className="inline-block h-2 w-2 bg-text-muted shrink-0" />
          <span>尚未配置提供方。请先添加 API 密钥或启动 Ollama 本地模型，然后这里会显示可用预设。</span>
        </div>
      ) : (
        presetProviders.map((provider) => (
          <div key={provider} className="mb-6 last:mb-0">
            <div className="mb-3 font-mono text-xs font-semibold uppercase tracking-wider text-text-secondary flex items-center gap-2">
              <span className="h-1.5 w-1.5 bg-accent" />
              <span>{PROVIDER_NAMES[provider] ?? provider}</span>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              {catalog.preset_names.map((name) => {
                const mapping = catalog.presets[provider]?.[name];
                if (!mapping) return null;
                const active = matchesPreset(provider, name);
                return (
                  <button
                    key={`${provider}-${name}`}
                    type="button"
                    disabled={busy}
                    onClick={() => applyRouting(mapping)}
                    aria-pressed={active}
                    className={`border p-4 text-left transition-all disabled:opacity-60 ${
                      active
                        ? "border-accent bg-accent/5 ring-1 ring-accent/30 shadow-sm"
                        : "border-border/80 bg-bg-surface hover:border-border hover:bg-bg-elevated/40"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="font-serif text-sm font-bold capitalize text-text-primary">
                        {({ fast: "快速", balanced: "均衡", best: "最佳质量" } as Record<string, string>)[name] ?? name}
                      </div>
                      {active && (
                        <span className="h-2 w-2 bg-accent shadow-xs shadow-accent/50" />
                      )}
                    </div>
                    <div className="mt-1 text-xs leading-relaxed text-text-muted">
                      {PRESET_COPY[name] ?? ""}
                    </div>
                    <div className="mt-2 font-mono text-[0.6875rem] text-text-muted border-t border-border/40 pt-2">
                      {relativeCost(mapping, byRoute) === 0
                        ? "免费"
                        : `约 $${relativeCost(mapping, byRoute).toFixed(2)} / 百万输出词元`}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        ))
      )}

      {/* 按角色展开模型配置。 */}
      {customizing && (
        <div className="mt-6 space-y-4 border-t border-border/60 pt-6">
          <div className="text-xs font-mono font-semibold uppercase tracking-wider text-text-secondary">
            角色模型分配
          </div>
          {catalog.roles.map((role) => {
            const selected = byRoute.get(current[role]);
            return (
              <div
                key={role}
                className="grid items-center gap-3 sm:grid-cols-[14rem_1fr] border border-border/60 bg-bg-base/40 p-3.5"
              >
                <div className="min-w-0">
                  <div className="text-[0.8125rem] font-semibold text-text-primary">
                    {ROLE_COPY[role].label}
                  </div>
                  <div className="text-xs leading-relaxed text-text-muted">
                    {ROLE_COPY[role].blurb}
                  </div>
                </div>
                <div>
                  <select
                    id={`model-${role}`}
                    aria-label={`${ROLE_COPY[role].label}的模型`}
                    className="input-base w-full text-xs"
                    value={current[role]}
                    disabled={busy}
                    onChange={(e) => applyRouting({ ...current, [role]: e.target.value })}
                  >
                    {Object.entries(groupedModels).map(([providerKey, models]) => (
                      <optgroup
                        key={providerKey}
                        label={PROVIDER_NAMES[providerKey] ?? providerKey}
                      >
                        {models.map((m) => {
                          const disabled =
                            !m.available ||
                            (m.provider !== "custom" && m.input_per_mtok === null) ||
                            (role === "executor" && !m.supports_tools);
                          return (
                            <option
                              key={m.route}
                              value={m.route}
                              disabled={disabled}
                            >
                              {m.display_name} — {formatPrice(m)}
                              {!m.available ? "（需要密钥）" : ""}
                              {role === "executor" && !m.supports_tools ? "（不支持工具调用）" : ""}
                            </option>
                          );
                        })}
                      </optgroup>
                    ))}
                  </select>
                  {selected?.notes && (
                    <p className="mt-1.5 text-xs leading-relaxed text-text-muted">
                      {selected.notes}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Section>
  );
}
