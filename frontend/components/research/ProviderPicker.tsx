"use client";

import { useCustomEndpointStatus, useLocalLLMStatus, useModelCatalog } from "@/hooks/queries";
import type { ModelRouting } from "@/lib/types";

/** 为本次研究选择模型后端类型。具体模型属于设置：变化少，且需要价格、上下文窗口和密钥信息；表单只选择自定义网关、本地模型或计费 API。默认按自定义→本地→API 选择当前可达者，在开始前解析并展示。这不是运行中的回退机制；选定后端失败时必须明确失败，不能暗中换提供方后只在最终报告里披露。 */

export type ProviderKind = "custom" | "local" | "api";

/** 辅助函数只声明实际读取的字段，比 ModelCatalog、LocalLLMStatus、CustomEndpointStatus 更窄，明确依赖并让调用方及测试无需构造完整目录。 */
export interface RoutingSource {
  user_routing?: ModelRouting | null;
  deployment_routing?: ModelRouting | null;
  models?: { route: string; provider: string; available: boolean }[];
}
export interface CustomSource {
  reachable?: boolean;
  models?: string[];
  configured_base_url?: string;
  hint?: string | null;
  error?: string | null;
}
export interface LocalSource {
  models?: { route: string | null; is_embedding: boolean; likely_underpowered?: boolean }[];
}

const ROLE_ORDER = ["planner", "executor", "critic", "synthesizer", "chat"] as const;

/** 全部角色共用一个路由，符合创建研究接口校验的请求形状。 */
export function routingFor(route: string): ModelRouting {
  return Object.fromEntries(ROLE_ORDER.map((r) => [r, route])) as ModelRouting;
}

/** provider:model 只按第一个冒号分割，保留 ollama:qwen2.5:7b 中的版本标签。 */
export function splitRoute(route: string): { provider: string; model: string } {
  const i = route.indexOf(":");
  if (i < 0) return { provider: route, model: "" };
  return { provider: route.slice(0, i), model: route.slice(i + 1) };
}

/** 判断费用是否可测量。目录无法为 custom 和 openrouter 定价，estimate_cost() 返回 0.0 并不代表免费，界面必须说明未计量。Ollama 本地推理确实免费，与未知费用不同，因此不在此列。 */
export function isUnpricedRoute(route: string | null | undefined): boolean {
  if (!route) return false;
  const { provider } = splitRoute(route);
  return provider === "custom" || provider === "openrouter";
}

export const PROVIDER_LABEL: Record<ProviderKind, string> = {
  custom: "自定义端点",
  local: "本地大模型",
  api: "API",
};

const PROVIDER_NOTE: Record<ProviderKind, string> = {
  custom: "兼容 OpenAI 的网关",
  local: "在本机运行",
  api: "Gemini、Claude、OpenAI 等",
};

/** 解析后端路由的三个来源，按优先级排序。 */
export function routeForKind(
  kind: ProviderKind,
  catalog: RoutingSource | undefined,
  custom: CustomSource | undefined,
  local: LocalSource | undefined,
): string | null {
  // 设置指定此后端时优先采用已保存路由，
  // 使表单尊重模型选择，而不是重新猜测。
  const saved = catalog?.user_routing?.planner ?? null;
  const deployed = catalog?.deployment_routing?.planner ?? null;
  const savedProvider = saved ? splitRoute(saved).provider : null;
  const deployedProvider = deployed ? splitRoute(deployed).provider : null;

  if (kind === "custom") {
    if (savedProvider === "custom") return saved;
    if (deployedProvider === "custom") return deployed;
    const first = custom?.models?.[0];
    return first ? `custom:${first}` : null;
  }
  if (kind === "local") {
    if (savedProvider === "ollama") return saved;
    if (deployedProvider === "ollama") return deployed;
    // 标记为仅适合对话的本地模型通常不足以返回可靠引用，
    // 只有未安装更大模型时才作为默认选择。
    const chat = (local?.models ?? []).filter((m) => !m.is_embedding && m.route);
    return (chat.find((m) => !m.likely_underpowered) ?? chat[0])?.route ?? null;
  }
  // API 必须是目录中已定价且用户实际可访问的提供方。
  if (savedProvider && !["custom", "ollama"].includes(savedProvider)) return saved;
  if (deployedProvider && !["custom", "ollama"].includes(deployedProvider)) return deployed;
  const firstApi = (catalog?.models ?? []).find(
    (m) => m.available && m.provider !== "ollama" && m.provider !== "custom",
  );
  return firstApi?.route ?? null;
}

/** 检查当前实际可达性，与仅已配置不同。 */
export function isKindReady(
  kind: ProviderKind,
  catalog: RoutingSource | undefined,
  custom: CustomSource | undefined,
  local: LocalSource | undefined,
): boolean {
  if (kind === "custom") return Boolean(custom?.reachable);
  if (kind === "local") return (local?.models ?? []).some((m) => !m.is_embedding);
  return Boolean(routeForKind("api", catalog, custom, local));
}

/** 研究开始前按自定义→本地→API 选择首个可达后端。探测尚在进行时返回 null，让调用方等待，避免按钮先显示随后又改变的默认值。 */
export function defaultKind(
  catalog: RoutingSource | undefined,
  custom: CustomSource | undefined,
  local: LocalSource | undefined,
): ProviderKind | null {
  for (const kind of ["custom", "local", "api"] as ProviderKind[]) {
    if (isKindReady(kind, catalog, custom, local)) return kind;
  }
  return null;
}

export function ProviderPicker({
  value,
  onChange,
  disabled,
}: {
  value: ProviderKind;
  onChange: (next: ProviderKind) => void;
  disabled?: boolean;
}) {
  const catalog = useModelCatalog();
  const custom = useCustomEndpointStatus();
  const local = useLocalLLMStatus();

  const kinds: ProviderKind[] = ["custom", "local", "api"];

  return (
    <div>
      <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="模型后端">
        {kinds.map((kind) => {
          const ready = isKindReady(kind, catalog.data, custom.data, local.data);
          const route = routeForKind(kind, catalog.data, custom.data, local.data);
          const active = value === kind;
          return (
            <button
              key={kind}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onChange(kind)}
              // bg-accent-soft 并非主题中的有效令牌，
              // 引用不存在的工具类会被静默丢弃，因此
              // 选中状态使用真实令牌并辅以 aria-checked。
              className={`min-w-[9rem] flex-1 border p-2.5 text-left ${
                active ? "border-accent bg-bg-elevated" : "border-border bg-bg-surface"
              }`}
            >
              <span className="flex items-center gap-1.5">
                {/* 状态同时由圆点与文字表达，确保读屏器无需依赖颜色即可识别。 */}
                <span
                  aria-hidden
                  className={`inline-block h-2 w-2 shrink-0 rounded-full ${
                    ready ? "bg-success" : "bg-danger"
                  }`}
                />
                <span className="text-sm font-medium text-text-primary">
                  {PROVIDER_LABEL[kind]}
                </span>
                {/* 状态说明包含后端名称，避免读屏顺序变化时无法得知哪个服务已停止。 */}
                <span className="sr-only">
                  {PROVIDER_LABEL[kind]}{ready ? "运行中" : "已停止"}
                </span>
              </span>
              <span className="mt-0.5 block text-xs text-text-muted">{PROVIDER_NOTE[kind]}</span>
              <span className="mt-1 block break-all font-mono text-[length:var(--text-micro)] text-text-secondary">
                {route ?? "尚未配置"}
              </span>
            </button>
          );
        })}
      </div>

      {/* 选定后端不可用时明确告知；保留选择并报告失败，不暗中替换提供方。 */}
      {!isKindReady(value, catalog.data, custom.data, local.data) && (
        <p role="alert" className="mt-2 text-xs text-danger">
          {PROVIDER_LABEL[value]} 目前无法连接。使用它启动研究会报错，不会自动切换到其他提供方。
        </p>
      )}

      <p className="mt-2 text-xs text-text-secondary">
        各后端使用的模型可在{" "}
        <span className="text-text-primary">设置 → 模型</span>中配置。
      </p>
    </div>
  );
}
