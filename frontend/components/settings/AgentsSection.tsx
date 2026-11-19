"use client";

import Link from "next/link";
import { useState } from "react";
import toast from "react-hot-toast";

import { RolePromptEditor } from "@/components/account/RolePromptEditor";
import { Section } from "@/components/account/Section";
import { ResetToDefault } from "@/components/settings/ResetToDefault";
import { useMe, useModelCatalog, usePromptDefaults, useUpdateProfile } from "@/hooks/queries";
import { ROLE_COPY, roleModelLabel } from "@/lib/agentRoles";
import { ApiError } from "@/lib/api";
import { customisedRoles } from "@/lib/promptOverrides";
import type { AgentRole } from "@/lib/types";

/** 设置中的智能体页面只管理行为（范围约定 §12），模型选择仍由模型页负责；这里只读展示当前模型并提供链接，避免同一设置出现两个编辑入口。可编辑角色严格来自 GET /models/prompt-defaults，即五个公开角色，不展示受保护的用途，也不提供创建新角色。一次只打开一个编辑器，避免长提示词掩盖每张卡片的默认或自定义状态。 */
export function AgentsSection() {
  const { data: me, isLoading: meLoading } = useMe();
  const defaults = usePromptDefaults();
  const { data: catalog } = useModelCatalog();
  const updateProfile = useUpdateProfile();
  const [editing, setEditing] = useState<AgentRole | null>(null);
  const [resetError, setResetError] = useState<{ role: AgentRole; message: string } | null>(
    null,
  );

  const overrides = me?.preferences.prompt_overrides ?? null;
  const customised = new Set(customisedRoles(overrides));

  // null 表示重置（§12）；API 按角色合并，不影响其他智能体。
  const write = async (role: AgentRole, text: string | null) => {
    await updateProfile.mutateAsync({ preferences: { prompt_overrides: { [role]: text } } });
  };

  async function reset(role: AgentRole): Promise<boolean> {
    setResetError(null);
    try {
      await write(role, null);
      toast.success(`${ROLE_COPY[role].label}已恢复内置提示词。`);
      return true;
    } catch (err) {
      setResetError({
        role,
        message: err instanceof ApiError ? err.message : "无法连接服务端。",
      });
      return false;
    }
  }

  const loading = meLoading || defaults.isLoading;

  return (
    <Section title="智能体" description="自定义研究智能体的行为。">
      <div className="space-y-5">
        <p className="text-xs leading-relaxed text-text-muted">
          智能体默认遵循内置指令，也可使用你保存的指令。新指令用于此后启动的研究；已开始的研究保留启动时的版本，追问对话则立即使用最新指令。验证包会完整记录指令，分享对象也能阅读。本版本使用 v2 格式，旧版验证器会拒绝，请使用当前版本核验。
        </p>

        {loading && <div className="h-40 animate-pulse border border-border bg-bg-surface" />}

        {!loading && (defaults.isError || !defaults.data || !me) && (
          <p className="text-sm text-text-secondary">
            无法加载智能体内置指令，当前不能编辑。未作任何更改。
          </p>
        )}

        {!loading && defaults.data && me && (
          <ul className="space-y-3" aria-label="智能体">
            {defaults.data.roles.map((shipped) => {
              const role = shipped.role;
              const isCustom = customised.has(role);
              const open = editing === role;
              const model = roleModelLabel(catalog, role);
              const saved =
                typeof overrides?.[role] === "string" ? (overrides[role] as string) : undefined;
              return (
                <li
                  key={role}
                  aria-label={ROLE_COPY[role].label}
                  className="border border-border/60 bg-bg-base/40 p-4"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-sm font-semibold text-text-primary">
                          {ROLE_COPY[role].label}
                        </h3>
                        <span
                          className={`badge font-mono text-[0.625rem] uppercase tracking-wider ${
                            isCustom
                              ? "border-accent/40 bg-accent/10 text-accent"
                              : "border-border text-text-muted"
                          }`}
                        >
                          {isCustom ? "已自定义" : "内置默认"}
                        </span>
                      </div>
                      <p className="text-xs leading-relaxed text-text-muted">
                        {ROLE_COPY[role].blurb}
                      </p>
                      <p className="font-mono text-[0.6875rem] text-text-muted">
                        模型：{model ?? "不可用"} ·{" "}
                        <Link href="/settings/models" className="text-accent hover:underline">
                          前往模型设置修改
                        </Link>
                      </p>
                    </div>
                    {!open && (
                      <div className="flex items-center gap-3">
                        {isCustom && (
                          <ResetToDefault
                            isDefault={editing !== null || updateProfile.isPending}
                            onReset={() => reset(role)}
                          />
                        )}
                        <button
                          type="button"
                          className="btn btn-secondary"
                          disabled={editing !== null || updateProfile.isPending}
                          onClick={() => {
                            setResetError(null);
                            setEditing(role);
                          }}
                        >
                          {isCustom ? "编辑指令" : "自定义"}
                        </button>
                      </div>
                    )}
                  </div>
                  {resetError?.role === role && (
                    <p role="alert" className="mt-2 text-xs text-danger">
                      {resetError.message}
                    </p>
                  )}
                  {open && (
                    <div className="mt-4 border-t border-border/40 pt-4">
                      <RolePromptEditor
                        label={ROLE_COPY[role].label}
                        shipped={shipped}
                        maxChars={defaults.data.max_chars}
                        saved={saved}
                        busy={updateProfile.isPending}
                        onSave={(text) => write(role, text)}
                        onClose={() => setEditing(null)}
                        onReset={async () => {
                          // 只有覆盖值确实清除后才关闭编辑器；请求被拒绝时，
                          // 在卡片上显示错误，并保留编辑器及其文本。
                          if (await reset(role)) setEditing(null);
                        }}
                      />
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </Section>
  );
}
