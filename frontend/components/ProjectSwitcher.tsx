"use client";

import { useEffect, useRef, useState } from "react";
import toast from "react-hot-toast";

import { useCreateProject } from "@/hooks/queries";
import { ApiError } from "@/lib/api";

import { useActiveProject } from "./ActiveProject";

/** 侧边栏项目选择器。支持展开与折叠两种模式，使用自定义弹层，避免原生 select 的样式限制。 */
interface ProjectSwitcherProps {
  collapsed?: boolean;
  /** 弹层展开方向。位于侧边栏顶部时向下展开；原先默认向上会超出视口顶部。 */
  menuDirection?: "up" | "down";
}

export function ProjectSwitcher({ collapsed = false, menuDirection = "up" }: ProjectSwitcherProps) {
  const { projects, activeId, setActiveId, isLoading } = useActiveProject();
  const create = useCreateProject();
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");

  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const activeProject = projects.find((p) => p.id === activeId) ?? projects[0];

  // 点击外部或按 Escape 时关闭。
  useEffect(() => {
    if (!open) return;

    const onPointerDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) {
        setOpen(false);
        setCreating(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (creating) {
          setCreating(false);
        } else {
          setOpen(false);
          triggerRef.current?.focus();
        }
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, creating]);

  // 开始创建项目时聚焦输入框。
  useEffect(() => {
    if (creating) {
      inputRef.current?.focus();
    }
  }, [creating]);

  const submitCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    try {
      const project = await create.mutateAsync({ name: trimmed });
      setActiveId(project.id);
      setName("");
      setCreating(false);
      setOpen(false);
      toast.success(`已创建项目“${project.name}”`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "无法创建该项目。");
    }
  };

  if (isLoading) {
    return (
      <div
        className={`h-9 animate-pulse border border-border bg-bg-elevated ${
          collapsed ? "w-9 mx-auto" : "w-full"
        }`}
        aria-hidden
      />
    );
  }

  return (
    <div ref={rootRef} className="relative w-full">
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={collapsed ? `Project: ${activeProject?.name ?? "无"}` : undefined}
        className={`group flex w-full items-center border bg-bg-surface text-text-primary transition-all duration-150 ${
          collapsed
            ? "h-9 w-9 justify-center mx-auto p-0 border-border hover:border-text-secondary"
            : "h-9 justify-between px-2.5 text-left text-xs border-border hover:border-text-secondary hover:bg-bg-elevated"
        } ${open ? "border-accent ring-1 ring-accent bg-bg-elevated" : ""}`}
      >
        {collapsed ? (
          <div className="flex h-6 w-6 items-center justify-center border border-border bg-accent-muted text-accent font-mono font-bold text-[0.6875rem]">
            {activeProject ? activeProject.name.charAt(0).toUpperCase() : "P"}
          </div>
        ) : (
          <>
            <div className="flex min-w-0 items-center gap-2">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center border border-border bg-accent-muted text-accent font-mono font-bold text-[0.625rem]">
                {activeProject ? activeProject.name.charAt(0).toUpperCase() : "P"}
              </span>
              <span className="truncate font-semibold text-text-primary text-[0.8125rem]">
                {activeProject ? activeProject.name : "选择项目"}
              </span>
            </div>
            <svg
              aria-hidden
              viewBox="0 0 20 20"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`h-3.5 w-3.5 shrink-0 text-text-muted transition-transform duration-200 group-hover:text-text-primary ${
                open ? "rotate-180 text-text-primary" : ""
              }`}
            >
              <path d="m6 8 4 4 4-4" />
            </svg>
          </>
        )}
      </button>

      {open && (
        <div
          role="listbox"
          aria-label="项目"
          className={`menu-surface animate-fade-in absolute z-50 shadow-xl ${
            collapsed
              ? menuDirection === "down"
                ? "left-full top-0 ml-2 w-64 origin-top-left"
                : "left-full bottom-0 ml-2 w-64 origin-bottom-left"
              : menuDirection === "down"
                ? "top-full left-0 mt-1.5 w-full min-w-[13.5rem] origin-top-left"
                : "bottom-full left-0 mb-1.5 w-full min-w-[13.5rem] origin-bottom-left"
          }`}
        >
          <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-bg-elevated/40">
            <span className="font-mono text-[length:var(--text-micro)] font-semibold uppercase tracking-wider text-text-muted">
              项目（{projects.length})
            </span>
            {!creating && (
              <button
                type="button"
                onClick={() => setCreating(true)}
                className="font-mono text-[0.6875rem] font-semibold text-accent hover:underline flex items-center gap-1 shrink-0"
              >
                <span>+ 新建</span>
              </button>
            )}
          </div>

          {/* 项目列表 */}
          <div className="max-h-56 overflow-y-auto py-1 divide-y divide-border/10">
            {projects.length === 0 ? (
              <div className="px-3 py-4 text-xs text-text-muted text-center">
                尚未创建项目
              </div>
            ) : (
              projects.map((p) => {
                const isSelected = p.id === activeId;
                return (
                  <button
                    key={p.id}
                    type="button"
                    role="option"
                    aria-selected={isSelected}
                    onClick={() => {
                      setActiveId(p.id);
                      setOpen(false);
                    }}
                    className={`flex w-full items-center justify-between px-3 py-2 text-xs text-left transition-colors ${
                      isSelected
                        ? "bg-accent-muted/70 font-semibold text-accent"
                        : "text-text-secondary hover:bg-bg-elevated hover:text-text-primary"
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <span
                        className={`flex h-5 w-5 shrink-0 items-center justify-center border font-mono font-bold text-[0.625rem] ${
                          isSelected
                            ? "border-accent bg-accent text-accent-contrast"
                            : "border-border bg-bg-elevated text-text-secondary"
                        }`}
                      >
                        {p.name.charAt(0).toUpperCase()}
                      </span>
                      <span className="truncate text-[0.8125rem]">{p.name}</span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0 ml-1.5">
                      {Boolean(p.session_count) && (
                        <span className="border border-border bg-bg-surface px-1 py-0.5 font-mono text-[0.625rem] text-text-muted font-normal shrink-0">
                          {p.session_count}
                        </span>
                      )}
                      {isSelected && (
                        <svg
                          aria-hidden
                          viewBox="0 0 20 20"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="h-3.5 w-3.5 text-accent shrink-0"
                        >
                          <path d="m4 10 4 4 8-8" />
                        </svg>
                      )}
                    </div>
                  </button>
                );
              })
            )}
          </div>

          {/* 新建项目表单 */}
          {creating ? (
            <form onSubmit={submitCreate} className="p-3 border-t border-border bg-bg-elevated">
              {/* 通过 htmlFor/id 关联标签，不能依赖视觉距离；占位符只是输入示例，输入后即消失，不能代替供读屏器使用的字段名称。 */}
              <label
                htmlFor="new-project-name"
                className="block font-mono text-[length:var(--text-micro)] font-semibold uppercase tracking-wider text-text-muted mb-1"
              >
                项目名称
              </label>
              <input
                id="new-project-name"
                ref={inputRef}
                value={name}
                onChange={(e) => setName(e.target.value.slice(0, 120))}
                placeholder="例如：mRNA 表观遗传学研究"
                className="input-base h-8 w-full text-xs mb-2 py-1 px-2"
              />
              <div className="flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setCreating(false);
                    setName("");
                  }}
                  className="px-2 py-1 text-xs text-text-muted hover:text-text-secondary"
                >
                  取消
                </button>
                <button
                  type="submit"
                  disabled={create.isPending || !name.trim()}
                  className="btn btn-primary h-7 px-3 text-xs"
                >
                  {create.isPending ? "创建中…" : "创建项目"}
                </button>
              </div>
            </form>
          ) : (
            <div className="p-1 border-t border-border bg-bg-elevated/20">
              <button
                type="button"
                onClick={() => setCreating(true)}
                className="flex w-full items-center gap-2 px-2.5 py-1.5 font-mono text-xs text-text-muted hover:bg-bg-elevated hover:text-text-primary transition-colors"
              >
                <svg
                  aria-hidden
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-3.5 w-3.5 text-accent"
                >
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
                <span>创建新项目</span>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
