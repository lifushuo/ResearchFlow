"use client";

import { useEffect, useId, useRef, useState } from "react";
import toast from "react-hot-toast";

import { ResetToDefault } from "@/components/settings/ResetToDefault";
import { ApiError } from "@/lib/api";
import { apiValidationMessage, promptEditorState } from "@/lib/promptOverrides";
import type { PromptDefault } from "@/lib/types";

/** 从“设置→智能体”中的角色卡片打开指令编辑器（范围约定 §12）。初始内容为用户覆盖值，否则为移除系统不可信内容框架后的默认提示词。清空不代表重置，空指令会被拒绝，必须引导使用重置控件。自定义角色重置时调用 onReset 清除覆盖；保存一份默认提示仍会被记录为自定义。默认角色重置时只丢弃未保存编辑，不发送请求。 */
export function RolePromptEditor({
  label,
  shipped,
  maxChars,
  saved,
  busy,
  onSave,
  onClose,
  onReset,
}: {
  label: string;
  shipped: PromptDefault;
  maxChars: number;
  saved: string | undefined;
  busy: boolean;
  onSave: (text: string) => Promise<void>;
  onClose: () => void;
  onReset: () => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const id = useId();
  const state = promptEditorState({
    saved,
    draft,
    defaultPrompt: shipped.default_prompt,
    maxChars,
  });

  // 编辑器由按钮打开，焦点应跟随用户操作进入输入框。
  useEffect(() => {
    box.current?.focus();
  }, []);

  const counterId = `${id}-count`;
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;

  async function save() {
    setError(null);
    try {
      await onSave(state.text);
      toast.success(`已保存${label}的指令，将用于此后启动的研究。`);
      onClose();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? apiValidationMessage(err.message)
          : "无法连接服务端，未作任何更改。",
      );
    }
  }

  function resetToShipped() {
    setError(null);
    if (state.customised) {
      onReset();
      return;
    }
    setDraft(null);
    box.current?.focus();
  }

  const atShipped = !state.customised && state.text === shipped.default_prompt;

  const describedBy = [counterId, state.dirty && state.blank ? hintId : null, error ? errorId : null]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="space-y-2">
      {shipped.untrusted_content_framed && (
        <p role="note" className="text-xs leading-relaxed text-text-muted">
          无论填写什么，系统都会在前后补充不可信网页内容规则：检索文本是数据，不能作为指令执行。这部分不属于可编辑内容，也不能移除。
        </p>
      )}
      <textarea
        ref={box}
        aria-label={`${label}的指令`}
        aria-describedby={describedBy}
        aria-invalid={state.tooLong || error !== null}
        className="textarea-base w-full font-mono text-xs leading-relaxed"
        rows={12}
        value={state.text}
        disabled={busy}
        onChange={(e) => {
          setDraft(e.target.value);
          setError(null);
        }}
      />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-3">
          <span
            id={counterId}
            className={`font-mono text-[0.6875rem] tabular-nums ${
              state.tooLong ? "text-danger" : "text-text-muted"
            }`}
          >
            {state.length.toLocaleString("zh-CN")} / {maxChars.toLocaleString("zh-CN")} 字符
            {state.tooLong && "——超出长度限制"}
          </span>
          <ResetToDefault isDefault={atShipped || busy} onReset={resetToShipped} />
        </div>
        <div className="flex items-center gap-2">
          <button type="button" className="btn btn-ghost" disabled={busy} onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!state.canSave || busy}
            onClick={save}
          >
            保存指令
          </button>
        </div>
      </div>
      {state.dirty && state.blank && (
        <p id={hintId} className="text-xs leading-relaxed text-text-secondary">
          清空提示词不代表重置，智能体必须有指令。{" "}
          {state.customised
            ? "使用“恢复默认”以还原内置提示词。"
            : "点击“取消”保留内置提示词。"}
        </p>
      )}
      {error && (
        <p id={errorId} role="alert" className="text-xs leading-relaxed text-danger">
          {error}
        </p>
      )}
    </div>
  );
}
