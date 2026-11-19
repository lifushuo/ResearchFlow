"use client";

/** 逐项恢复默认（internal/07 Phase 3）。自定义设置必须有清晰可见的还原路径；仅当当前值不同于默认值时显示，避免无意义的重置入口。 */
export function ResetToDefault({
  isDefault,
  onReset,
}: {
  isDefault: boolean;
  onReset: () => void;
}) {
  if (isDefault) return null;
  return (
    <button
      type="button"
      onClick={onReset}
      className="font-mono text-[0.6875rem] text-accent hover:underline"
    >
      恢复默认
    </button>
  );
}
