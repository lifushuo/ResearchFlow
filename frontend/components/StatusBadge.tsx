import { statusMeta } from "@/lib/status";
import type { SessionStatus } from "@/lib/types";

/** 颜色只使用主题令牌（docs/07 的主题章节）；状态文字统一定义于 lib/status.ts，保证徽标与历史筛选一致。 */

export function StatusBadge({ status }: { status: SessionStatus }) {
  const { token, label, pulse } = statusMeta(status);
  const c = `var(--${token})`;
  return (
    <span
      className="badge font-mono text-[0.6875rem] font-semibold uppercase tracking-wider"
      style={{
        color: c,
        backgroundColor: `color-mix(in srgb, ${c} 10%, var(--bg-surface))`,
        borderColor: `color-mix(in srgb, ${c} 30%, var(--border))`,
      }}
    >
      <span
        className="status-marker"
        style={{
          backgroundColor: c,
          animation: pulse ? "pulse-ring 1.4s ease-in-out infinite" : undefined,
        }}
        aria-hidden
      />
      {label}
    </span>
  );
}
