/** 标识由脚本模型生成、而非真实研究的演示会话（docs/17 §6.2）。使用警告色而非普通强调色或灰色，因为演示会使卡片上的其他研究结论失去实际意义；标题补充完整说明，不能仅靠短徽标表达未进行研究。 */
export function DemoBadge({ className = "" }: { className?: string }) {
  const c = "var(--warning)";
  return (
    <span
      className={`badge font-mono text-[0.6875rem] font-semibold uppercase tracking-wider ${className}`}
      style={{
        color: c,
        backgroundColor: `color-mix(in srgb, ${c} 10%, var(--bg-surface))`,
        borderColor: `color-mix(in srgb, ${c} 35%, var(--border))`,
      }}
      title="使用脚本模型和固定示例来源，未进行实际研究或核验。"
    >
      演示
    </span>
  );
}
