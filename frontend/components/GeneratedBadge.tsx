/** 区分自动保存的报告与上传来源（app/services/report_corpus.py，docs/12 M10）。生成报告永不用于检索证据，research_engine/corpus.py 无条件排除 origin=generated；因此徽标必须让用户一眼分清哪些材料可作为研究来源。 */
export function GeneratedBadge({ className = "" }: { className?: string }) {
  const c = "var(--warning)";
  return (
    <span
      className={`badge font-mono text-[0.6875rem] font-semibold uppercase tracking-wider ${className}`}
      style={{
        color: c,
        backgroundColor: `color-mix(in srgb, ${c} 10%, var(--bg-surface))`,
        borderColor: `color-mix(in srgb, ${c} 35%, var(--border))`,
      }}
      title="由本项目中已批准的报告自动保存；与上传文档不同，不会作为研究证据使用。"
    >
      生成内容
    </span>
  );
}
