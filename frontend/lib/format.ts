/** 用于界面展示的纯格式化函数，已由单元测试覆盖。 */

export function researchDepthLabel(depth: string): string {
  return ({ fast: "快速", balanced: "均衡", comprehensive: "全面" } as Record<string, string>)[depth] ?? depth;
}

export function formatCost(usd: number | null | undefined): string {
  const v = usd ?? 0;
  if (v === 0) return "$0.00";
  if (v < 0.01) return `$${v.toFixed(4)}`;
  return `$${v.toFixed(2)}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}秒`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}分 ${s}秒`;
}

export function formatNumber(n: number | null | undefined): string {
  return new Intl.NumberFormat("zh-CN").format(n ?? 0);
}

export function relativeTime(iso: string, now: number = Date.now()): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return iso;
  const diff = Math.max(0, now - t);
  const s = Math.floor(diff / 1000);
  if (s < 45) return "刚刚";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}分钟前`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}小时前`;
  const d = Math.floor(h / 24);
  if (d < 30) return `${d}天前`;
  const mo = Math.floor(d / 30);
  if (mo < 12) return `${mo}个月前`;
  return `${Math.floor(mo / 12)}年前`;
}
