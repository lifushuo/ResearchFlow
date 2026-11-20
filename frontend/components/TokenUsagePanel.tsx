import type { TokenUsage } from "@/lib/token-usage";

const number = (value: number | null | undefined) => value == null ? "未披露" : value.toLocaleString("zh-CN");
const roles: Record<string, string> = { planner: "规划", executor: "执行", critic: "证据评审", synthesizer: "报告生成", judge: "独立评分" };

export function TokenUsagePanel({ usage, title = "模型 Token 用量" }: { usage?: TokenUsage | null; title?: string }) {
  return <section aria-label={title} className="min-w-0 rounded-lg border border-border bg-bg-elevated p-4 space-y-3">
    <h3 className="font-semibold">{title}</h3>
    {usage?.models.length ? <>
      <p className="text-sm">已披露合计 <strong className="font-mono tabular-nums">{number(usage.known_total_tokens)}</strong> Token · 输入 {number(usage.known_input_tokens)} / 输出 {number(usage.known_output_tokens)} · 已记录 {usage.calls} 次调用</p>
      <div className="overflow-x-auto"><table className="w-full min-w-[640px] text-left text-sm">
        <thead><tr>{["模型 / 用途", "输入 Token", "输出 Token", "已披露合计", "调用次数", "用量缺失"].map((label) => <th key={label} className="p-2 font-medium">{label}</th>)}</tr></thead>
        <tbody>{usage.models.map((row, index) => <tr key={index} className="border-t border-border align-top">
          <td className="p-2"><p className="break-all font-mono">{row.llm_mode === "fake" ? "脚本模型（未调用供应商）" : row.served_model || "实际模型未披露"}</p>
            {row.configured_model && <p className="text-xs text-text-muted break-all">配置路由：{row.configured_model}</p>}
            <p className="text-xs text-text-muted">{row.kind === "judge" ? "Judge 评分" : "研究"} · {row.roles.map((role) => roles[role] || role).join("、")}</p>
            {row.failed_calls > 0 && <p className="text-xs text-warning">{row.failed_calls} 次异常 / 取消</p>}</td>
          {[row.known_input_tokens, row.known_output_tokens, row.known_total_tokens, row.calls, row.missing_usage_calls].map((value, i) => <td key={i} className="p-2 font-mono tabular-nums whitespace-nowrap">{number(value)}</td>)}
        </tr>)}</tbody>
      </table></div>
      {usage.missing_usage_calls > 0 && <p className="text-sm text-warning">{usage.missing_usage_calls} 次调用缺少完整用量，上述数字仅为已披露部分。</p>}
    </> : <p className="text-sm text-text-muted">尚无按模型用量记录；历史总 Token 不能倒推出每个模型的消耗。</p>}
    <p className="text-xs text-text-muted">{usage?.collection_incomplete ? "采集可能不完整。" : ""}只汇总已观测的模型调用；恢复的旧批次不计入本轮新增用量。未披露不等于 0，Token 不等于费用；未披露的 SDK 重试、向量化和工具服务用量不在其中。</p>
  </section>;
}
