import { apiFetch } from "@/lib/api";

export type ClosureRun = { run_id: string; dataset_id: string; execution_mode: string; state: string; source_pipeline_id: string | null; provenance_json: { contract?: string } };
export type Gate = { verdict: string; measured: number; total_recorded: number; pass_rate: number | null; p95_latency_ms: number | null; counts: Record<string, number>; policy_sha256: string };
export type Comparison = { comparable: boolean; reasons?: string[]; regressions?: string[]; improvements?: string[] };

const base = "/evaluation/closure";
export const closureGet = <T,>(path: string) => apiFetch<T>(base + path);
export const closurePost = <T,>(path: string, body?: unknown) => apiFetch<T>(base + path, {
  method: "POST", headers: { "X-ResearchFlow-Eval-Action": "run-pipeline-v1.3" }, body,
});

const messages: Array<[RegExp, string]> = [
  [/unfinished|unmeasured or blocked/, "运行未完成、被阻塞或未测量，不能证明修复通过。"],
  [/hash mismatch|does not match/, "输入指纹不一致，不能将此结果作为同一案例的复测。"],
  [/Dataset or execution mode/, "数据集或执行模式不一致，不能用于复测。"],
  [/Resource not found/, "记录不存在，或当前账号无权访问。"],
  [/Closure storage unavailable/, "闭环存储暂不可用，请检查服务端评测卷配置。"],
];
export function closureError(error: unknown): string {
  const raw = error instanceof Error ? error.message : "";
  return messages.find(([pattern]) => pattern.test(raw))?.[1] ?? "操作未完成，请稍后重试或查看技术详情。";
}

export function closureState(value: string): string {
  return ({
    passed: "通过", failed: "失败", blocked: "阻塞", unmeasured: "未测量", open: "未完成", finished: "已完成", incomplete: "不完整",
    fake: "脚本模型受控运行", synthetic: "合成契约评分", real: "真实模型或向量化服务" } as Record<string, string>)[value] ?? value;
}
