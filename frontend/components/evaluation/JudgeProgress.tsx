import type { JudgeProgress as Progress, ReportState } from "@/lib/evaluation-diagnostics";

const count = (value: number | null | undefined) => value ?? "未测量";
const ratio = (part: number | null | undefined, whole: number | null | undefined) => `${count(part)} / ${count(whole)}`;
const percent = (value: number | null | undefined) => value == null ? "未测量" : `${(value * 100).toFixed(2)}%`;

export function JudgeProgress({ progress: j }: { progress: Progress }) {
  return <div className="space-y-1 text-sm">
    <p>{j.case_id} · {j.trace_complete ? "轨迹记录已闭合" : "轨迹记录不完整"} · {j.scoring_complete ? "全量评分已完成" : "全量评分尚未确认完成"}</p>
    <p>成功批次 {ratio(j.successful_batches, j.total_batches)} · 已尝试批次 {ratio(j.attempted_batches, j.total_batches)}</p>
    <p>全量 Claim 覆盖率 {percent(j.full_judge_coverage)} · 已尝试范围覆盖率 {percent(j.attempted_judge_coverage ?? j.observed_judge_coverage)}</p>
    <p>全量判定进度 {ratio(j.judged_claims_total, j.total_claims)} · 未判定 {count(j.unjudged_claims_total)}</p>
    <p>本轮已请求 {count(j.new_requested_batches ?? j.observed_application_level_calls)} · 请求成功返回 {count(j.successful_requests)} · 有效完整评分批次 {count(j.valid_result_batches)}</p>
    {j.batch_measurements && <details><summary>逐批耗时与用量</summary><p>已披露输入 / 输出 Token：{count(j.known_input_tokens)} / {count(j.known_output_tokens)}；{count(j.batches_missing_usage)} 批用量缺失，成本未测量。</p><ul>{j.batch_measurements.map((b) => <li key={b.batch_no}>第 {b.batch_no} 批：{b.elapsed_ms == null ? "耗时未测量" : `${(b.elapsed_ms / 1000).toFixed(2)} 秒`}；输入 / 输出 Token {count(b.tokens_input)} / {count(b.tokens_output)}</li>)}</ul>{j.wait_observations?.map((w, i) => <p key={i}>批间等待：{w.wait_reason || "旧记录未披露原因"} · {w.elapsed_ms == null ? "未测量" : `${(w.elapsed_ms / 1000).toFixed(2)} 秒`}</p>)}</details>}
    {j.total_claims == null && <p className="text-text-muted">历史记录缺少全量输入，不能由已尝试范围推算整体覆盖率。</p>}
  </div>;
}

export function ReportEvaluationState({ state: s }: { state: ReportState }) {
  const reason = s.stop_reason?.startsWith("judge_deadline") ? "阶段预算耗尽" : s.stop_reason?.startsWith("batch_in_progress") ? "批次正由其他执行器处理" : s.stop_reason || "未记录";
  return <div className="rounded-lg border border-border p-3 text-sm space-y-1">
    <p className="font-medium">{s.case_id || "本次报告"} · 分阶段评测状态</p>
    <p>研究草稿：{s.research === "draft_saved" ? "已生成并保存" : "保存状态未确认"}</p>
    <p>Judge 评分：{s.judge === "completed" ? "完整完成" : s.judge === "partial" ? "部分完成" : "未确认"} · 评分截止原因：{reason}</p>
    <p>评分覆盖进度：{ratio(s.judged_claims, s.total_claims)} · 已提交保存 {count(s.committed_batches)} 批 / {count(s.committed_claims)} 条</p>
    <p>恢复能力：{s.resume === "artifacts_validated" ? "保存产物只读校验通过，可供续评；启动前仍需核对当前模型配置" : "未通过产物校验，不能直接续评"}</p>
    <p>最终完整评测：{s.evaluation === "completed" ? "已完成评分" : "未完成"} · 质量门禁：{s.quality_gate === "passed" ? "放行" : s.quality_gate === "pending_quality_checks" ? "仍需核对质量判据" : "不放行"}</p>
  </div>;
}
