import type { Diagnostic } from "./evaluation-diagnostics";

const phaseNames: Record<string, string> = {
  judge_request: "等待评分模型回复", judge_wait: "评分批次之间的等待", judge_wait_inferred: "评分阶段（等待位置为推断）",
  judge: "引用评分", postprocessing: "评分后的结果整理", after_graph: "研究之后的处理", planner: "研究任务规划",
  executor: "搜索和收集资料", critic: "证据检查", synthesizer: "撰写报告", finalizer: "保存最终报告",
};
export const diagnosisPhase = (value: string | null) => value ? phaseNames[value] || "具体调用（技术详情见下方）" : "尚未定位到具体步骤";
const seconds = (value: number) => `${value.toFixed(2)} 秒`;

export function explainDiagnostic(data: Diagnostic) {
  const facts: string[] = [];
  const timeline = data.timeline;
  if (data.elapsed_seconds != null) facts.push(`这轮总共执行了 ${seconds(data.elapsed_seconds)}。`);
  const research = timeline?.stages.find((s) => s.name === "研究图");
  const grading = timeline?.stages.filter((s) => s.finished_batches != null) || [];
  if (research) facts.push(`已记录的研究流程用了 ${seconds(research.duration_s)}。`);
  for (const s of grading) {
    facts.push(`引用评分阶段已用 ${seconds(s.duration_s)}，已完成 ${s.finished_batches} 批评分。`);
    if (s.recorded_wait_s != null) facts.push(`其中，明确记录的批次间等待占 ${seconds(s.recorded_wait_s)}。`);
    if (s.completed_request_s != null) facts.push(`已经完成的评分请求合计用了 ${seconds(s.completed_request_s)}；尚未结束的请求不包含在这个数里。`);
  }
  if (!data.has_issue) return { title: "本轮没有登记执行故障", summary: "可以查看本轮实际测得的结果；受控故障案例通过只表示符合预期。", facts, boundary: "各指标仍按自己的评测口径解释。", next: data.next_actions };
  if (data.reason === "runner_timeout") {
    const inJudge = data.failure_phase?.startsWith("judge");
    const requests = data.judge_traces.flatMap((t) => t.events || []).filter((e) => e.event === "batch_started");
    const dated = requests.filter((e) => e.timestamp && Number.isFinite(Date.parse(e.timestamp)));
    const last = dated.length ? dated.sort((a, b) => Date.parse(a.timestamp!) - Date.parse(b.timestamp!)).at(-1)
      : data.judge_traces.length === 1 ? requests.at(-1) : undefined;
    if (data.failure_phase === "judge_request" && last?.batch_no != null) facts.push(`停止前已开始第 ${last.batch_no} 批评分，但没有记录到该请求结束。`);
    if (data.failure_phase === "judge_request" && last?.timestamp && data.execution_observation.finished_at) {
      const elapsed = (Date.parse(data.execution_observation.finished_at) - Date.parse(last.timestamp)) / 1000;
      if (Number.isFinite(elapsed) && elapsed >= 0 && data.elapsed_seconds != null && elapsed <= data.elapsed_seconds) facts.push(`最后一批评分开始约 ${seconds(elapsed)}后，整轮评测被停止；这不等于单次模型请求超时。`);
    }
    return {
      title: "评测达到总时限，被系统停止",
      summary: inJudge && data.dataset_id === "report-benchmark" && research
        ? "研究流程已经结束，后续的引用评分还没完成，整轮评测就用完了允许的时间。"
        : `系统在「${diagnosisPhase(data.failure_phase)}」阶段停止了这轮评测，因为整轮执行已达到总时限。`,
      facts,
      boundary: data.failure_phase === "judge_request" ? "这能确定停止时还在等待评分回复；不能据此认定模型额度不足，也不能把早先已恢复的网页错误当成终止原因。"
        : data.failure_phase === "judge_wait_inferred" ? "旧记录缺少等待开始事件，停止位置仍有推断成分；不能据此认定模型服务商发生故障。"
          : "总时限被触发是已知事实；具体调用是否另有故障，要看它自己的错误与结束记录。",
      next: inJudge ? ["先检查评分批次之间的等待安排，以及研究和评分共用的时间预算。", "修改后只复测这一道案例，保留本轮记录，用新的结果验证是否完成。"] : data.next_actions,
    };
  }
  return { title: "本轮评测没有正常完成", summary: timeline?.summary || data.candidates[0]?.cause || "已保存执行失败记录，需要结合下面的具体异常排查。", facts,
    boundary: "模型可以根据现有轨迹提出待验证的解释；最终需用修复后的新运行验证。", next: data.next_actions };
}
