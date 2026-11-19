// Only known presentation values are translated; IDs, evidence and unknown diagnostics stay intact.
const labels: Readonly<Record<string, string>> = {
  outcome: "报告质量", reliability: "回放可靠性", retrieval: "检索", grounding: "引用支撑", executor: "执行器",
  completion_rate: "任务完成率", judge_coverage: "评分覆盖率", coverage: "判定覆盖率",
  citation_support_among_judged: "引用支持率（仅已判定）", citation_support_rate: "引用支持率（仅已判定）",
  latency_p95_completed_s: "已完成报告的 95% 分位耗时", api_cost_usd: "预估接口成本（美元）",
  case_pass_rate: "受控回放通过率", contract_pass_rate: "契约通过率", accuracy: "准确率",
  measured: "已测量", unmeasured: "未测量", invalid: "无效", pass: "通过", passed: "通过", fail: "失败", failed: "失败",
  completed: "已完成", complete: "已完成", ok: "正常", success: "成功", error: "错误", pending: "待处理",
  skipped: "已跳过", unjudged: "未判定", yes: "支持", no: "不支持", unknown: "未知",
  awaiting_approval: "等待报告审批", awaiting_plan: "等待计划审批", executor_checkpoint: "执行器检查点",
  cancelled: "已取消", interrupted: "正常中断", running: "执行中", queued: "排队中", execution_error: "执行异常",
  healthy: "正常流程", planner_invalid_schema: "规划结果格式错误", planner_rate_limit: "模拟规划器限流",
  planner_upstream_error: "模拟规划器上游异常", strict: "严格顺序匹配", unordered: "无序匹配", subset: "子集匹配",
  ai: "人工智能", technology: "技术", science: "科学", business: "商业", policy: "政策", health: "健康",
  environment: "环境", energy: "能源", economics: "经济", general: "通用",
  climate: "气候", physics: "物理", software: "软件", semantic: "语义检索", lexical: "关键词检索", distractor: "干扰项检索",
  summary: "评测摘要", unified: "统一评测结果", none: "无统一评分文件",
  planner: "规划器", critic: "证据审查器", synthesizer: "报告生成器", finalizer: "最终报告整理",
  plan_gate: "计划审批", hitl_gate: "报告审批", failer: "失败处理", graph_node: "工作流节点", llm: "模型调用",
  tool: "工具调用", span_start: "片段开始", span_end: "片段结束", trace_start: "轨迹开始", trace_end: "轨迹结束",
  web_search: "搜索资料", read_webpage: "读取网页", submit_evidence: "提交证据",
  controlled_fault_handling_not_task_completion: "受控故障处理契约，不是研究完成率",
  report_quality_not_controlled_fault_handling: "研究报告质量，不是故障处理通过率",
  grounding_benchmark_not_report_citation_support: "结论与证据关系基准，不是报告整体引用支持率",
  corpus_retrieval: "全库检索", selection_before_cap: "指定文件检索（限制调整前）", selection_after_cap: "指定文件检索（限制调整后）",
  selection_retrieval: "指定文件检索", controlled_simulation: "受控模拟", controlled_fixture: "合成测试样本",
  not_reproduced_from_historical_run: "尚未从历史运行复现", not_a_historical_replay: "不属于历史故障重放",
  false_accept: "错误放行率", false_reject: "错误拒绝率", contradicted_recall: "矛盾召回率",
  false_accept_wilson_upper: "错误放行率置信上界", unmeasured_share: "未测量占比", unmeasured_rate: "未测量占比",
  passage_recall: "证据段落召回率", span_accuracy: "证据片段定位准确率", leakage: "范围外文件占比",
  negative_recall: "不支持样本识别率", gate_passed: "门禁通过", meets_threshold: "达到阈值",
  case_count: "案例数", passed_count: "通过案例数", failed_count: "失败案例数", total: "总数",
  judged: "已判定数", judged_count: "已判定数", unjudged_count: "未判定数", claims_judged: "已判定结论数",
  claims_unjudged: "未判定结论数", true_positive: "支持样本正确判定数", true_negative: "不支持样本正确判定数",
  false_positive: "不支持样本误判为支持数", false_negative: "支持样本误判为不支持数",
  tp: "支持样本正确判定数", tn: "不支持样本正确判定数", fp: "不支持样本误判为支持数", fn: "支持样本误判为不支持数",
  precision: "精确率", recall: "召回率", selection_correct: "工具选择正确", arguments_correct: "必需参数匹配",
  trajectory_correct: "调用顺序正确", forbidden_attempt_count: "禁止工具调用次数", status: "测量状态",
  value: "数值", unit: "单位", n: "有效样本数", note: "口径说明", metrics: "指标", aggregate: "汇总",
  ratio: "比例", seconds: "秒", usd: "美元", count: "数量", boolean: "是否", true: "是", false: "否",
  task_completion: "任务完成率", cases: "案例数", claims: "结论数",
  judge: "评分模型", accuracy_on_judged: "准确率（仅已判定）", positive_recall: "支持样本识别率",
  confusion: "正确与错误判定数", split_counts: "各划分样本数", controlled_regression: "受控回归",
  invalid_metric: "指标无效", queries: "查询数", avg_source_count: "平均来源数", avg_uncited_claims: "平均未引用结论数",
  avg_resolution_rate: "平均引用定位率", avg_cost_usd: "原记录平均成本（美元）", avg_latency_s: "平均耗时（秒）",
  precision_at_5: "前 5 项精确率", latency_s: "耗时（秒）", gate_pass_rate: "门禁通过率",
  dev: "开发集", holdout: "独立留出集", split: "数据划分", baseline: "基线", candidate: "候选版本",
};

export function evaluationLabel(value: string): string {
  if (Object.hasOwn(labels, value)) return labels[value];
  const rank = /^(recall|precision|ndcg)(?:_at_|@)(\d+)$/.exec(value);
  if (rank) return `前 ${rank[2]} 项${({ recall: "召回率", precision: "精确率", ndcg: "排序质量" } as Record<string, string>)[rank[1]]}`;
  return value;
}

const notes: Readonly<Record<string, string>> = {
  "Claim-weighted conditional support; unjudged excluded; not the official mean across reports": "按结论数加权，仅计算已判定结论；不等于官方逐报告平均支持率。",
  "Nearest-rank research-only P95; Judge time excluded; unmeasured if any completed latency is missing": "按最近秩法统计研究耗时的 95% 分位值，不含评分耗时；成功任务有耗时缺失时保持未测量。",
  "Original pricing not verified; a reported zero is not free API usage": "原始计价未经验证；记录为零不代表接口调用免费。",
  "All Replay cases, including expected Agent failures": "包含全部回放案例，也包括预期研究失败的场景。",
  "Recorded aggregate; n counts available per-query observations for this metric, not all queries": "沿用原始汇总值；样本数仅统计该指标有有效观测的查询，不一定是全部查询数。",
  "Recorded conditional rate; source validity retained; no unified release gate": "沿用原始条件分母和有效性标记；不参与统一发布门禁。",
  "Judge returned only YES for all measured claims; calibrate against labeled negative controls.": "评分模型对所有已判定结论均给出支持，应结合有标签的负例校准。",
  "Unjudged claims were excluded from citation-support denominator; inspect judge batch logs.": "引用支持率分母不包含未判定结论，请另查评分覆盖率和批次日志。",
  "0.0 cost with nonzero token counts may mean unpriced provider, not zero actual spend.": "用量非零而成本为零，可能是供应商尚未计价，并不代表没有实际费用。",
  "Report JSON lacks cited evidence snippets and full runtime spans; root cause is not provable from this artifact alone.": "历史报告缺少引用证据片段和完整执行轨迹，仅凭此产物无法确认根因。",
  "Legacy Judge parser out-of-range Claim labels are an OPEN defect; this adapter does not fix or rescore them.": "该历史产物记录了旧评分解析器的越界编号缺陷；本视图不会修改或重新计算历史成绩。",
  "Original calibration samples are not linked; case count not verified from source bytes": "尚未关联原始校准样本，案例数未经源文件核对。",
  "Source case count differs from registry declaration": "源文件中的案例数与登记数不一致。",
  "One or more supporting files unavailable": "一个或多个支撑文件不可用。",
  "Dataset source unavailable or invalid": "数据集源文件不可用或无效。",
};

export function evaluationNote(value: string): string {
  return Object.hasOwn(notes, value) ? notes[value] : value;
}

export function evaluationValue(value: unknown): string {
  if (value == null) return "未测量";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "string") return evaluationLabel(value);
  return String(value);
}
