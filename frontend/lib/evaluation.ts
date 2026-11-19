// The API keeps English detail strings (interface contract). Translate the known
// ones for display; keep any unknown original text so failures stay diagnosable.
const ERROR_MESSAGES: Array<[RegExp, string]> = [
  [/Invalid evaluation trace/i, "轨迹文件无效"],
  [/Evaluation run not found/i, "未找到该评测运行"],
  [/Trace not found/i, "未找到该轨迹"],
  [/Evaluation datasets unavailable or invalid/i, "评测数据集清单不可用或无效"],
  [/Evaluation dataset not found/i, "未找到该评测数据集"],
  [/Invalid dataset registry or pagination/i, "数据集清单或分页参数无效"],
  [/runner_not_configured/i, "离线运行器尚未配置（RF_EVAL_JOB_ROOT）"],
  [/runner_storage_unavailable/i, "离线任务存储卷不可用"],
  [/dataset_not_executable/i, "该评测集不支持在线运行"],
  [/invalid_request_id/i, "请求编号无效"],
  [/idempotency_conflict/i, "同一请求编号已用于其他任务"],
  // v1.3 unified pipeline: backend pipeline errors and worker failure reasons.
  [/Explicit evaluation action required/i, "缺少显式统一评测动作标记，请求已拒绝"],
  [/Pipeline not found/i, "未找到该统一评测任务"],
  [/unknown_dataset|invalid_dataset_selection/i, "评测集选择无效"],
  [/unknown_report_case_id|invalid_report_case_id/i, "找不到指定的报告案例，未启动评测"],
  [/duplicate_report_case_id/i, "原始报告数据集存在重复案例标识，未启动评测"],
  [/case_id_requires_report_benchmark_only/i, "指定案例时只能选择研究报告评测"],
  [/case_id_requires_max_cases_1/i, "指定案例时案例上限必须为 1"],
  [/max_cases_out_of_range_1_to_20/i, "案例数量超出允许范围（1～20 或完整数据集）"],
  [/explicit_paid_confirmation_required/i, "请先勾选真实模型评测费用确认"],
  [/explicit_full_dataset_confirmation_required/i, "完整数据集需要单独勾选费用确认"],
  [/invalid_paid_confirmation|unexpected_paid_confirmation/i, "费用确认参数无效"],
  [/online_runner_disabled/i, "在线执行未启用，请在服务端配置后再试"],
  [/one_active_pipeline_per_user/i, "当前账号已有统一评测任务在执行，请等待其结束"],
  [/pipeline_queue_full|pipeline_history_full/i, "统一评测队列已满，请稍后重试"],
  [/Not authenticated/i, "请先登录"],
  [/diagnostic_storage_unavailable/i, "诊断记录暂时无法读取，请稍后刷新。尚未提交本次操作。"],
];

export function evaluationErrorMessage(error: unknown, fallback: string): string {
  const raw = error instanceof Error ? error.message : "";
  for (const [pattern, text] of ERROR_MESSAGES) {
    if (pattern.test(raw)) return text;
  }
  return raw ? `${fallback}（${raw}）` : fallback;
}
