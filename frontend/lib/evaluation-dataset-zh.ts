/**
 * Evaluation dataset PRESENTATION ONLY.
 *
 * Stable IDs belong to backend registry, pipeline jobs, stored results and SHA256 inventories.
 * Never translate or mutate IDs, dataset files, Gold labels, metrics, or API payloads here.
 * Unknown IDs fail open to the provided fallback rather than inventing a name.
 */
import type { DatasetId } from "@/lib/evaluation-pipeline-v13";

export type DatasetZhInfo = Readonly<{
  label: string;
  english: string;
  short: string;
  description: string;
  metrics: string;
}>;

export const DATASET_ZH = {
  "report-benchmark": {
    label: "研究报告质量评测",
    english: "Report Benchmark",
    short: "检查研究任务是否完成，报告是否有证据支持",
    description: "运行完整研究流程，先生成报告，再由独立评分模型判断引用是否支持结论。未判定的结论单独统计，不计作已验证。",
    metrics: "任务完成率、评分覆盖率、已判定结论的引用支持率、95% 分位耗时",
  },
  "corpus-retrieval": {
    label: "全库知识检索评测",
    english: "Corpus Retrieval",
    short: "检查能否从整个测试语料库找到相关文档和证据片段",
    description: "将实际检索结果与参考文档相关性标签比较，检查是否找全、是否找准、排序是否合理，以及证据片段是否定位正确。",
    metrics: "前若干项召回率、精确率、排序质量、证据片段定位准确率",
  },
  "selection-retrieval": {
    label: "指定文件检索评测",
    english: "Selection Retrieval",
    short: "检查检索是否限定在选中文件内，并找全证据段落",
    description: "只允许检索用户选中的文件，检查结果的相关性、是否返回范围外的文件，以及同一文档中的多段证据是否找全。",
    metrics: "召回率、精确率、排序质量、范围外文件占比、证据段落召回率",
  },
  "grounding-v1": {
    label: "证据支持与矛盾评测",
    english: "Grounding",
    short: "区分证据支持、证据不足和证据矛盾",
    description: "逐条检查结论与引用证据的关系，关注错误放行、错误拒绝、矛盾识别和无法判定。参考标签由独立模型生成，并非全部来自人工标注。",
    metrics: "错误放行率、错误拒绝率、未测量占比、矛盾召回率、错误放行率置信上界",
  },
  "judge-calibration": {
    label: "评分模型可靠性校准",
    english: "Judge Calibration",
    short: "检查负责评分的模型能否分辨支持与不支持",
    description: "使用带有支持或不支持标签的校准样本，分为开发集与独立留出集。短样本上的校准准确率不能等同于真实长报告上的准确率。",
    metrics: "准确率、判定覆盖率、正确与错误判定数、不支持样本识别率",
  },
  "planner-replay-v1": {
    label: "规划器故障回放测试",
    english: "Planner Replay",
    short: "检查正常规划、格式错误、模拟限流及上游异常的处理",
    description: "在实际编译后的工作流中注入脚本模型响应或模拟异常，检查调用次数、状态与执行轨迹。故障测试通过表示处理符合预期，不代表研究成功，也不能证明真实网络重试行为。",
    metrics: "预期行为检查、应用层调用次数、状态流转、执行轨迹完整性",
  },
  "executor-regression-v1": {
    label: "工具执行可靠性回归",
    english: "Executor Regression",
    short: "检查搜索、读取、提交证据，以及异常、去重和取消",
    description: "通过脚本模型和受控网络响应运行实际执行器，检查工具依赖、非法网址、内部地址访问防护、超时、重复处理和取消。成绩不代表付费模型的工具选择准确率。",
    metrics: "七场景预期行为断言、错误处理、证据路径、安全与取消",
  },
  "tool-contract-v1": {
    label: "工具调用契约评分测试",
    english: "Tool Contract",
    short: "检查合成调用记录中的工具、参数、顺序和越权问题",
    description: "将合成工具调用与参考调用比较，包含合法样本与刻意违规样本。通过表示满足所配置的工具契约，不代表真实模型的工具调用准确率；参数检查也不等于完整结构校验。",
    metrics: "工具选择、必需参数匹配、调用顺序、禁止工具识别",
  },
} satisfies Readonly<Record<DatasetId, DatasetZhInfo>>;

// 注册表里不进入 Pipeline 能力清单的数据集（自进化开发集）：同样只展示，不改 ID 与载荷。
const NON_PIPELINE_ZH = {
  "planner-evolution-reviewed": {
    label: "Planner 自进化开发集",
    english: "Planner Evolution Dev Set",
    short: "训练/验证/留出三组题目，人工标注必答维度",
    description: "Planner Prompt 自进化（GEPA）的开发集。由规划演化 CLI 与独立评分模型打分，不由评测 Worker 运行；配套文件标注研究报告评测的必答维度，供坏例回流预填。",
    metrics: "逐题必答维度覆盖率（训练/验证/留出）、留出集提升幅度",
  },
} satisfies Readonly<Record<string, DatasetZhInfo>>;

function datasetInfo(id: string): DatasetZhInfo | undefined {
  if (Object.hasOwn(DATASET_ZH, id)) return DATASET_ZH[id as DatasetId];
  return Object.hasOwn(NON_PIPELINE_ZH, id) ? NON_PIPELINE_ZH[id as keyof typeof NON_PIPELINE_ZH] : undefined;
}

export function getDatasetLabel(id: string, fallback?: string): string {
  return datasetInfo(id)?.label ?? fallback ?? id;
}
export function getDatasetShort(id: string, fallback = "评测数据集"): string {
  return datasetInfo(id)?.short ?? fallback;
}
export function getDatasetDescription(id: string, fallback = ""): string {
  return datasetInfo(id)?.description ?? fallback;
}
export function getDatasetMetrics(id: string, fallback = ""): string {
  return datasetInfo(id)?.metrics ?? fallback;
}
export function getDatasetEnglish(id: string): string {
  return datasetInfo(id)?.english ?? id;
}
