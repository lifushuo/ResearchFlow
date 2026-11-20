import { apiFetch } from "@/lib/api";

// 自进化闭环：评测中心坏例回流、开发数据集合并视图与实验证据投影。
// 与评测闭环同一约定：读操作需要登录，写操作必须携带显式动作头。
const base = "/evaluation/evolution";
const BACKFLOW_ACTION = "planner-backflow-v1";

export type EvolutionCase = {
  case_id: string;
  group: string;
  split: "train" | "validation" | "holdout";
  query: string;
  depth: string;
  required_aspects: string[];
  source_bad_case_id: string | null;
};

export type EvolutionOverview = {
  dataset: {
    seed_case_count: number;
    backflow_case_count: number;
    review_record_count: number;
    group_count: number | null;
    splits: Record<string, number> | null;
    dataset_sha256: string | null;
    error: string | null;
  };
  benchmark: { case_count: number; annotated_case_count: number };
  bad_cases: { open_benchmark_count: number; flowed_count: number };
  experiments: {
    directory_count: number;
    last_holdout_gate_passed: boolean | null;
    last_report_gate_passed: boolean | null;
    release_count: number;
  };
};

export type EvolutionDatasetState = {
  seed_case_count: number;
  backflow_case_count: number;
  review_record_count: number;
  unreadable_record_count?: number;
  dataset_sha256: string | null;
  error: string | null;
  splits?: Record<string, number>;
  group_count?: number;
  dataset?: { schema_version: number; dataset_id: string; provenance: string; cases: EvolutionCase[] };
};

export type BenchmarkAnnotations = {
  schema_version: number;
  dataset_id: string;
  source: string;
  note: string;
  aspects: Record<string, { required_aspects: string[]; rationale: string }>;
};

export type EvolutionBadCase = {
  bad_case_id: string;
  case_id: string;
  error_key: string;
  review_status: string;
  first_run_id: string;
  last_run_id: string;
  last_seen_ms: number;
  query: string | null;
  depth: string;
  suggested_aspects: string[] | null;
};

export type EvolutionExperimentRow = {
  name: string;
  experiment:
    | {
        created_at: string;
        evidence_kind: string;
        metric_version: string;
        optimizer: { name: string; version: string } | null;
        limits: Record<string, unknown> | null;
        dataset_sha256: string | null;
        case_count: number;
        routes: Record<string, unknown> | null;
        judge_settings: Record<string, unknown> | null;
        validation_gain: number | null;
        holdout_gate: { passed?: boolean; reasons?: string[]; delta?: number | null; baseline_coverage?: number | null; candidate_coverage?: number | null } | null;
        candidate: string | null;
        has_prompt_diff: boolean;
      }
    | null;
  report_gate:
    | {
        passed: boolean | null;
        cases: number | null;
        served_judge: string | null;
        judge_identity_status: string | null;
        baseline: Record<string, unknown> | null;
        candidate: Record<string, unknown> | null;
      }
    | null;
  release:
    | {
        version: string | null;
        approved_by: string | null;
        approved_at: string | null;
        prompt_sha256: string | null;
        parent_prompt_sha256: string | null;
      }
    | null;
};

export type BackflowRequest = {
  case_id: string;
  group: string;
  aspects: string[];
  review_note: string;
  stage: "planner" | "retrieval" | "report";
  bad_case_id?: string;
  manual_query?: string;
  manual_depth?: "fast" | "balanced" | "comprehensive";
};

export type BackflowResult = {
  case: EvolutionCase;
  review: {
    stage: string;
    review_note: string;
    reviewed_at: string;
    source: { type: string; bad_case_id: string | null; benchmark_case_id: string | null; run_id: string | null; error_key: string | null };
    benchmark_aspects: string[] | null;
    task_evidence_status: string;
  };
};

export function evolutionOverview() {
  return apiFetch<EvolutionOverview>(`${base}/overview`);
}

export function evolutionDataset() {
  return apiFetch<EvolutionDatasetState>(`${base}/dataset`);
}

export function evolutionAnnotations() {
  return apiFetch<BenchmarkAnnotations>(`${base}/annotations`);
}

export function evolutionBadCases() {
  return apiFetch<{ bad_cases: EvolutionBadCase[]; annotated_cases: number }>(`${base}/bad-cases`);
}

export function evolutionExperiments() {
  return apiFetch<EvolutionExperimentRow[]>(`${base}/experiments`);
}

export function evolutionBackflow(body: BackflowRequest) {
  return apiFetch<BackflowResult>(`${base}/backflow`, {
    method: "POST",
    headers: { "X-ResearchFlow-Eval-Action": BACKFLOW_ACTION },
    body,
  });
}

const messages: Array<[RegExp, string]> = [
  [/Explicit backflow action/, "缺少显式回流动作头，请求被拒绝。"],
  [/基准题源不一致/, "坏例最近一次输入与基准题源不一致，请先重新运行评测再回流。"],
  [/必须来自已测量的/, "坏例必须来自已测量的研究报告评测运行。"],
  [/该坏例已回流/, "这条坏例已经回流过，不能重复进入开发集。"],
  [/案例 ID|分组|必答维度|复核结论|研究问题|归因阶段/, "回流内容未通过校验，请按提示修改后重试。"],
  [/已存在|回流记录/, "回流记录已存在或目录状态异常。"],
  [/Evaluation storage unavailable|评测卷/, "评测卷暂不可用，请稍后重试或检查服务端配置。"],
  [/Resource not found/, "记录不存在，或当前账号无权访问。"],
];

export function evolutionError(error: unknown): string {
  const raw = error instanceof Error ? error.message : "";
  return messages.find(([pattern]) => pattern.test(raw))?.[1] ?? "操作未完成，请稍后重试或查看技术详情。";
}

export const stageLabel: Record<string, string> = {
  planner: "Planner 漏项（进开发集）",
  retrieval: "检索环节问题（仅存证）",
  report: "报告环节问题（仅存证）",
};

export const splitLabel: Record<string, string> = {
  train: "训练集",
  validation: "验证集",
  holdout: "留出集",
};
