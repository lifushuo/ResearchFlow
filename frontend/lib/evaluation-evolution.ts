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

export type BenchmarkCase = {
  case_id: string;
  query: string;
  depth: string;
  required_aspects: string[] | null;
  rationale: string | null;
};

export type BenchmarkCaseSet = {
  dataset_id: string;
  case_count: number;
  annotated_case_count: number;
  cases: BenchmarkCase[];
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
  stage: "planner" | "planner_complete" | "retrieval" | "report";
  bad_case_id?: string;
  evaluation_run_id?: string;
  evaluation_case_id?: string;
  manual_query?: string;
  manual_depth?: "fast" | "balanced" | "comprehensive";
  planner_tasks?: string[];
  missing_aspects?: string[];
  evidence_reference?: string;
  review_confirmed?: boolean;
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
    planner_tasks: string[];
    missing_aspects: string[];
    evidence_reference: string | null;
    evidence_sha256: string | null;
    review_confirmed: boolean;
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

export type PlannerEvidence = {
  status: "recorded" | "unavailable";
  run_id: string;
  reason?: string;
  case_id?: string;
  tasks: { id: string; query: string }[];
  prompt_sha256?: string | null;
  trace_id?: string | null;
  evidence_reference?: string;
};

export function evolutionPlannerEvidence(badCaseId: string) {
  return apiFetch<PlannerEvidence>(`${base}/bad-cases/${encodeURIComponent(badCaseId)}/planner-evidence`);
}

export type ReviewableEvaluationCase = {
  run_id: string;
  case_id: string;
  result_status: "passed" | "failed" | "blocked" | "unmeasured";
  query: string;
  depth: string;
  required_aspects: string[];
};

export function evolutionReviewableCases() {
  return apiFetch<{ cases: ReviewableEvaluationCase[] }>(`${base}/reviewable-cases`);
}

export function evolutionReviewableCaseEvidence(runId: string, caseId: string) {
  return apiFetch<PlannerEvidence>(`${base}/reviewable-cases/${encodeURIComponent(runId)}/${encodeURIComponent(caseId)}/planner-evidence`);
}

export type PlannerJudgeSuggestion = {
  status: "suggested";
  run_id: string;
  case_id: string;
  evidence_reference: string;
  question: string;
  judge_route: string;
  served_judge: string | null;
  judge_tokens: { input: number; output: number } | null;
  coverage: number;
  automatic_label: false;
  aspects: {
    aspect: string;
    aspect_index: number;
    covered: boolean;
    task_ids: number[];
    reason: string;
  }[];
};

export function evolutionPlannerJudge(
  runId: string, caseId: string, evidenceReference: string,
) {
  return apiFetch<PlannerJudgeSuggestion>(
    `${base}/reviewable-cases/${encodeURIComponent(runId)}/${encodeURIComponent(caseId)}/judge-suggestion`,
    {
      method: "POST",
      headers: { "X-ResearchFlow-Eval-Action": "planner-judge-v1" },
      body: { evidence_reference: evidenceReference, acknowledge_cost: true },
    },
  );
}

export function evolutionBenchmarkCases() {
  return apiFetch<BenchmarkCaseSet>(`${base}/benchmark-cases`);
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
  [/Planner 漏项|Planner 任务|漏项维度|证据编号|任务证据/, "Planner 漏项必须附真实任务列表、未覆盖维度和 Trace 证据；请先完成人工复核。"],
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

// GEPA jobs: only immutable server-approved options reach the backend.
export type GEPAScores = { coverage: number; specificity: number; plan_quality: number; redundancy: number; quality: number };
export type GEPAFrozen = { request_timeouts?: { planner: number; judge: number; reflection: number }; metric_version: string; metric_rules_sha256: string; baseline_sha256: string; min_gain: number; routes: { pipeline: Record<string,string>; judge: string; reflection: string } };
export type GEPABudget = { configured: { total_cap: number; quotas: Record<string,number> }; used_total: number | null; remaining_total: number | null; unknown: string[] | null; unknown_details?: { slot_id: string; role: string; cause_type: string | null }[]; simulated: boolean; error?: string };
export type GEPAComparison = { baseline: GEPAScores | null; candidate: GEPAScores | null; quality_delta: number | null; same_prompt: boolean };

export type GEPAJob = {
  task_kind?: "search" | "planner_final_ab";
  source_job_id?: string | null;
  frozen?: GEPAFrozen | null;
  request_cap?: number | null;
  dataset_preset?: string;
  job_id: string;
  mode: "demo" | "live";
  metric_version: "v1" | "v2" | "v3";
  max_evals: number;
  proposals: number;
  seed: number;
  dataset_sha256: string | null;
  state: "queued" | "running" | "cancelling" | "cancelled" | "succeeded" | "quality_rejected" | "pending_final" | "no_candidate" | "failed";
  phase: string;
  error_type?: string | null;
  quality_gate_passed?: boolean | null;
  created_at: string;
  updated_at: string;
};

export type GEPAResult = {
  task_kind?: "planner_final_ab";
  source_job_id?: string;
  release_ready?: boolean;
  comparison?: GEPAComparison;
  outcome?: string;
  search_only?: boolean;
  rejection_reasons?: string[] | null;
  budget?: GEPABudget | null;
  job_id: string;
  available: boolean;
  quality_gate_passed?: boolean | null;
  holdout?: {
    passed?: boolean;
    reasons?: string[];
    delta?: number | null;
    baseline_coverage?: number | null;
    candidate_coverage?: number | null;
  };
  baseline_sha256?: string;
  candidate_sha256?: string | null;
  prompt_diff?: string;
  validation_gain?: number | null;
  evidence_kind?: string;
};

export function evolutionGEPAJobs() {
  return apiFetch<{ jobs: GEPAJob[] }>(`${base}/jobs`);
}

export type GEPAFinalPreview = {
  source_job_id: string; candidate_sha256: string; baseline_sha256: string;
  fake_enabled: boolean; live_enabled: boolean; live_disabled_reason: string | null;
  case_count: number; evaluations: number; request_cap: number;
  quotas: { planner: number; judge: number; reflection: number }; min_gain: number;
};
export function evolutionFinalPreview(jobId: string) {
  return apiFetch<GEPAFinalPreview>(`${base}/jobs/${encodeURIComponent(jobId)}/final/preview`);
}
export function evolutionStartFinal(jobId: string, body: { request_id: string; mode: "demo" | "live"; consent: string | null }) {
  return apiFetch<GEPAJob>(`${base}/jobs/${encodeURIComponent(jobId)}/final`, {
    method: "POST", headers: { "X-ResearchFlow-Eval-Action": "planner-final-ab-v1" }, body,
  });
}

export function evolutionGEPACapabilities() {
  return apiFetch<{ demo_enabled: boolean; live_enabled: boolean; max_evals_limit: number; proposals_limit: number; estimated_api_cost_is_not_a_hard_dollar_cap: boolean }>(`${base}/jobs/capabilities`);
}

export function evolutionStartGEPA(body: {
  request_id: string;
  mode: "demo" | "live";
  metric_version: "v1" | "v2" | "v3";
  max_evals: number;
  proposals: number;
  seed: number;
  dataset_preset?: "smoke" | "full";
  consent?: string | null;
}) {
  return apiFetch<GEPAJob>(`${base}/jobs`, {
    method: "POST",
    headers: { "X-ResearchFlow-Eval-Action": "planner-gepa-optimize-v1" },
    body,
  });
}

export function evolutionCancelGEPA(jobId: string) {
  return apiFetch<GEPAJob>(`${base}/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
    headers: { "X-ResearchFlow-Eval-Action": "planner-gepa-cancel-v1" },
  });
}

export function evolutionGEPAResult(jobId: string) {
  return apiFetch<GEPAResult>(`${base}/jobs/${encodeURIComponent(jobId)}/result`);
}

// P2: read-only GEPA audit evidence. The backend never returns raw queries,
// prompts, provider messages, or invented Pareto parent-selection steps.
export type GEPAObservationEvent = {
  sequence: number;
  kind: string;
  recorded_at?: string;
  prompt_sha256?: string;
  case_key?: string | null;
  split?: "train" | "validation" | "holdout";
  coverage?: number | null;
  scores?: GEPAScores | null;
  measured?: boolean;
  task_count?: number | null;
  response_sha256?: string | null;
  candidate_sha256?: string | null;
  accepted_for_holdout?: boolean;
  holdout_passed?: boolean | null;
  error_type?: string;
  slot_id?: string | null;
  role?: string | null;
  cause_type?: string | null;
};

export type GEPAAudit = {
  task_kind?: "search" | "planner_final_ab";
  source_job_id?: string | null;
  frozen?: GEPAFrozen | null;
  budget?: GEPABudget | null;
  job_id: string;
  mode: "demo" | "live";
  state: string;
  phase: string | null;
  dataset_sha256: string | null;
  holdout_sha256: string | null;
  limits: { max_evals: number; proposals: number; seed: number };
  journal: {
    available: boolean;
    error: string | null;
    event_count: number;
    events_truncated: boolean;
    events: GEPAObservationEvent[];
    candidates: {
      sha256: string;
      samples: number;
      measured: number;
      errors: number;
      splits: Record<string, { count: number; measured: number; mean_coverage: number | null; scores?: GEPAScores | null }>;
    }[];
    observation_count: number;
    reflection_count: number;
    holdout_observed: boolean;
    observed_tokens: Record<string, number | null>;
    dataset_id: string | null;
    model_routes: { planner: string | null; judge: string | null; reflection: string | null } | null;
    pareto_parent_selection: "not_recorded";
    token_scope: string;
  };
  holdout_policy: string;
  result_integrity: "verified" | "not_available" | "mismatch";
  result_versions: { optimizer: string | null; optimizer_version: string; metric_version: string;
    baseline_sha256: string | null; candidate_sha256: string | null; report_dataset_sha256: string | null } | null;
  full_report_ab: string;
};

export function evolutionGEPAAudit(jobId: string) {
  return apiFetch<GEPAAudit>(`${base}/jobs/${encodeURIComponent(jobId)}/audit`);
}
