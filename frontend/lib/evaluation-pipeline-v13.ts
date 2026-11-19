import { apiFetch } from "@/lib/api";

export type DatasetId =
  | "report-benchmark" | "corpus-retrieval" | "selection-retrieval"
  | "grounding-v1" | "judge-calibration" | "planner-replay-v1"
  | "executor-regression-v1" | "tool-contract-v1";
export type SuiteCapability = { dataset_id: DatasetId; lane: "online" | "offline"; ready: boolean; reason: string };
export type PipelineStep = {
  dataset_id: DatasetId;
  lane: "online" | "offline";
  state: "pending" | "running" | "completed" | "failed" | "blocked";
  reason: string | null;
  case_count: number | null;
  elapsed_seconds: number | null;
  metrics: Record<string, unknown> | null;
  closure_run_id?: string | null;
  closure_status?: string;
};
export type Pipeline = {
  pipeline_id: string;
  datasets: DatasetId[];
  max_cases: number;
  case_id?: string;
  state: "queued" | "running" | "completed" | "completed_with_issues" | "failed";
  steps: PipelineStep[];
  created_at: string;
  updated_at: string;
  paid_execution_authorized: boolean;
  not_a_single_score: true;
};
export type Capabilities = {
  schema_version: number;
  online_enabled: boolean;
  paid_confirm_token_required: true;
  not_a_single_score: true;
  online_config_source?: "rf_eval_override" | "env" | "disabled";
  judge_config_source?: "rf_eval_override" | "env" | "none";
  judge_route_configured?: boolean;
  datasets: SuiteCapability[];
};
export const fetchPipelineCapabilities = () => apiFetch<Capabilities>("/evaluation/pipelines/capabilities");
export const fetchPipelines = () => apiFetch<{ pipelines: Pipeline[] }>("/evaluation/pipelines");
export const fetchPipeline = (id: string) => apiFetch<Pipeline>(`/evaluation/pipelines/${encodeURIComponent(id)}`);
export const createPipeline = (body: {
  datasets: DatasetId[];
  request_id: string;
  max_cases: number;
  consent: string | null;
  case_id?: string;
}) => apiFetch<Pipeline>("/evaluation/pipelines", {
  method: "POST", headers: { "X-ResearchFlow-Eval-Action": "run-pipeline-v1.3" }, body,
});
