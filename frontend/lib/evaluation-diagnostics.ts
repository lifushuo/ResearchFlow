import { apiFetch } from "./api";
import type { DatasetId } from "./evaluation-pipeline-v13";

export type Candidate = { cause: string; confidence: string; evidence_ids: string[]; verification: string };
export type JudgeProgress = {
  case_id?: string; phase?: string; trace_complete: boolean; observed_batches?: number;
  observed_application_level_calls?: number; judge_coverage?: number | null; observed_judge_coverage?: number | null;
  total_claims?: number | null; total_batches?: number | null; successful_batches?: number;
  attempted_batches?: number; new_requested_batches?: number; responses_received?: number | null;
  successful_requests?: number | null; valid_result_batches?: number; judged_claims_total?: number | null;
  unjudged_claims_total?: number | null; full_judge_coverage?: number | null; attempted_judge_coverage?: number | null;
  scoring_complete?: boolean; problem_batches?: Record<string, unknown>[];
  batch_measurements?: { batch_no: number; status: string; elapsed_ms: number | null; tokens_input: number | null; tokens_output: number | null; cost_usd: number | null }[];
  known_input_tokens?: number; known_output_tokens?: number; batches_missing_usage?: number;
  wait_observations?: { batch_no: number; wait_reason?: string | null; elapsed_ms?: number | null }[];
  events?: { event: string; seq: number; batch_no?: number; timestamp?: string | null }[];
};
export type ReportState = {
  case_id?: string; phase?: string; research: string; judge: string; stop_reason?: string | null;
  total_claims?: number; judged_claims?: number | null; committed_claims?: number; committed_batches?: number;
  resume: string; evaluation: string; quality_gate: string;
};
export type TraceEvent = { seq: number; timestamp?: string | null; case_id: string; span_id: string | null; name: string | null; kind: string | null; status: string | null; duration_ms: number | null; error_type: string | null; metadata: Record<string, string | number>; details?: Record<string, unknown> };
export type UnifiedTrace = {
  sources: { id: string; sha256: string | null; run_id: string | null; case_id: string | null; status: string }[];
  events: (Partial<TraceEvent> & { evidence_id: string; source: string; event?: string; elapsed_ms?: number; parent_span_id?: string | null })[];
  coverage: { loaded_events: number; content_events: number; content_truncated_events: number; not_collected: string[]; note: string };
};
export type Timeline = {
  summary: string; phase: string | null; certainty: string; evidence_ids: string[]; worker_wall_s: number | null;
  stages: { name: string; duration_s: number; completed_request_s?: number | null; outside_completed_request_s?: number | null; recorded_wait_s?: number | null; finished_batches?: number; closed?: boolean; basis: string }[];
  operations: { kind: string; name: string; calls: number; sum_s: number | null; max_s: number | null; errors: number }[];
  slowest: { case_id: string; kind: string; name: string; duration_ms: number; status: string; evidence_id: string }[];
  symptoms: { name: string; count: number; error_type: string | null; workflow_continued: boolean; evidence_ids: string[] }[];
  retry_observations: { planner_attempts: number | null; extra_executor_rounds: number | null; sdk_retry_attempts: null };
  parallel_note: string; wait_note: string;
};
export type Diagnostic = {
  token_usage?: import("./token-usage").TokenUsage | null;
  available_case_ids?: string[];
  case_evidence?: { report_excerpt: string; report_truncated: boolean; claims: { text: string; judged: boolean; supported: boolean | null }[]; claims_truncated: boolean };
  pipeline_id: string; dataset_id: DatasetId; case_id: string | null; state: string; reason: string | null;
  elapsed_seconds: number | null; has_issue: boolean; root_cause_confirmed: false;
  execution_observation: { started_at: string | null; finished_at: string | null; exit_code: number | null; timed_out: boolean | null };
  collection: string; failure_phase: string | null; quality_measurement: string; evidence_fingerprint: string;
  facts: { id: string; text: string; span_id?: string }[]; candidates: Candidate[]; next_actions: string[]; missing_evidence: string[];
  traces: { trace_id?: string; case_id?: string; trace_status: string; open_spans?: string[]; events: TraceEvent[]; sha256?: string }[];
  judge_traces: JudgeProgress[];
  report_states?: ReportState[];
  cases: { case_id: string; status: string; latency_s: number | null; claim_count: number | null; metrics?: Record<string, unknown>; error?: string | null }[];
  metrics: Record<string, unknown>;
  timeline?: Timeline;
  unified_trace?: UnifiedTrace;

};
export type Incident = {
  incident_id: string; pipeline_id: string; dataset_id: DatasetId; case_id: string | null;
  reason: string; occurrences: string[]; summary: string; collection: string;
};
export const getDiagnostics = (pipe: string, dataset: DatasetId, caseId?: string) => apiFetch<Diagnostic>(`/evaluation/pipelines/${encodeURIComponent(pipe)}/diagnostics/${encodeURIComponent(dataset)}${caseId ? `?case_id=${encodeURIComponent(caseId)}` : ""}`);
export const getIncidents = (pipe?: string, dataset?: DatasetId | "") => {
  const query = new URLSearchParams(); if (pipe) query.set("pipeline_id", pipe); if (dataset) query.set("dataset_id", dataset);
  return apiFetch<{ incidents: Incident[] }>(`/evaluation/closure/execution-incidents${query.size ? `?${query}` : ""}`);
};
export function openDiagnostics(pipe: string, dataset: DatasetId, caseId?: string) {
  window.dispatchEvent(new CustomEvent("rf-evaluation-diagnostics", { detail: { pipe, dataset, caseId } }));
}
