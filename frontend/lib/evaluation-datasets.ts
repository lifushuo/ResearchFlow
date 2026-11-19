import { apiFetch } from "@/lib/api";

export type DatasetEntry = {
  id: string;
  title: string;
  origin: "upstream" | "custom";
  category: string;
  split: string;
  collection: string;
  source: string | null;
  declared_count: number;
  verified_count: number | null;
  count_basis: "source_verified" | "registry_declared" | "reported_only" | "unmeasured";
  source_status: string;
  source_sha256: string | null;
  supporting_file_count: number;
  supporting_files_valid: boolean | null;
  execution: string;
  score_family: string;
  description: string;
  warnings: string[];
};

export type DatasetCase = {
  case_id: string;
  index: number;
  question_preview?: string;
  kind?: string;
  domain?: string;
  depth?: string;
  match_mode?: string;
};

export type DatasetDetail = DatasetEntry & {
  cases: DatasetCase[];
  offset: number;
  limit: number;
  has_more: boolean;
};

export function evaluationDatasets() {
  return apiFetch<{ schema_version: number; registry_sha256: string; read_only: boolean; datasets: DatasetEntry[] }>(
    "/evaluation/datasets",
  );
}

export function evaluationDataset(id: string, offset = 0, limit = 50) {
  return apiFetch<{ schema_version: number; registry_sha256: string; read_only: boolean; dataset: DatasetDetail }>(
    `/evaluation/datasets/${encodeURIComponent(id)}?offset=${offset}&limit=${limit}`,
  );
}
