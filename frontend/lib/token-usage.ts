export type ModelTokenUsage = {
  kind: "research" | "judge";
  llm_mode: "real" | "fake" | "unknown";
  served_model: string | null;
  configured_model: string | null;
  roles: string[];
  calls: number;
  failed_calls: number;
  missing_usage_calls: number;
  known_input_tokens: number | null;
  known_output_tokens: number | null;
  known_total_tokens: number | null;
};

export type TokenUsage = {
  schema_version: number;
  scope: "observed_application_calls";
  models: ModelTokenUsage[];
  calls: number;
  missing_usage_calls: number;
  collection_incomplete: boolean;
  known_input_tokens: number | null;
  known_output_tokens: number | null;
  known_total_tokens: number | null;
};
