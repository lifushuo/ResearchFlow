import { evaluationLabel, evaluationNote, evaluationValue } from "@/lib/evaluation-display-zh";

export function EvaluationMetrics({ metrics, nested = false }: { metrics: Record<string, unknown>; nested?: boolean }) {
  return <dl className={`mt-2 grid min-w-0 gap-2 ${nested ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-2"}`}>
    {Object.entries(metrics).map(([key, value]) => <div key={key} className="min-w-0 rounded border border-border p-2">
      <dt className="text-text-muted break-words">{evaluationLabel(key)}</dt>
      <dd className="mt-1 break-words text-text-primary">
        {value !== null && typeof value === "object"
          ? <EvaluationMetrics metrics={value as Record<string, unknown>} nested />
          : key === "note" && typeof value === "string" ? evaluationNote(value) : evaluationValue(value)}
      </dd>
    </div>)}
  </dl>;
}
