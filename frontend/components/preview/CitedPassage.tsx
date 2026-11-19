"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import {
  citedDocumentPath,
  citedDocumentUrl,
  kindForFilename,
} from "@/components/preview/DocumentPreview";
import { PreviewDrawer } from "@/components/preview/PreviewDrawer";
import { apiFetch } from "@/lib/api";
import type { CorpusLocator } from "@/lib/corpusLocator";

type Extracted = {
  filename: string;
  kind: string;
  text: string;
  segments: [number, string][];
  extraction_mode: "verbatim" | "linearized";
};

/** The segment a character offset falls in — the server's `segment_label`, for display. */
function labelAt(segments: [number, string][], offset: number): string | null {
  let label: string | null = null;
  for (const [start, name] of segments) if (start <= offset) label = name;
  return label;
}

/**
 * A corpus citation shown in place: the document's canonical extracted text — exactly what
 * the quote was verified against — with the cited span highlighted and scrolled into view
 * (RFC V3.1 §12). Text is rendered as text, never as markup, so a document cannot inject
 * anything into the page. A linearized document says so beside the quote.
 */
export function CitedPassage({
  owner,
  locator,
}: {
  /** The run or session that cited it: `/runs/{id}` or `/research/{id}`. */
  owner: string;
  locator: CorpusLocator;
}) {
  const mark = useRef<HTMLElement>(null);
  const [original, setOriginal] = useState(false);
  const { data, error, isLoading } = useQuery({
    queryKey: ["cited-document", owner, locator.documentId],
    queryFn: () => apiFetch<Extracted>(citedDocumentPath(owner, locator.documentId)),
  });
  useEffect(() => {
    mark.current?.scrollIntoView({ block: "center" });
  }, [data]);

  if (isLoading) return <p className="text-xs text-text-muted">正在打开引用片段…</p>;
  if (error || !data)
    return (
      <p className="text-xs text-danger">
        暂时无法读取语料库中的此文档。研究记录仍保留引用证据，但当前无法重新打开完整文件。
      </p>
    );

  const start = Math.max(0, Math.min(locator.start ?? 0, data.text.length));
  const end = Math.max(start, Math.min(locator.end ?? start, data.text.length));
  const segment = labelAt(data.segments, start);
  const where = segment?.replace(/^page (\d+)$/i, "第 $1 页") ?? (locator.page ? `第 ${locator.page} 页` : null);
  // The extracted text is what the quote was checked against; the PDF is what a reader
  // trusts. Offered only for a PDF, opened at the cited page when the locator has one.
  const pdf = kindForFilename(data.filename) === "pdf";
  return (
    <div className="space-y-2">
      <p className="text-xs text-text-secondary">
        <span className="font-semibold text-text-primary">{data.filename}</span>
        {where ? ` — ${where}` : ""}
        {data.extraction_mode === "linearized" &&
          " · 引用自此文件表格、幻灯片或单元格转换后的文本"}
      </p>
      {pdf && (
        <button
          type="button"
          onClick={() => setOriginal(true)}
          className="text-xs text-accent hover:underline"
        >
          {locator.page ? `打开 PDF 第 ${locator.page} 页 →` : "打开 PDF →"}
        </button>
      )}
      {pdf && original && (
        <PreviewDrawer
          open
          onClose={() => setOriginal(false)}
          url={citedDocumentUrl(owner, locator.documentId)}
          filename={data.filename}
          downloadable
          subtitle={locator.page ? `引用自第 ${locator.page} 页` : "引用自语料库"}
          page={locator.page}
        />
      )}
      <pre className="max-h-96 overflow-auto whitespace-pre-wrap border border-border bg-bg-elevated p-3 text-xs leading-relaxed text-text-secondary">
        {data.text.slice(Math.max(0, start - 2000), start)}
        <mark ref={mark} className="bg-warning-soft text-text-primary">
          {data.text.slice(start, end)}
        </mark>
        {data.text.slice(end, end + 2000)}
      </pre>
    </div>
  );
}
