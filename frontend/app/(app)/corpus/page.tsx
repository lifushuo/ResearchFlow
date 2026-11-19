"use client";

import { useRef, useState } from "react";
import toast from "react-hot-toast";

import { useActiveProject } from "@/components/ActiveProject";
import { CorpusTree } from "@/components/corpus/CorpusTree";
import { GeneratedBadge } from "@/components/GeneratedBadge";
import { documentUrl } from "@/components/preview/DocumentPreview";
import { PreviewDrawer } from "@/components/preview/PreviewDrawer";
import { EmptyState } from "@/components/ui/EmptyState";
import {
  useCorpusDocuments,
  useCorpusStatus,
  useCorpusTree,
  useDeleteDocument,
  useUploadDocument,
} from "@/hooks/queries";
import { ApiError } from "@/lib/api";
import { filesFromDrop } from "@/lib/dropFiles";

/**
 * Corpus management (docs/12 M10).
 *
 * Uploads run one file at a time on purpose. Each ingest embeds every chunk, so firing a
 * folder of PDFs concurrently would stampede the embedding endpoint — and on a local
 * model, thrash a single set of weights. A queue with visible per-file state is both
 * kinder to the backend and more honest: you can see which file is being worked and which
 * one failed, instead of one spinner that hides a partial failure.
 */

/** What the backend will accept: exactly `documents.FORMATS`. A backend test reads this
 * list and fails when the two differ, so the picker never offers a file the upload refuses
 * or refuses one it would accept. */
const ACCEPTED = [
  ".pdf",
  ".html",
  ".htm",
  ".md",
  ".markdown",
  ".txt",
  ".text",
  ".rst",
  ".json",
  ".docx",
  ".xlsx",
  ".xlsm",
  ".xls",
  ".ods",
  ".pptx",
  ".odt",
  ".epub",
  ".rtf",
  ".csv",
  ".tsv",
  ".ipynb",
  ".py",
  ".java",
  ".r",
  ".m",
  ".jl",
  ".c",
  ".h",
  ".cpp",
  ".hpp",
  ".cc",
  ".cs",
  ".js",
  ".ts",
  ".tsx",
  ".jsx",
  ".go",
  ".rs",
  ".rb",
  ".scala",
  ".kt",
  ".swift",
  ".sql",
  ".sh",
  ".ps1",
  ".yaml",
  ".yml",
  ".toml",
  ".ini",
  ".cfg",
  ".xml",
  ".tex",
  ".sty",
  ".cls",
  ".bib",
];
const MAX_BYTES = 25 * 1024 * 1024;

type Outcome = "queued" | "uploading" | "done" | "failed" | "skipped";

interface QueueItem {
  key: string;
  file: File;
  /** Set when the file arrived via a folder pick, so its origin stays visible. */
  path?: string;
  state: Outcome;
  detail?: string;
}

function formatBytes(bytes?: number | null): string | null {
  if (bytes === null || bytes === undefined) return null;
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${Math.round(kb)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

/** Reject locally what the server would reject anyway. A folder pick can hand us hundreds
 *  of unrelated files; uploading 25 MB to be told "wrong format" wastes the user's time
 *  and the embedding endpoint's. */
function rejectionReason(file: File): string | null {
  if (!ACCEPTED.includes(extensionOf(file.name))) {
    return `不支持此格式；支持：${ACCEPTED.join("、")}`;
  }
  if (file.size > MAX_BYTES) return `文件过大；最大 ${formatBytes(MAX_BYTES)}`;
  if (file.size === 0) return "文件为空";
  return null;
}

const STATE_STYLE: Record<Outcome, { label: string; token: string }> = {
  queued: { label: "等待上传", token: "text-muted" },
  uploading: { label: "上传中", token: "info" },
  done: { label: "已添加", token: "success" },
  skipped: { label: "已跳过", token: "warning" },
  failed: { label: "失败", token: "danger" },
};

function QueueRow({ item }: { item: QueueItem }) {
  const style = STATE_STYLE[item.state];
  const color = `var(--${style.token})`;
  return (
    <li className="flex items-center gap-3 px-3 py-2 font-mono text-xs">
      <span
        className="min-w-0 flex-1 truncate text-text-primary"
        title={item.path ?? item.file.name}
      >
        {item.path ?? item.file.name}
      </span>
      <span className="shrink-0 tabular-nums text-text-muted">{formatBytes(item.file.size)}</span>
      <span
        className="shrink-0 border px-1.5 py-0.5 text-[0.625rem] font-semibold uppercase tracking-wider"
        style={{
          color,
          backgroundColor: `color-mix(in srgb, ${color} 10%, var(--bg-surface))`,
          borderColor: `color-mix(in srgb, ${color} 30%, var(--border))`,
        }}
        title={item.detail}
      >
        {style.label}
      </span>
    </li>
  );
}

export default function CorpusPage() {
  // Which document the drawer is showing, or null. Identified by id + filename because
  // the preview needs both: the id builds the URL, the filename picks the renderer.
  const [preview, setPreview] = useState<{ id: string; filename: string } | null>(null);
  const { activeId, active } = useActiveProject();
  const { data: status, refetch: refetchStatus } = useCorpusStatus(activeId);
  const { data: docs, isLoading: docsLoading } = useCorpusDocuments(activeId);
  const { data: tree } = useCorpusTree(activeId);
  const upload = useUploadDocument();
  const del = useDeleteDocument();

  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  // Read inside the upload loop, so a ref: state would be the value captured when it began.
  const stop = useRef(false);
  const [stopping, setStopping] = useState(false);

  if (!activeId) {
    return (
      <EmptyState
        title="尚未选择项目"
        description="请先选择或创建项目，再管理其语料库。"
      />
    );
  }

  const runQueue = async (files: { file: File; path?: string }[]) => {
    if (!files.length) return;
    const items: QueueItem[] = files.map(({ file, path }, i) => {
      const reason = rejectionReason(file);
      return {
        key: `${file.name}-${file.size}-${i}`,
        file,
        path,
        state: reason ? "skipped" : "queued",
        detail: reason ?? undefined,
      };
    });
    setQueue(items);
    setBusy(true);
    stop.current = false;
    setStopping(false);

    let added = 0;
    let failed = 0;
    let stopped = 0;
    for (const item of items) {
      if (item.state === "skipped") continue;
      // Stopping waits for the file in flight: a half-sent upload is never ingested, so the
      // one already going either lands whole or fails, and nothing after it is sent.
      if (stop.current) {
        stopped += 1;
        setQueue((q) =>
          q.map((x) =>
            x.key === item.key ? { ...x, state: "skipped", detail: "已停止，未上传" } : x,
          ),
        );
        continue;
      }
      setQueue((q) => q.map((x) => (x.key === item.key ? { ...x, state: "uploading" } : x)));
      try {
        await upload.mutateAsync({
          projectId: activeId,
          file: item.file,
          relativePath: item.path,
        });
        added += 1;
        setQueue((q) => q.map((x) => (x.key === item.key ? { ...x, state: "done" } : x)));
      } catch (err) {
        failed += 1;
        const detail = err instanceof ApiError ? err.message : "上传失败";
        // One bad file must not abandon the rest of a folder — record it and continue.
        setQueue((q) => q.map((x) => (x.key === item.key ? { ...x, state: "failed", detail } : x)));
      }
    }

    setBusy(false);
    setStopping(false);
    refetchStatus();
    const skipped = items.filter((i) => i.state === "skipped").length;
    if (added) toast.success(`已添加 ${added} 份文档`);
    if (failed || skipped) {
      toast.error(
        [failed ? `${failed} 份失败` : "", skipped ? `${skipped} 份已跳过` : ""]
          .filter(Boolean)
          .join(", "),
      );
    }
    if (stopped) toast(`已停止：${stopped} 份未上传`);
  };

  const fromInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(e.target.files ?? []).map((file) => ({
      file,
      // webkitRelativePath is set only by the folder picker; it files each document in
      // the matching folders (created on the server as needed).
      path: (file as File & { webkitRelativePath?: string }).webkitRelativePath || undefined,
    }));
    e.target.value = "";
    void runQueue(picked);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (busy) return;
    // Walks dropped folders where the browser allows it; elsewhere the files still upload,
    // without their folders (lib/dropFiles.ts).
    void filesFromDrop(e.dataTransfer).then(runQueue);
  };

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`确定删除“${filename}”吗？`)) return;
    try {
      await del.mutateAsync({ projectId: activeId, docId });
      toast.success(`已删除“${filename}”`);
      refetchStatus();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "删除失败。");
    }
  };

  const doneCount = queue.filter((q) => q.state === "done").length;

  return (
    <div className="space-y-8">
      <section aria-labelledby="corpus-management">
        <h1
          id="corpus-management"
          className="mb-1 font-serif text-2xl font-bold tracking-tight text-text-primary"
        >
          语料库管理
        </h1>
        <p className="mb-6 max-w-2xl text-sm leading-relaxed text-text-muted">
          上传文档后，可将研究限定在本地语料库内。当前项目：
          <strong className="font-semibold text-text-primary">{active?.name}</strong>。本项目已批准的报告也会自动归档于此，并标注为
          <span className="border border-border bg-bg-elevated px-1.5 py-0.2 font-mono text-[0.6875rem] font-semibold uppercase tracking-wider text-text-secondary">自动生成</span>，
          不会被循环用作后续问题的研究证据。
        </p>

        {/* 1. Drag & Drop Upload Zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            if (!busy) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className="mb-8 border-2 border-dashed p-8 text-center transition-all shadow-xs"
          style={{
            borderColor: dragging ? "var(--accent)" : "var(--border)",
            backgroundColor: dragging
              ? "color-mix(in srgb, var(--accent) 8%, var(--bg-surface))"
              : "var(--bg-surface)",
          }}
        >
          <div className="mx-auto flex h-11 w-11 items-center justify-center border border-border bg-bg-elevated text-lg text-accent">
            📄
          </div>
          <p className="mt-3 text-sm font-semibold text-text-primary">
            将文件或文件夹拖到这里上传
          </p>
          <div className="mt-2 flex flex-wrap items-center justify-center gap-1.5 font-mono text-[length:var(--text-micro)] text-text-muted">
            {ACCEPTED.map((ext) => (
              <span key={ext} className="border border-border bg-bg-elevated px-1.5 py-0.5">
                {ext}
              </span>
            ))}
            <span className="ml-1 text-text-muted">· 单个文件最大 {formatBytes(MAX_BYTES)}</span>
          </div>

          <div className="mt-5 flex flex-wrap items-center justify-center gap-2.5">
            <button
              type="button"
              className="btn btn-primary px-4 py-2 text-xs font-medium"
              disabled={busy}
              onClick={() => fileInput.current?.click()}
            >
              {busy && <span className="spinner" />}
              选择文件
            </button>
            <button
              type="button"
              className="btn btn-secondary px-4 py-2 text-xs font-medium"
              disabled={busy}
              onClick={() => folderInput.current?.click()}
            >
              选择文件夹
            </button>
          </div>

          <input
            ref={fileInput}
            type="file"
            multiple
            accept={ACCEPTED.join(",")}
            onChange={fromInput}
            className="hidden"
          />
          {/* `webkitdirectory` is non-standard but is what every current browser
              implements for directory selection, and React has no typing for it. */}
          <input
            ref={folderInput}
            type="file"
            multiple
            onChange={fromInput}
            className="hidden"
            {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
          />
        </div>

        {tree && (
          <div className="mb-8">
            <CorpusTree tree={tree} projectId={activeId} />
          </div>
        )}

        {/* 2. Upload Queue */}
        {queue.length > 0 && (
          <div className="mb-8 border border-border bg-bg-surface shadow-xs">
            <div className="flex items-center justify-between border-b border-border bg-bg-elevated/40 px-3.5 py-2.5">
              <span className="font-mono text-xs font-semibold uppercase tracking-wider text-text-secondary">
                上传队列（已完成 {doneCount}/{queue.length}）
              </span>
              {busy && (
                <button
                  type="button"
                  disabled={stopping}
                  onClick={() => {
                    stop.current = true;
                    setStopping(true);
                  }}
                  className="font-mono text-xs text-text-muted transition-colors hover:text-text-primary disabled:opacity-60"
                >
                  {stopping ? "当前文件完成后停止…" : "停止上传"}
                </button>
              )}
              {!busy && (
                <button
                  type="button"
                  onClick={() => setQueue([])}
                  className="font-mono text-xs text-text-muted transition-colors hover:text-text-primary"
                >
                  清空队列
                </button>
              )}
            </div>
            <ul className="max-h-60 divide-y divide-border overflow-y-auto">
              {queue.map((item) => (
                <QueueRow key={item.key} item={item} />
              ))}
            </ul>
          </div>
        )}

        {preview && activeId && (
          <PreviewDrawer
            open
            onClose={() => setPreview(null)}
            url={documentUrl(activeId, preview.id)}
            filename={preview.filename}
            downloadable
          />
        )}

        {/* 3. Main Split Grid (Documents & Stats) */}
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            <div className="flex items-baseline justify-between gap-2">
              <h2 className="font-serif text-lg font-bold tracking-tight text-text-primary">
                文档列表
              </h2>
              {docs && docs.length > 0 && (
                <span className="border border-border bg-bg-elevated px-2 py-0.5 font-mono text-xs text-text-muted">
                  {docs.length} 份文档
                </span>
              )}
            </div>

            {docsLoading ? (
              <div className="h-28 animate-pulse border border-border bg-bg-elevated" />
            ) : docs && docs.length > 0 ? (
              <ul className="divide-y divide-border border border-border bg-bg-surface shadow-xs">
                {docs.map((doc) => {
                  const size = formatBytes(doc.size_bytes);
                  return (
                    <li
                      key={doc.id}
                      className="group flex items-center justify-between gap-3 p-4 transition-colors hover:bg-bg-elevated/60"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs text-text-muted" aria-hidden>
                            📄
                          </span>
                          <p className="truncate font-serif text-sm font-semibold text-text-primary group-hover:text-accent">
                            {doc.filename}
                          </p>
                          {doc.origin === "generated" && <GeneratedBadge className="shrink-0" />}
                        </div>
                        <div className="mt-1 flex flex-wrap items-center gap-2 font-mono text-xs text-text-muted">
                          <span>{doc.chunks} 个片段</span>
                          {size && <span>· {size}</span>}
                          {doc.created_at && <span>· {new Date(doc.created_at).toLocaleDateString("zh-CN")}</span>}
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-1.5">
                        {doc.downloadable ? (
                          <button
                            type="button"
                            onClick={() => setPreview({ id: doc.id, filename: doc.filename })}
                            className="btn btn-secondary px-2.5 py-1 font-mono text-xs font-medium text-accent hover:border-accent"
                          >
                            预览
                          </button>
                        ) : (
                          <span
                            className="px-2 py-0.5 font-mono text-xs text-text-muted"
                            title="此文档上传时尚未保留原文件，已提取的文本仍可检索。"
                          >
                            仅存文本
                          </span>
                        )}
                        <button
                          onClick={() => handleDelete(doc.id, doc.filename)}
                          disabled={del.isPending}
                          className="px-2 py-1 font-mono text-xs font-medium text-text-muted transition-colors hover:text-danger"
                        >
                          删除
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <div className="border border-dashed border-border bg-bg-surface p-10 text-center shadow-xs">
                <p className="text-sm font-medium text-text-secondary">尚未上传文档</p>
                <p className="mt-1 text-xs text-text-muted">
                  将文件拖到上方区域，或点击“选择文件”建立语料库。
                </p>
              </div>
            )}
          </div>

          {/* Right Sidebar: Corpus Stats & Privacy */}
          <div className="space-y-4">
            <h2 className="font-serif text-lg font-bold tracking-tight text-text-primary">
              语料库统计
            </h2>
            <div className="card space-y-4 border border-border bg-bg-surface p-5 shadow-xs">
              <div className="border-b border-border/60 pb-3">
                <div className="font-mono text-xs uppercase tracking-wider text-text-muted">
                  文档总数
                </div>
                <div className="mt-1 font-mono text-2xl font-bold tabular-nums text-text-primary">
                  {status ? status.documents : "—"}
                </div>
              </div>
              <div className="border-b border-border/60 pb-3">
                <div className="font-mono text-xs uppercase tracking-wider text-text-muted">
                  片段总数
                </div>
                <div className="mt-1 font-mono text-2xl font-bold tabular-nums text-text-primary">
                  {status ? status.chunks : "—"}
                </div>
              </div>
              <div>
                <div className="font-mono text-xs uppercase tracking-wider text-text-muted">
                  向量嵌入模型
                </div>
                <div className="mt-1.5 truncate border border-border bg-bg-elevated px-2 py-1 font-mono text-xs font-semibold text-accent">
                  {status?.current_model || "未配置"}
                </div>
              </div>

              <div className="border-t border-border/60 pt-3">
                <div className="flex items-center gap-1.5 text-xs text-text-muted font-mono">
                  <span className="h-2 w-2 rounded-full bg-success" aria-hidden />
                  本地存储
                </div>
                <p className="mt-1 text-[length:var(--text-micro)] text-text-muted leading-relaxed">
                  文档经向量化后，存储在当前项目的本地语料库中。
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
