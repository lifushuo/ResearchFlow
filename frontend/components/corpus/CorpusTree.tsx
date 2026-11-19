"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { FolderDialog, type FolderAction } from "@/components/corpus/FolderDialog";
import {
  buildTree,
  filterTree,
  folderState,
  selectionRequest,
  toggleDocument,
  toggleFolder,
  type CorpusTree as Tree,
  type FolderNode,
  type TreeDocument,
  type TreeFolder,
} from "@/lib/corpusTree";

const actionLabels: Record<FolderAction, string> = { rename: "重命名", move: "移动", delete: "删除" };

/**
 * The corpus as folders, with checkboxes for choosing what a research run may read
 * (RFC V3.1 §12). The selection lives in this component only — never in web storage — and
 * becomes a request on the research form's URL, where the server resolves and freezes it.
 *
 * ARIA tree pattern: the list is `role="tree"`, every row a `treeitem` with `aria-level`,
 * folders `aria-expanded`, and checkboxes `aria-checked="mixed"` when partly selected.
 */
export function CorpusTree({ tree, projectId }: { tree: Tree; projectId: string }) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [acting, setActing] = useState<{ folder: TreeFolder; action: FolderAction } | null>(
    null,
  );
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Set<string>>(new Set());
  const { roots, loose } = useMemo(() => buildTree(tree), [tree]);
  const shown = useMemo(() => filterTree(roots, query), [roots, query]);
  const looseShown = query.trim()
    ? loose.filter((d) => d.filename.toLowerCase().includes(query.trim().toLowerCase()))
    : loose;

  const request = selectionRequest(roots, loose, selected);
  const href =
    "/research?" +
    new URLSearchParams({
      ...(request.folder_ids.length ? { folders: request.folder_ids.join(",") } : {}),
      ...(request.doc_keys.length ? { files: request.doc_keys.join(",") } : {}),
    }).toString();

  if (!roots.length && !loose.length) return null;

  const toggleOpen = (id: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const renderDoc = (doc: TreeDocument, level: number) => (
    <li key={`${level}-${doc.id}`} role="treeitem" aria-level={level} aria-selected={selected.has(doc.doc_key as string)}>
      <label className="flex cursor-pointer items-center gap-2 py-1 text-sm text-text-primary">
        <input
          type="checkbox"
          checked={selected.has(doc.doc_key as string)}
          onChange={() => setSelected((s) => toggleDocument(doc.doc_key as string, s))}
        />
        <span className="truncate">{doc.filename}</span>
      </label>
    </li>
  );

  const renderFolder = (node: FolderNode, level: number) => {
    const state = folderState(node, selected);
    const expanded = open.has(node.folder.id) || !!query.trim();
    return (
      <li
        key={node.folder.id}
        role="treeitem"
        aria-level={level}
        aria-expanded={expanded}
        aria-selected={state === "checked"}
      >
        <div className="flex items-center gap-2 py-1 text-sm">
          <button
            type="button"
            className="w-5 text-text-muted"
            aria-label={`${expanded ? "收起" : "展开"} ${node.folder.name}`}
            onClick={() => toggleOpen(node.folder.id)}
          >
            {expanded ? "▾" : "▸"}
          </button>
          <input
            type="checkbox"
            aria-label={`选择“${node.folder.name}”中的所有文件`}
            aria-checked={state === "mixed" ? "mixed" : state === "checked"}
            checked={state === "checked"}
            ref={(el) => {
              if (el) el.indeterminate = state === "mixed";
            }}
            onChange={() => setSelected((s) => toggleFolder(node, s))}
          />
          <span className="font-semibold text-text-primary">{node.folder.name}</span>
          <span className="ml-auto flex gap-1">
            {(["rename", "move", "delete"] as const).map((action) => (
              <button
                key={action}
                type="button"
                className="btn btn-ghost px-1.5 py-0.5 text-xs"
                aria-label={`${actionLabels[action]} ${node.folder.name}`}
                onClick={() => setActing({ folder: node.folder, action })}
              >
                {actionLabels[action]}
              </button>
            ))}
          </span>
        </div>
        {expanded && (
          <ul role="group" className="ml-6">
            {node.folders.map((f) => renderFolder(f, level + 1))}
            {node.documents.map((d) => renderDoc(d, level + 1))}
          </ul>
        )}
      </li>
    );
  };

  return (
    <section aria-labelledby="corpus-select" className="border border-border bg-bg-surface p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h2 id="corpus-select" className="font-serif text-lg font-bold text-text-primary">
          选择研究文件
        </h2>
        <input
          type="search"
          placeholder="按名称筛选"
          aria-label="按名称筛选文件和文件夹"
          className="border border-border bg-bg-elevated px-2 py-1 text-sm"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      {acting && (
        <FolderDialog
          projectId={projectId}
          tree={tree}
          folder={acting.folder}
          action={acting.action}
          onClose={() => setActing(null)}
        />
      )}
      <ul role="tree" aria-label="语料库文件夹和文件" aria-multiselectable="true">
        {shown.map((n) => renderFolder(n, 1))}
        {looseShown.map((d) => renderDoc(d, 1))}
      </ul>
      <div className="mt-4 flex items-center gap-3">
        {selected.size > 0 ? (
          <Link href={href} className="btn btn-primary">
            研究所选文件（{selected.size}）
          </Link>
        ) : (
          <span className="text-sm text-text-muted">
            选择文件或文件夹，再提出仅依据这些资料回答的问题。
          </span>
        )}
        {selected.size > 0 && (
          <button type="button" className="btn" onClick={() => setSelected(new Set())}>
            清除选择
          </button>
        )}
      </div>
    </section>
  );
}
