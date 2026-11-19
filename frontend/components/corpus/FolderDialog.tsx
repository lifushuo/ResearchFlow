"use client";

import { useRef, useState } from "react";
import toast from "react-hot-toast";

import { useDeleteFolder, useUpdateFolder } from "@/hooks/queries";
import { useDialogFocus } from "@/hooks/useDialogFocus";
import { ApiError } from "@/lib/api";
import {
  folderDeletion,
  moveTargets,
  type CorpusTree,
  type TreeFolder,
} from "@/lib/corpusTree";

export type FolderAction = "rename" | "move" | "delete";

const TOP_LEVEL = "";

/**
 * Rename, move or delete one corpus folder (RFC V3.1 §7.1).
 *
 * An in-page dialog rather than the browser's `confirm`/`prompt`: those are drawn by the
 * webview, and the three desktop webviews do not draw them alike, so a confirmation the user
 * must read before files are deleted is rendered by the page itself.
 *
 * The delete confirmation states what will go before anything is sent — the subfolders, and
 * the files filed nowhere else — and the toast afterwards reports the server's own count.
 */
export function FolderDialog({
  projectId,
  tree,
  folder,
  action,
  onClose,
}: {
  projectId: string;
  tree: CorpusTree;
  folder: TreeFolder;
  action: FolderAction;
  onClose: () => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const firstRef = useRef<HTMLInputElement & HTMLSelectElement & HTMLButtonElement>(null);
  useDialogFocus(true, onClose, panelRef, firstRef);

  const update = useUpdateFolder();
  const remove = useDeleteFolder();
  const [name, setName] = useState(folder.name);
  const [parent, setParent] = useState<string>(folder.parent_id ?? TOP_LEVEL);
  const pending = update.isPending || remove.isPending;

  const fail = (err: unknown) =>
    toast.error(err instanceof ApiError ? err.message : "操作未成功，未做任何更改。");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (action === "rename") {
      const next = name.trim();
      if (!next || next === folder.name) return onClose();
      update.mutate(
        { projectId, folderId: folder.id, name: next },
        { onSuccess: () => onClose(), onError: fail },
      );
    } else if (action === "move") {
      const next = parent === TOP_LEVEL ? null : parent;
      if (next === (folder.parent_id ?? null)) return onClose();
      update.mutate(
        { projectId, folderId: folder.id, parentId: next },
        { onSuccess: () => onClose(), onError: fail },
      );
    } else {
      remove.mutate(
        { projectId, folderId: folder.id },
        {
          onSuccess: (done) => {
            toast.success(
              `已删除 ${done.folders_deleted} 个文件夹和 ${done.documents_deleted} 份文档`,
            );
            onClose();
          },
          onError: fail,
        },
      );
    }
  };

  const title =
    action === "rename"
      ? `重命名“${folder.name}”`
      : action === "move"
        ? `移动“${folder.name}”`
        : `确定删除“${folder.name}”吗？`;
  const doomed = action === "delete" ? folderDeletion(tree, folder.id) : null;

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center p-4">
      <button
        type="button"
        aria-label="关闭且不做更改"
        tabIndex={-1}
        onClick={onClose}
        className="absolute inset-0 bg-black/40"
      />
      <div
        ref={panelRef}
        role={action === "delete" ? "alertdialog" : "dialog"}
        aria-modal="true"
        aria-labelledby="folder-dialog-title"
        aria-describedby={doomed ? "folder-dialog-consequence" : undefined}
        tabIndex={-1}
        className="relative w-full max-w-md border border-border bg-bg-surface p-5 shadow-xl"
      >
        <form onSubmit={submit}>
          <h2 id="folder-dialog-title" className="font-serif text-lg font-bold text-text-primary">
            {title}
          </h2>

          {action === "rename" && (
            <label className="mt-3 block text-sm text-text-secondary">
              新名称
              <input
                ref={firstRef}
                value={name}
                onChange={(e) => setName(e.target.value)}
                maxLength={255}
                className="mt-1 block w-full border border-border bg-bg-elevated px-2 py-1 text-text-primary"
              />
            </label>
          )}

          {action === "move" && (
            <label className="mt-3 block text-sm text-text-secondary">
              移动到
              <select
                ref={firstRef}
                value={parent}
                onChange={(e) => setParent(e.target.value)}
                className="mt-1 block w-full border border-border bg-bg-elevated px-2 py-1 text-text-primary"
              >
                <option value={TOP_LEVEL}>顶层目录</option>
                {moveTargets(tree, folder.id).map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.name}
                  </option>
                ))}
              </select>
            </label>
          )}

          {doomed && (
            <div id="folder-dialog-consequence" className="mt-3 space-y-2 text-sm text-text-secondary">
              <p>
                {doomed.subfolders > 0
                  ? `将删除此文件夹及其 ${doomed.subfolders} 个子文件夹。`
                  : "将删除此文件夹。"}
                <strong className="text-text-primary">
                  {doomed.files > 0
                    ? `仅存放于此处的 ${doomed.files} 份文档及其所有版本也将删除。`
                    : "不会删除任何文档。"}
                </strong>
              </p>
              <p>
                同时归属其他文件夹的文档会保留。引用过已删除文档的研究仍保留当时收集的引文。
              </p>
            </div>
          )}

          <div className="mt-5 flex justify-end gap-2">
            <button
              ref={action === "delete" ? firstRef : undefined}
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
              disabled={pending}
            >
              取消
            </button>
            <button
              type="submit"
              className={action === "delete" ? "btn btn-danger" : "btn btn-primary"}
              disabled={pending || (action === "rename" && !name.trim())}
            >
              {action === "rename" ? "重命名" : action === "move" ? "移动" : "删除文件夹"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
