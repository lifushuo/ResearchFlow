import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { CorpusTree } from "@/lib/corpusTree";

import { FolderDialog } from "./FolderDialog";

/**
 * The folder actions a user can take on the Corpus page (RFC V3.1 §7.1). A delete states what
 * it removes before anything is sent, so these tests read the confirmation, not just the
 * request.
 */

const update = vi.fn();
const remove = vi.fn();

vi.mock("@/hooks/queries", () => ({
  useUpdateFolder: () => ({ mutate: update, isPending: false }),
  useDeleteFolder: () => ({ mutate: remove, isPending: false }),
}));

const TREE: CorpusTree = {
  folders: [
    { id: "f-thesis", parent_id: null, name: "Thesis" },
    { id: "f-ch1", parent_id: "f-thesis", name: "Ch1" },
    { id: "f-notes", parent_id: null, name: "Notes" },
  ],
  documents: [],
  memberships: [
    { doc_key: "k1", folder_id: "f-ch1", display_name: "a.pdf" },
    { doc_key: "k2", folder_id: "f-thesis", display_name: "b.pdf" },
    { doc_key: "k2", folder_id: "f-notes", display_name: "b.pdf" },
  ],
};

function open(action: "rename" | "move" | "delete", folderId = "f-thesis") {
  const onClose = vi.fn();
  const folder = TREE.folders.find((f) => f.id === folderId)!;
  render(
    <FolderDialog projectId="p1" tree={TREE} folder={folder} action={action} onClose={onClose} />,
  );
  return onClose;
}

beforeEach(() => {
  update.mockReset();
  remove.mockReset();
});

describe("FolderDialog", () => {
  it("says what a delete removes before sending it, and starts on Cancel", () => {
    open("delete");
    const dialog = screen.getByRole("alertdialog", { name: "确定删除“Thesis”吗？" });
    // k1 is filed only beneath Thesis; k2 is also in Notes, so it stays.
    expect(dialog).toHaveTextContent("将删除此文件夹及其 1 个子文件夹。");
    expect(dialog).toHaveTextContent("仅存放于此处的 1 份文档及其所有版本也将删除。");
    expect(dialog).toHaveTextContent("同时归属其他文件夹的文档会保留。");
    expect(screen.getByRole("button", { name: "取消" })).toHaveFocus();
    expect(remove).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "删除文件夹" }));
    expect(remove).toHaveBeenCalledWith({ projectId: "p1", folderId: "f-thesis" }, expect.anything());
  });

  it("says when a delete removes no files", () => {
    open("delete", "f-notes");
    expect(screen.getByRole("alertdialog")).toHaveTextContent("不会删除任何文档。");
  });

  it("closes on Escape without sending anything", () => {
    const onClose = open("delete");
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
    expect(remove).not.toHaveBeenCalled();
  });

  it("renames to the trimmed name, and sends nothing for an unchanged one", () => {
    const onClose = open("rename");
    const field = screen.getByLabelText("新名称");
    expect(field).toHaveFocus();
    fireEvent.change(field, { target: { value: "  Thesis  " } });
    fireEvent.click(screen.getByRole("button", { name: "重命名" }));
    expect(update).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();

    fireEvent.change(field, { target: { value: " Dissertation " } });
    fireEvent.click(screen.getByRole("button", { name: "重命名" }));
    expect(update).toHaveBeenCalledWith(
      { projectId: "p1", folderId: "f-thesis", name: "Dissertation" },
      expect.anything(),
    );
  });

  it("moves anywhere but into itself or beneath itself", () => {
    open("move");
    const options = screen.getAllByRole("option").map((o) => o.textContent);
    expect(options).toEqual(["顶层目录", "Notes"]);
    fireEvent.change(screen.getByLabelText("移动到"), { target: { value: "f-notes" } });
    fireEvent.click(screen.getByRole("button", { name: "移动" }));
    expect(update).toHaveBeenCalledWith(
      { projectId: "p1", folderId: "f-thesis", parentId: "f-notes" },
      expect.anything(),
    );
  });
});
