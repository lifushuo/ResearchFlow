import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import CorpusPage from "./page";

/**
 * Stopping a folder upload part-way (RFC V3.1 G-2). The file in flight finishes or fails on
 * its own — a half-sent upload is never ingested — and nothing after it is sent.
 */

let release: () => void = () => {};
const mutateAsync = vi.fn(
  () =>
    new Promise<void>((resolve) => {
      release = resolve;
    }),
);

vi.mock("@/components/ActiveProject", () => ({
  useActiveProject: () => ({ activeId: "p1", active: { id: "p1", name: "P" } }),
}));

vi.mock("@/hooks/queries", () => ({
  useCorpusStatus: () => ({ data: undefined, refetch: vi.fn() }),
  useCorpusDocuments: () => ({ data: [], isLoading: false }),
  useCorpusTree: () => ({ data: undefined }),
  useUploadDocument: () => ({ mutateAsync }),
  useDeleteDocument: () => ({ mutate: vi.fn() }),
}));

function folderOf(names: string[]): File[] {
  return names.map((name) => {
    const file = new File(["text"], name, { type: "text/plain" });
    Object.defineProperty(file, "webkitRelativePath", { value: `Thesis/${name}` });
    return file;
  });
}

describe("CorpusPage upload queue", () => {
  it("stops after the file in flight and sends nothing after it", async () => {
    const { container } = render(<CorpusPage />);
    const input = container.querySelector<HTMLInputElement>("input[webkitdirectory]")!;
    fireEvent.change(input, { target: { files: folderOf(["a.txt", "b.txt", "c.txt"]) } });

    fireEvent.click(await screen.findByRole("button", { name: "停止上传" }));
    expect(screen.getByRole("button", { name: "当前文件完成后停止…" })).toBeDisabled();

    await act(async () => release());

    expect(mutateAsync).toHaveBeenCalledTimes(1);
    expect(screen.getByText("上传队列（已完成 1/3）")).toBeInTheDocument();
    const stopped = screen.getAllByTitle("已停止，未上传");
    expect(stopped).toHaveLength(2);
    expect(screen.getByRole("button", { name: "清空队列" })).toBeInTheDocument();
  });
});
