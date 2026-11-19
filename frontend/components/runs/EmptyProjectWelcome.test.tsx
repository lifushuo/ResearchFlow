import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { EmptyProjectWelcome } from "./EmptyProjectWelcome";

describe("EmptyProjectWelcome", () => {
  it("names the project and offers both first steps, neither presented as required first", () => {
    render(<EmptyProjectWelcome projectName="Agent Memory" />);
    expect(
      screen.getByRole("heading", { name: /Agent Memory 已准备好开展首次研究/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "开始研究" })).toHaveAttribute(
      "href",
      "/research",
    );
    expect(screen.getByRole("link", { name: "添加语料材料" })).toHaveAttribute(
      "href",
      "/corpus",
    );
    expect(screen.getByText(/无需固定先后顺序/)).toBeInTheDocument();
  });

  it("does not claim nothing was ever asked, since archived sessions are excluded from the check", () => {
    render(<EmptyProjectWelcome projectName="Agent Memory" />);
    expect(screen.getByText(/还没有进行中的研究或来源材料/)).toBeInTheDocument();
    expect(screen.queryByText(/从未提出过任何问题/)).not.toBeInTheDocument();
  });
});
