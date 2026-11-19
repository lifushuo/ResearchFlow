import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { FirstRunNotice } from "./FirstRunNotice";

/** 提示只能提供当前构建确实支持的免密钥路径。自托管服务可使用 ./start.sh --fake，桌面端没有此替代入口；不能引导用户勾选实际不存在的演示选项。 */

const readiness = vi.hoisted(() => ({ value: { ready: false, local_reachable: false } as object }));
const host = vi.hoisted(() => ({ desktop: false }));

vi.mock("@/hooks/queries", () => ({ useReadiness: () => ({ data: readiness.value }) }));
vi.mock("@/lib/desktop", () => ({
  get isDesktop() {
    return host.desktop;
  },
}));

beforeEach(() => {
  readiness.value = { ready: false, local_reachable: false };
  host.desktop = false;
});

describe("the keyless path the notice offers", () => {
  it("names the real demo command on a self-hosted stack", () => {
    render(<FirstRunNotice />);
    expect(screen.getByText("./start.sh --fake")).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent(/标记为演示/);
  });

  it("never points at a Demo run control under Options, which does not exist", () => {
    render(<FirstRunNotice />);
    expect(screen.getByRole("note")).not.toHaveTextContent(/演示研究/);
    expect(screen.getByRole("note")).not.toHaveTextContent(/under Options/);
  });

  it("offers no keyless path on the desktop, which has none", () => {
    host.desktop = true;
    render(<FirstRunNotice />);
    expect(screen.getByRole("note")).not.toHaveTextContent(/start\.sh/);
    expect(screen.getByRole("note")).not.toHaveTextContent(/演示研究/);
    expect(screen.getByRole("link", { name: "连接模型" })).toHaveAttribute(
      "href",
      "/settings",
    );
  });

  it("is not shown once a model can be reached", () => {
    readiness.value = { ready: true, local_reachable: false };
    const { container } = render(<FirstRunNotice />);
    expect(container).toBeEmptyDOMElement();
  });
});
