import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DemoNotice } from "./DemoNotice";

vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: React.ReactNode }) => (
    <a href={href}>{children}</a>
  ),
}));

/** 研究任务和会话共用同一警告，避免文案漂移。不能把去设置当成唯一补救：服务端本身处于演示模式时，全部研究都是演示，设置中的密钥不会被使用。 */
describe("DemoNotice", () => {
  it("names what did not happen before anything else", () => {
    render(<DemoNotice />);
    const notice = screen.getByRole("note", { name: /演示研究/i });
    expect(notice).toHaveTextContent("演示研究——未使用大模型进行实际研究");
    expect(notice).toHaveTextContent(
      "你的问题未发送给模型。本报告是脚本演示",
    );
  });

  it("says where demo mode comes from when it is the server, not the user", () => {
    render(<DemoNotice />);
    expect(screen.getByRole("note", { name: /演示研究/i })).toHaveTextContent(
      /服务端本身处于演示模式/,
    );
    expect(screen.getByRole("link", { name: "设置" })).toHaveAttribute("href", "/settings");
  });
});
