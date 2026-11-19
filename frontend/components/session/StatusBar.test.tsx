import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { SessionDetail } from "@/lib/types";

import { StatusBar } from "./StatusBar";

/** 会话状态栏不是描述列表，不能直接使用 dt/dd。其 Stat 由 div/span 构成；工作区的同名基础组件则要求调用方提供 dl。本测试固定这一差异，避免日后替换组件时悄悄引入无效结构。 */

const session = {
  created_at: "2026-09-27T00:00:00Z",
  elapsed_seconds: 12,
  total_cost_usd: 0.01,
} as unknown as SessionDetail;

describe("StatusBar", () => {
  it("shows its three figures", () => {
    render(<StatusBar session={session} events={[]} running={false} />);
    for (const label of ["已用时间", "费用", "任务"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it("has no term or description outside a dl", () => {
    const { container } = render(<StatusBar session={session} events={[]} running={false} />);
    for (const el of container.querySelectorAll("dt, dd")) {
      expect(el.closest("dl")).not.toBeNull();
    }
  });
});
