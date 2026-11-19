import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AccountMenu } from "./AccountMenu";
import { firstNameOf } from "@/lib/user";
import type { User } from "@/lib/types";

vi.mock("next/navigation", () => ({ usePathname: () => "/research", useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }));
vi.mock("next-themes", () => ({ useTheme: () => ({ resolvedTheme: "light", setTheme: vi.fn() }) }));
vi.mock("@/hooks/queries", () => ({
  useLogout: () => ({ mutate: vi.fn(), isPending: false }),
  // 模拟支持账户的宿主，以渲染完整菜单；这些测试检查
  // 界面在功能最完整时也不能显示邮箱地址。
  useCapabilities: () => ({ accounts: true }),
}));

const EMAIL = "ada.lovelace@analytical-engine.org";

function user(overrides: Partial<User> = {}): User {
  return {
    id: "u1",
    email: EMAIL,
    display_name: "Ada",
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  } as User;
}

/** 侧边栏会一直出现在共享屏幕、截图和旁观者视野中。邮箱既是身份标识，也可能是其他服务的登录凭据；此处显示名称和头像已经足够，邮箱只放在用户主动打开的个人资料页。 */
describe("AccountMenu keeps the address out of the chrome", () => {
  it("does not render the email in the resting sidebar row", () => {
    render(<AccountMenu user={user()} />);
    expect(screen.getByText("Ada")).toBeInTheDocument();
    expect(screen.queryByText(EMAIL)).toBeNull();
  });

  it("does not render the email once the menu is opened either", async () => {
    render(<AccountMenu user={user()} />);
    await userEvent.click(screen.getByRole("button", { name: /ada/i }));

    expect(screen.getByRole("menu", { name: "账户" })).toBeInTheDocument();
    expect(screen.queryByText(EMAIL)).toBeNull();
    // 邮箱只在个人资料中显示，因此保留一步直达该页面的入口。
    expect(screen.getByRole("menuitem", { name: /个人资料/i })).toHaveAttribute(
      "href",
      "/profile",
    );
  });

  it("does not leak the address through a tooltip when collapsed", () => {
    const { container } = render(<AccountMenu user={user()} collapsed />);
    const titles = [...container.querySelectorAll("[title]")].map((n) => n.getAttribute("title"));
    expect(titles.some((t) => t?.includes("@"))).toBe(false);
  });

  it("still names an account that has no display name, without showing the address", () => {
    render(<AccountMenu user={user({ display_name: null })} />);
    // 这里只展示推导出的本地名称，不是可用于发送邮件的完整地址。
    expect(screen.getByText(firstNameOf(user({ display_name: null })))).toBeInTheDocument();
    expect(screen.queryByText(EMAIL)).toBeNull();
  });
});
