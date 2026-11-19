import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { AgentRole, ModelCatalog, PromptDefaults, User } from "@/lib/types";

import { AgentsSection } from "./AgentsSection";

/** 智能体设置只管理行为，与模型选择分离（范围约定 §12）。仅展示五个公开角色及内置/自定义状态，一次打开一个编辑器；恢复默认通过 ResetToDefault 发送 null。模型只读展示并链接到修改位置，不创建角色，也不暴露受保护用途。 */

const ROLES: AgentRole[] = ["planner", "executor", "critic", "synthesizer", "chat"];
const LABELS = ["规划器", "执行器", "评审器", "综合器", "追问对话"];

const DEFAULTS: PromptDefaults = {
  max_chars: 2500,
  roles: ROLES.map((role) => ({
    role,
    default_prompt: `Shipped ${role} prompt.`,
    untrusted_content_framed: role !== "planner",
  })),
};

const CATALOG = {
  roles: ROLES,
  models: [
    {
      route: "anthropic:claude-sonnet",
      provider: "anthropic",
      model_id: "claude-sonnet",
      display_name: "Claude Sonnet",
    },
  ],
  effective_routing: Object.fromEntries(ROLES.map((r) => [r, "anthropic:claude-sonnet"])),
} as unknown as ModelCatalog;

const updateProfile = vi.hoisted(() => ({ mutateAsync: vi.fn(), isPending: false }));
const state = vi.hoisted(() => ({
  me: null as User | null,
  defaults: { data: undefined as PromptDefaults | undefined, isLoading: false, isError: false },
}));

vi.mock("@/hooks/queries", () => ({
  useMe: () => ({ data: state.me, isLoading: false }),
  usePromptDefaults: () => state.defaults,
  useModelCatalog: () => ({ data: CATALOG }),
  useUpdateProfile: () => updateProfile,
}));

function user(prompt_overrides: Partial<Record<AgentRole, string>> | null): User {
  return {
    id: "u1",
    email: "a@example.com",
    is_active: true,
    created_at: "2026-01-01T00:00:00Z",
    display_name: null,
    avatar_url: null,
    monthly_token_limit: 0,
    api_key_provider: null,
    api_key_hint: null,
    api_key_label: null,
    api_key_set_at: null,
    preferences: { prompt_overrides },
  };
}

function show(overrides: Partial<Record<AgentRole, string>> | null = null) {
  state.me = user(overrides);
  state.defaults = { data: DEFAULTS, isLoading: false, isError: false };
  return render(<AgentsSection />);
}

const card = (label: string) => screen.getByRole("listitem", { name: label });

beforeEach(() => {
  updateProfile.mutateAsync.mockReset().mockResolvedValue(user(null));
});

describe("the page", () => {
  it("says what it is for", () => {
    show();
    expect(screen.getByText("自定义研究智能体的行为。")).toBeInTheDocument();
  });

  it("lists exactly the five public agents, in order", () => {
    show();
    const items = within(screen.getByRole("list", { name: "智能体" })).getAllByRole("listitem");
    expect(items.map((li) => li.getAttribute("aria-label"))).toEqual(LABELS);
  });

  it("states that customization is recorded in the artifact and needs a current verifier", () => {
    show();
    expect(screen.getByText(/验证包会完整记录指令/)).toHaveTextContent(
      /使用 v2 格式，旧版验证器会拒绝/,
    );
  });

  it("opens no editor until asked", () => {
    show();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("offers no way to create an agent", () => {
    show();
    expect(screen.queryByRole("button", { name: /创建|添加|新建/ })).not.toBeInTheDocument();
  });

  it("never names a protected purpose, even with every editor opened in turn", async () => {
    const { container } = show({ critic: "Grade harshly." });
    for (const label of LABELS) {
      const button = within(card(label)).getByRole("button", {
        name: /自定义|编辑指令/,
      });
      await userEvent.click(button);
      await userEvent.click(screen.getByRole("button", { name: "取消" }));
    }
    for (const name of ["citation_verify", "contradiction_detector", "synthesizer.repair", "chat.project"]) {
      expect(container.textContent).not.toContain(name);
    }
  });
});

describe("each agent's state", () => {
  it("marks a shipped agent as shipped and offers Customize", () => {
    show();
    const planner = card("规划器");
    expect(within(planner).getByText("内置默认")).toBeInTheDocument();
    expect(within(planner).getByRole("button", { name: "自定义" })).toBeInTheDocument();
    expect(within(planner).queryByRole("button", { name: "恢复默认" })).not.toBeInTheDocument();
  });

  it("marks a customized agent and offers Edit instructions and Reset to default", () => {
    show({ critic: "Grade harshly." });
    const critic = card("评审器");
    expect(within(critic).getByText("已自定义")).toBeInTheDocument();
    expect(within(critic).getByRole("button", { name: "编辑指令" })).toBeInTheDocument();
    expect(within(critic).getByRole("button", { name: "恢复默认" })).toBeInTheDocument();
    expect(within(card("规划器")).getByText("内置默认")).toBeInTheDocument();
  });
});

describe("editing", () => {
  it("Customize opens that agent's editor on the shipped text", async () => {
    show();
    await userEvent.click(within(card("执行器")).getByRole("button", { name: "自定义" }));
    expect(screen.getByRole("textbox", { name: "执行器的指令" })).toHaveValue(
      "Shipped executor prompt.",
    );
  });

  it("Edit instructions opens that agent's editor on the saved text", async () => {
    show({ critic: "Grade harshly." });
    await userEvent.click(within(card("评审器")).getByRole("button", { name: "编辑指令" }));
    expect(screen.getByRole("textbox", { name: "评审器的指令" })).toHaveValue(
      "Grade harshly.",
    );
  });

  it("keeps one editor open at a time", async () => {
    show();
    await userEvent.click(within(card("规划器")).getByRole("button", { name: "自定义" }));
    expect(screen.getAllByRole("textbox")).toHaveLength(1);
    expect(within(card("评审器")).getByRole("button", { name: "自定义" })).toBeDisabled();
  });

  it("saves that agent's text alone", async () => {
    show();
    await userEvent.click(within(card("评审器")).getByRole("button", { name: "自定义" }));
    const box = screen.getByRole("textbox", { name: "评审器的指令" });
    await userEvent.clear(box);
    await userEvent.type(box, "Grade harshly.");
    await userEvent.click(screen.getByRole("button", { name: "保存指令" }));
    expect(updateProfile.mutateAsync).toHaveBeenCalledWith({
      preferences: { prompt_overrides: { critic: "Grade harshly." } },
    });
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("keeps the API's validation: a refusal shows inline and the editor stays open", async () => {
    updateProfile.mutateAsync.mockRejectedValue(
      new (await import("@/lib/api")).ApiError(
        422,
        "Value error, prompt override for 'critic' is 2501 characters; the maximum is 2500",
      ),
    );
    show();
    await userEvent.click(within(card("评审器")).getByRole("button", { name: "自定义" }));
    await userEvent.type(screen.getByRole("textbox"), " more");
    await userEvent.click(screen.getByRole("button", { name: "保存指令" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "prompt override for 'critic' is 2501 characters; the maximum is 2500",
    );
    expect(screen.getByRole("textbox")).toBeInTheDocument();
  });
});

describe("resetting", () => {
  it("shows one Reset to default per agent at a time — the card's yields to the editor's", async () => {
    show({ critic: "Grade harshly." });
    const critic = card("评审器");
    expect(within(critic).getAllByRole("button", { name: "恢复默认" })).toHaveLength(1);
    await userEvent.click(within(critic).getByRole("button", { name: "编辑指令" }));
    expect(within(critic).getAllByRole("button", { name: "恢复默认" })).toHaveLength(1);
  });

  it("can reset from inside an open editor, then closes it", async () => {
    show({ critic: "Grade harshly." });
    await userEvent.click(within(card("评审器")).getByRole("button", { name: "编辑指令" }));
    await userEvent.click(within(card("评审器")).getByRole("button", { name: "恢复默认" }));
    expect(updateProfile.mutateAsync).toHaveBeenCalledWith({
      preferences: { prompt_overrides: { critic: null } },
    });
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("sends null for that agent alone", async () => {
    show({ critic: "Grade harshly.", planner: "Plan in three steps." });
    await userEvent.click(within(card("评审器")).getByRole("button", { name: "恢复默认" }));
    expect(updateProfile.mutateAsync).toHaveBeenCalledWith({
      preferences: { prompt_overrides: { critic: null } },
    });
  });
});

describe("the model, read-only", () => {
  it("shows each agent's current model and links to Models to change it", () => {
    show();
    const planner = card("规划器");
    expect(within(planner).getByText(/模型：Claude Sonnet/)).toBeInTheDocument();
    expect(within(planner).getByRole("link", { name: "前往模型设置修改" })).toHaveAttribute(
      "href",
      "/settings/models",
    );
  });

  it("offers no model selector of its own", () => {
    show();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });
});

describe("loading and failure", () => {
  it("shows a placeholder while loading", () => {
    state.me = user(null);
    state.defaults = { data: undefined, isLoading: true, isError: false };
    render(<AgentsSection />);
    expect(screen.queryByRole("list", { name: "智能体" })).not.toBeInTheDocument();
  });

  it("says the instructions could not be loaded, and offers nothing to edit", () => {
    state.me = user(null);
    state.defaults = { data: undefined, isLoading: false, isError: true };
    render(<AgentsSection />);
    expect(screen.getByText(/无法加载智能体内置指令/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Customize|Edit instructions/ })).not.toBeInTheDocument();
  });
});
