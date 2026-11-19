import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { AgentRole, ModelCatalog } from "@/lib/types";

import { ModelPicker } from "./ModelPicker";

/** 模型设置只负责各角色使用哪个模型，智能体指令在独立页面管理。测试保证选择器仍提供五个角色的模型选择，同时不包含指令编辑器。 */

const ROLES: AgentRole[] = ["planner", "executor", "critic", "synthesizer", "chat"];
const ROUTING = Object.fromEntries(ROLES.map((r) => [r, "google:gemini"])) as Record<
  AgentRole,
  string
>;

const CATALOG: ModelCatalog = {
  roles: ROLES,
  models: [
    {
      route: "google:gemini",
      provider: "google",
      model_id: "gemini",
      display_name: "Gemini",
      input_per_mtok: 1,
      output_per_mtok: 2,
      context_window: null,
      max_output_tokens: null,
      supports_tools: true,
      supports_structured_output: true,
      notes: "",
      available: true,
    },
  ],
  presets: {},
  preset_names: [],
  available_providers: ["google"],
  effective_routing: ROUTING,
  user_routing: null,
  deployment_routing: ROUTING,
};

const usePromptDefaults = vi.hoisted(() => vi.fn());
const useMe = vi.hoisted(() => vi.fn());

vi.mock("@/hooks/queries", () => ({
  useModelCatalog: () => ({ data: CATALOG, isLoading: false }),
  useCustomEndpointStatus: () => ({ data: { models: [], reachable: false } }),
  useSetModelRouting: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useResetModelRouting: () => ({ mutateAsync: vi.fn(), isPending: false }),
  usePromptDefaults,
  useMe,
}));

describe("设置 → 模型", () => {
  it("still offers a model selector for each of the five roles", async () => {
    render(<ModelPicker />);
    await userEvent.click(screen.getByRole("button", { name: "自定义" }));
    for (const label of ["规划器", "执行器", "评审器", "综合器", "追问对话"]) {
      expect(screen.getByRole("combobox", { name: `${label}的模型` })).toBeInTheDocument();
    }
  });

  it("renders no agent instruction editor, mark or statement", async () => {
    const { container } = render(<ModelPicker />);
    await userEvent.click(screen.getByRole("button", { name: "自定义" }));
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /指令/ })).not.toBeInTheDocument();
    expect(container.textContent).not.toMatch(/自定义指令|验证包/);
  });

  it("does not load agent instructions at all", () => {
    render(<ModelPicker />);
    expect(usePromptDefaults).not.toHaveBeenCalled();
    expect(useMe).not.toHaveBeenCalled();
  });
});
