import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  defaultKind,
  isKindReady,
  isUnpricedRoute,
  ProviderPicker,
  routeForKind,
  routingFor,
  splitRoute,
} from "./ProviderPicker";

/** 研究表单只决定模型后端类型及其可用性。测试要求展示真实路由、明确标记不可用端点，并在选定后端失败时禁止暗中替换提供方。 */

const catalog = vi.hoisted(() => ({ value: {} as Record<string, unknown> }));
const custom = vi.hoisted(() => ({ value: {} as Record<string, unknown> }));
const local = vi.hoisted(() => ({ value: {} as Record<string, unknown> }));

vi.mock("@/hooks/queries", () => ({
  useModelCatalog: () => catalog.value,
  useCustomEndpointStatus: () => custom.value,
  useLocalLLMStatus: () => local.value,
}));

const CATALOG = {
  user_routing: null,
  deployment_routing: routingFor("custom:auto/best-fast"),
  models: [
    {
      route: "google:gemini-2.5-pro",
      provider: "google",
      display_name: "Gemini 2.5 Pro",
      available: true,
    },
  ],
};
const LOCAL = {
  models: [{ name: "qwen2.5:7b", route: "ollama:qwen2.5:7b", is_embedding: false }],
};
const CUSTOM = { reachable: true, models: ["auto/best-fast"], configured_base_url: "http://x/v1" };

function setUp({ cat = CATALOG, cus = CUSTOM, loc = LOCAL } = {}) {
  catalog.value = { data: cat, isLoading: false };
  custom.value = { data: cus, isLoading: false };
  local.value = { data: loc, isLoading: false };
  return { cat, cus, loc };
}

describe("backend resolution", () => {
  it("prefers the custom endpoint, then local, then API", () => {
    // 使用用户指定的优先级，也与按钮显示顺序一致。
    expect(defaultKind(CATALOG, CUSTOM, LOCAL)).toBe("custom");
    expect(defaultKind(CATALOG, { ...CUSTOM, reachable: false }, LOCAL)).toBe("local");
    expect(
      defaultKind(CATALOG, { ...CUSTOM, reachable: false }, { models: [] }),
    ).toBe("api");
  });

  it("returns null while nothing is reachable, rather than guessing", () => {
    // 等待探测完成比先显示选中状态又突然改变更稳妥。
    const bare = { user_routing: null, deployment_routing: null, models: [] };
    expect(defaultKind(bare, { ...CUSTOM, reachable: false }, { models: [] })).toBeNull();
  });

  it("honours a model chosen in Settings over the deployment's own default", () => {
    // 模型选择迁移到设置后，研究表单必须确实遵循已保存配置。
    const withSaved = { ...CATALOG, user_routing: routingFor("custom:my/pinned-model") };
    expect(routeForKind("custom", withSaved, CUSTOM, LOCAL)).toBe("custom:my/pinned-model");
  });

  it("falls back to the deployment's configured route when Settings names none", () => {
    expect(routeForKind("custom", CATALOG, CUSTOM, LOCAL)).toBe("custom:auto/best-fast");
  });

  it("picks an installed local model by its exact tag", () => {
    expect(routeForKind("local", CATALOG, CUSTOM, LOCAL)).toBe("ollama:qwen2.5:7b");
    expect(splitRoute("ollama:qwen2.5:7b")).toEqual({ provider: "ollama", model: "qwen2.5:7b" });
  });

  it("does not default to a model Settings calls chat only while a larger one is installed", () => {
    const installed = {
      models: [
        { route: "ollama:deepseek-r1:1.5b", is_embedding: false, likely_underpowered: true },
        { route: "ollama:nomic-embed-text:latest", is_embedding: true, likely_underpowered: false },
        { route: "ollama:qwen2.5:7b", is_embedding: false, likely_underpowered: false },
      ],
    };
    expect(routeForKind("local", { user_routing: null }, CUSTOM, installed)).toBe(
      "ollama:qwen2.5:7b",
    );
    const onlySmall = { models: installed.models.slice(0, 2) };
    expect(routeForKind("local", { user_routing: null }, CUSTOM, onlySmall)).toBe(
      "ollama:deepseek-r1:1.5b",
    );
  });

  it("reports a backend with nothing configured as null, not as an empty route", () => {
    expect(routeForKind("local", CATALOG, CUSTOM, { models: [] })).toBeNull();
    expect(isKindReady("local", CATALOG, CUSTOM, { models: [] })).toBe(false);
  });

  it("knows which backends have no measurable cost", () => {
    expect(isUnpricedRoute("custom:auto/best-fast")).toBe(true);
    expect(isUnpricedRoute("openrouter:meta/llama")).toBe(true);
    // 本地推理确实免费，与费用未知是不同结论。
    expect(isUnpricedRoute("ollama:qwen2.5:7b")).toBe(false);
    expect(isUnpricedRoute("google:gemini-2.5-pro")).toBe(false);
  });
});

describe("ProviderPicker", () => {
  it("offers exactly three backends", () => {
    setUp();
    render(<ProviderPicker value="custom" onChange={vi.fn()} />);
    expect(screen.getAllByRole("radio")).toHaveLength(3);
    for (const label of ["自定义端点", "本地大模型", "API"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it("shows the route each backend will actually use", () => {
    setUp();
    render(<ProviderPicker value="custom" onChange={vi.fn()} />);
    expect(screen.getByText("custom:auto/best-fast")).toBeInTheDocument();
    expect(screen.getByText("ollama:qwen2.5:7b")).toBeInTheDocument();
  });

  it("marks a stopped endpoint stopped in text, not only in colour", () => {
    // 圆点仅用于装饰，读屏器依赖状态文字。
    setUp({ cus: { ...CUSTOM, reachable: false } });
    render(<ProviderPicker value="local" onChange={vi.fn()} />);
    expect(screen.getByText("自定义端点已停止")).toBeInTheDocument();
    expect(screen.getByText("本地大模型运行中")).toBeInTheDocument();
  });

  it("says a chosen dead backend will fail rather than switching provider", () => {
    // 此规则防止暗中替换提供方，避免最终报告的模型归属
    // 成为用户首次发现替换发生的位置。
    setUp({ cus: { ...CUSTOM, reachable: false } });
    render(<ProviderPicker value="custom" onChange={vi.fn()} />);
    expect(screen.getByRole("alert")).toHaveTextContent(/启动研究会报错，不会自动切换/i);
  });

  it("does not warn when the chosen backend is up", () => {
    setUp();
    render(<ProviderPicker value="custom" onChange={vi.fn()} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("reports the user's choice", async () => {
    setUp();
    const onChange = vi.fn();
    render(<ProviderPicker value="custom" onChange={onChange} />);
    await userEvent.click(screen.getByRole("radio", { name: /本地大模型/i }));
    expect(onChange).toHaveBeenCalledWith("local");
  });

  it("points at Settings for the model itself", () => {
    setUp();
    render(<ProviderPicker value="custom" onChange={vi.fn()} />);
    expect(screen.getByText(/设置 → 模型/)).toBeInTheDocument();
  });
});
