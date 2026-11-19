import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { RunGraph } from "@/lib/types";

import RunPage from "./page";

/** 研究页中的全部 dt/dd 都必须位于 dl 内。证据链概览曾将四组名称和值放入普通网格 div，读屏器因缺少描述列表结构只能读成零散文本。此测试渲染概览会出现的完成状态，并核查页面上的全部描述项，而不只检查一个区块。 */

const current = vi.hoisted(() => ({ graph: null as unknown }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams("id=run-1"),
}));
vi.mock("@/components/ActiveProject", () => ({ useActiveProject: () => ({ active: null }) }));
vi.mock("@/hooks/runs", async (original) => ({
  ...(await original<typeof import("@/hooks/runs")>()),
  useRun: () => ({
    data: current.graph,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
    isFetching: false,
  }),
  useRunStream: () => ({ events: [], degraded: false }),
}));
// 工作区内容另有测试；此处只检查页头的 HTML 结构。
vi.mock("@/components/runs/RunWorkspace", () => ({
  RunWorkspace: () => null,
  CancelButton: () => null,
}));
vi.mock("@/components/runs/RunProgress", () => ({ RunProgress: () => null }));

function graph(over: Partial<RunGraph> = {}): RunGraph {
  const base: RunGraph = {
    run: {
      id: "run-1",
      project_id: "p1",
      question: "Does grounding help?",
      status: "COMPLETED",
      depth: "fast",
      corpus_mode: false,
      demo: true,
      skip_plan_gate: true,
      model_routing: null,
      cost_usd: 0.01,
      tokens_input: 1,
      tokens_output: 1,
      elapsed_seconds: 2,
      citation_resolution_rate: null,
      error_message: null,
      cancelled_at: null,
      created_at: "2026-08-18T00:00:00Z",
      updated_at: "2026-08-18T00:00:00Z",
    },
    plans: [],
    sources: [
      {
        id: "s1",
        url: "https://a.invalid/one",
        title: "Paper One",
        kind: "WEB",
        retrieval_status: "FETCHED",
        citation_index: 1,
        corpus_document_id: null,
      },
      {
        id: "s2",
        url: "https://a.invalid/two",
        title: "Never referenced",
        kind: "WEB",
        retrieval_status: "FETCHED",
        citation_index: null,
        corpus_document_id: null,
      },
    ],
    evidence: [
      {
        id: "e1",
        source_id: "s1",
        sequence: 1,
        task_id: "1",
        snippet: "Grounding raised accuracy by nine points.",
        content_hash: "a".repeat(64),
        key_fact: "accuracy up",
        provenance_state: "UNCHECKED",
        attested_against: null,
        attestation_run_at: null,
      },
    ],
    revisions: [
      {
        id: "r1",
        version: 1,
        report_markdown: "# Findings\n\nGrounding raised accuracy [1].",
        report_hash: "b".repeat(64),
        evidence_watermark: 1,
        cited_indices: [1],
        created_at: "2026-08-18T00:00:00Z",
      },
    ],
    claims: [
      {
        id: "c1",
        revision_id: "r1",
        position: 0,
        text: "Grounding raised accuracy [1].",
        extraction_method: "DERIVED_FROM_REPORT",
        verification_state: "UNCHECKED",
        verification_method: "NOT_RUN",
        lineage_id: null,
      },
      {
        id: "c2",
        revision_id: "r1",
        position: 1,
        text: "An assertion nothing backs.",
        extraction_method: "DERIVED_FROM_REPORT",
        verification_state: "UNCHECKED",
        verification_method: "NOT_RUN",
        lineage_id: null,
      },
    ],
    claim_evidence_links: [
      { id: "l1", claim_id: "c1", evidence_id: "e1", stance: "SUPPORTS", origin: "CITATION_MARKER" },
    ],
    contradictions: [],
    reviews: [],
    artifact: null,
  };
  return { ...base, ...over };
}

describe("the run page's description lists", () => {
  it("shows the Evidence Chain Overview for a finished run", () => {
    current.graph = graph();
    render(<RunPage />);
    expect(screen.getByText("Evidence Chain Overview")).toBeInTheDocument();
  });

  it("puts every term and description inside a dl", () => {
    current.graph = graph();
    const { container } = render(<RunPage />);
    const pairs = container.querySelectorAll("dt, dd");
    expect(pairs.length).toBeGreaterThanOrEqual(8); // 概览中的四项指标。
    for (const el of pairs) {
      expect(el.closest("dl"), `${el.tagName} "${el.textContent}" has no dl`).not.toBeNull();
    }
  });
});

/** 演示报告必须明确说明它没有研究用户提出的问题。演示无论输入为何都返回带引用的固定内容，单靠两字徽标容易使读者误认为是针对自身问题的研究，因此页面还必须用完整句子说明。 */
describe("a demo run's page", () => {
  it("says plainly that nothing was researched and the question never reached a model", () => {
    current.graph = graph({ run: { ...graph().run, demo: true } });
    render(<RunPage />);
    const notice = screen.getByRole("note", { name: /演示研究/i });
    expect(notice).toHaveTextContent("演示研究——未使用大模型进行实际研究");
    expect(notice).toHaveTextContent(
      "你的问题未发送给模型。本报告是脚本演示",
    );
  });

  it("is not shown on a run a model actually researched", () => {
    current.graph = graph({ run: { ...graph().run, demo: false } });
    render(<RunPage />);
    expect(screen.queryByRole("note", { name: /演示研究/i })).toBeNull();
  });
});
