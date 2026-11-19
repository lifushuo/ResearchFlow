import type { AgentRole, ModelCatalog } from "@/lib/types";

/** 五个公开角色的名称与职责统一定义于此，供“设置→模型”和“设置→智能体”共用，避免同一角色在不同页面被描述成不同职责。 */
export const ROLE_COPY: Record<AgentRole, { label: string; blurb: string }> = {
  planner: { label: "规划器", blurb: "将问题拆分为可独立研究的任务。" },
  executor: { label: "执行器", blurb: "执行搜索并收集证据。" },
  critic: { label: "评审器", blurb: "评估证据质量，并将不足的部分退回补充。" },
  synthesizer: { label: "综合器", blurb: "撰写带引用的研究报告。" },
  chat: { label: "追问对话", blurb: "围绕已完成的报告回答后续问题。" },
};

/** 仅用于展示角色当前使用的模型。读取目录中的有效路由，反映此刻研究实际会调用的模型；目录未列出的自定义模型显示原始 ID，未加载或未配置则返回 null，由调用方明确提示，不能猜测。 */
export function roleModelLabel(catalog: ModelCatalog | undefined, role: AgentRole): string | null {
  const route = catalog?.effective_routing[role];
  if (!route) return null;
  const listed = catalog.models.find((m) => m.route === route);
  if (listed) return listed.display_name;
  const colon = route.indexOf(":");
  return colon >= 0 ? route.slice(colon + 1) : route;
}
