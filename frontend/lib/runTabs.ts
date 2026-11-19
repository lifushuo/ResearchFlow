/** 工作区标签页的统一定义。使用普通模块，避免页面为校验 ?tab= 而从客户端 RunWorkspace.tsx 导入，使整个工作区及 react-markdown 重复进入客户端引用图。标签顺序强调报告是证据链的结果，而非研究的全部。 */
export const TABS = [
  "plan",
  "report",
  "claims",
  "evidence",
  "sources",
  "contradictions",
  "review",
  "artifact",
] as const;

export type Tab = (typeof TABS)[number];

/** 判断 ?tab= 是否指向真实标签页；未知值回退，不抛异常。 */
export function isTab(value: string | null | undefined): value is Tab {
  return Boolean(value) && (TABS as readonly string[]).includes(value as string);
}
