"use client";

import { isDesktop } from "@/lib/desktop";
import EvolutionWorkspace from "@/components/evolution/EvolutionWorkspace";

export default function EvolutionPage() {
  // 自进化读写评测卷（RF_EVAL_JOB_ROOT）与闭环数据库，与评测中心同属仅服务端能力。
  if (isDesktop)
    return (
      <p className="text-sm text-text-muted">自进化依赖评测中心的评测卷与闭环数据，仅在自托管服务端提供。</p>
    );
  return <EvolutionWorkspace />;
}
