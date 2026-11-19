"use client";

import { isDesktop } from "@/lib/desktop";
import LiveEvaluationWorkspace from "@/components/evaluation/LiveEvaluationWorkspace";

export default function EvaluationPage() {
  if (isDesktop) return <p className="text-sm text-text-muted">评测中心仅在自托管服务端提供，需要登录账号与隔离评测 Worker。</p>;
  return <LiveEvaluationWorkspace />;
}
