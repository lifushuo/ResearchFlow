"use client";

import Link from "next/link";

import { useReadiness } from "@/hooks/queries";
import { isDesktop } from "@/lib/desktop";

/** 用户尚无可用研究模型时显示（docs/17 §8a）。是否显示由当前配置计算，不能存储已看过引导的标记，因为撤销密钥或启动本地模型后条件会改变。引导统一进入设置，不另建密钥输入向导，以免两套配置界面随提供方变化而漂移。 */
export function FirstRunNotice() {
  const { data } = useReadiness();

  // 数据未返回不等于尚未配置；请求进行中或失败时，
  // 不能误告用户缺少实际上可能已经存在的配置。
  if (!data || data.ready) return null;

  return (
    <div
      role="note"
      className="border px-4 py-3.5"
      style={{
        borderColor: "color-mix(in srgb, var(--accent) 35%, var(--border))",
        backgroundColor: "color-mix(in srgb, var(--accent) 6%, var(--bg-surface))",
      }}
    >
      <p className="text-sm font-semibold text-text-primary">
        连接模型，开始研究
      </p>
      <p className="mt-1 max-w-2xl text-sm leading-relaxed text-text-secondary">
        {data.local_reachable ? (
          <>
            本地模型服务正在运行，可用于文档嵌入。若需要更好的研究质量，请在设置中添加 Anthropic、Google 或 OpenAI 的提供方密钥。
          </>
        ) : (
          <>
            只需提供方 API 密钥或本地模型服务之一即可开始；网页搜索会回退到无需密钥的服务，数据库也已内置。
          </>
        )}
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Link href="/settings" className="btn btn-primary">
          连接模型
        </Link>
        {/* The keyless path this build actually has, never a control that does not exist.
            A self-hosted stack has one: `./start.sh --fake` runs the real graph on scripted
            models and fixture sources and stamps every run as a demo — and it is what
            `start.sh` falls back to on its own when `.env` has no provider key. The desktop
            app has no such switch, so on the desktop this offers nothing rather than a path
            that is not there. */}
        {!isDesktop && (
          <span className="font-mono text-xs text-text-muted">
            也可以在不提供密钥的情况下启动服务，使用{" "}
            <code className="font-semibold">./start.sh --fake</code> ——脚本模型搭配固定示例来源，所有研究都会标记为演示。
          </span>
        )}
      </div>
    </div>
  );
}
