import Link from "next/link";

/** 在演示报告旁显示整行警告（docs/17 §6.2）。演示不论问题为何都返回固定内容，单靠列表徽标不足以防止误读。必须先说明问题未发送给模型。研究任务和会话共用组件；不能只让用户去设置添加密钥，因为服务端若处于演示模式，保存的密钥也不会被调用。 */
export function DemoNotice() {
  return (
    <div
      role="note"
      aria-label="演示研究"
      className="border px-4 py-3"
      style={{
        borderColor: "color-mix(in srgb, var(--warning) 35%, var(--border))",
        backgroundColor: "color-mix(in srgb, var(--warning) 8%, var(--bg-surface))",
      }}
    >
      <p
        className="font-mono text-xs font-semibold uppercase tracking-wider"
        style={{ color: "var(--warning)" }}
      >
        ⚠ 演示研究——未使用大模型进行实际研究
      </p>
      <p className="mt-1 text-sm leading-relaxed text-text-secondary">
        你的问题未发送给模型。本报告是脚本演示：使用固定内容和示例来源，让你无需 API 密钥即可体验流程，并不回答你提出的问题。导出文件也带有演示标记。要进行真实研究，请前往{" "}
        <Link href="/settings" className="text-accent hover:underline">
          设置
        </Link>
        连接模型；如果所有研究都返回演示内容，则服务端本身处于演示模式，需要配置模型提供方。
      </p>
    </div>
  );
}
