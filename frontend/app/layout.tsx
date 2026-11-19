import type { Metadata } from "next";

import { OG_IMAGE, basePath, isPagesBuild, siteUrl } from "@/lib/pages-build";
import { Inter, JetBrains_Mono } from "next/font/google";

import { Providers } from "@/components/Providers";
import "./globals.css";

// 通过 next/font 自托管字体，避免外部 CSS 阻塞渲染（docs/03、docs/07 的主题章节）。
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  // siteUrl 始终提供合法绝对地址；非 Pages 构建也有回退值，
  // 详见 lib/pages-build.ts，尽管只有 Pages 构建需要用于公开索引。
  // Next 需要 metadataBase 将相对 OG_IMAGE 路径
  // 解析为绝对地址；否则会回退到默认本机地址并在每次构建时警告。
  // 服务端和桌面端不面向爬虫（见 robots.ts），使用回退地址
  // 比让全部构建反复产生无意义警告更合适。
  metadataBase: new URL(siteUrl),
  title: {
    default: "研究助手——引用有据，结论可核查",
    // 应用于各页面自己的 title；首页若只需默认标题，
    // 应省略 title 而不是重复声明，见 page.tsx。
    template: "%s · 研究助手",
  },
  description:
    "多智能体研究流水线：规划问题、收集引用证据、评审并综合成可供人工审核的报告。",
  // 图标路径显式加上 basePath：服务端和桌面端为空，
  // GitHub Pages 为 /<repo>。Next 不会自动为这些元数据路径加前缀，
  // 因此必须避免浏览器把 /icon.svg 解析到域名根目录，
  // 否则会比 Pages 实际文件位置高一级，即便图标本身存在且可访问，
  // 页面中的链接也会返回 404。
  icons: {
    icon: `${basePath}/icon.svg`,
    shortcut: `${basePath}/icon.svg`,
    apple: `${basePath}/icon.svg`,
  },
  // 这里只提供结构默认值，不设置 title 和 description；
  // 由各页面通过 Next 的逐路由回退填充，
  // 原因见 lib/pages-build.ts 的 pageUrls。siteName、
  // type 和 card 向未自行重写 openGraph/twitter 的页面继承，
  // 当前应用的全部页面均采用此方式。
  openGraph: {
    type: "website",
    siteName: "Multi-Agent Research Assistant",
    locale: "zh_CN",
    images: [OG_IMAGE],
  },
  twitter: {
    card: "summary_large_image",
    images: [OG_IMAGE],
  },
  // 与 robots.ts 配合：即使自托管反向代理丢弃或改写
  // /robots.txt，页面中的 robots 元标签仍可限制索引。
  // 只有 Pages 构建应被搜索引擎发现。
  robots: isPagesBuild
    ? { index: true, follow: true }
    : { index: false, follow: false },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="zh-CN"
      className={`${inter.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-bg-base text-text-primary antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
