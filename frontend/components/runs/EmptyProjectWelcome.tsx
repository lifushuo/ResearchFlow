"use client";

import Link from "next/link";

/** 全新项目没有研究、语料库或记忆；使用一个清晰的下一步入口替代三个空面板，page.tsx 用此组件替换关注事项、活跃研究和项目健康。分别提供发起研究和上传材料两个入口，因为默认网页研究无需语料库，用户也可以先上传材料。 */
export function EmptyProjectWelcome({ projectName }: { projectName: string }) {
  return (
    <section aria-labelledby="empty-project" className="card py-10 text-center">
      <h2
        id="empty-project"
        className="font-serif text-xl font-bold tracking-tight text-text-primary"
      >
        {projectName} 已准备好开展首次研究
      </h2>
      {/* "Nothing active here" rather than "nothing has ever been asked": the emptiness
          check reads the *unarchived* session list, and archiving is explicitly recoverable
          rather than deletion — so a user who archived an old project's sessions would be
          told their work never happened. */}
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-text-secondary">
        这里还没有进行中的研究或来源材料。可以先提出问题，也可以先添加材料，二者无需固定先后顺序。
      </p>
      <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
        <Link href="/research" className="btn btn-primary">
          开始研究
        </Link>
        <Link href="/corpus" className="btn btn-secondary">
          添加语料材料
        </Link>
      </div>
    </section>
  );
}
