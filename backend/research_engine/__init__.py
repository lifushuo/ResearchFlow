"""research_engine：不依赖具体宿主的研究引擎。

LangGraph 流程为规划 → 执行 ⇄ 评审 → 综合 → 人工审批中断 → 定稿，
并提供工具、检索链、模型工厂和引用数据结构。图契约见 docs/architecture/04-agent-architecture.md。

引擎不能导入 app，也不自行读取进程环境。宿主通过 RunConfig 注入配置：
服务端使用 app.runtime.install_process_default，桌面端从本地配置和系统钥匙串构造。
事件发送器与用户密钥同样通过端口注入，因此同一张图可以运行在服务端或桌面端。
边界由 tests/test_engine_boundary.py 检查，契约见 docs/13 §3。"""

import sys

# 从源码直接运行时，pip 的 Python 版本约束不会生效；先明确报出所需版本，
# 避免旧解释器遇到 Python 3.14 语法后只给出难懂的 SyntaxError。
if sys.version_info < (3, 14):  # noqa: UP036 — 在这里实际检查最低版本，不能假定安装器已经检查
    raise RuntimeError(f"research_engine needs Python 3.14+, this is {sys.version.split()[0]}")
