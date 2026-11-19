"""公开主机的产品能力，由客户端查询，不再靠构建标记猜测。

能力差异是设计决定，例如桌面没有项目记忆；功能缺口则是缺陷，KNOWN_DESKTOP_GAPS 必须为空。未支持能力返回带 capability 的 501，而不是把正确请求误报成 404。
只依赖 Pydantic，两种主机共用。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Capabilities(BaseModel):
    """相同响应结构，按主机返回实际能力。

    只描述用户能做什么，不暴露 Postgres、SQLite、Celery 等基础设施选择。"""

    # 是否提供注册、登录、资料、密码和账号用量管理。
    # 桌面是启动令牌保护的单一本地用户，没有账号体系；不能从是否限流推断是否有账号。
    accounts: bool

    # 是否可检索项目内已批准报告；当前实现依赖 pgvector，桌面不提供。
    project_memory: bool

    # 是否支持引用项目全部获批报告的聊天，依赖项目记忆能力。
    project_chat: bool

    # 是否提供服务端 PDF 渲染；桌面改用 WebView 打印，此接口按设计返回 501。
    server_pdf: bool

    # 明确报告是否实行请求限流。桌面单用户使用不同边界，不提供限流；不能用永远放行的假实现冒充安全控制。
    rate_limits: bool

    # 是否能启动或停止本地 Ollama。桌面可以；容器不应访问宿主机进程表。
    local_llm_control: bool

    # 是否能查询新版本，并非自动升级。桌面安装包需要检查更新；服务端由运维决定何时拉镜像。
    update_check: bool

    # 明确密钥保存位置；两个主机都保存密钥，但位置不同，不能用布尔值表达。
    byok_storage: Literal["encrypted_column", "os_keychain"]

    # 主机身份仅用于支持排查，客户端应读取明确的能力字段，不根据主机名推断能力。
    host: Literal["server", "desktop"] = Field(
        description="Identifies the host. Branch on the capabilities above, never on this."
    )


SERVER = Capabilities(
    accounts=True,
    project_memory=True,
    project_chat=True,
    server_pdf=True,
    rate_limits=True,
    local_llm_control=False,
    update_check=False,
    byok_storage="encrypted_column",
    host="server",
)

DESKTOP = Capabilities(
    accounts=False,
    project_memory=False,
    project_chat=False,
    server_pdf=False,
    rate_limits=False,
    local_llm_control=True,
    update_check=True,
    byok_storage="os_keychain",
    host="desktop",
)
