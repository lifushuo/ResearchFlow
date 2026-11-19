"""统一描述业务拒绝原因，不依赖 HTTP 传输。

业务层只判断不存在、越权、关卡不匹配或功能不支持，不决定 HTTP/gRPC 状态。FastAPI 映射统一在 app/services/error_responses.py，由两种主机安装。
本模块禁止导入 FastAPI、主机基础设施或 desktop，保持应用层可共用。"""

from __future__ import annotations


class AppError(Exception):
    """预期的业务拒绝，不同于意外崩溃。

    detail 面向用户，应说明哪个限制被触发及超出多少，不能只写模糊的请求错误。"""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFound(AppError):
    """对象不存在或不属于当前用户，两者使用同一错误。

    不区分可避免泄露他人的研究确实存在；服务端与桌面都按 404 处理。"""


class Conflict(AppError):
    """请求结构正确，但当前状态不允许操作。

    例如必须在匹配的人工关卡恢复，否则计划负载可能被报告关卡误当成拒绝。"""


class Invalid(AppError):
    """输入结构正确但业务不能接受，例如模型不可路由或任务列表为空。

    与框架在进入业务前拒绝的格式错误不同。"""


class Unprocessable(AppError):
    """输入结构正确、业务无法处理，对外返回 422。

    与 Invalid 含义近似，但保留已有契约：运行接口使用 422，语料接口使用 400。统一状态码属于可见行为变更，不能借重构悄悄修改。"""


class PayloadTooLarge(AppError):
    """请求可以理解，但大小超过允许范围。"""


class DependencyUnavailable(AppError):
    """所需向量服务或模型供应商暂不可达。

    这是依赖故障，不是用户输入有误，应提示稍后重试，而不是要求用户改文件。"""


class CapabilityUnavailable(AppError):
    """当前主机按产品设计不支持此能力。

    例如桌面没有 pgvector 项目记忆或服务端 PDF 依赖；不能误报成用户请求错误或系统崩溃。capability 字段供客户端识别缺少哪种能力，不必解析提示文字。"""

    def __init__(self, detail: str, *, capability: str) -> None:
        super().__init__(detail)
        self.capability = capability
