"""统一将领域错误映射为 HTTP 状态，供两个主机共用。

错误含义在 app/errors.py，传输方式在这里。只依赖 FastAPI 和标准库，不引入 app.config、app.db，避免破坏桌面导入。"""

from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.errors import (
    AppError,
    CapabilityUnavailable,
    Conflict,
    DependencyUnavailable,
    Invalid,
    NotFound,
    PayloadTooLarge,
    Unprocessable,
)

# 全部状态码契约集中在此映射表。
#:
# 以错误类对象而非类名作键，防止重命名意外改变状态码。
ERROR_STATUS: dict[type[AppError], int] = {
    NotFound: 404,
    Conflict: 409,
    Invalid: 400,
    Unprocessable: 422,
    PayloadTooLarge: 413,
    DependencyUnavailable: 503,
    CapabilityUnavailable: 501,
}

# 未映射错误按 500 处理，这是映射遗漏的程序缺陷，不能用看似正常的 4xx 掩盖。
UNMAPPED_STATUS = 500


def status_for(exc: AppError) -> int:
    """沿继承顺序查找状态映射；更具体的 NotFound 子类自然继承 404，无需重复登记。"""
    for cls in type(exc).__mro__:
        if cls in ERROR_STATUS:
            return ERROR_STATUS[cls]
    return UNMAPPED_STATUS


def error_body(exc: AppError) -> dict:
    """沿用客户端已识别的 detail 响应结构，不另加包装层，避免无意义改变接口。"""
    body = {"detail": exc.detail}
    if isinstance(exc, CapabilityUnavailable):
        # 提供机器可读的能力标识，客户端不必解析说明文字来判断缺少哪个功能。
        body["capability"] = exc.capability
    return body


async def _handle(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=status_for(exc), content=error_body(exc))


def install_error_handlers(app) -> None:
    """为一个 FastAPI 应用安装领域错误处理器，两种主机共用。

    挂载 runs.py 的应用必须调用，否则业务拒绝会变成未处理的 500；例如越权应为 404，而不是崩溃。"""
    app.add_exception_handler(AppError, _handle)


# 公开此处理器供验证，确保两个主机注册的是同一函数，而不是各自实现。
install_error_handlers.handler = _handle


def body_validation_error(exc: ValidationError, *, at: tuple[str, ...]) -> RequestValidationError:
    """把局部请求体的 Pydantic 校验失败转换成 FastAPI 标准请求错误。

    桌面手动校验 preferences 时也应返回与服务端相同的 422：位置从 body 开始，携带拒绝输入，由默认处理器渲染，并省略额外 URL。"""
    return RequestValidationError(
        [{**err, "loc": ("body", *at, *err["loc"])} for err in exc.errors(include_url=False)]
    )
