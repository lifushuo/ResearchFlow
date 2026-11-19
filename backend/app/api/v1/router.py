"""组合 v1 API 路由。

/version 放在共用路由中，因为它描述整个构建，不属于某个功能。统一使用 /api/v1 路径，让服务端与桌面客户端用同一方式查询版本，无需根据主机类型分支。"""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.corpus import router as corpus_router
from app.api.v1.models import router as models_router
from app.api.v1.projects import router as projects_router
from app.api.v1.research import router as research_router
from app.api.v1.runs import router as runs_router
from app.api.v1.threads import router as threads_router
from app.schemas.capabilities import SERVER, Capabilities
from research_engine.build_info import build_info

api_router = APIRouter()


@api_router.get("/capabilities", response_model=Capabilities, tags=["Health"])
async def capabilities() -> Capabilities:
    """报告本主机实际支持的能力，客户端不再根据构建标记猜测。"""
    return SERVER


@api_router.get("/version", tags=["Health"])
async def version():
    """报告当前构建版本与来源提交。

    服务端无需登录，因为不披露比 OpenAPI 更多的敏感信息，便于维护者定位部署。桌面仍通过每次启动令牌保护。
    没有构建标记时保持 unknown，不能从当前工作区猜版本，原因见 research_engine/build_info.py。"""
    return build_info().as_dict()


api_router.include_router(auth_router)
api_router.include_router(projects_router)
api_router.include_router(research_router)
api_router.include_router(corpus_router)
api_router.include_router(chat_router)
api_router.include_router(threads_router)
api_router.include_router(models_router)
api_router.include_router(runs_router)
