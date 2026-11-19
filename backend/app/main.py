from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app import metrics
from app.api.v1.evaluation import router as evaluation_router
from app.api.v1.router import api_router
from app.config import settings
from app.db.base import engine
from app.db.redis import close_redis_pool, get_redis, init_redis_pool
from app.logconfig import configure_logging
from app.runtime import install_process_default
from app.services.error_responses import install_error_handlers
from app.services.security_headers import SecurityHeadersMiddleware
from research_engine.llm_factory import validate_pricing

# 在第一条日志之前配置 structlog，确保 correlation_id 能随上下文贯穿 API、Celery 和研究引擎。
configure_logging(json_output=settings.is_production)

logger = structlog.get_logger()

# 应用版本同时用于 OpenAPI 和 /health 返回值。
#:
# 版本共用一个常量，避免健康检查和实际部署版本不一致。
# 发布新标签时，与 desktop/tauri.conf.json、frontend/lib/releases.ts 一起更新。
APP_VERSION = "3.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和退出时的资源管理。

    表结构由 Alembic 管理，这里不执行 create_all（docs/03、docs/05 §2）。配置导入时先校验，启动时再检查模型计价，避免错误路由在接受请求后才暴露。
    引擎不读取 app.config，因此主机要先安装 RunConfig，再让其他组件使用它（docs/13 §2）。"""
    logger.info("startup", message="Starting Multi-Agent Research Assistant API")
    install_process_default()
    validate_pricing()
    await init_redis_pool()
    logger.info("startup", message="Redis initialized ✅ (run `alembic upgrade head` for schema)")
    yield
    await close_redis_pool()
    logger.info("shutdown", message="Server shutting down")


app = FastAPI(
    title="Multi-Agent Research Assistant API",
    version=APP_VERSION,
    description="Self-hostable research assistant with an auditable human-in-the-loop gate.",
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    lifespan=lifespan,
)

app.add_middleware(SecurityHeadersMiddleware)

# 领域错误通过统一映射转换为 HTTP 状态码，桌面端也使用同一映射。
# 共享处理函数在两种主机上的错误语义必须一致。
install_error_handlers(app)

# 跨域只为本地开发提供便利：开发前端和 API 在不同端口，仅放行指定来源。
# 正式部署通过 Next.js 同源代理访问 API，不允许浏览器跨域调用（docs/06 §6）。
if not settings.is_production:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

# 注册带版本前缀的 API 路由
app.include_router(api_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    """存活检查：确认进程在运行，供容器 HEALTHCHECK 使用。"""
    return {"status": "ok", "version": APP_VERSION}


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """就绪检查：数据库迁移和 Redis 均可用才返回成功，否则返回 503（docs/09 §7）。
    Compose 据此决定是否启动依赖它的 Worker 和前端。"""
    checks = {"database": False, "redis": False}
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:  # noqa: BLE001
        logger.warning("readiness_db_down", error=str(e))
    try:
        await get_redis().ping()
        checks["redis"] = True
    except Exception as e:  # noqa: BLE001
        logger.warning("readiness_redis_down", error=str(e))

    if not all(checks.values()):
        return JSONResponse(status_code=503, content={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}


@app.get("/metrics", tags=["Health"])
async def metrics_endpoint() -> Response:
    """公开本进程的 Prometheus 聚合计数。

    放在 /health 旁边而不是 /api/v1 下，因为这是运维接口，可由反向代理统一屏蔽。仅服务端提供，桌面端仍记录指标但不开放抓取端点。
    无需登录的前提是指标只含封闭标签与聚合数量，不含研究正文、任意文本或用户、项目、运行 ID；该约束由 app/metrics.py 保证。"""
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)
