from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base  # noqa: F401 — 导入全部模型，让 Alembic 能发现完整表结构

logger = structlog.get_logger()

engine = create_async_engine(
    settings.database_url,
    echo=False,  # 开发时可设为 True，输出 SQL 查询日志
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # 每次取连接前检查可用性，失效时重新连接
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def init_db() -> None:
    """创建数据表，仅用于开发；生产环境使用 Alembic 迁移。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db_init", message="Database tables created/verified")


async def get_db() -> AsyncGenerator[AsyncSession]:
    """为请求提供异步数据库会话；成功提交，异常回滚，最后关闭连接。"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
