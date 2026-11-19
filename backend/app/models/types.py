"""同时适配 Postgres 与 SQLite 的字段类型（docs/12 M9，docs/13 §7）。

模型共享，但数据库不同；with_variant 保留服务端 JSONB、原生 UUID，桌面改用文本 JSON 和 CHAR(32) UUID。向量记忆表仍仅限 Postgres。"""

from __future__ import annotations

from sqlalchemy import JSON, BigInteger, Uuid
from sqlalchemy.dialects import sqlite
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

# 类型实例构造后不可变，可安全共用，SQLAlchemy 按列需要复制。
JsonType = JSON().with_variant(JSONB(), "postgresql")
UuidType = Uuid(as_uuid=True).with_variant(PG_UUID(as_uuid=True), "postgresql")

# 自增大整数主键：Postgres 使用 BIGINT/SERIAL，SQLite 必须用 INTEGER PRIMARY KEY 才能自动填值。
# 否则桌面日志和审计插入会因 ID 未提供而失败。
BigIntAutoType = BigInteger().with_variant(sqlite.INTEGER(), "sqlite")
