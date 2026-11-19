"""两个主机共用的运行请求结构。

只依赖 Pydantic，不导入 app.config、app.db；否则桌面通过服务端路由取结构时会因缺少数据库和 JWT 配置而启动失败。"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class CorpusSelectionRequest(BaseModel):
    """所选文件运行的请求范围（RFC V3.1 §8.1）。

    逻辑文档与文件夹在创建时解析为具体版本并冻结；限制列表长度，防止一次请求无限扩大解析工作量。"""

    folder_ids: list[str] = Field(default_factory=list, max_length=5000)
    doc_keys: list[str] = Field(default_factory=list, max_length=5000)


class CreateRunRequest(BaseModel):
    """研究运行的领域入口参数。"""

    project_id: uuid.UUID
    question: str = Field(min_length=1, max_length=2000)
    depth: str = Field("balanced", description="fast | balanced | comprehensive")
    corpus_mode: bool = False
    # 配置文件选择即意味着语料模式；若同时明确 corpus_mode=false，直接拒绝，不替用户猜意图。
    corpus_selection: CorpusSelectionRequest | None = None
    # 离线保证要求每次模型调用都留在本机，否则不启动。
    # 区别于只限制证据来源的文件选择；离线同时启用语料模式，不读取网页。
    offline: bool = False
    skip_plan_gate: bool = True
    topic_seeds: list | None = None
    outline_template: str | None = None
    # 每个代理角色可指定 provider:model。None 沿用用户偏好及部署默认，不强迫不关心模型的调用方填写五个角色。
    #:
    # 路由校验需要目录和服务端配置，放在实际主机处理器中，不在共用请求模型里导入。
    # 最终结构与供应商是否合法由 model_routing.validate 决定。
    model_routing: dict[str, str] | None = None
    # 是否分发 Worker；False 只创建领域记录，用于暂存运行或进程内驱动测试。
    dispatch: bool = True


class PlanReviewRequest(BaseModel):
    decision: str = Field("APPROVED", description="APPROVED | REWORK_REQUESTED | REJECTED")
    feedback: str | None = None
    # 人工编辑后的任务列表，None 表示未编辑；[] 表示全部排除，应拒绝而不能当成未修改。
    # 编辑新增计划版本，保留真正执行的设计。
    tasks: list[dict] | None = None
    dispatch: bool = True


class ReportReviewRequest(BaseModel):
    """对报告修订作决定，dispatch 决定是否调度返工恢复。"""

    revision_version: int | None = Field(
        None, description="Which revision was reviewed. Defaults to the latest."
    )
    decision: str = Field(..., description="APPROVED | REWORK_REQUESTED | REJECTED")
    feedback: str | None = None
    dispatch: bool = True
