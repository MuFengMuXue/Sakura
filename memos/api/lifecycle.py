# api/lifecycle.py
"""
应用生命周期管理（启动/关闭钩子）
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from .service_registry import ServiceRegistry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("系统启动")

    # 1. 创建服务注册表实例（同步初始化所有组件）
    registry = ServiceRegistry()
    
    # 2. 异步初始化所有需要连接的组件（Qdrant、记忆、BM25、调度器等）
    await registry.ainit()

    # 3. 存入应用状态，供路由使用
    app.state.registry = registry

    logger.info("MemOS 服务启动成功!")

    yield  # 应用运行期间

    # ---------- 关闭 ----------
    logger.info("正在关闭记忆系统...")
    await registry.shutdown()
    logger.info("服务已关闭")