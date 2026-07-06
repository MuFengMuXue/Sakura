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
    await registry.ainit()

    # 2. 存入应用状态，供路由使用
    app.state.registry = registry

    # 3. 异步加载记忆数据（偏好、工具、图像）
    logger.info("加载记忆数据...")
    await registry.load_memories()

    # 4. 重建 BM25 索引（如果启用）
    logger.info("重建 BM25 索引...")
    await registry.rebuild_bm25()

    # 5. 启动异步调度器
    logger.info("启动调度器...")
    await registry.start_scheduler()

    logger.info("MemOS 服务启动成功!")

    yield  # 应用运行期间

    # ---------- 关闭 ----------
    logger.info("正在关闭记忆系统...")
    await registry.shutdown()
    logger.info("服务已关闭")