"""
处理生命周期
"""
# api/lifespan.py
"""
应用生命周期管理（启动/关闭钩子）
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from .service_registry import get_service_registry

logger = logging.getLogger(__name__)


async def startup():
    """服务启动时执行"""
    print("系统启动")

    # 1. 获取服务注册表（会触发所有组件的初始化）
    registry = get_service_registry()
    print("组件初始化完成")

    # 2. 加载记忆管理器的持久化数据
    print("加载记忆数据...")
    await registry.load_memories()

    # 3. 重建 BM25 索引（如果启用）
    print("重建 BM25 索引...")
    await registry.rebuild_bm25()

    # 4. 启动异步调度器（如果启用）
    print("启动调度器...")
    await registry.start_scheduler()
    print("MemOS 服务启动成功!")


async def shutdown():
    """服务关闭时执行"""
    print("\n正在关闭记忆系统...")
    registry = get_service_registry()
    await registry.shutdown()
    print("[服务已关闭")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期上下文管理器
    用法：在 FastAPI 实例化时传入 lifespan=lifespan
    """
    await startup()
    yield  # 应用运行期间
    await shutdown()