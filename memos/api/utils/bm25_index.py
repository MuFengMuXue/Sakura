"""
有关BM25的工具
"""
from ..service_registry import ServiceRegistry
from ..dependencies import get_registry
from fastapi import Depends
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def update_bm25_index(memory_id: str, content: str,registry: ServiceRegistry = Depends(get_registry),) -> None:
    """
    异步更新 BM25 索引（将同步调用转入线程池）
    """
    bm25 = registry.bm25
    if bm25 and hasattr(bm25, 'add_document'):
        try:
            # 将同步 I/O 操作转移到线程池，避免阻塞事件循环
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,  # 使用默认线程池
                bm25.add_document,
                memory_id,
                content
            )
            logger.debug(f"BM25 索引已更新: {memory_id}")
        except Exception as e:
            logger.error(f"BM25 索引更新失败 (id={memory_id}): {e}")

async def rebuild_bm25_index(bm25_searcher, qdrant_client):
    if not bm25_searcher:
        return
    if qdrant_client and qdrant_client.is_available():
        documents = qdrant_client.get_all_memories(limit=10000)
        bm25_searcher.build_index(documents)