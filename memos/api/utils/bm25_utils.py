"""
BM25 索引操作工具
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def update_bm25_index(
    memory_id: str,
    content: str,
    registry,  # 由调用方传入
) -> None:
    """
    异步更新 BM25 索引（将同步调用转入线程池）
    """
    bm25 = registry.bm25
    if bm25 and hasattr(bm25, 'add_document'):
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                bm25.add_document,
                memory_id,
                content
            )
            logger.debug(f"BM25 索引已更新: {memory_id}")
        except Exception as e:
            logger.error(f"BM25 索引更新失败 (id={memory_id}): {e}")
    else:
        logger.warning("BM25 未启用或 add_document 方法不存在，跳过更新")


async def remove_bm25_document(
    memory_id: str,
    registry,  # 由调用方传入
) -> None:
    """
    异步从 BM25 索引移除记忆（将同步调用转入线程池）
    """
    bm25 = registry.bm25
    if bm25 and hasattr(bm25, 'remove_document'):
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                bm25.remove_document,
                memory_id
            )
            logger.debug(f"BM25 索引已移除: {memory_id}")
        except Exception as e:
            logger.error(f"BM25 索引移除失败 (id={memory_id}): {e}")
    else:
        logger.warning("BM25 未启用或 remove_document 方法不存在，跳过移除")


async def rebuild_bm25_index(
    registry,  # 由调用方传入
) -> None:
    """
    重建 BM25 索引
    """
    bm25 = registry.bm25
    qdrant = registry.qdrant
    if not bm25 or not qdrant:
        return
    if qdrant.is_available():
        documents = await qdrant.get_all_memories(limit=10000)
        # 如果 build_index 是同步方法，放入线程池
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, bm25.build_index, documents)
        logger.info("BM25 索引重建完成")