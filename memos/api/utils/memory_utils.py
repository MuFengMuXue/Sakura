"""
记忆对象操作工具
提供记忆的获取、格式化、使用统计更新等功能
"""
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime

from .payload_utils import payload_of
from .payload_utils import normalize_duplicate_action
from .bm25_utils import remove_bm25_document

logger = logging.getLogger(__name__)


async def get_memory(
    memory_id: str,
    registry,  # 由调用方传入，不使用 Depends
    include_deleted: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    从 Qdrant 获取单条记忆（支持获取已删除的）。

    Args:
        memory_id: 记忆 ID
        registry: 服务注册表（由调用方传入）
        include_deleted: 是否包含已软删除的记忆

    Returns:
        原始记忆字典（包含 id, vector, payload），若不存在返回 None
    """
    if not registry.qdrant or not registry.qdrant.is_available():
        return None

    # 尝试常规获取
    memory = await registry.qdrant.get_memory(memory_id)
    if memory or not include_deleted:
        return memory

    # 若需要包含已删除，尝试直接检索（某些客户端支持 retrieve）
    try:
        if hasattr(registry.qdrant, 'retrieve'):
            results = await registry.qdrant.retrieve(
                ids=[memory_id],
                with_payload=True,
                with_vectors=True
            )
            if results:
                point = results[0]
                return {
                    'id': point.id,
                    'vector': point.vector,
                    'payload': point.payload or {}
                }
    except Exception as e:
        logger.debug(f"检索已删除记忆失败 {memory_id}: {e}")

    return None


def flatten_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 Qdrant 原始记忆结构展平为前端友好的格式。

    Args:
        memory: Qdrant 返回的 {id, vector, payload} 结构

    Returns:
        展平后的字典，包含 id, content, importance, created_at, updated_at,
        memory_type, layer, status, access_count, last_accessed_at, tags, payload 等
    """
    payload = payload_of(memory)

    return {
        'id': memory.get('id'),
        'content': payload.get('content', memory.get('content', '')),
        'importance': payload.get('importance', memory.get('importance', 0.5)),
        'created_at': payload.get('created_at', memory.get('created_at')),
        'updated_at': payload.get('updated_at', memory.get('updated_at')),
        'memory_type': payload.get('memory_type', memory.get('memory_type', 'general')),
        'layer': payload.get('layer', 'LongTermMemory'),
        'status': payload.get('status', 'active'),
        'access_count': payload.get('access_count', 0),
        'last_accessed_at': payload.get('last_accessed_at'),
        'tags': payload.get('tags', memory.get('tags', [])),
        'payload': payload,
    }


async def update_memory_usage(
    memory_ids: List[str],
    registry,  # 由调用方传入，不使用 Depends
) -> None:
    """
    异步回写命中的记忆访问计数，不阻塞检索热路径。

    通常在搜索接口中调用，使用 asyncio.create_task 后台执行。

    Args:
        memory_ids: 被命中的记忆 ID 列表
        registry: 服务注册表（由调用方传入）
    """
    if not registry.qdrant or not registry.qdrant.is_available():
        return

    # 去重，避免重复更新
    unique_ids = list(dict.fromkeys([mid for mid in memory_ids if mid]))

    for memory_id in unique_ids:
        try:
            # 如果 qdrant 客户端有专用的 update_usage 方法，直接调用
            if hasattr(registry.qdrant, 'update_usage'):
                await registry.qdrant.update_usage(memory_id)
            else:
                # 否则手动增加 access_count
                memory = await registry.qdrant.get_memory(memory_id)
                if memory:
                    payload = payload_of(memory)
                    payload['access_count'] = payload.get('access_count', 0) + 1
                    payload['last_accessed_at'] = datetime.now().isoformat()
                    await registry.qdrant.update_memory(memory_id, payload)
        except Exception as e:
            logger.debug(f"回写记忆使用计数失败 {memory_id}: {e}")


async def dispose_merged_duplicate(
    duplicate_id: str,
    keeper_id: str,
    duplicate_action: str,
    registry,  # 由调用方传入，不使用 Depends
    similarity: Optional[float] = None,
) -> bool:
    """
    处理合并后的重复记忆，根据动作归档/软删除/物理删除，并同步移除 BM25 索引。

    Args:
        duplicate_id: 被合并的重复记忆 ID
        keeper_id: 保留的记忆 ID
        duplicate_action: 处理动作（archive/soft_delete/delete）
        registry: 服务注册表（由调用方传入）
        similarity: 相似度分数（可选）
    """
    qdrant = registry.qdrant
    if not qdrant:
        return False

    action = normalize_duplicate_action(duplicate_action)

    if action == "delete":
        success = await qdrant.delete_memory(duplicate_id)
        if success:
            await remove_bm25_document(duplicate_id, registry)
        return success

    # 记录合并元数据（便于追溯）
    merge_metadata = {
        'merged_into': keeper_id,
        'merged_at': datetime.now().isoformat(),
        'deduplicate_action': action,
        'deduplicate_similarity': round(similarity, 4) if similarity is not None else None,
    }
    await qdrant.update_memory(duplicate_id, merge_metadata)

    if action == "archive":
        success = await qdrant.archive_memory(duplicate_id, reason='deduplicate')
    else:  # soft_delete
        success = await qdrant.soft_delete_memory(duplicate_id, reason='deduplicate')

    if success:
        await remove_bm25_document(duplicate_id, registry)
    return success