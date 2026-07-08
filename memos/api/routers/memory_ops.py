# api/routes/memory_ops.py
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
from datetime import datetime

from ..service_registry import ServiceRegistry
from ..dependencies import get_registry
from ..utils import (
    get_memory,
    flatten_memory,
    remove_bm25_document,
    update_bm25_index,
    normalize_layer,
    MEMORY_LAYERS,
    payload_of,
)
from ..schemas import GetMemoryByIdsRequest, DeleteMemoryRequest, RecoverMemoryRequest

import logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Memory Operations"])


# ==================== 获取单条记忆 ====================
@router.get("/memory/{memory_id}")
async def get_memory_by_id(
    memory_id: str,
    include_deleted: bool = Query(False, description="是否包含已删除的记忆"),
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    获取单条记忆的详细信息

    Args:
        memory_id: 记忆 ID
        include_deleted: 是否包含已软删除的记忆
    """
    memory = await get_memory(memory_id, include_deleted=include_deleted, registry=registry)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")

    return flatten_memory(memory)


# ==================== 批量获取记忆 ====================
@router.post("/get_memory_by_ids")
async def get_memory_by_ids(
    request: GetMemoryByIdsRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    批量获取记忆

    Args:
        request: 包含 memory_ids 列表和可选的 user_id
    """
    qdrant = registry.qdrant
    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")

    memories = []
    for memory_id in request.memory_ids:
        memory = await get_memory(
            memory_id,
            include_deleted=request.include_deleted,
            registry=registry,
        )
        if memory:
            item = flatten_memory(memory)
            # 如果指定了 user_id，过滤不属于该用户的记忆
            if request.user_id and item.get('payload', {}).get('user_id') not in (None, request.user_id):
                continue
            memories.append(item)

    return {
        "memories": memories,
        "count": len(memories),
    }


# ==================== 分层统计 ====================
@router.get("/memory/layers")
async def get_memory_layers(
    user_id: Optional[str] = Query(None, description="用户ID（可选）"),
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    按生命周期层与状态统计记忆分布

    Args:
        user_id: 用户 ID，不指定则使用配置中的默认用户
    """
    qdrant = registry.qdrant
    config = registry.config

    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")

    user_id = user_id if user_id is not None else config.users.default_user_id

    memories = await qdrant.get_all_memories(
        user_id=user_id,
        include_archived=True,
        include_deleted=False,
        limit=10000,
    )

    by_layer = {layer: 0 for layer in MEMORY_LAYERS}
    by_status = {}

    for mem in memories:
        payload = payload_of(mem)
        layer = normalize_layer(
            mem.get('layer') or payload.get('layer'),
            default='LongTermMemory',
        )
        status = mem.get('status') or payload.get('status', 'active')

        by_layer[layer] = by_layer.get(layer, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1

    return {
        "user_id": user_id,
        "layers": by_layer,
        "statuses": by_status,
        "total": len(memories),
    }


# ==================== 恢复记忆 ====================
@router.post("/memory/{memory_id}/restore")
async def restore_memory(
    memory_id: str,
    user_id: Optional[str] = Query(None, description="用户ID（可选）"),
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    恢复归档或软删除的记忆，并重新加入 BM25 索引

    Args:
        memory_id: 记忆 ID
        user_id: 用户 ID（用于权限校验）
    """
    qdrant = registry.qdrant
    config = registry.config

    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")

    # 获取记忆（包含已删除的）
    memory = await get_memory(memory_id, include_deleted=True, registry=registry)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")

    payload = payload_of(memory)
    check_user_id = user_id if user_id is not None else config.users.default_user_id

    # 权限校验
    if payload.get('user_id') not in (None, check_user_id):
        raise HTTPException(status_code=403, detail="无权恢复该记忆")

    # 恢复记忆
    if hasattr(qdrant, 'recover_memory'):
        success = await qdrant.recover_memory(memory_id)
    else:
        # 降级：手动更新状态
        success = await qdrant.update_memory(
            memory_id,
            {'status': 'active', 'deleted_at': None, 'archived_at': None}
        )

    if success:
        # 重新加入 BM25 索引
        content = payload.get('content', '')
        if content:
            await update_bm25_index(memory_id, content, registry)
        return {"status": "success", "memory_id": memory_id}

    return {"status": "failed", "memory_id": memory_id}


# ==================== 产品风格删除（兼容接口） ====================
@router.post("/delete_memory")
async def delete_memory_product(
    request: DeleteMemoryRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    产品风格删除接口，默认软删除

    Args:
        request: 包含 memory_id, user_id, reason, hard
    """
    qdrant = registry.qdrant
    config = registry.config

    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")

    # 获取记忆（包含已删除的）
    memory = await get_memory(
        request.memory_id,
        include_deleted=True,
        registry=registry,
    )
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")

    payload = payload_of(memory)
    check_user_id = request.user_id if request.user_id is not None else config.users.default_user_id

    # 权限校验
    if payload.get('user_id') not in (None, check_user_id):
        raise HTTPException(status_code=403, detail="无权删除该记忆")

    # 执行删除
    if request.hard:
        success = await qdrant.delete_memory(request.memory_id)
    else:
        if hasattr(qdrant, 'soft_delete_memory'):
            success = await qdrant.soft_delete_memory(
                request.memory_id,
                reason=request.reason,
            )
        else:
            success = await qdrant.delete_memory(request.memory_id)

    if success:
        # 从 BM25 移除
        await remove_bm25_document(request.memory_id, registry)

    return {
        "status": "success" if success else "failed",
        "memory_id": request.memory_id,
        "hard": request.hard,
    }


# ==================== 恢复记忆（产品风格） ====================
@router.post("/recover_memory")
async def recover_memory_product(
    request: RecoverMemoryRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    恢复软删除记忆（产品风格）

    Args:
        request: 包含 memory_id 或 delete_record_id，以及 user_id
    """
    qdrant = registry.qdrant
    config = registry.config

    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")

    target_id = request.memory_id

    # 如果通过 delete_record_id 查找
    if not target_id and request.delete_record_id:
        memories = await qdrant.get_all_memories(
            user_id=request.user_id or config.users.default_user_id,
            include_deleted=True,
            limit=10000,
        )
        for mem in memories:
            payload = payload_of(mem)
            if payload.get('delete_record_id') == request.delete_record_id:
                target_id = mem.get('id')
                break

    if not target_id:
        raise HTTPException(status_code=404, detail="未找到可恢复记忆")

    # 获取记忆
    memory = await get_memory(target_id, include_deleted=True, registry=registry)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")

    payload = payload_of(memory)
    check_user_id = request.user_id if request.user_id is not None else config.users.default_user_id

    # 权限校验
    if payload.get('user_id') not in (None, check_user_id):
        raise HTTPException(status_code=403, detail="无权恢复该记忆")

    # 恢复记忆
    if hasattr(qdrant, 'recover_memory'):
        success = await qdrant.recover_memory(target_id)
    else:
        success = await qdrant.update_memory(
            target_id,
            {'status': 'active', 'deleted_at': None, 'archived_at': None}
        )

    if success:
        content = payload.get('content', '')
        if content:
            await update_bm25_index(target_id, content, registry)
        return {"status": "success", "memory_id": target_id}

    return {"status": "failed", "memory_id": target_id}