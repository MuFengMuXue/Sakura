# api/routes/deduplicate.py
from fastapi import APIRouter, HTTPException, Depends, Query
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ..dependencies import get_registry
from ..service_registry import ServiceRegistry
from ..utils import (
    merge_memories,
    choose_deduplicate_keeper,
    normalize_duplicate_action,
    dispose_merged_duplicate,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Deduplicate"])


@router.post("/deduplicate")
async def deduplicate_memories(
    threshold: float = Query(0.90, description="相似度阈值"),
    by_type: bool = Query(True, description="是否按 memory_type 分组去重"),
    duplicate_action: str = Query(
        "soft_delete",
        description="重复项处理: archive=归档, soft_delete=软删除, delete=物理删除"
    ),
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    去重记忆，支持按类型分组，使用 LLM 智能合并，并处理重复项。
    """
    qdrant = registry.qdrant
    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")

    # 获取所有记忆
    memories = await qdrant.get_all_memories(limit=10000)
    if len(memories) < 2:
        return {"status": "success", "merged_count": 0, "by_type": by_type}

    # 归一化动作
    action = normalize_duplicate_action(duplicate_action)

    # 已处理的记忆 ID（被合并掉的重复项）
    processed_ids = set()
    merged_count = 0
    type_stats = {}

    if by_type:
        # 按 memory_type 分组
        type_groups = {}
        for mem in memories:
            mem_type = mem.get('memory_type', 'general')
            type_groups.setdefault(mem_type, []).append(mem)

        logger.info(f"按类型分组去重（阈值: {threshold}）")
        for mem_type, group in type_groups.items():
            logger.info(f"{mem_type}: {len(group)} 条记忆")

        for mem_type, group in type_groups.items():
            if len(group) < 2:
                continue

            group_merged = 0

            for i, mem_i in enumerate(group):
                if mem_i['id'] in processed_ids:
                    continue

                full_i = await qdrant.get_memory(mem_i['id'])
                if not full_i or 'vector' not in full_i:
                    continue

                emb_i = np.array(full_i['vector'])

                for j in range(i + 1, len(group)):
                    mem_j = group[j]
                    if mem_j['id'] in processed_ids:
                        continue

                    full_j = await qdrant.get_memory(mem_j['id'])
                    if not full_j or 'vector' not in full_j:
                        continue

                    emb_j = np.array(full_j['vector'])
                    similarity = float(cosine_similarity([emb_i], [emb_j])[0][0])

                    if similarity >= threshold:
                        logger.info(f"[{mem_type}] 发现相似记忆 (相似度: {similarity:.2%})")
                        logger.info(f"  记忆1: {mem_i.get('content', '')[:50]}...")
                        logger.info(f"  记忆2: {mem_j.get('content', '')[:50]}...")

                        # 选择保留方（根据重要度、层级、访问次数等）
                        keeper = choose_deduplicate_keeper(full_i, full_j)
                        duplicate = full_j if keeper['id'] == full_i['id'] else full_i

                        # 使用 LLM 合并
                        merge_success = await merge_memories(
                            keeper_id=keeper['id'],
                            content_a=keeper.get('content', ''),
                            content_b=duplicate.get('content', ''),
                            registry=registry,
                        )

                        if merge_success:
                            # 处理重复项（归档/软删除/物理删除）
                            await dispose_merged_duplicate(
                                duplicate_id=duplicate['id'],
                                keeper_id=keeper['id'],
                                duplicate_action=action,
                                similarity=similarity,
                                registry=registry,
                            )
                            processed_ids.add(duplicate['id'])
                            merged_count += 1
                            group_merged += 1

                            # 如果 keeper 是当前 mem_i，更新其向量和内容，继续比较
                            if keeper['id'] == mem_i['id']:
                                # 重新获取更新后的记忆
                                full_i = await qdrant.get_memory(mem_i['id'])
                                if full_i and 'vector' in full_i:
                                    emb_i = np.array(full_i['vector'])
                            else:
                                # 如果 keeper 是 mem_j，则 mem_i 已被合并为重复项，跳出内层循环
                                break
                        else:
                            logger.warning(f"[{mem_type}] LLM合并失败，两条均保留")

            if group_merged > 0:
                type_stats[mem_type] = group_merged

    else:
        # 全局去重
        logger.info(f"全局去重（阈值: {threshold}）")

        for i, mem_i in enumerate(memories):
            if mem_i['id'] in processed_ids:
                continue

            full_i = await qdrant.get_memory(mem_i['id'])
            if not full_i or 'vector' not in full_i:
                continue

            emb_i = np.array(full_i['vector'])

            for j in range(i + 1, len(memories)):
                mem_j = memories[j]
                if mem_j['id'] in processed_ids:
                    continue

                full_j = await qdrant.get_memory(mem_j['id'])
                if not full_j or 'vector' not in full_j:
                    continue

                emb_j = np.array(full_j['vector'])
                similarity = float(cosine_similarity([emb_i], [emb_j])[0][0])

                if similarity >= threshold:
                    logger.info(f"发现相似记忆 (相似度: {similarity:.2%})")
                    logger.info(f"  记忆1: {mem_i.get('content', '')[:50]}...")
                    logger.info(f"  记忆2: {mem_j.get('content', '')[:50]}...")

                    keeper = choose_deduplicate_keeper(full_i, full_j)
                    duplicate = full_j if keeper['id'] == full_i['id'] else full_i

                    merge_success = await merge_memories(
                        keeper_id=keeper['id'],
                        content_a=keeper.get('content', ''),
                        content_b=duplicate.get('content', ''),
                        registry=registry,
                    )

                    if merge_success:
                        await dispose_merged_duplicate(
                            duplicate_id=duplicate['id'],
                            keeper_id=keeper['id'],
                            duplicate_action=action,
                            similarity=similarity,
                            registry=registry,
                        )
                        processed_ids.add(duplicate['id'])
                        merged_count += 1

                        if keeper['id'] == mem_i['id']:
                            full_i = await qdrant.get_memory(mem_i['id'])
                            if full_i and 'vector' in full_i:
                                emb_i = np.array(full_i['vector'])
                        else:
                            break
                    else:
                        logger.warning("LLM合并失败，两条均保留")

    logger.info(f"去重完成！合并 {merged_count} 条记忆")

    return {
        "status": "success",
        "merged_count": merged_count,
        "remaining_count": len(memories) - len(processed_ids),
        "by_type": by_type,
        "duplicate_action": action,
        "type_stats": type_stats if by_type else None,
    }