# api/routes/reclassify.py
from fastapi import APIRouter, HTTPException, Query, Depends
import uuid
from datetime import datetime
from typing import Optional

from ..service_registry import ServiceRegistry
from ..utils import (
    encode_text,
    extract_memories,
    update_bm25_index,
    remove_bm25_document,
)
from ..dependencies import get_registry
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reclassify"])


@router.post("/reclassify")
async def reclassify_all_memories(
    dry_run: bool = False,
    limit: int = 10000,
    user_id: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    批量重新分类所有历史记忆

    将每条现有记忆重新用 LLM 拆分+分类，生成新的分类记忆，保留原始时间戳。
    原记忆会被归档（可恢复），新记忆经过去重后添加。

    Args:
        dry_run: 是否只预览不执行（True 时只返回预览结果，不修改数据）
        limit: 处理的最大记忆数量
        user_id: 指定用户 ID（可选，不指定则使用各记忆自己的 user_id）
    """
    qdrant = registry.qdrant
    embedder = registry.embedder
    config = registry.config

    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=500, detail="存储不可用")

    if embedder is None:
        raise HTTPException(status_code=500, detail="Embedding 模型未加载")

    # 获取所有记忆
    all_memories = await qdrant.get_all_memories(limit=limit)
    total_original = len(all_memories)

    if total_original == 0:
        return {"status": "success", "message": "没有记忆需要处理"}

    logger.info(f"开始重新分类 {total_original} 条历史记忆...")

    new_memories_to_add = []    # 待添加的新记忆
    original_ids_to_archive = []  # 待归档的原记忆 ID
    failed_count = 0
    skipped_count = 0
    type_distribution = {}

    # 逐条处理
    for idx, mem in enumerate(all_memories):
        original_id = mem['id']
        original_content = mem.get('content', '')
        original_time = mem.get('created_at') or mem.get('timestamp', datetime.now().isoformat())
        original_importance = mem.get('importance', 0.5)
        mem_user_id = mem.get('user_id') or config.users.default_user_id

        # 如果指定了 user_id，只处理该用户的记忆
        if user_id and mem_user_id != user_id:
            continue

        if not original_content or len(original_content) < 5:
            skipped_count += 1
            continue

        logger.info(f"[{idx+1}/{total_original}] 处理: {original_content[:50]}...")

        try:
            # 调用 LLM 提取记忆（不传 context_summary）
            result = await extract_memories(original_content, registry=registry)
            extracted_memories = result.get('memories', [])

            if not extracted_memories:
                logger.warning("提取失败，保留原记忆")
                failed_count += 1
                continue

            # 处理提取出的每条新记忆
            for new_mem in extracted_memories:
                new_content = new_mem.get('content', '').strip()
                if not new_content or len(new_content) < 5:
                    continue

                memory_type = new_mem.get('memory_type', 'general')
                tags = new_mem.get('tags', [])
                importance = new_mem.get('importance', original_importance)

                # 统计类型分布
                type_distribution[memory_type] = type_distribution.get(memory_type, 0) + 1

                new_memories_to_add.append({
                    'content': new_content,
                    'memory_type': memory_type,
                    'tags': tags,
                    'importance': importance,
                    'created_at': original_time,    # 继承原始时间戳
                    'user_id': mem_user_id,
                    'original_id': original_id,
                    'reclassified': True,
                })

                type_label = {
                    'preference': '偏好',
                    'fact': '事实',
                    'episodic': '情景',
                    'semantic': '语义',
                    'procedural': '程序性',
                    'general': '通用'
                }.get(memory_type, memory_type)
                logger.info(f"  ✅ [{type_label}] {new_content[:40]}...")

            # 标记原记忆待归档
            original_ids_to_archive.append(original_id)

        except Exception as e:
            logger.error(f"处理失败: {e}")
            failed_count += 1
            continue

    # 预览模式
    if dry_run:
        logger.info("预览模式：不执行实际修改")
        return {
            "status": "preview",
            "dry_run": True,
            "original_count": total_original,
            "will_archive": len(original_ids_to_archive),
            "will_add": len(new_memories_to_add),
            "failed": failed_count,
            "skipped": skipped_count,
            "type_distribution": type_distribution,
            "sample_new_memories": [
                {
                    "content": m['content'][:100],
                    "memory_type": m['memory_type'],
                    "tags": m['tags'],
                    "created_at": m['created_at']
                }
                for m in new_memories_to_add[:10]
            ]
        }

    # 执行模式
    logger.info("开始写入新记忆...")
    added_count = 0
    duplicate_skipped = 0

    for new_mem in new_memories_to_add:
        content = new_mem['content']
        vector = await encode_text(content, registry.embedder)

        # 去重检查（相似度 ≥ 0.95 视为重复，跳过）
        similar = await qdrant.find_similar(
            vector, threshold=0.95, user_id=new_mem['user_id']
        )
        if similar:
            duplicate_skipped += 1
            continue

        # 添加新记忆（不显式设置 layer，与 v2 一致）
        memory_id = str(uuid.uuid4())
        payload = {
            'content': content,
            'user_id': new_mem['user_id'],
            'importance': new_mem['importance'],
            'memory_type': new_mem['memory_type'],
            'tags': new_mem['tags'],
            'created_at': new_mem['created_at'],   # 使用原始时间
            'merge_count': 0,
            'processed': True,
            'reclassified': True,
            'original_id': new_mem.get('original_id'),
        }

        await qdrant.add_memory(memory_id, vector, payload)
        # 更新 BM25 索引
        await update_bm25_index(memory_id, content, registry)
        added_count += 1

    # 归档原记忆（保留可恢复性，与 v2 一致）
    if original_ids_to_archive:
        logger.info(f"归档 {len(original_ids_to_archive)} 条原记忆...")
        for memory_id in original_ids_to_archive:
            # 优先使用 archive_memory，否则使用 soft_delete_memory
            if hasattr(qdrant, 'archive_memory'):
                await qdrant.archive_memory(memory_id, reason='reclassified')
            elif hasattr(qdrant, 'soft_delete_memory'):
                await qdrant.soft_delete_memory(memory_id, reason='reclassified')
            else:
                # 降级：直接删除（但原 v2 至少会 soft_delete）
                await qdrant.delete_memory(memory_id)
            # 从 BM25 索引移除
            await remove_bm25_document(memory_id, registry)

    logger.info("重新分类完成！")
    logger.info(f"原记忆: {total_original} 条")
    logger.info(f"新记忆: {added_count} 条")
    logger.info(f"去重跳过: {duplicate_skipped} 条")
    logger.info(f"失败: {failed_count} 条")

    return {
        "status": "success",
        "original_count": total_original,
        "archived_count": len(original_ids_to_archive),
        "added_count": added_count,
        "duplicate_skipped": duplicate_skipped,
        "failed": failed_count,
        "skipped": skipped_count,
        "type_distribution": type_distribution,
    }