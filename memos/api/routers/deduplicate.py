# api/routes/deduplicate.py
from fastapi import APIRouter, HTTPException
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..service_registry import get_service_registry
from ..utils import merge_memories

router = APIRouter(tags=["Deduplicate"])

@router.post("/deduplicate")
async def deduplicate_memories(
    threshold: float = 0.90,
    by_type: bool = True
):
    """去重（支持按记忆类型分组）
    
    Args:
        threshold: 相似度阈值，默认 0.90
        by_type: 是否按 memory_type 分组去重，默认 True
                 - True: 只在同类型记忆之间去重（推荐）
                 - False: 全局去重（所有记忆之间比较）
    """
    registry = get_service_registry()
    qdrant = registry.qdrant
    
    if not qdrant or not qdrant.is_available():
        return {"status": "error", "message": "存储不可用"}
    
    # 获取所有记忆
    memories = qdrant.get_all_memories(limit=10000)
    
    if len(memories) < 2:
        return {"status": "success", "merged_count": 0, "by_type": by_type}
    
    deleted_ids = set()
    merged_count = 0
    type_stats = {}
    
    if by_type:
        # 按 memory_type 分组
        type_groups = {}
        for mem in memories:
            mem_type = mem.get('memory_type', 'general')
            type_groups.setdefault(mem_type, []).append(mem)
        
        print(f"按类型分组去重（阈值: {threshold}）")
        for mem_type, group in type_groups.items():
            print(f"{mem_type}: {len(group)} 条记忆")
        
        # 在每个类型组内去重
        for mem_type, group in type_groups.items():
            if len(group) < 2:
                continue
            
            group_deleted = 0
            
            for i, mem_i in enumerate(group):
                if mem_i['id'] in deleted_ids:
                    continue
                
                full_mem_i = qdrant.get_memory(mem_i['id'])
                if not full_mem_i or 'vector' not in full_mem_i:
                    continue
                
                emb_i = np.array(full_mem_i['vector'])
                
                for j in range(i + 1, len(group)):
                    mem_j = group[j]
                    if mem_j['id'] in deleted_ids:
                        continue
                    
                    full_mem_j = qdrant.get_memory(mem_j['id'])
                    if not full_mem_j or 'vector' not in full_mem_j:
                        continue
                    
                    emb_j = np.array(full_mem_j['vector'])
                    similarity = float(cosine_similarity([emb_i], [emb_j])[0][0])
                    
                    if similarity >= threshold:
                        print(f"[{mem_type}] 发现相似记忆 (相似度: {similarity:.2%})")
                        print(f"记忆1: {mem_i.get('content', '')[:50]}...")
                        print(f"记忆2: {mem_j.get('content', '')[:50]}...")
                        
                        # 使用 LLM 智能合并
                        merge_success = await merge_memories(
                            keeper_id=mem_i['id'],
                            content_a=mem_i.get('content', ''),
                            content_b=mem_j.get('content', '')
                        )
                        
                        if merge_success:
                            deleted_ids.add(mem_j['id'])
                            merged_count += 1
                            group_deleted += 1
                        else:
                            print(f"[{mem_type}] LLM合并失败，两条均保留")
            
            if group_deleted > 0:
                type_stats[mem_type] = group_deleted
    else:
        # 全局去重
        print(f"全局去重（阈值: {threshold}）")
        
        for i, mem_i in enumerate(memories):
            if mem_i['id'] in deleted_ids:
                continue
            
            full_mem_i = qdrant.get_memory(mem_i['id'])
            if not full_mem_i or 'vector' not in full_mem_i:
                continue
            
            emb_i = np.array(full_mem_i['vector'])
            
            for j in range(i + 1, len(memories)):
                mem_j = memories[j]
                if mem_j['id'] in deleted_ids:
                    continue
                
                full_mem_j = qdrant.get_memory(mem_j['id'])
                if not full_mem_j or 'vector' not in full_mem_j:
                    continue
                
                emb_j = np.array(full_mem_j['vector'])
                similarity = float(cosine_similarity([emb_i], [emb_j])[0][0])
                
                if similarity >= threshold:
                    print(f"发现相似记忆 (相似度: {similarity:.2%})")
                    print(f"记忆1: {mem_i.get('content', '')[:50]}...")
                    print(f"记忆2: {mem_j.get('content', '')[:50]}...")
                    
                    merge_success = await merge_memories(
                        keeper_id=mem_i['id'],
                        content_a=mem_i.get('content', ''),
                        content_b=mem_j.get('content', '')
                    )
                    
                    if merge_success:
                        deleted_ids.add(mem_j['id'])
                        merged_count += 1
                    else:
                        print(f"LLM合并失败，两条均保留")
    
    # 删除重复记忆
    if deleted_ids:
        qdrant.delete_memories_batch(list(deleted_ids))
    
    print(f"去重完成！合并 {merged_count} 条记忆")
    
    return {
        "status": "success",
        "merged_count": merged_count,
        "remaining_count": len(memories) - len(deleted_ids),
        "by_type": by_type,
        "type_stats": type_stats if by_type else None
    }