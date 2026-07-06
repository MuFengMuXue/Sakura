# api/routes/reclassify.py
from fastapi import APIRouter, HTTPException, Query,Depends
import uuid
from datetime import datetime
from typing import Optional

from ..service_registry import ServiceRegistry
from ..utils import encode_text, extract_memories
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
    """批量重新分类所有历史记忆
    
    将每条现有记忆重新用 LLM 拆分+分类，生成新的分类记忆，保留原始时间戳。
    
    Args:
        dry_run: 是否只预览不执行（True 时只返回预览结果，不修改数据）
        limit: 处理的最大记忆数量
    """
    qdrant = registry.qdrant
    embedder = registry.embedder
    config = registry.config
    
    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=500, detail="存储不可用")
    
    if embedder is None:
        raise HTTPException(status_code=500, detail="Embedding 模型未加载")
    
    user_id = user_id if user_id is not None else config.users.default_user_id
    
    try:
        # 1. 获取所有记忆
        all_memories = await qdrant.get_all_memories(limit=limit)
        total_original = len(all_memories)
        
        if total_original == 0:
            return {"status": "success", "message": "没有记忆需要处理"}
        logging.info(f"开始处理 {total_original} 条历史记忆...")
        
        new_memories_to_add = []  # 待添加的新记忆
        original_ids_to_delete = []  # 待删除的原记忆 ID
        failed_count = 0
        skipped_count = 0
        type_distribution = {}
        
        # 2. 逐条处理
        for idx, mem in enumerate(all_memories):
            original_id = mem['id']
            original_content = mem.get('content', '')
            original_time = mem.get('created_at') or mem.get('timestamp', datetime.now().isoformat())
            original_importance = mem.get('importance', 0.5)
            
            if not original_content or len(original_content) < 5:
                skipped_count += 1
                continue
            
            logging.info(f"\n[{idx+1}/{total_original}] 处理: {original_content[:50]}...")
            
            try:
                # 调用 LLM 提取函数
                result = await extract_memories(original_content,registry)
                extracted_memories = result.get('memories', [])
                
                if not extracted_memories:
                    logging.warning(f"提取失败，保留原记忆")
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
                        'created_at': original_time,
                        'user_id': user_id,
                        'original_id': original_id,
                        'reclassified': True
                    })
                    
                    type_label = {'preference': '偏好', 'fact': '事实', 'episodic': '情景', 
                                 'semantic': '语义', 'procedural': '程序性', 'general': '通用'}.get(memory_type, memory_type)
                    logging.info(f"[{type_label}] {new_content[:40]}...")
                
                # 标记原记忆待删除
                original_ids_to_delete.append(original_id)
                
            except Exception as e:
                logging.error(f"处理失败: {e}")
                failed_count += 1
                continue
        
        # 3. 预览模式 - 只返回结果不执行
        if dry_run:
            logging.info(f"不执行实际修改")
            
            return {
                "status": "preview",
                "dry_run": True,
                "original_count": total_original,
                "will_delete": len(original_ids_to_delete),
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
                    } for m in new_memories_to_add[:10]
                ]
            }
        
        # 4. 执行模式 - 添加新记忆并删除原记忆
        logging.info(f"开始写入新记忆...")
        
        added_count = 0
        duplicate_skipped = 0
        
        for new_mem in new_memories_to_add:
            content = new_mem['content']
            vector = await encode_text(content,registry)
            
            # 去重检查
            similar = await qdrant.find_similar(vector, threshold=0.95, user_id=new_mem['user_id'])
            if similar:
                duplicate_skipped += 1
                continue
            
            # 添加新记忆
            memory_id = str(uuid.uuid4())
            payload = {
                'content': content,
                'user_id': new_mem['user_id'],
                'importance': new_mem['importance'],
                'memory_type': new_mem['memory_type'],
                'tags': new_mem['tags'],
                'created_at': new_mem['created_at'],
                'merge_count': 0,
                'processed': True,
                'reclassified': True,
                'original_id': new_mem.get('original_id')
            }
            
            await qdrant.add_memory(memory_id, vector, payload)
            added_count += 1
        
        # 删除原记忆
        if original_ids_to_delete:
            logging.info(f"\n删除 {len(original_ids_to_delete)} 条原记忆...")
            await qdrant.delete_memories_batch(original_ids_to_delete)
        
        logging.info(f"重新分类完成！")
        logging.info(f"原记忆: {total_original} 条")
        logging.info(f"新记忆: {added_count} 条")
        logging.info(f"去重跳过: {duplicate_skipped} 条")
        logging.warning(f"失败: {failed_count} 条")
        
        return {
            "status": "success",
            "original_count": total_original,
            "deleted_count": len(original_ids_to_delete),
            "added_count": added_count,
            "duplicate_skipped": duplicate_skipped,
            "failed": failed_count,
            "skipped": skipped_count,
            "type_distribution": type_distribution
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"重新分类失败: {str(e)}")