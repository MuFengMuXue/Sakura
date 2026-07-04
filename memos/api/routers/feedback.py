# api/routes/feedback.py
from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid

from ..schemas import MemoryFeedbackRequest
from ..service_registry import get_service_registry
from ..utils import encode_text

router = APIRouter(tags=["Feedback"])

@router.post("/memory/feedback")
async def submit_memory_feedback(request: MemoryFeedbackRequest):
    """提交记忆反馈（修正/补充/删除/合并）
    
    feedback_type:
    - correct: 修正记忆内容
    - supplement: 补充信息
    - delete: 标记删除
    - merge: 合并到其他记忆
    """
    registry = get_service_registry()
    qdrant = registry.qdrant
    
    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")
    
    try:
        # 获取原记忆
        original = qdrant.get_memory(request.memory_id)
        if not original:
            raise HTTPException(status_code=404, detail="记忆不存在")
        
        if request.feedback_type == "correct":
            # 修正记忆内容
            if not request.correction:
                raise HTTPException(status_code=400, detail="修正内容不能为空")
            
            # 更新内容
            new_vector = await encode_text(request.correction)
            payload = original.get('payload', {})
            payload['content'] = request.correction
            payload['updated_at'] = datetime.now().isoformat()
            payload['correction_history'] = payload.get('correction_history', [])
            payload['correction_history'].append({
                'original': original.get('payload', {}).get('content'),
                'corrected_at': datetime.now().isoformat(),
                'reason': request.reason
            })
            
            qdrant.update_memory(request.memory_id, payload, new_vector)
            
            return {
                "status": "success",
                "action": "corrected",
                "memory_id": request.memory_id,
                "new_content": request.correction
            }
        
        elif request.feedback_type == "supplement":
            # 补充信息
            if not request.correction:
                raise HTTPException(status_code=400, detail="补充内容不能为空")
            
            payload = original.get('payload', {})
            original_content = payload.get('content', '')
            supplemented_content = f"{original_content}\n[补充] {request.correction}"
            
            new_vector = await encode_text(supplemented_content)
            payload['content'] = supplemented_content
            payload['updated_at'] = datetime.now().isoformat()
            
            qdrant.update_memory(request.memory_id, payload, new_vector)
            
            return {
                "status": "success",
                "action": "supplemented",
                "memory_id": request.memory_id,
                "new_content": supplemented_content
            }
        
        elif request.feedback_type == "delete":
            # 标记删除
            success = qdrant.delete_memory(request.memory_id)
            
            return {
                "status": "success" if success else "failed",
                "action": "deleted",
                "memory_id": request.memory_id
            }
        
        elif request.feedback_type == "merge":
            # 合并到其他记忆（需要 correction 字段指定目标记忆 ID）
            if not request.correction:
                raise HTTPException(status_code=400, detail="请指定目标记忆 ID")
            
            target_id = request.correction
            target = qdrant.get_memory(target_id)
            if not target:
                raise HTTPException(status_code=404, detail="目标记忆不存在")
            
            # 合并内容
            original_content = original.get('payload', {}).get('content', '')
            target_content = target.get('payload', {}).get('content', '')
            merged_content = f"{target_content}\n[合并自 {request.memory_id}] {original_content}"
            
            # 更新目标记忆
            new_vector = await encode_text(merged_content)
            target_payload = target.get('payload', {})
            target_payload['content'] = merged_content
            target_payload['updated_at'] = datetime.now().isoformat()
            target_payload['merge_count'] = target_payload.get('merge_count', 0) + 1
            target_payload['merged_from'] = target_payload.get('merged_from', [])
            target_payload['merged_from'].append(request.memory_id)
            
            qdrant.update_memory(target_id, target_payload, new_vector)
            
            # 删除原记忆
            qdrant.delete_memory(request.memory_id)
            
            return {
                "status": "success",
                "action": "merged",
                "source_id": request.memory_id,
                "target_id": target_id,
                "merged_content": merged_content
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"未知的反馈类型: {request.feedback_type}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory/{memory_id}/history")
async def get_memory_history(memory_id: str):
    """获取记忆的修改历史"""
    registry = get_service_registry()
    qdrant = registry.qdrant
    
    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")
    
    memory = qdrant.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    payload = memory.get('payload', {})
    
    return {
        "memory_id": memory_id,
        "current_content": payload.get('content'),
        "created_at": payload.get('created_at'),
        "updated_at": payload.get('updated_at'),
        "merge_count": payload.get('merge_count', 0),
        "merged_from": payload.get('merged_from', []),
        "correction_history": payload.get('correction_history', [])
    }