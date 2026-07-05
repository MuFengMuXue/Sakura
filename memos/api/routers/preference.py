# api/routes/preferences.py
from fastapi import APIRouter, HTTPException, Query,Depends
from typing import Optional

from ..schemas import AddPreferenceRequest, ExtractPreferencesRequest
from ..service_registry import ServiceRegistry
from ..dependencies import get_registry

router = APIRouter(tags=["Preferences"])

@router.get("/preferences")
async def get_preferences(
    category: Optional[str] = Query(default=None, description="类别过滤"),
    preference_type: Optional[str] = Query(default=None, description="类型过滤: like/dislike"),
    registry: ServiceRegistry = Depends(get_registry),
):
    """获取用户偏好列表"""

    preference_memory = registry.preference_memory
    if not preference_memory:
        return {"preferences": [], "message": "偏好记忆未初始化"}

    try:
        from memories.preference_memory import PreferenceCategory, PreferenceType
        
        cat = PreferenceCategory(category) if category else None
        ptype = PreferenceType(preference_type) if preference_type else None
        
        # 注意：不传 user_id，因为实例化时已绑定
        prefs = await preference_memory.get_preferences(category=cat, preference_type=ptype)
        return {
            "preferences": [p.model_dump() for p in prefs],
            "count": len(prefs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preferences")
async def add_preference(request: AddPreferenceRequest,
                         registry: ServiceRegistry = Depends(get_registry),):
    """添加用户偏好"""
    preference_memory = registry.preference_memory
    if not preference_memory:
        raise HTTPException(status_code=503, detail="偏好记忆未初始化")

    try:
        from memories.preference_memory import PreferenceCategory, PreferenceType
        
        pref = await preference_memory.add_preference(
            item=request.item,
            category=PreferenceCategory(request.category),
            preference_type=PreferenceType(request.preference_type),
            strength=request.strength
        )
        return {
            "status": "success",
            "preference": pref.model_dump(),
            "message": f"已添加偏好: {'喜欢' if request.preference_type == 'like' else '不喜欢'}{request.item}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/preferences/{pref_id}")
async def delete_preference(pref_id: str,registry: ServiceRegistry = Depends(get_registry),):
    """删除偏好"""
    preference_memory = registry.preference_memory
    if not preference_memory:
        raise HTTPException(status_code=503, detail="偏好记忆未初始化")

    try:
        success = await preference_memory.delete_preference(pref_id)
        if success:
            return {"status": "success", "message": "偏好已删除"}
        else:
            raise HTTPException(status_code=404, detail="未找到该偏好")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preferences/summary")
async def get_preference_summary(registry: ServiceRegistry = Depends(get_registry),):
    """获取偏好摘要"""
    preference_memory = registry.preference_memory
    if not preference_memory:
        return {"summary": {}, "message": "偏好记忆未初始化"}

    try:
        summary = await preference_memory.get_summary()
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preferences/extract")
async def extract_preferences(request: ExtractPreferencesRequest,registry: ServiceRegistry = Depends(get_registry),):
    """从文本中提取偏好（使用 LLM）"""
    config = registry.config
    llm_cfg = config.llm.config
    preference_memory = registry.preference_memory

    if not llm_cfg.api_key or not llm_cfg.model or not llm_cfg.base_url:
        raise HTTPException(status_code=503, detail="LLM 未配置")

    try:
        from utils.entity_extractor import PreferenceExtractor
        
        # 直接传入主 LLM 配置，无 fallback
        extractor = PreferenceExtractor(llm_config=llm_cfg.dict())
        extracted = await extractor.extract_preferences(request.text)
        
        # 自动添加提取到的偏好
        added_count = 0
        if preference_memory and extracted:
            from memories.preference_memory import PreferenceCategory, PreferenceType
            for like in extracted.get('likes', []):
                try:
                    cat = PreferenceCategory(like.get('category', 'other'))
                    await preference_memory.add_preference(
                        item=like.get('item', ''),
                        category=cat,
                        preference_type=PreferenceType.LIKE,
                        strength=like.get('strength', 0.8)
                    )
                    added_count += 1
                except:
                    pass
            for dislike in extracted.get('dislikes', []):
                try:
                    cat = PreferenceCategory(dislike.get('category', 'other'))
                    await preference_memory.add_preference(
                        item=dislike.get('item', ''),
                        category=cat,
                        preference_type=PreferenceType.DISLIKE,
                        strength=dislike.get('strength', 0.8)
                    )
                    added_count += 1
                except:
                    pass
        return {
            "status": "success",
            "extracted": extracted,
            "added_count": added_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preferences/search")
async def search_preferences(query: str, top_k: int = 5,registry: ServiceRegistry = Depends(get_registry),):
    """搜索相关偏好"""
    preference_memory = registry.preference_memory
    if not preference_memory:
        return {"preferences": [], "message": "偏好记忆未初始化"}

    try:
        prefs = await preference_memory.search_preferences(query, top_k)
        return {
            "preferences": [p.model_dump() for p in prefs],
            "count": len(prefs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))