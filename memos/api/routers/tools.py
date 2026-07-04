# api/routes/tools.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from ..schemas import RecordToolUsageRequest
from ..service_registry import get_service_registry

router = APIRouter(tags=["Tools"])

@router.delete("/tools/{record_id}")
async def delete_tool_record(record_id: str):
    """删除工具使用记录"""
    registry = get_service_registry()
    tool_memory = registry.tool_memory
    if not tool_memory:
        raise HTTPException(status_code=503, detail="工具记忆未初始化")
    
    try:
        success = await tool_memory.delete_record(record_id)
        if success:
            return {"status": "success", "message": "记录已删除"}
        else:
            raise HTTPException(status_code=404, detail="未找到该记录")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools/stats")
async def get_tool_stats():
    """获取工具使用统计"""
    registry = get_service_registry()
    tool_memory = registry.tool_memory
    if not tool_memory:
        return {"status": "disabled", "message": "工具记忆未启用"}
    
    stats = await tool_memory.get_stats()
    return {
        "status": "enabled",
        **stats
    }

@router.post("/tools/record")
async def record_tool_usage(request: RecordToolUsageRequest):
    """记录工具使用"""
    registry = get_service_registry()
    tool_memory = registry.tool_memory
    if not tool_memory:
        raise HTTPException(status_code=503, detail="工具记忆未启用")
    
    try:
        from memories.tool_memory import ToolCategory
        
        # 转换类别
        try:
            category = ToolCategory(request.tool_category)
        except ValueError:
            category = ToolCategory.OTHER
        
        # 注意：tool_memory 在初始化时已绑定 user_id（来自配置）
        record = await tool_memory.record_usage(
            tool_name=request.tool_name,
            tool_category=category,
            parameters=request.parameters or {},
            success=request.success,
            result_summary=request.result_summary,
            context=request.context,
            user_intent=request.user_intent
        )
        
        return {
            "status": "success",
            "record_id": record.id,
            "tool_name": record.tool_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools/frequently-used")
async def get_frequently_used_tools(
    category: Optional[str] = None,
    top_k: int = 10
):
    """获取常用工具列表"""
    registry = get_service_registry()
    tool_memory = registry.tool_memory
    if not tool_memory:
        return {"tools": [], "message": "工具记忆未启用"}
    
    try:
        from memories.tool_memory import ToolCategory
        
        cat = ToolCategory(category) if category else None
        tools = await tool_memory.get_frequently_used_tools(cat, top_k)
        
        return {
            "tools": [
                {
                    "tool_name": t.tool_name,
                    "category": t.tool_category.value,
                    "use_count": t.use_count,
                    "success_rate": round(t.success_rate, 2),
                    "last_used": t.last_used_at.isoformat()
                }
                for t in tools
            ],
            "count": len(tools)
        }
    except Exception as e:
        return {"tools": [], "error": str(e)}

@router.get("/tools/recent")
async def get_recent_tool_usage(
    tool_name: Optional[str] = None,
    limit: int = 20
):
    """获取最近工具使用记录"""
    registry = get_service_registry()
    tool_memory = registry.tool_memory
    if not tool_memory:
        return {"records": [], "message": "工具记忆未启用"}
    
    records = await tool_memory.get_recent_usage(tool_name, limit)
    return {
        "records": [
            {
                "id": r.id,
                "tool_name": r.tool_name,
                "category": r.tool_category.value,
                "success": r.success,
                "result_summary": r.result_summary,
                "user_intent": r.user_intent,
                "used_at": r.used_at.isoformat()
            }
            for r in records
        ],
        "count": len(records)
    }

@router.get("/tools/suggest/{tool_name}")
async def suggest_tool_parameters(tool_name: str):
    """根据历史使用建议工具参数"""
    registry = get_service_registry()
    tool_memory = registry.tool_memory
    if not tool_memory:
        return {"suggestions": {}, "message": "工具记忆未启用"}
    
    suggestions = await tool_memory.suggest_parameters(tool_name)
    return {"tool_name": tool_name, "suggestions": suggestions}