# api/routes/graph.py
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
import uuid

from ..schemas import AddEntityRequest, AddRelationRequest
from ..service_registry import ServiceRegistry
from ..dependencies import get_registry

import logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Graph"])


# ---------- 统计 ----------
@router.get("/graph/stats")
async def get_graph_stats(
    registry: ServiceRegistry = Depends(get_registry),
):
    """获取知识图谱统计"""
    graph = registry.graph
    if not graph or not graph.is_available():
        return {"status": "disabled", "message": "知识图谱未启用"}

    stats = graph.get_stats()  # 同步调用，无 await
    return {
        "status": "enabled",
        "entity_count": stats.get('entity_count', 0),
        "relation_count": stats.get('relation_count', 0),
    }


# ---------- 实体管理 ----------
@router.get("/graph/entities")
async def list_entities(
    entity_type: Optional[str] = None,
    limit: int = 50,
    user_id: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """列出实体"""
    graph = registry.graph
    config = registry.config

    if not graph or not graph.is_available():
        return {"entities": [], "message": "知识图谱未启用"}

    user_id = user_id if user_id is not None else config.users.default_user_id
    entities = graph.list_entities(user_id=user_id, entity_type=entity_type, limit=limit)  # 同步
    return {"entities": entities, "count": len(entities)}


@router.get("/graph/entity/{entity_id}")
async def get_entity(
    entity_id: str,
    registry: ServiceRegistry = Depends(get_registry),
):
    """获取实体详情"""
    graph = registry.graph
    if not graph or not graph.is_available():
        raise HTTPException(status_code=503, detail="图数据库未启用")

    entity = graph.get_entity(entity_id)  # 同步
    if entity:
        return entity
    raise HTTPException(status_code=404, detail="实体不存在")


@router.post("/graph/entity")
async def add_entity(
    request: AddEntityRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """添加实体"""
    graph = registry.graph
    config = registry.config

    if not graph or not graph.is_available():
        raise HTTPException(status_code=503, detail="图数据库未启用")

    user_id = request.user_id if request.user_id is not None else config.users.default_user_id
    entity_id = request.entity_id or str(uuid.uuid4())

    success = graph.add_entity(  # 同步
        entity_id=entity_id,
        entity_type=request.entity_type,
        name=request.name,
        properties=request.properties,
        user_id=user_id,
    )

    if success:
        return {"status": "success", "entity_id": entity_id, "name": request.name}
    raise HTTPException(status_code=500, detail="添加实体失败")


@router.delete("/graph/entity/{entity_id}")
async def delete_entity(
    entity_id: str,
    registry: ServiceRegistry = Depends(get_registry),
):
    """删除实体"""
    graph = registry.graph
    if not graph or not graph.is_available():
        raise HTTPException(status_code=503, detail="图数据库未启用")

    success = graph.delete_entity(entity_id)  # 同步
    if success:
        return {"status": "success", "message": f"已删除实体 {entity_id}"}
    raise HTTPException(status_code=404, detail="实体不存在或删除失败")


# ---------- 关系管理 ----------
@router.post("/graph/relation")
async def add_relation(
    request: AddRelationRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """添加关系"""
    graph = registry.graph
    if not graph or not graph.is_available():
        raise HTTPException(status_code=503, detail="图数据库未启用")

    success = graph.add_relation(  # 同步
        source_id=request.source_id,
        target_id=request.target_id,
        relation_type=request.relation_type,
        properties=request.properties,
    )

    if success:
        return {
            "status": "success",
            "relation": f"{request.source_id} -[{request.relation_type}]-> {request.target_id}",
        }
    raise HTTPException(status_code=400, detail="添加关系失败（请确保两个实体都存在）")


@router.get("/graph/entity/{entity_id}/relations")
async def get_entity_relations(
    entity_id: str,
    direction: str = Query(default="both", pattern="^(in|out|both)$"),
    registry: ServiceRegistry = Depends(get_registry),
):
    """获取实体的所有关系"""
    graph = registry.graph
    if not graph or not graph.is_available():
        raise HTTPException(status_code=503, detail="图数据库未启用")

    relations = graph.get_relations(entity_id, direction)  # 同步
    return {"entity_id": entity_id, "relations": relations, "count": len(relations)}


# ---------- 图谱搜索 ----------
@router.post("/graph/search")
async def graph_search(
    entity_names: List[str],
    user_id: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """根据实体名称搜索图谱"""
    graph = registry.graph
    config = registry.config

    if not graph or not graph.is_available():
        return {"results": [], "message": "图数据库未启用"}

    user_id = user_id if user_id is not None else config.users.default_user_id
    results = graph.search_by_entities(entity_names, user_id)  # 同步
    return {"results": results, "count": len(results)}


@router.post("/graph/query/related")
async def find_related(
    entity_id: str,
    max_depth: int = 2,
    registry: ServiceRegistry = Depends(get_registry),
):
    """查找相关实体"""
    graph = registry.graph
    if not graph or not graph.is_available():
        return {"related": [], "message": "知识图谱未启用"}

    related = graph.find_related_entities(entity_id, max_depth)  # 同步
    return {"entity_id": entity_id, "related": related}


@router.get("/graph/path")
async def find_path(
    source_id: str,
    target_id: str,
    max_length: int = 5,
    registry: ServiceRegistry = Depends(get_registry),
):
    """查找两个实体之间的路径"""
    graph = registry.graph
    if not graph or not graph.is_available():
        raise HTTPException(status_code=503, detail="图数据库未启用")

    path = graph.find_path(source_id, target_id, max_length)  # 同步
    if path:
        return {"path": path, "length": len(path)}
    return {"path": None, "message": "未找到路径"}


# ---------- 额外：列出所有关系 ----------
@router.get("/graph/relations")
async def list_relations(
    limit: int = 500,
    user_id: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """列出所有关系（用于图谱可视化）"""
    graph = registry.graph
    config = registry.config

    if not graph or not graph.is_available():
        return {"relations": [], "message": "知识图谱未启用"}

    user_id = user_id if user_id is not None else config.users.default_user_id
    try:
        relations = graph.list_all_relations(user_id=user_id, limit=limit)  # 同步
        return {"relations": relations, "count": len(relations)}
    except Exception as e:
        logger.error(f"获取关系列表失败: {e}")
        return {"relations": [], "error": str(e)}