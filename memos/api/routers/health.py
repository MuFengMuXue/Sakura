# api/routes/health.py
from fastapi import APIRouter, Depends
from ..service_registry import ServiceRegistry
from ..dependencies import get_registry

router = APIRouter(tags=["Health"])


@router.get("/")
async def root(
    registry: ServiceRegistry = Depends(get_registry),
):
    return {
        "service": "MemOS API",
        "version": "2.0.0",
        "status": "running",
        "storage": "qdrant" if registry.qdrant and registry.qdrant.is_available() else "memory",
        "graph": "neo4j" if registry.graph and registry.graph.is_available() else "disabled"
    }


@router.get("/health")
async def health_check(
    registry: ServiceRegistry = Depends(get_registry),
):
    memory_count = 0
    if registry.qdrant and registry.qdrant.is_available():
        memory_count = await registry.qdrant.count_memories()  # 异步，需要 await

    return {
        "status": "healthy",
        "model_loaded": registry.embedder is not None,
        "qdrant_available": registry.qdrant is not None and registry.qdrant.is_available(),
        "neo4j_available": registry.graph is not None and registry.graph.is_available(),
        "memory_count": memory_count
    }


@router.get("/stats")
async def get_statistics(
    registry: ServiceRegistry = Depends(get_registry),
):
    stats = {
        "total_count": 0,
        "today_count": 0,
        "week_count": 0,
        "avg_importance": 0,
        "storage_type": "qdrant" if registry.qdrant else "memory",
        "graph_enabled": registry.graph is not None
    }

    # Qdrant 统计（异步）
    if registry.qdrant and registry.qdrant.is_available():
        info = await registry.qdrant.get_collection_info()  # 异步，需要 await
        stats["total_count"] = info.get('points_count', 0)

    # 图谱统计（同步）
    if registry.graph and registry.graph.is_available():
        graph_stats = registry.graph.get_stats()  # 同步，不需要 await
        stats["entity_count"] = graph_stats.get('entity_count', 0)
        stats["relation_count"] = graph_stats.get('relation_count', 0)

    return stats