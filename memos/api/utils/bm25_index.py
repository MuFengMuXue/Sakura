"""
有关BM25的工具
"""
from ..service_registry import get_service_registry


def update_bm25_index(memory_id: str, content: str):
    registry = get_service_registry()
    if registry.bm25 and hasattr(registry.bm25, 'add_document'):
        try:
            registry.bm25.add_document(memory_id, content)
        except Exception as e:
            print(f"BM25 索引更新失败: {e}")


async def rebuild_bm25_index():
    registry = get_service_registry()
    if not registry.bm25 or not registry.qdrant:
        return
 