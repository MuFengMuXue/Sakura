"""
有关文本向量化的工具
"""
from typing import List, Optional
from ..service_registry import get_service_registry


async def encode_text(text: str, embedder: Optional[object] = None) -> List[float]:
    """
    处理单条
    """
    if embedder is None:
        registry = get_service_registry()
        embedder = registry.embedder
    if embedder is None:
        raise RuntimeError("Embedder 未初始化")
    result = await embedder.encode([text])
    return result[0] if result else []


async def encode_texts(texts: List[str], embedder: Optional[object] = None) -> List[List[float]]:
    """
    处理多条
    """
    if not texts:
        return []
    if embedder is None:
        registry = get_service_registry()
        embedder = registry.embedder
    if embedder is None:
        raise RuntimeError("Embedder 未初始化")
    return await embedder.encode(texts)