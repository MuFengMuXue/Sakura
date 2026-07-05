"""
有关文本向量化的工具
"""
from fastapi import Depends
from ..dependencies import get_registry
from typing import List, Optional
from ..service_registry import ServiceRegistry


async def encode_text(text: str,registry: ServiceRegistry = Depends(get_registry),embedder: Optional[object] = None,) -> List[float]:
    """
    处理单条
    """
    if embedder is None:
        embedder = registry.embedder
    if embedder is None:
        raise RuntimeError("Embedder 未初始化")
    result = await embedder.encode([text])
    return result[0] if result else []


async def encode_texts(texts: List[str],registry: ServiceRegistry = Depends(get_registry), embedder: Optional[object] = None,) -> List[List[float]]:
    """
    处理多条
    """
    if not texts:
        return []
    if embedder is None:
        embedder = registry.embedder
    if embedder is None:
        raise RuntimeError("Embedder 未初始化")
    return await embedder.encode(texts)