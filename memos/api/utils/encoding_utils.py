"""
文本向量化工具
"""
from typing import List, Optional


async def encode_text(
    text: str,
    embedder: Optional[object] = None,
) -> List[float]:
    """
    将单条文本编码为向量

    Args:
        text: 要编码的文本
        embedder: Embedding 客户端实例（必须提供）

    Returns:
        向量列表
    """
    if embedder is None:
        raise RuntimeError("Embedder 未提供")
    result = await embedder.encode([text])
    return result[0] if result else []


async def encode_texts(
    texts: List[str],
    embedder: Optional[object] = None,
) -> List[List[float]]:
    """
    将多条文本编码为向量

    Args:
        texts: 要编码的文本列表
        embedder: Embedding 客户端实例（必须提供）

    Returns:
        向量列表的列表
    """
    if not texts:
        return []
    if embedder is None:
        raise RuntimeError("Embedder 未提供")
    return await embedder.encode(texts)


# 便捷函数：从 registry 获取 embedder 后调用
async def encode_text_with_registry(
    text: str,
    registry,
) -> List[float]:
    """从 registry 获取 embedder 后编码文本"""
    if registry.embedder is None:
        raise RuntimeError("Embedder 未初始化")
    return await encode_text(text, registry.embedder)


async def encode_texts_with_registry(
    texts: List[str],
    registry,
) -> List[List[float]]:
    """从 registry 获取 embedder 后编码文本列表"""
    if registry.embedder is None:
        raise RuntimeError("Embedder 未初始化")
    return await encode_texts(texts, registry.embedder)