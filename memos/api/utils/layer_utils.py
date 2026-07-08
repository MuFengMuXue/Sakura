"""
记忆生命周期层工具
定义层名称、权重、排序，并提供层规范化与推断函数
"""
from typing import Optional

# 层名称列表
MEMORY_LAYERS = ["WorkingMemory", "LongTermMemory", "UserMemory"]

# 层权重默认值（用于搜索时的加权）
DEFAULT_LAYER_WEIGHTS = {
    "WorkingMemory": 0.05,
    "LongTermMemory": 0.15,
    "UserMemory": 0.25,
}

# 层排序（用于去重时选择保留方，值越大优先级越高）
LAYER_RANK = {"WorkingMemory": 0, "LongTermMemory": 1, "UserMemory": 2}


def normalize_layer(layer: Optional[str], default: str = "WorkingMemory") -> str:
    """
    规范化生命周期层名称。
    
    如果传入的 layer 在 MEMORY_LAYERS 中，则原样返回；否则返回 default。
    """
    if layer in MEMORY_LAYERS:
        return layer
    return default


def infer_memory_layer(
    memory_type: Optional[str] = None,
    source: Optional[str] = None,
    scope: Optional[str] = None,
    explicit_layer: Optional[str] = None,
) -> str:
    """
    根据写入来源推断新记忆应归属的生命周期层。
    
    优先级：
    1. explicit_layer（显式指定）
    2. 如果 memory_type == 'preference' 或 source == 'user_profile' 或 scope == 'user_profile' -> UserMemory
    3. 否则 -> WorkingMemory
    
    Args:
        memory_type: 记忆类型（如 'preference'）
        source: 来源（如 'user_profile'）
        scope: 作用域（如 'user_profile'）
        explicit_layer: 显式指定的层
    
    Returns:
        层名称（一定是 MEMORY_LAYERS 中的一项）
    """
    if explicit_layer in MEMORY_LAYERS:
        return explicit_layer
    
    if memory_type == "preference" or source == "user_profile" or scope == "user_profile":
        return "UserMemory"
    
    return "WorkingMemory"