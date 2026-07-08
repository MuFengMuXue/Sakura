"""
Payload 操作工具函数
提供记忆 payload 的提取、列表处理、去重评分、操作规范化等
"""
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .time_utils import parse_iso_datetime, safe_float, safe_int
from .layer_utils import LAYER_RANK, normalize_layer


# 去重允许的操作类型
DEDUPLICATE_ACTIONS = {"archive", "soft_delete", "delete"}


def payload_of(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    安全获取记忆的 payload 字典。
    
    Args:
        memory: 记忆字典（可能包含 'payload' 字段，也可能没有）
    
    Returns:
        payload 字典，如果不存在则返回空字典
    """
    payload = memory.get('payload')
    return payload if isinstance(payload, dict) else {}


def ensure_list(value: Any) -> List[Any]:
    """
    确保值为列表类型。
    
    如果 value 为 None，返回空列表；
    如果已是列表，原样返回；
    否则包装为单元素列表。
    
    Args:
        value: 任意值
    
    Returns:
        列表
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def merge_unique_list(*values: Any) -> List[Any]:
    """
    合并多个值/列表，并去重（保持首次出现的顺序）。
    
    Args:
        *values: 任意数量的值或列表
    
    Returns:
        合并去重后的列表
    """
    merged = []
    for value in values:
        for item in ensure_list(value):
            if item is not None and item not in merged:
                merged.append(item)
    return merged


def deduplicate_keeper_score(memory: Dict[str, Any]) -> Tuple[float, int, int, int, int]:
    """
    计算用于去重时选择保留方的评分（元组，值越高越应保留）。
    
    评分维度（按优先级排序）：
    1. importance（重要性）
    2. layer 层级（UserMemory > LongTermMemory > WorkingMemory）
    3. access_count（访问次数）
    4. merge_count（合并次数）
    5. created_at（创建时间，越新越好）
    
    Args:
        memory: 记忆字典（包含 id, content, payload 等）
    
    Returns:
        用于比较的评分元组，按优先级从高到低排列
    """
    payload = payload_of(memory)
    
    # 获取创建时间（越新越好，使用负数排序）
    created_at = parse_iso_datetime(
        payload.get('created_at') or memory.get('created_at')
    ) or datetime.max
    created_key = -(
        created_at.toordinal() * 86400
        + created_at.hour * 3600
        + created_at.minute * 60
        + created_at.second
    )
    
    # 获取 layer 层级（值越大越优先）
    layer = normalize_layer(
        payload.get('layer') or memory.get('layer'),
        default='LongTermMemory'
    )
    
    return (
        safe_float(payload.get('importance', memory.get('importance', 0.5)), 0.5),
        LAYER_RANK.get(layer, 1),
        safe_int(payload.get('access_count', memory.get('access_count', 0)), 0),
        safe_int(payload.get('merge_count', memory.get('merge_count', 0)), 0),
        created_key,
    )


def choose_deduplicate_keeper(
    left: Dict[str, Any],
    right: Dict[str, Any]
) -> Dict[str, Any]:
    """
    根据去重评分选择保留方（评分较高的记忆将被保留）。
    
    Args:
        left: 第一条记忆字典
        right: 第二条记忆字典
    
    Returns:
        评分较高的记忆字典（保留方）
    """
    return left if deduplicate_keeper_score(left) >= deduplicate_keeper_score(right) else right


def normalize_duplicate_action(action: Optional[str]) -> str:
    """
    规范化去重处理动作。
    
    支持的 action：
    - "archive": 归档（可从归档列表恢复）
    - "soft_delete": 软删除（默认，标记删除但可恢复）
    - "delete": 物理删除（不可恢复）
    
    Args:
        action: 原始动作字符串（不区分大小写，支持 'hard_delete' 自动映射到 'delete'）
    
    Returns:
        规范化后的动作
    
    Raises:
        HTTPException: 当 action 不在允许范围内时抛出 400 错误
    """
    from fastapi import HTTPException
    
    if action is None:
        return "soft_delete"
    
    normalized = action.strip().lower().replace("-", "_")
    if normalized == "hard_delete":
        normalized = "delete"
    
    if normalized not in DEDUPLICATE_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail="duplicate_action 必须是 archive、soft_delete 或 delete"
        )
    
    return normalized