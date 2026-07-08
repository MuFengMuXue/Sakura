"""
日期时间工具函数
提供 ISO 时间解析、安全类型转换、时间衰减/频率加分等
"""
from datetime import datetime
from typing import Any, Optional
import math


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    """
    容错解析 ISO 时间字符串或 datetime 对象。
    
    Args:
        value: 可能是字符串（ISO格式）、datetime 对象或 None
    
    Returns:
        解析后的 datetime（naive，即无时区信息），若解析失败返回 None
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        try:
            # 尝试解析，支持 'Z' 结尾
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全地将任意值转换为 float，转换失败返回默认值。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    安全地将任意值转换为 int，转换失败返回默认值。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def recency_boost_for(payload: dict, weight: float) -> float:
    """
    按最近访问/创建时间计算时间衰减加分。
    
    Args:
        payload: 记忆的 payload 字典
        weight: 基础权重（最大加分值）
    
    Returns:
        加分值（0 到 weight 之间）
    """
    if weight <= 0:
        return 0.0
    
    # 优先使用 last_accessed_at，否则使用 created_at 或 timestamp
    ts = parse_iso_datetime(
        payload.get('last_accessed_at') 
        or payload.get('created_at') 
        or payload.get('timestamp')
    )
    if not ts:
        return 0.0
    
    age_days = max((datetime.now() - ts).total_seconds() / 86400, 0)
    return weight * math.exp(-age_days / 30.0)


def frequency_boost_for(payload: dict, weight: float) -> float:
    """
    按访问次数计算饱和加分（采用对数饱和）。
    
    Args:
        payload: 记忆的 payload 字典
        weight: 基础权重（最大加分值）
    
    Returns:
        加分值（0 到 weight 之间）
    """
    if weight <= 0:
        return 0.0
    
    access_count = max(safe_int(payload.get('access_count'), 0), 0)
    # 饱和函数：当访问次数达到10次时，接近 weight
    return min(weight, weight * math.log1p(access_count) / math.log(11))