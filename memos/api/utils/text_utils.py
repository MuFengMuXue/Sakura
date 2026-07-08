"""
文本处理工具函数
提供文本规范化、截断、摘要处理等纯文本操作
"""
from typing import Any, Optional

MAX_CONTEXT_SUMMARY_CHARS = 6000


def normalize_context_summary(*values: Any) -> Optional[str]:
    """
    从多个来源中选取第一个非空摘要，并截断至安全长度。
    
    原代码中用于处理 LLM 记忆提取时的历史摘要，避免 prompt 过长。
    
    Args:
        *values: 多个可能包含摘要的值（字符串或 None）
    
    Returns:
        处理后的摘要字符串，若所有值均为空则返回 None
    """
    # 选取第一个非空字符串
    for value in values:
        if not isinstance(value, str):
            continue
        summary = value.strip()
        if not summary:
            continue
        
        # 如果长度在安全范围内，直接返回
        if len(summary) <= MAX_CONTEXT_SUMMARY_CHARS:
            return summary
        
        # 过长则保留开头和结尾各一半，中间加入提示
        half = MAX_CONTEXT_SUMMARY_CHARS // 2
        return (
            summary[:half].rstrip()
            + "\n...[历史摘要过长，已保留开头和结尾]...\n"
            + summary[-half:].lstrip()
        )
    
    return None


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    截断文本到指定长度，并添加后缀。
    
    Args:
        text: 原始文本
        max_length: 最大长度（包含后缀）
        suffix: 截断时添加的后缀
    
    Returns:
        截断后的文本
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_whitespace(text: str) -> str:
    """
    清理文本中的多余空白字符。
    
    将多个连续空格/换行合并为单个空格，并去除首尾空白。
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    return " ".join(text.split())


def extract_role_label(role: str) -> str:
    """
    将角色名转换为对话标签。
    
    原代码中用于将 'user' 转换为 '主人'，'assistant' 转换为 '肥牛'。
    
    Args:
        role: 角色名（如 'user', 'assistant', 'system'）
    
    Returns:
        对应的中文标签
    """
    role_map = {
        "user": "主人",
        "assistant": "肥牛",
        "system": "系统",
        "function": "工具",
        "tool": "工具",
    }
    return role_map.get(role.lower(), role)