"""
JSON 和 LLM 响应解析工具
提供 JSON 截断修复、memories 解析、Chat Completions 请求体构建等
"""
import json
import re
from typing import Dict, Any, List, Optional


def balance_truncated_json(text: str) -> Optional[str]:
    """
    尽力补全被截断的 JSON：定位最后一个完整的对象/数组位置后配平括号。

    Args:
        text: 可能被截断的 JSON 字符串

    Returns:
        补全后的 JSON 字符串，若无法补全返回 None
    """
    text = (text or '').strip()
    if not text:
        return None

    in_string = False
    escape = False
    brace = 0
    bracket = 0
    last_complete = -1
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            brace += 1
        elif ch == '}':
            brace -= 1
            if brace >= 0:
                last_complete = i
        elif ch == '[':
            bracket += 1
        elif ch == ']':
            bracket -= 1
            if bracket >= 0:
                last_complete = i

    if brace == 0 and bracket == 0 and not in_string:
        return text
    if last_complete <= 0:
        return None

    truncated = text[:last_complete + 1]
    open_brace = 0
    open_bracket = 0
    in_str2 = False
    esc2 = False
    for ch in truncated:
        if esc2:
            esc2 = False
            continue
        if ch == '\\':
            esc2 = True
            continue
        if ch == '"':
            in_str2 = not in_str2
            continue
        if in_str2:
            continue
        if ch == '{':
            open_brace += 1
        elif ch == '}':
            open_brace -= 1
        elif ch == '[':
            open_bracket += 1
        elif ch == ']':
            open_bracket -= 1
    if open_brace < 0 or open_bracket < 0:
        return None
    return truncated + (']' * open_bracket) + ('}' * open_brace)


def parse_memories_json(response_text: str) -> Optional[List[Dict[str, Any]]]:
    """
    从 LLM 原始响应中尽力解析出 memories 列表。

    返回 list 表示解析成功（空列表代表模型确实没提取到记忆）；
    返回 None 表示无法解析（空响应/非 JSON/修复失败），调用方应重试或换模型。
    """
    if not response_text:
        return None

    text = response_text.strip()

    # 去除 markdown 代码块围栏 ```json ... ```
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z0-9]*\s*', '', text)
        text = re.sub(r'\s*```$', '', text).strip()

    def _to_memories(obj):
        if isinstance(obj, dict):
            mem = obj.get('memories', [])
            return mem if isinstance(mem, list) else None
        if isinstance(obj, list):
            return obj
        return None

    # 1) 直接解析整段
    try:
        parsed = json.loads(text)
        result = _to_memories(parsed)
        if result is not None:
            return result
    except Exception:
        pass

    # 2) 正则截取包含 "memories" 字段的 JSON 对象（含截断修复）
    match = re.search(r'\{[\s\S]*"memories"[\s\S]*\}', text)
    if match:
        snippet = match.group()
        try:
            parsed = json.loads(snippet)
            result = _to_memories(parsed)
            if result is not None:
                return result
        except Exception:
            fixed = balance_truncated_json(snippet)
            if fixed:
                try:
                    parsed = json.loads(fixed)
                    result = _to_memories(parsed)
                    if result is not None:
                        return result
                except Exception:
                    pass

    # 3) 对整段做截断修复后再解析
    fixed_all = balance_truncated_json(text)
    if fixed_all:
        try:
            parsed = json.loads(fixed_all)
            result = _to_memories(parsed)
            if result is not None:
                return result
        except Exception:
            pass

    return None


def build_chat_completion_payload(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    model_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    构建兼容 DeepSeek 思考模式的 Chat Completions 请求体

    Args:
        model: 模型名称
        messages: 消息列表
        max_tokens: 最大输出 token 数
        temperature: 温度参数
        model_config: 可包含 thinking_mode, reasoning_effort 等

    Returns:
        请求 payload 字典
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }

    model_config = model_config or {}
    # DeepSeek 默认关闭思考：未显式配置（或为空）时按 disabled 处理
    thinking_mode = str(model_config.get("thinking_mode") or "disabled").lower()
    if thinking_mode in ("enabled", "disabled"):
        payload["thinking"] = {"type": thinking_mode}

    reasoning_effort = model_config.get("reasoning_effort")
    # reasoning_effort 仅在思考开启时有意义；关思考时发送它会触发超时
    if reasoning_effort and thinking_mode == "enabled":
        payload["reasoning_effort"] = str(reasoning_effort)

    # 只有思考关闭时才设置 temperature
    if thinking_mode != "enabled":
        payload["temperature"] = temperature

    return payload