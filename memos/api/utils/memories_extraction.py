"""
记忆提取函数
"""
import asyncio
import aiohttp
from typing import Dict, Any, Optional
import logging

from .json_utils import build_chat_completion_payload, parse_memories_json
from .text_utils import normalize_context_summary
from .time_utils import safe_float

logger = logging.getLogger(__name__)

VALID_MEMORY_TYPES = ['preference', 'fact', 'episodic', 'semantic', 'procedural', 'general']


async def extract_memories(
    conversation: str,
    registry,  # 由调用方传入，不使用 Depends
    context_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用 LLM 从对话中提取结构化记忆。

    Args:
        conversation: 当前对话文本（多轮，含角色标签）
        registry: 服务注册表（由调用方传入）
        context_summary: 可选的历史压缩摘要，用于提供背景参考

    Returns:
        {"memories": [{"content": "...", "importance": 0.9, "memory_type": "preference", "tags": ["食物"]}]}
    """
    llm_cfg = registry.config.llm

    if not llm_cfg.api_key or not llm_cfg.model or not llm_cfg.base_url:
        logger.warning("LLM 配置不完整，无法提取记忆")
        return {"memories": []}

    # 处理摘要
    context_summary = normalize_context_summary(context_summary)
    context_summary_section = ""
    if context_summary:
        context_summary_section = f"""
历史压缩摘要（仅供理解当前对话背景）：
{context_summary}

注意：
- 历史摘要只用于理解代词、简称、延续话题和人物关系。
- 不要仅凭历史摘要生成新记忆，也不要把历史摘要整段改写成记忆。
- 只有当前待总结对话明确提到、确认、更新或修正的信息，才可以提取为记忆。
- 如果当前对话与历史摘要冲突，以当前对话为准。
"""

    prompt = f"""你是记忆提取专家。从以下多轮对话中提取关键事实，并按类型严格分类。

身份说明：
- "主人"是使用AI的真人用户
- "沐樱"是AI助手

提取规则：
1. 用自然的中文描述要点，每条记忆15-80字
2. 忽略无意义的闲聊
3. 判断重要度（0.1-1.0）
4. **严格按以下类型分类记忆（memory_type）**：
   - preference: 用户表达的喜好/偏好/厌恶
   - fact: 用户的客观个人信息
   - episodic: 具体的事件或经历
   - semantic: 用户了解/学习的知识概念
   - procedural: 用户的技能/习惯/日常规律
   - general: 无法归入以上任何类别的记忆
5. 提取相关标签（tags），1-3个关键词

分类优先级：preference > fact > episodic > procedural > semantic > general

{context_summary_section}

当前待总结对话：
{conversation}

请返回 JSON：
{{"memories": [
  {{"content": "主人喜欢吃辣的食物", "importance": 0.9, "memory_type": "preference", "tags": ["食物", "口味"]}},
  {{"content": "主人今天去了健身房锻炼", "importance": 0.6, "memory_type": "episodic", "tags": ["健身", "运动"]}},
  {{"content": "主人的生日是5月20日", "importance": 0.95, "memory_type": "fact", "tags": ["生日", "个人信息"]}}
]}}
"""

    # 读取提取专用的 max_tokens（默认 8000）
    extract_max_tokens = 8000
    if hasattr(llm_cfg, 'max_tokens') and llm_cfg.max_tokens:
        extract_max_tokens = int(llm_cfg.max_tokens)

    base_url = llm_cfg.base_url.rstrip('/')
    
    # 超时时间列表（逐步增加）
    timeouts = [90, 180]

    for attempt, timeout_seconds in enumerate(timeouts, 1):
        try:
            logger.info(f"调用 LLM 提取记忆 (第{attempt}次, 超时{timeout_seconds}s)")
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {llm_cfg.api_key}",
                    "Content-Type": "application/json"
                }

                # 构建 payload（支持思考模式）
                model_config = {
                    'name': '主模型',
                    'model': llm_cfg.model,
                    'thinking_mode': getattr(llm_cfg, 'thinking_mode', 'disabled'),
                    'reasoning_effort': getattr(llm_cfg, 'reasoning_effort', None),
                }
                
                payload = build_chat_completion_payload(
                    model=llm_cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=extract_max_tokens,
                    temperature=0.3,
                    model_config=model_config
                )

                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(f"LLM 返回 HTTP {resp.status}: {error_text[:200]}")
                        continue

                    result = await resp.json()
                    message = result.get('choices', [{}])[0].get('message', {}) or {}
                    content_text = (message.get('content') or '').strip()
                    reasoning_text = (message.get('reasoning_content') or '').strip()
                    usage = result.get('usage', {}) or {}
                    
                    logger.info(
                        f"LLM 调用成功 "
                        f"(content {len(content_text)}字, reasoning {len(reasoning_text)}字, "
                        f"completion_tokens={usage.get('completion_tokens', '?')})"
                    )

                    # 解析响应（优先 content，若为空且 reasoning 有内容则用 reasoning）
                    memories = parse_memories_json(content_text)
                    if memories is None and reasoning_text:
                        logger.info("content 解析失败，改用 reasoning_content")
                        memories = parse_memories_json(reasoning_text)

                    if memories is None:
                        preview = (content_text or reasoning_text)[:500].replace('\n', ' | ')
                        logger.warning(f"无法解析出 memories，原始响应(前500字): {preview}")
                        continue

                    if not memories:
                        logger.info("模型返回空 memories（判定为无可记内容）")

                    valid_memories = []
                    for mem in memories:
                        if isinstance(mem, dict) and mem.get('content'):
                            content = str(mem['content']).strip()
                            if len(content) < 5:
                                continue
                            importance = safe_float(mem.get('importance'), 0.5)
                            importance = max(0.1, min(1.0, importance))

                            memory_type = mem.get('memory_type', 'general')
                            if memory_type not in VALID_MEMORY_TYPES:
                                memory_type = 'general'

                            tags = mem.get('tags', [])
                            if not isinstance(tags, list):
                                tags = []

                            valid_memories.append({
                                "content": content,
                                "importance": importance,
                                "memory_type": memory_type,
                                "tags": tags,
                            })

                    return {"memories": valid_memories}

        except asyncio.TimeoutError:
            logger.warning(f"LLM 调用超时 (第{attempt}次, {timeout_seconds}s)")
            continue
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            continue

    logger.error("LLM 调用全部失败，返回空")
    return {"memories": []}