"""
记忆提取函数
"""
import re
import json
import asyncio
import aiohttp
from typing import Dict, Any

from ..service_registry import get_service_registry

VALID_MEMORY_TYPES = ['preference', 'fact', 'episodic', 'semantic', 'procedural', 'general']


async def extract_memories(conversation: str) -> Dict[str, Any]:
    """
    使用 LLM 从对话中提取结构化记忆
    返回: {"memories": [{"content": "...", "importance": 0.9, "memory_type": "preference", "tags": ["食物"]}]}
    """
    registry = get_service_registry()
    llm_cfg = registry.config.llm

    if not llm_cfg.api_key or not llm_cfg.model or not llm_cfg.base_url:
        print("LLM 配置不完整，无法提取记忆")
        return {"memories": []}

    base_url = llm_cfg.base_url.rstrip('/')

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

对话内容：
{conversation}

请返回 JSON：
{{"memories": [
  {{"content": "主人喜欢吃辣的食物", "importance": 0.9, "memory_type": "preference", "tags": ["食物", "口味"]}},
  {{"content": "主人今天去了健身房锻炼", "importance": 0.6, "memory_type": "episodic", "tags": ["健身", "运动"]}},
  {{"content": "主人的生日是5月20日", "importance": 0.95, "memory_type": "fact", "tags": ["生日", "个人信息"]}}
]}}
"""

    timeouts = [60, 120]

    for attempt, timeout_seconds in enumerate(timeouts, 1):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {llm_cfg.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": llm_cfg.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3
                }
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        response_text = result['choices'][0]['message']['content'].strip()

                        try:
                            parsed = json.loads(response_text)
                        except:
                            json_match = re.search(r'\{[\s\S]*"memories"[\s\S]*\}', response_text)
                            if json_match:
                                parsed = json.loads(json_match.group())
                            else:
                                continue

                        memories = parsed.get('memories', [])
                        valid_memories = []

                        for mem in memories:
                            if isinstance(mem, dict) and mem.get('content'):
                                content = str(mem['content']).strip()
                                try:
                                    importance = float(mem.get('importance', 0.5))
                                except:
                                    importance = 0.5
                                importance = max(0.1, min(1.0, importance))

                                memory_type = mem.get('memory_type', 'general')
                                if memory_type not in VALID_MEMORY_TYPES:
                                    memory_type = 'general'

                                tags = mem.get('tags', [])
                                if not isinstance(tags, list):
                                    tags = []

                                if len(content) >= 5:
                                    valid_memories.append({
                                        "content": content,
                                        "importance": importance,
                                        "memory_type": memory_type,
                                        "tags": tags
                                    })

                        return {"memories": valid_memories}

        except asyncio.TimeoutError:
            print(f"LLM 调用超时 (尝试 {attempt}/{len(timeouts)})")
            continue
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            continue

    return {"memories": []}