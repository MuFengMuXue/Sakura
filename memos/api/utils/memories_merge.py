"""
记忆合并函数
"""
# api/utils/merge_memory.py
import asyncio
import aiohttp
from datetime import datetime
from fastapi import Depends
from ..dependencies import get_registry
from ..service_registry import ServiceRegistry
from .text_encoding import encode_text
from .bm25_index import update_bm25_index


async def merge_memories(keeper_id: str, content_a: str, content_b: str,registry: ServiceRegistry = Depends(get_registry)) -> bool:
    """
    使用 LLM 智能合并两条相似记忆
    """
    llm_cfg = registry.config.llm.config

    if not llm_cfg.api_key or not llm_cfg.model or not llm_cfg.base_url:
        print("LLM 未配置，无法合并记忆")
        return False

    base_url = llm_cfg.base_url.rstrip('/')

    prompt = f"""合并以下两条相似的记忆，保留所有有价值的信息，去除重复内容：

已有记忆：{content_a}
新增信息：{content_b}

合并后的记忆（保留所有细节，用分号分隔要点）："""

    max_retries = 3
    timeouts = [60, 90, 120]

    for attempt in range(max_retries):
        try:
            timeout_seconds = timeouts[attempt]
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {llm_cfg.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": llm_cfg.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.2
                }
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        merged_content = result['choices'][0]['message']['content'].strip()

                        new_vector = await encode_text(merged_content)

                        full_mem = await registry.qdrant.get_memory(keeper_id)
                        if not full_mem:
                            print(f"找不到记忆 {keeper_id}，合并失败")
                            return False

                        keeper_payload = full_mem.get('payload', {})
                        keeper_payload['content'] = merged_content
                        keeper_payload['updated_at'] = datetime.now().isoformat()
                        keeper_payload['merge_count'] = keeper_payload.get('merge_count', 0) + 1

                        await registry.qdrant.update_memory(keeper_id, keeper_payload, new_vector)
                        update_bm25_index(keeper_id, merged_content)

                        print(f"LLM合并成功 (第 {keeper_payload['merge_count']} 次): {merged_content[:50]}...")
                        return True
                    else:
                        error_text = await resp.text()
                        print(f"LLM API 返回错误 {resp.status}: {error_text[:200]}")
                        return False

        except asyncio.TimeoutError:
            print(f"LLM 合并超时 (第 {attempt + 1}/{max_retries} 次, {timeouts[attempt]}秒)")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
        except Exception as e:
            print(f"LLM 合并异常: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(3)

    print(f"LLM 合并失败（已重试 {max_retries} 次），两条记忆均保留")
    return False