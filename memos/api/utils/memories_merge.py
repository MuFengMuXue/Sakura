"""
记忆合并函数
"""
import asyncio
import aiohttp
from datetime import datetime
import logging

from .encoding_utils import encode_text
from .bm25_utils import update_bm25_index
from .json_utils import build_chat_completion_payload

logger = logging.getLogger(__name__)


async def merge_memories(
    keeper_id: str,
    content_a: str,
    content_b: str,
    registry,  # 由调用方传入，不使用 Depends
) -> bool:
    """
    使用 LLM 智能合并两条相似记忆。

    Args:
        keeper_id: 要保留的记忆 ID（合并后的内容写入这条）
        content_a: 第一条记忆的内容
        content_b: 第二条记忆的内容
        registry: 服务注册表（由调用方传入）

    Returns:
        True 表示合并成功，False 表示失败（两条记忆均保留）
    """
    llm_cfg = registry.config.llm

    if not llm_cfg.api_key or not llm_cfg.model or not llm_cfg.base_url:
        logger.warning("LLM 未配置，无法合并记忆")
        return False

    base_url = llm_cfg.base_url.rstrip('/')

    prompt = f"""合并以下两条相似的记忆，保留所有有价值的信息，去除重复内容：

已有记忆：{content_a}
新增信息：{content_b}

合并后的记忆（保留所有细节，用分号分隔要点）："""

    # 读取合并专用的 max_tokens（默认 2000）
    max_tokens = getattr(llm_cfg, 'max_tokens', 2000)
    if not isinstance(max_tokens, int):
        max_tokens = 2000

    # 构建模型配置（支持思考模式）
    model_config = {
        'thinking_mode': getattr(llm_cfg, 'thinking_mode', 'disabled'),
        'reasoning_effort': getattr(llm_cfg, 'reasoning_effort', None),
    }

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

                # 使用工具函数构建 payload（支持思考模式）
                payload = build_chat_completion_payload(
                    model=llm_cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.2,
                    model_config=model_config,
                )

                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        merged_content = result['choices'][0]['message']['content'].strip()

                        # 生成新向量（传入 embedder）
                        new_vector = await encode_text(merged_content, registry.embedder)

                        # 获取保留方记忆
                        full_mem = await registry.qdrant.get_memory(keeper_id)
                        if not full_mem:
                            logger.error(f"找不到记忆 {keeper_id}，合并失败")
                            return False

                        # 更新 payload
                        keeper_payload = full_mem.get('payload', {})
                        keeper_payload['content'] = merged_content
                        keeper_payload['updated_at'] = datetime.now().isoformat()
                        keeper_payload['merge_count'] = keeper_payload.get('merge_count', 0) + 1

                        # 更新存储
                        await registry.qdrant.update_memory(keeper_id, keeper_payload, new_vector)
                        await update_bm25_index(keeper_id, merged_content, registry)

                        logger.info(
                            f"LLM合并成功 (第 {keeper_payload['merge_count']} 次): {merged_content[:50]}..."
                        )
                        return True
                    else:
                        error_text = await resp.text()
                        logger.warning(f"LLM API 返回错误 {resp.status}: {error_text[:200]}")
                        return False

        except asyncio.TimeoutError:
            logger.warning(f"LLM 合并超时 (第 {attempt + 1}/{max_retries} 次, {timeouts[attempt]}秒)")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"LLM 合并异常: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(3)

    logger.error(f"LLM 合并失败（已重试 {max_retries} 次），两条记忆均保留")
    return False