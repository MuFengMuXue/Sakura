"""
云端embedding模型
"""

import aiohttp
import asyncio
from typing import List
import logging

logger = logging.getLogger(__name__)

class CloudEmbeddingClient:
    """云端 Embedding API 客户端（支持 OpenAI 兼容格式）"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 30,
        max_retries: int = 2
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """异步批量编码文本，返回向量列表"""
        if not texts:
            return []
        return await self._encode_openai(texts)
    
    async def _encode_openai(self, texts: List[str]) -> List[List[float]]:
        """OpenAI 兼容 API（SiliconFlow / DeepSeek / 智谱 等）"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # 按 index 排序，确保顺序正确
                            items = sorted(data["data"], key=lambda x: x.get("index", 0))
                            embeddings = [item["embedding"] for item in items]
                            return embeddings
                        else:
                            error_text = await resp.text()
                            logger.warning(f"Embedding API 错误 {resp.status}: {error_text}")
                except asyncio.TimeoutError:
                    logger.warning(f"Embedding API 超时（尝试 {attempt+1}/{self.max_retries}）")
                except Exception as e:
                    logger.warning(f"Embedding API 异常: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
            raise RuntimeError("Embedding API 请求失败，已重试")
