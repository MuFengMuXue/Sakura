"""
云端 Reranker 客户端（支持 Cohere / SiliconFlow 等兼容 API）
"""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CloudRerankerClient:
    """
    云端 Rerank API 客户端（OpenAI 兼容格式，通常类似 Cohere 的 /rerank 接口）

    使用示例：
        client = CloudRerankerClient(
            api_key="your-api-key",
            base_url="https://api.siliconflow.cn/v1",
            model="BAAI/bge-reranker-v2-m3",
        )
        results = await client.rerank(
            query="什么是机器学习？",
            documents=["文本1", "文本2", ...],
            top_n=3
        )
        # results: [{"index": 1, "relevance_score": 0.95}, ...]
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 30,
        max_retries: int = 2,
        rerank_path: str = "/rerank",  # 可自定义 API 路径
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.rerank_path = rerank_path

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = False,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        url = f"{self.base_url}{self.rerank_path}"
        logger.info(f"Rerank 请求 URL: {url}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        if return_documents:
            payload["return_documents"] = True

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
                            # 解析响应：通常返回 {"results": [...]}
                            results = data.get("results", [])
                            if not results and "data" in data:  # 某些 API 可能用 data 字段
                                results = data["data"]
                            # 确保按分数降序排列（API 通常已排序，但以防万一）
                            results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
                            return results
                        else:
                            error_text = await resp.text()
                            logger.warning(
                                f"Rerank API 错误 {resp.status}: {error_text} "
                                f"(尝试 {attempt+1}/{self.max_retries})"
                            )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Rerank API 超时（尝试 {attempt+1}/{self.max_retries}）"
                    )
                except Exception as e:
                    logger.warning(
                        f"Rerank API 异常: {e} （尝试 {attempt+1}/{self.max_retries}）"
                    )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)  # 退避延迟

            raise RuntimeError("Rerank API 请求失败，已重试")

    async def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[str]:
        """
        快捷方法：仅返回重排序后的文档列表（不含分数）
        """
        results = await self.rerank(query, documents, top_n, return_documents=True)
        # 如果 API 返回了文档原文，直接提取
        if results and "document" in results[0]:
            return [item["document"]["text"] for item in results]
        # 否则根据 index 从原始文档中提取
        indices = [item["index"] for item in results]
        return [documents[i] for i in indices]