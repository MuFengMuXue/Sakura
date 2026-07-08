# qdrant_client.py - MemOS Qdrant 向量数据库客户端（异步版本）
"""
Qdrant 向量数据库客户端封装
提供记忆的向量存储、检索、更新、删除等功能
所有 I/O 操作均为异步，使用 asyncio.to_thread 包装同步 Qdrant 调用
"""

import os
import asyncio
import uuid
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, PointStruct,
        Filter, FieldCondition, MatchValue, Range,
        UpdateStatus, PayloadSchemaType
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("qdrant-client 未安装，将使用内存存储模式")

logger = logging.getLogger(__name__)


class MemosQdrantClient:
    """MemOS 的 Qdrant 向量数据库客户端（异步版本）"""

    def __init__(
        self,
        path: str = "./memos_data/qdrant",
        collection_name: str = "memories",
        vector_size: int = 768,
        use_memory: bool = False
    ):
        """
        初始化 Qdrant 客户端

        Args:
            path: Qdrant 本地存储路径
            collection_name: 集合名称
            vector_size: 向量维度
            use_memory: 是否使用内存模式（不持久化）
        """
        self.path = path
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.use_memory = use_memory
        self.client = None
        self._initialized = False

        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant 不可用，请安装 qdrant-client")
            return

        self._init_client()

    def _init_client(self):
        """同步初始化 Qdrant 客户端（仅创建实例，不执行 I/O）"""
        try:
            if self.use_memory:
                self.client = QdrantClient(":memory:")
                logger.info("Qdrant 内存模式已启动")
            else:
                os.makedirs(self.path, exist_ok=True)
                self.client = QdrantClient(path=self.path)
                logger.info(f"Qdrant 本地模式已启动: {self.path}")

            self._initialized = True

        except Exception as e:
            logger.error(f"Qdrant 初始化失败: {e}")
            raise

    async def initialize(self):
        """异步初始化（创建集合和索引）"""
        if not self.is_available():
            return False
        try:
            await self._ensure_collection()
            return True
        except Exception as e:
            logger.error(f"异步初始化失败: {e}")
            return False

    async def _ensure_collection(self):
        """确保集合存在，不存在则创建"""
        try:
            collections = await asyncio.to_thread(
                self.client.get_collections
            )
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建集合: {self.collection_name}")
                await self._create_payload_indexes()
            else:
                logger.info(f"集合已存在: {self.collection_name}")
                await self._create_payload_indexes()

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise

    async def _create_payload_indexes(self):
        """创建 Payload 索引以加速过滤查询"""
        index_fields = [
            ("user_id", PayloadSchemaType.KEYWORD),
            ("memory_type", PayloadSchemaType.KEYWORD),
            ("tags", PayloadSchemaType.KEYWORD),
            ("layer", PayloadSchemaType.KEYWORD),
            ("status", PayloadSchemaType.KEYWORD),
        ]

        created = 0
        for field_name, schema in index_fields:
            try:
                await asyncio.to_thread(
                    self.client.create_payload_index,
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema
                )
                created += 1
            except Exception as e:
                logger.debug(f"Payload 索引 {field_name} 未创建（可能已存在）: {e}")

        logger.info(f"Payload 索引检查完成，新增 {created} 个")

    def is_available(self) -> bool:
        """检查 Qdrant 是否可用"""
        return QDRANT_AVAILABLE and self._initialized and self.client is not None

    @staticmethod
    def _infer_default_layer(payload: Dict[str, Any]) -> str:
        """根据来源/类型推断新写入记忆的生命周期层。"""
        explicit_layer = payload.get('layer')
        if explicit_layer in {'WorkingMemory', 'LongTermMemory', 'UserMemory'}:
            return explicit_layer

        memory_type = payload.get('memory_type', 'general')
        source = payload.get('source')
        scope = payload.get('scope')
        if memory_type == 'preference' or source == 'user_profile' or scope == 'user_profile':
            return 'UserMemory'
        return 'WorkingMemory'

    def _prepare_payload_defaults(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """补齐生命周期元数据，保持旧调用方无需感知新字段。"""
        payload.setdefault('created_at', datetime.now().isoformat())
        payload.setdefault('importance', 0.5)
        payload.setdefault('status', 'active')
        payload.setdefault('layer', self._infer_default_layer(payload))
        payload.setdefault('access_count', 0)
        payload.setdefault('last_accessed_at', None)
        return payload

    # ==================== 记忆操作（异步） ====================

    async def add_memory(
        self,
        memory_id: str,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> bool:
        """
        添加单条记忆

        Args:
            memory_id: 记忆唯一 ID
            vector: 向量表示
            payload: 元数据（content, user_id, importance 等）

        Returns:
            是否成功
        """
        if not self.is_available():
            logger.error("Qdrant 不可用")
            return False

        try:
            payload = self._prepare_payload_defaults(payload)

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            logger.debug(f"添加记忆成功: {memory_id}")
            return True

        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
            return False

    async def add_memories_batch(
        self,
        memories: List[Dict[str, Any]]
    ) -> int:
        """
        批量添加记忆

        Args:
            memories: 记忆列表，每个元素包含 id, vector, payload

        Returns:
            成功添加的数量
        """
        if not self.is_available():
            logger.error("Qdrant 不可用")
            return 0

        try:
            points = []
            for mem in memories:
                payload = self._prepare_payload_defaults(mem.get('payload', {}))
                points.append(
                    PointStruct(
                        id=mem['id'],
                        vector=mem['vector'],
                        payload=payload
                    )
                )

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"批量添加 {len(points)} 条记忆成功")
            return len(points)

        except Exception as e:
            logger.error(f"批量添加记忆失败: {e}")
            return 0

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.5,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        importance_min: Optional[float] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
        status: Optional[str] = None,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        搜索相似记忆

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            score_threshold: 相似度阈值
            user_id: 用户 ID 过滤
            memory_type: 记忆类型过滤
            memory_types: 多记忆类型过滤
            tags: 标签过滤
            importance_min: 最低重要度
            layer: 单层过滤
            layers: 多层过滤
            status: 状态过滤
            include_archived: 是否包含归档

        Returns:
            记忆列表，包含 id, content, similarity, payload
        """
        if not self.is_available():
            logger.error("Qdrant 不可用")
            return []

        try:
            # 构建过滤条件
            filter_conditions = []
            must_not_conditions = []

            if status:
                filter_conditions.append(
                    FieldCondition(
                        key="status",
                        match=MatchValue(value=status)
                    )
                )
            else:
                must_not_conditions.append(
                    FieldCondition(
                        key="status",
                        match=MatchValue(value="deleted")
                    )
                )
                if not include_archived:
                    must_not_conditions.append(
                        FieldCondition(
                            key="status",
                            match=MatchValue(value="archived")
                        )
                    )

            if user_id:
                filter_conditions.append(
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                )

            if memory_type:
                filter_conditions.append(
                    FieldCondition(
                        key="memory_type",
                        match=MatchValue(value=memory_type)
                    )
                )

            if layer:
                filter_conditions.append(
                    FieldCondition(
                        key="layer",
                        match=MatchValue(value=layer)
                    )
                )

            if tags:
                for tag in tags:
                    filter_conditions.append(
                        FieldCondition(
                            key="tags",
                            match=MatchValue(value=tag)
                        )
                    )

            if importance_min is not None:
                filter_conditions.append(
                    FieldCondition(
                        key="importance",
                        range=Range(gte=importance_min)
                    )
                )

            query_filter = Filter(must=filter_conditions, must_not=must_not_conditions)

            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter
            )

            memories = []
            for hit in results.points:
                payload = hit.payload or {}
                payload_status = payload.get('status', 'active')
                if payload_status == 'deleted' or (payload_status == 'archived' and not include_archived and status != 'archived'):
                    continue
                if memory_types and payload.get('memory_type', 'general') not in memory_types:
                    continue
                payload_layer = payload.get('layer', 'LongTermMemory')
                if layers and payload_layer not in layers:
                    continue
                memories.append({
                    'id': hit.id,
                    'content': payload.get('content', ''),
                    'similarity': round(hit.score, 4),
                    'importance': payload.get('importance', 0.5),
                    'created_at': payload.get('created_at'),
                    'updated_at': payload.get('updated_at'),
                    'memory_type': payload.get('memory_type', 'general'),
                    'layer': payload_layer,
                    'status': payload_status,
                    'access_count': payload.get('access_count', 0),
                    'last_accessed_at': payload.get('last_accessed_at'),
                    'tags': payload.get('tags', []),
                    'entity_ids': payload.get('entity_ids', []),
                    'payload': payload
                })

            logger.debug(f"搜索返回 {len(memories)} 条结果")
            return memories

        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return []

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        获取单条记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            记忆数据，包含 id, content, payload
        """
        if not self.is_available():
            return None

        try:
            results = await asyncio.to_thread(
                self.client.retrieve,
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
                with_vectors=True
            )

            if results:
                point = results[0]
                payload = point.payload or {}
                if payload.get('status') == 'deleted':
                    return None
                return {
                    'id': point.id,
                    'content': payload.get('content', ''),
                    'vector': point.vector,
                    'payload': payload
                }
            return None

        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            return None

    async def get_all_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        batch_size: int = 1000,
        include_deleted: bool = False,
        include_archived: bool = False,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        获取记忆列表

        Args:
            user_id: 用户 ID 过滤
            limit: 返回数量，None 或 <=0 表示不限制
            batch_size: 分批 scroll 的批次大小

        Returns:
            记忆列表
        """
        if not self.is_available():
            return []

        try:
            filter_conditions = []
            must_not_conditions = []

            if user_id:
                filter_conditions.append(
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                )
            if memory_type:
                filter_conditions.append(
                    FieldCondition(
                        key="memory_type",
                        match=MatchValue(value=memory_type)
                    )
                )
            if tags:
                for tag in tags:
                    filter_conditions.append(
                        FieldCondition(
                            key="tags",
                            match=MatchValue(value=tag)
                        )
                    )
            if status:
                filter_conditions.append(
                    FieldCondition(
                        key="status",
                        match=MatchValue(value=status)
                    )
                )
            else:
                if not include_deleted:
                    must_not_conditions.append(
                        FieldCondition(
                            key="status",
                            match=MatchValue(value="deleted")
                        )
                    )
                if not include_archived:
                    must_not_conditions.append(
                        FieldCondition(
                            key="status",
                            match=MatchValue(value="archived")
                        )
                    )
            if layer:
                filter_conditions.append(
                    FieldCondition(
                        key="layer",
                        match=MatchValue(value=layer)
                    )
                )

            query_filter = Filter(must=filter_conditions, must_not=must_not_conditions) if filter_conditions or must_not_conditions else None

            unlimited = limit is None or limit <= 0
            memories = []
            next_offset = None

            while True:
                if unlimited:
                    page_limit = batch_size
                else:
                    remaining = limit - len(memories)
                    if remaining <= 0:
                        break
                    page_limit = min(batch_size, remaining)

                results, next_offset = await asyncio.to_thread(
                    self.client.scroll,
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=page_limit,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False
                )

                for point in results:
                    payload = point.payload or {}
                    payload_status = payload.get('status', 'active')
                    if status and payload_status != status:
                        continue
                    if not status:
                        if not include_deleted and payload_status == 'deleted':
                            continue
                        if not include_archived and payload_status == 'archived':
                            continue
                    if memory_types and payload.get('memory_type', 'general') not in memory_types:
                        continue
                    payload_layer = payload.get('layer', 'LongTermMemory')
                    if layers and payload_layer not in layers:
                        continue
                    memories.append({
                        'id': point.id,
                        'content': payload.get('content', ''),
                        'importance': payload.get('importance', 0.5),
                        'created_at': payload.get('created_at'),
                        'updated_at': payload.get('updated_at'),
                        'memory_type': payload.get('memory_type', 'general'),
                        'layer': payload_layer,
                        'status': payload_status,
                        'access_count': payload.get('access_count', 0),
                        'last_accessed_at': payload.get('last_accessed_at'),
                        'tags': payload.get('tags', []),
                        'merge_count': payload.get('merge_count', 0),
                        'payload': payload
                    })

                if not results or next_offset is None:
                    break

            return memories

        except Exception as e:
            logger.error(f"获取记忆列表失败: {e}")
            return []

    async def update_memory(
        self,
        memory_id: str,
        payload_updates: Dict[str, Any],
        new_vector: Optional[List[float]] = None
    ) -> bool:
        """
        更新记忆

        Args:
            memory_id: 记忆 ID
            payload_updates: 要更新的字段
            new_vector: 新的向量（可选）

        Returns:
            是否成功
        """
        if not self.is_available():
            return False

        try:
            payload_updates['updated_at'] = datetime.now().isoformat()

            await asyncio.to_thread(
                self.client.set_payload,
                collection_name=self.collection_name,
                points=[memory_id],
                payload=payload_updates
            )

            if new_vector:
                existing = await self.get_memory(memory_id)
                if existing:
                    merged_payload = {**existing['payload'], **payload_updates}
                    await asyncio.to_thread(
                        self.client.upsert,
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=memory_id,
                                vector=new_vector,
                                payload=merged_payload
                            )
                        ]
                    )

            logger.debug(f"更新记忆成功: {memory_id}")
            return True

        except Exception as e:
            logger.error(f"更新记忆失败: {e}")
            return False

    async def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            是否成功
        """
        if not self.is_available():
            return False

        try:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=[memory_id]
            )
            logger.debug(f"删除记忆成功: {memory_id}")
            return True

        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    async def soft_delete_memory(
        self,
        memory_id: str,
        delete_record_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> bool:
        """软删除记忆，保留 payload 以便恢复。"""
        return await self.update_memory(
            memory_id,
            {
                'status': 'deleted',
                'deleted_at': datetime.now().isoformat(),
                'delete_record_id': delete_record_id or str(uuid.uuid4()),
                'feedback_reason': reason
            }
        )

    async def archive_memory(
        self,
        memory_id: str,
        archive_record_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> bool:
        """归档记忆，保留 payload 且默认从搜索/BM25 中排除。"""
        return await self.update_memory(
            memory_id,
            {
                'status': 'archived',
                'archived_at': datetime.now().isoformat(),
                'archive_record_id': archive_record_id or str(uuid.uuid4()),
                'feedback_reason': reason
            }
        )

    async def recover_memory(self, memory_id: str) -> bool:
        """恢复软删除或归档的记忆。"""
        return await self.update_memory(
            memory_id,
            {
                'status': 'active',
                'deleted_at': None,
                'archived_at': None,
                'recovered_at': datetime.now().isoformat()
            }
        )

    async def update_usage(self, memory_id: str, increment: int = 1) -> bool:
        """递增访问计数并刷新最近访问时间。"""
        memory = await self.get_memory(memory_id)
        if not memory:
            return False
        payload = memory.get('payload', {}) or {}
        try:
            access_count = int(payload.get('access_count', 0) or 0) + increment
        except Exception:
            access_count = increment
        return await self.update_memory(
            memory_id,
            {
                'access_count': max(access_count, 0),
                'last_accessed_at': datetime.now().isoformat()
            }
        )

    async def delete_memories_batch(self, memory_ids: List[str]) -> int:
        """
        批量删除记忆

        Args:
            memory_ids: 记忆 ID 列表

        Returns:
            成功删除的数量
        """
        if not self.is_available():
            return 0

        try:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=memory_ids
            )
            logger.info(f"批量删除 {len(memory_ids)} 条记忆成功")
            return len(memory_ids)

        except Exception as e:
            logger.error(f"批量删除记忆失败: {e}")
            return 0

    # ==================== 统计和维护（异步） ====================

    async def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        if not self.is_available():
            return {}

        try:
            info = await asyncio.to_thread(
                self.client.get_collection,
                self.collection_name
            )
            points_count = getattr(info, 'points_count', 0) or 0
            vectors_count = getattr(info, 'vectors_count', points_count) or points_count
            status = info.status.value if hasattr(info.status, 'value') else str(info.status)
            return {
                'name': self.collection_name,
                'vectors_count': vectors_count,
                'points_count': points_count,
                'status': status,
                'vector_size': self.vector_size
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}

    async def count_memories(
        self,
        user_id: Optional[str] = None,
        include_archived: bool = False,
        include_deleted: bool = False,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        layer: Optional[str] = None,
        layers: Optional[List[str]] = None
    ) -> int:
        """
        统计记忆数量

        Args:
            user_id: 用户 ID 过滤
            include_archived: 是否包含归档记忆
            include_deleted: 是否包含软删除记忆
            status: 精确状态过滤（active/archived/deleted）
            memory_type: 精确记忆类型过滤
            memory_types: 多记忆类型过滤
            layer: 精确生命周期层过滤
            layers: 多生命周期层过滤

        Returns:
            记忆数量
        """
        if not self.is_available():
            return 0

        try:
            if memory_types or layers:
                if memory_types and len(memory_types) == 1 and not memory_type:
                    memory_type = memory_types[0]
                    memory_types = None
                if layers and len(layers) == 1 and not layer:
                    layer = layers[0]
                    layers = None
                if memory_types or layers:
                    return len(await self.get_all_memories(
                        user_id=user_id,
                        limit=0,
                        include_deleted=include_deleted,
                        include_archived=include_archived,
                        status=status,
                        memory_type=memory_type,
                        memory_types=memory_types,
                        layer=layer,
                        layers=layers
                    ))

            must_conditions = []
            must_not_conditions = []

            if status:
                must_conditions.append(
                    FieldCondition(
                        key="status",
                        match=MatchValue(value=status)
                    )
                )
            else:
                if not include_deleted:
                    must_not_conditions.append(
                        FieldCondition(
                            key="status",
                            match=MatchValue(value="deleted")
                        )
                    )
                if not include_archived:
                    must_not_conditions.append(
                        FieldCondition(
                            key="status",
                            match=MatchValue(value="archived")
                        )
                    )

            if user_id:
                must_conditions.append(
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                )
            if memory_type:
                must_conditions.append(
                    FieldCondition(
                        key="memory_type",
                        match=MatchValue(value=memory_type)
                    )
                )
            if layer:
                must_conditions.append(
                    FieldCondition(
                        key="layer",
                        match=MatchValue(value=layer)
                    )
                )

            result = await asyncio.to_thread(
                self.client.count,
                collection_name=self.collection_name,
                count_filter=Filter(must=must_conditions, must_not=must_not_conditions)
            )
            return result.count

        except Exception as e:
            logger.error(f"统计记忆数量失败: {e}")
            return 0

    async def find_similar(
        self,
        vector: List[float],
        threshold: float = 0.95,
        exclude_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        查找相似记忆（用于去重）

        Args:
            vector: 向量
            threshold: 相似度阈值
            exclude_id: 排除的 ID
            user_id: 用户 ID

        Returns:
            最相似的记忆，或 None
        """
        results = await self.search(
            query_vector=vector,
            top_k=5,
            score_threshold=threshold,
            user_id=user_id
        )

        for result in results:
            if exclude_id and result['id'] == exclude_id:
                continue
            return result

        return None

    def close(self):
        """关闭连接（同步方法）"""
        if self.client:
            try:
                self.client.close()
                logger.info("Qdrant 连接已关闭")
            except:
                pass

