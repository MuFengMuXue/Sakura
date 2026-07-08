"""
注册所有的服务
"""
import os
import sys
from typing import Optional
import asyncio

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sentence_transformers import SentenceTransformer
import torch

from storage.qdrant_client import MemosQdrantClient
from storage.networkx_graph import NetworkXGraphClient
from storage.neo4j_client import MemosNeo4jClient
from utils.search_utils import (BM25Searcher,Reranker)
from .utils.reranker_client import CloudRerankerClient

from memories.preference_memory import PreferenceMemory
from memories.tool_memory import ToolMemory
from memories.image_memory import ImageMemory
from utils.document_loader import DocumentLoader
from utils.entity_extractor import EntityExtractor
from core.scheduler import MemScheduler

from .load_config import get_config, AppConfig
import logging
logger = logging.getLogger(__name__)

class ServiceRegistry:
    """应用服务注册表，管理所有外部依赖的生命周期"""
    
    def __init__(self):
        self.config: AppConfig = get_config()
        self.memory_evolution = None
        self.evolution_loop_task = None
        self._create_components()
    
    def _create_components(self):
        logger.info("创建组件...")
        self._init_embedding()
        self._init_qdrant()
        self._init_graph()
        self._init_bm25()
        self._init_memories()
        self._init_document_loader()
        self._init_scheduler()
        self._init_entity_extractor()
        self._init_reranker()
        self._init_evolution()
        self._init_others()
        logger.info("创建完成")
        
    
    def _init_embedding(self):
        cfg = self.config.embedding
        if cfg.use_local_model:
            if not cfg.model_path:
                raise RuntimeError("use_local_model 为 true 但未配置 model_path")
            logger.info("加载 Embedding 模型...")
            model_path = cfg.model_path
            if not os.path.isabs(model_path):
                model_path = os.path.join(_project_root, model_path)
            model_path = os.path.normpath(model_path)
            self.embedder = SentenceTransformer(model_path)
            if torch.cuda.is_available():
                self.embedder = self.embedder.to('cuda')
                logger.info("模型已加载 (GPU)")
            else:
                self.embedder = self.embedder.to('cpu')
                logger.info("模型已加载 (CPU)")
        else:
            if not cfg.base_url or not cfg.api_key:
                raise RuntimeError("云端 Embedding 需要配置 api_key 和 base_url")
            from .utils.embedding_client import CloudEmbeddingClient
            base_url = cfg.base_url.rstrip('/')
            if base_url.endswith('/embeddings'):
                base_url = base_url[:-10]
            self.embedder = CloudEmbeddingClient(
                api_key=cfg.api_key,
                base_url=base_url,
                model=cfg.model,
                timeout=cfg.timeout,
                max_retries=cfg.max_retries,
            )
            logger.info("云端 Embedding 加载完成")
    
    def _init_reranker(self):
        if not self.config.search.enable_reranker:
            self.reranker = None
            logger.info("重排序器未启用")
            return 
        logger.info("初始化重排序器")

        reranker_cfg = self.config.reranker

        if reranker_cfg.use_local_model:
            if not reranker_cfg.model_path:
                logger.warning("model_path为空，回退禁用")
                self.reranker = None
                return 
            try:
                model_path = reranker_cfg.model_path
                if not os.path.isabs(model_path):
                    model_path = os.path.join(_project_root, model_path)
                model_path = os.path.normpath(model_path)
            
            # 初始化本地 CrossEncoder
                self.reranker = Reranker(model_path)
                if self.reranker.is_available():
                    logger.info(f"本地重排序器已加载: {model_path}")
                else:
                    logger.warning("本地重排序器不可用，回退禁用")
                    self.reranker = None
            except Exception as e:
                logger.error(f"本地重排序器初始化失败: {e}")
                self.reranker = None
            return
        if not reranker_cfg.base_url or not reranker_cfg.api_key or not reranker_cfg.model:
            logger.warning("配置缺失,回退禁用")
            self.reranker = None
            return 
        try:
            timeout = getattr(reranker_cfg,'timeout',30)
            max_retries = getattr(reranker_cfg,'max_retries',2)
            self.reranker = CloudRerankerClient(
            api_key=reranker_cfg.api_key,
            base_url=reranker_cfg.base_url,
            model=reranker_cfg.model,
            timeout=timeout,
            max_retries=max_retries,
            rerank_path="/rerank"
            )
        except Exception as e:
            logger.error(f"云端重排序器初始化失败: {e}")
            self.reranker = None

    def _init_qdrant(self):
        logger.info("创建 Qdrant实例...")
        try:
            vec_cfg = self.config.storage.vector
            path = vec_cfg.path
            if not os.path.isabs(path):
                path = os.path.join(_project_root, path)
            path = os.path.normpath(path)
            vector_size = self.config.storage.vector.vector_size   
            collection_name = vec_cfg.collection_name
            self.qdrant = MemosQdrantClient(
                path=path,
                collection_name=collection_name,
                vector_size=vector_size
            )
            logger.info("Qdrant实例已创建，等待异步初始化")
        except Exception as e:
            logger.error(f"Qdrant 初始化失败: {e}")
            self.qdrant = None
    
    def _init_graph(self):
        graph_cfg = self.config.storage.graph
        if not graph_cfg.enable:
            self.graph = None
            logger.info("图数据库未启用")
            return
        logger.info(f"初始化图数据库 ({graph_cfg.type})...")
        try:
            if graph_cfg.type == "networkx":
                path = graph_cfg.path
                if not os.path.isabs(path):
                    path = os.path.join(_project_root, path)
                path = os.path.normpath(path)
                self.graph = NetworkXGraphClient(data_path=path)
            elif graph_cfg.type == "neo4j":
                uri = getattr(graph_cfg, 'uri', 'bolt://localhost:7687')
                user = getattr(graph_cfg, 'user', 'neo4j')
                password = getattr(graph_cfg, 'password', 'password')
                self.graph = MemosNeo4jClient(uri=uri, user=user, password=password)
            else:
                raise ValueError(f"未知的图类型: {graph_cfg.type}")
            if self.graph.is_available():
                logger.info("图数据库已就绪")
            else:
                logger.warning("图数据库不可用")
                self.graph = None
        except Exception as e:
            logger.error(f"图数据库初始化失败: {e}")
            self.graph = None
    
    def _init_bm25(self):
        if not self.config.search.enable_bm25:
            self.bm25 = None
            logger.info("BM25 索引未启用")
            return
        logger.info("初始化 BM25 索引...")
        try:
            self.bm25 = BM25Searcher()
            logger.info("BM25 索引实例已创建（待重建）")
        except Exception as e:
            logger.error(f"BM25 初始化失败: {e}")
            self.bm25 = None
    
    def _init_memories(self):
        logger.info("初始化偏好记忆管理器...")
        try:
            self.preference_memory = PreferenceMemory(
                user_id=self.config.users.default_user_id,
                vector_storage=self.qdrant,
                graph_storage=self.graph,
                embedder=self.embedder
            )
            logger.info("偏好记忆管理器已创建（待加载）")
        except Exception as e:
            logger.error(f"偏好记忆管理器初始化失败: {e}")
            self.preference_memory = None
        
        logger.info("初始化工具记忆管理器...")
        try:
            self.tool_memory = ToolMemory(
                user_id=self.config.users.default_user_id,
                vector_storage=self.qdrant
            )
            logger.info("工具记忆管理器已创建（待加载）")
        except Exception as e:
            logger.error(f"工具记忆管理器初始化失败: {e}")
            self.tool_memory = None
        
        img_cfg = self.config.image
        if img_cfg.enable:
            logger.info("初始化图像记忆管理器...")
            try:
                storage_path = img_cfg.storage_path
                if not os.path.isabs(storage_path):
                    storage_path = os.path.join(_project_root, storage_path)
                storage_path = os.path.normpath(storage_path)
                self.image_memory = ImageMemory(
                    storage_path=storage_path,
                    vector_storage=self.qdrant,
                    embedder=self.embedder,
                    llm_config=self.config.llm.dict(),
                    use_clip=img_cfg.use_clip,
                    max_image_size=img_cfg.max_size_mb * 1024 * 1024
                )
                logger.info("图像记忆管理器已创建（待加载元数据）")
            except Exception as e:
                logger.error(f"图像记忆管理器初始化失败: {e}")
                self.image_memory = None
        else:
            self.image_memory = None
            logger.info("图像记忆未启用")
    
    def _init_document_loader(self):
        logger.info("初始化文档加载器...")
        try:
            kb_cfg = self.config.kb
            self.document_loader = DocumentLoader(
                chunk_size=kb_cfg.chunk_size,
                chunk_overlap=kb_cfg.chunk_overlap
            )
            logger.info("文档加载器已就绪")
        except Exception as e:
            logger.error(f"文档加载器初始化失败: {e}")
            self.document_loader = None
    
    def _init_scheduler(self):
        sched_cfg = self.config.scheduler
        if not sched_cfg.enable:
            self.scheduler = None
            logger.info("异步调度器未启用")
            return
        logger.info("初始化异步任务调度器...")
        try:
            self.scheduler = MemScheduler(
                use_redis=sched_cfg.use_redis,
                redis_url=sched_cfg.redis_url,
                max_workers=sched_cfg.max_workers,
                quota_per_user=sched_cfg.quota_per_user
            )
            logger.info("调度器实例已创建（待启动）")
        except Exception as e:
            logger.error(f"调度器初始化失败: {e}")
            self.scheduler = None
    
    def _init_entity_extractor(self):
        ent_cfg = self.config.entity_extraction
        if not ent_cfg.enable:
            self.entity_extractor = None
            logger.info("实体提取器未启用")
            return
        logger.info("初始化实体提取器...")
        try:
            llm_dict = self.config.llm.dict()
            self.entity_extractor = EntityExtractor(llm_config=llm_dict)
            logger.info("实体提取器已就绪")
        except Exception as e:
            logger.error(f"实体提取器初始化失败: {e}")
            self.entity_extractor = None
    
    def _init_evolution(self):
        if not self.config.evolution.enable:
            self.memory_evolution = None
            logger.info("记忆演化未启用")
            return 
        if not self.qdrant or not self.qdrant.is_available():
            logger.warning("qdrant不可用，无法初始化演化引擎")
            return 
        try:
            from core.evolution import MemoryEvolution
            self.memory_evolution = MemoryEvolution(
            qdrant_client=self.qdrant,
            config=self.config.evolution.model_dump()
            )
            logger.info("记忆自演化引擎已就绪")
        except Exception as e:
            logger.error(f"记忆自演化引擎初始化失败: {e}")
            self.memory_evolution = None
    
    def _init_others(self):
        pass

    async def ainit(self):
        """
        异步初始化所有需要连接/加载的组件（Qdrant、记忆、BM25、调度器等）
        该方法应在 FastAPI 启动事件中调用
        """
        logger.info("开始异步初始化...")
        # 1. 初始化 Qdrant（建表、建索引）
        if self.qdrant:
            try:
                await self.qdrant.initialize()
                logger.info("Qdrant 已就绪")
            except Exception as e:
                logger.error(f"Qdrant 异步初始化失败: {e}")
                self.qdrant = None
        
        # 2. 加载记忆数据
        await self.load_memories()
        
        # 3. 重建 BM25（如果启用）
        await self.rebuild_bm25()
        
        # 4. 启动调度器
        await self.start_scheduler()

        #启动演化引擎
        await self.start_evolution_loop()
        
        # 5. 打印最终状态
        self._print_status()
        logger.info("异步初始化完成")
    
    def _print_status(self):
       
        logger.info(f"Embedding:   {'已加载' if self.embedder else '未加载'}")
        logger.info(f"Qdrant:      {'已启用' if self.qdrant and self.qdrant.is_available() else '未启用'}")
        logger.info(f"Graph:       {'已启用' if self.graph and self.graph.is_available() else '未启用'}")
        logger.info(f"BM25:        {'已启用' if self.bm25 else '未启用'}")
        logger.info(f"Preference:  {'已创建' if self.preference_memory else '未创建'}")
        logger.info(f"Tool:        {'已创建' if self.tool_memory else '未创建'}")
        logger.info(f"Image:       {'已创建' if self.image_memory else '未创建'}")
        logger.info(f"Loader:      {'已创建' if self.document_loader else '未创建'}")
        logger.info(f"Scheduler:   {'已创建' if self.scheduler else '未创建'}")
        logger.info(f"Entity:      {'已创建' if self.entity_extractor else '未创建'}")
        logger.info(f"Reranker:    {'已加载' if self.reranker else '未加载'}")
    # ---------- 异步加载方法 ----------
    async def load_memories(self):
        if self.preference_memory:
            try:
                await self.preference_memory.load()
                logger.info("偏好记忆已加载")
            except Exception as e:
                logger.error(f"偏好记忆加载失败: {e}")
        if self.tool_memory:
            try:
                await self.tool_memory.load()
                logger.info("工具记忆已加载")
            except Exception as e:
                logger.error(f"工具记忆加载失败: {e}")
        if self.image_memory:
            try:
                await self.image_memory.load_metadata()
                logger.info("图像记忆已加载")
            except Exception as e:
                logger.error(f"图像记忆加载失败: {e}")
    
    async def rebuild_bm25(self):
        if not self.bm25 or not self.qdrant:
            return
        try:
            from .utils.bm25_utils import rebuild_bm25_index
            await rebuild_bm25_index(self)
            logger.info("BM25 索引重建完成")
        except Exception as e:
            logger.error(f"BM25 索引重建失败: {e}")
    
    async def start_scheduler(self):
        if self.scheduler:
            try:
                await self.scheduler.start()
                self.scheduler.register_handler('evolve_memory', self._handle_evolve_memory_task)
                logger.info("调度器已启动")
            except Exception as e:
                logger.error(f"调度器启动失败: {e}")
    
    async def start_evolution_loop(self):
        """启动记忆演化后台循环"""
        if not self.config.evolution.enable:
            return
        if not self.memory_evolution:
            logger.warning("记忆演化引擎未初始化，无法启动循环")
            return
        if not self.scheduler:
            logger.warning("调度器未启用，无法启动演化循环")
            return

        self.evolution_loop_task = asyncio.create_task(
        self._evolution_periodic_loop()
        )  
        logger.info("记忆演化后台循环已启动")

    async def _handle_evolve_memory_task(self, task):
        """处理记忆自演化任务"""
        if not self.memory_evolution:
            return {'status': 'disabled', 'message': '记忆自演化未启用'}
        payload = task.payload or {}
        user_id = payload.get('user_id') or self.config.users.default_user_id
        limit = payload.get('limit', 10000)
        try:
            result = await self.memory_evolution.evolve(user_id=user_id, limit=limit)
             # 演化完成后重建 BM25
            if self.bm25:
                await self.rebuild_bm25()
            # 记录完成时间（使用 evolution.py 中的函数）
            from .routers.evolution import _record_evolution_completed
            _record_evolution_completed(result=result)
            return result
        except Exception as e:
            logger.error(f"演化任务执行失败: {e}")
            return {'status': 'error', 'message': str(e)}
        
    async def _evolution_periodic_loop(self):
        """后台周期性演化循环"""
        while True:
            try:
            # 从路由模块导入状态函数
                from .routers.evolution import  _seconds_until_next_evolution

                interval = max(int(self.config.evolution.evolve_interval), 60)
                wait_seconds = _seconds_until_next_evolution(interval)

                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                    continue

            # 提交演化任务到调度器
                await self.scheduler.submit(
                    task_type='evolve_memory',
                    payload={'user_id': self.config.users.default_user_id},
                    user_id=self.config.users.default_user_id,
                    timeout=int(self.config.evolution.timeout or 600)
                )
                logger.info("已提交到期补跑的记忆演化任务")
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"周期性记忆演化任务提交失败: {e}")
                await asyncio.sleep(60)

    async def shutdown(self):
        logger.info("正在关闭所有资源...")
        if self.evolution_loop_task:
            try:
                self.evolution_loop_task.cancel()
                await asyncio.gather(self.evolution_loop_task, return_exceptions=True)
                logger.info("演化后台循环已停止")
            except Exception as e:
                logger.warning(f"停止演化循环失败: {e}")
        if self.scheduler:
            try:
                await self.scheduler.stop()
                logger.info("调度器已停止")
            except:
                pass
        if self.qdrant:
            try:
                await self.qdrant.close()
            except:
                pass
        if self.graph and hasattr(self.graph, 'close'):
            try:
                self.graph.close()
            except:
                pass
        logger.info("资源清理完成")
