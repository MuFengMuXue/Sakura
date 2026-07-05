"""
注册所有的服务
"""
import os
import sys
from typing import Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sentence_transformers import SentenceTransformer
import torch

from storage.qdrant_client import MemosQdrantClient
from storage.networkx_graph import NetworkXGraphClient
from storage.neo4j_client import MemosNeo4jClient
from utils.search_utils import BM25Searcher
from memories.preference_memory import PreferenceMemory
from memories.tool_memory import ToolMemory
from memories.image_memory import ImageMemory
from utils.document_loader import DocumentLoader
from utils.entity_extractor import EntityExtractor
from core.scheduler import MemScheduler

from .load_config import get_config, AppConfig


class ServiceRegistry:
    """应用服务注册表，管理所有外部依赖的生命周期"""
    
    def __init__(self):
        self.config: AppConfig = get_config()
        self._init_components()
    
    def _init_components(self):
        print("初始化依赖组件...")
        self._init_embedding()
        self._init_qdrant()
        self._init_graph()
        self._init_bm25()
        self._init_memories()
        self._init_document_loader()
        self._init_scheduler()
        self._init_entity_extractor()
        self._init_others()
        print("初始化完成")
        self._print_status()
    
    def _init_embedding(self):
        cfg = self.config.embedding
        if cfg.use_local_model:
            if not cfg.model_path:
                raise RuntimeError("use_local_model 为 true 但未配置 model_path")
            print("加载 Embedding 模型...")
            model_path = cfg.model_path
            if not os.path.isabs(model_path):
                model_path = os.path.join(_project_root, model_path)
            model_path = os.path.normpath(model_path)
            self.embedder = SentenceTransformer(model_path)
            if torch.cuda.is_available():
                self.embedder = self.embedder.to('cuda')
                print("模型已加载 (GPU)")
            else:
                self.embedder = self.embedder.to('cpu')
                print("模型已加载 (CPU)")
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
            print("云端 Embedding 加载完成")

    def _init_qdrant(self):
        print("初始化 Qdrant...")
        try:
            vec_cfg = self.config.storage.vector
            path = vec_cfg.path
            if not os.path.isabs(path):
                path = os.path.join(_project_root, path)
            path = os.path.normpath(path)
            vector_size = self.config.storage.vector.vector_size   # 统一从 embedding 取
            collection_name = vec_cfg.collection_name
            self.qdrant = MemosQdrantClient(
                path=path,
                collection_name=collection_name,
                vector_size=vector_size
            )
            if self.qdrant.is_available():
                print("Qdrant 已就绪")
            else:
                print("Qdrant 不可用")
        except Exception as e:
            print(f"Qdrant 初始化失败: {e}")
            self.qdrant = None
    
    def _init_graph(self):
        graph_cfg = self.config.storage.graph
        if not graph_cfg.enable:
            self.graph = None
            print("图数据库未启用")
            return
        print(f"初始化图数据库 ({graph_cfg.type})...")
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
                print("图数据库已就绪")
            else:
                print("图数据库不可用")
                self.graph = None
        except Exception as e:
            print(f"图数据库初始化失败: {e}")
            self.graph = None
    
    def _init_bm25(self):
        if not self.config.search.enable_bm25:
            self.bm25 = None
            print("BM25 索引未启用")
            return
        print("初始化 BM25 索引...")
        try:
            self.bm25 = BM25Searcher()
            print("BM25 索引实例已创建（待重建）")
        except Exception as e:
            print(f"BM25 初始化失败: {e}")
            self.bm25 = None
    
    def _init_memories(self):
        print("初始化偏好记忆管理器...")
        try:
            self.preference_memory = PreferenceMemory(
                user_id=self.config.users.default_user_id,
                vector_storage=self.qdrant,
                graph_storage=self.graph,
                embedder=self.embedder
            )
            print("偏好记忆管理器已创建（待加载）")
        except Exception as e:
            print(f"偏好记忆管理器初始化失败: {e}")
            self.preference_memory = None
        
        print("初始化工具记忆管理器...")
        try:
            self.tool_memory = ToolMemory(
                user_id=self.config.users.default_user_id,
                vector_storage=self.qdrant
            )
            print("工具记忆管理器已创建（待加载）")
        except Exception as e:
            print(f"工具记忆管理器初始化失败: {e}")
            self.tool_memory = None
        
        img_cfg = self.config.image
        if img_cfg.enable:
            print("初始化图像记忆管理器...")
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
                print("图像记忆管理器已创建（待加载元数据）")
            except Exception as e:
                print(f"图像记忆管理器初始化失败: {e}")
                self.image_memory = None
        else:
            self.image_memory = None
            print("图像记忆未启用")
    
    def _init_document_loader(self):
        print("初始化文档加载器...")
        try:
            kb_cfg = self.config.kb
            self.document_loader = DocumentLoader(
                chunk_size=kb_cfg.chunk_size,
                chunk_overlap=kb_cfg.chunk_overlap
            )
            print("文档加载器已就绪")
        except Exception as e:
            print(f"文档加载器初始化失败: {e}")
            self.document_loader = None
    
    def _init_scheduler(self):
        sched_cfg = self.config.scheduler
        if not sched_cfg.enable:
            self.scheduler = None
            print("异步调度器未启用")
            return
        print("初始化异步任务调度器...")
        try:
            self.scheduler = MemScheduler(
                use_redis=sched_cfg.use_redis,
                redis_url=sched_cfg.redis_url,
                max_workers=sched_cfg.max_workers,
                quota_per_user=sched_cfg.quota_per_user
            )
            print("调度器实例已创建（待启动）")
        except Exception as e:
            print(f"调度器初始化失败: {e}")
            self.scheduler = None
    
    def _init_entity_extractor(self):
        ent_cfg = self.config.entity_extraction
        if not ent_cfg.enable:
            self.entity_extractor = None
            print("实体提取器未启用")
            return
        print("初始化实体提取器...")
        try:
            llm_dict = self.config.llm.dict()
            self.entity_extractor = EntityExtractor(llm_config=llm_dict)
            print("实体提取器已就绪")
        except Exception as e:
            print(f"实体提取器初始化失败: {e}")
            self.entity_extractor = None
    
    def _init_others(self):
        pass
    
    def _print_status(self):
       
        print(f"Embedding:   {'已加载' if self.embedder else '未加载'}")
        print(f"Qdrant:      {'已启用' if self.qdrant and self.qdrant.is_available() else '未启用'}")
        print(f"Graph:       {'已启用' if self.graph and self.graph.is_available() else '未启用'}")
        print(f"BM25:        {'已启用' if self.bm25 else '未启用'}")
        print(f"Preference:  {'已创建' if self.preference_memory else '未创建'}")
        print(f"Tool:        {'已创建' if self.tool_memory else '未创建'}")
        print(f"Image:       {'已创建' if self.image_memory else '未创建'}")
        print(f"Loader:      {'已创建' if self.document_loader else '未创建'}")
        print(f"Scheduler:   {'已创建' if self.scheduler else '未创建'}")
        print(f"Entity:      {'已创建' if self.entity_extractor else '未创建'}")
    
    # ---------- 异步加载方法 ----------
    async def load_memories(self):
        if self.preference_memory:
            try:
                await self.preference_memory.load()
                print("偏好记忆已加载")
            except Exception as e:
                print(f"偏好记忆加载失败: {e}")
        if self.tool_memory:
            try:
                await self.tool_memory.load()
                print("工具记忆已加载")
            except Exception as e:
                print(f"工具记忆加载失败: {e}")
        if self.image_memory:
            try:
                await self.image_memory.load_metadata()
                print("图像记忆已加载")
            except Exception as e:
                print(f"图像记忆加载失败: {e}")
    
    async def rebuild_bm25(self):
        if not self.bm25 or not self.qdrant:
            return
        try:
            from .utils.bm25_index import rebuild_bm25_index
            await rebuild_bm25_index(self.bm25, self.qdrant)
            print("BM25 索引重建完成")
        except Exception as e:
            print(f"BM25 索引重建失败: {e}")
    
    async def start_scheduler(self):
        if self.scheduler:
            try:
                await self.scheduler.start()
                print("调度器已启动")
            except Exception as e:
                print(f"调度器启动失败: {e}")
    
    async def shutdown(self):
        print("正在关闭所有资源...")
        if self.scheduler:
            try:
                await self.scheduler.stop()
                print("调度器已停止")
            except:
                pass
        if self.qdrant:
            try:
                self.qdrant.close()
            except:
                pass
        if self.graph and hasattr(self.graph, 'close'):
            try:
                self.graph.close()
            except:
                pass
        print("资源清理完成")
