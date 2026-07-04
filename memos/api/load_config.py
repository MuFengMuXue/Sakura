"""
从config/settings.yaml加载配置文件
"""
import os
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path

# --- 子配置 ---

class LLMConfig(BaseModel):
    model: str
    api_key: str
    base_url: str
    temperature: float

class EmbeddingConfig(BaseModel):
    use_local_model: bool
    model_path: Optional[str] = None
    model: str
    base_url: str
    api_key: str
    vector_size: int 
    timeout: int
    max_retries: int

class VectorStorageConfig(BaseModel):
    type: str 
    path: str  
    collection_name: str
    vector_size: int 

class GraphStorageConfig(BaseModel):
    enable: bool 
    type: str
    path: str


class StorageConfig(BaseModel):
    vector: VectorStorageConfig
    graph: GraphStorageConfig

class SearchConfig(BaseModel):
    top_k: int 
    similarity_threshold: float
    enable_bm25: bool 
    bm25_weight: float 
    enable_graph_query: bool 
    graph_max_depth: int 
    importance_weight: float 
    type_weight_factor: float 

class EntityExtractionConfig(BaseModel):
    enable: bool 
    auto_extract_on_add: bool 

class SchedulerConfig(BaseModel):
    enable: bool 
    use_redis: bool
    redis_url: str 
    max_workers: int 
    quota_per_user: int 

class ImageConfig(BaseModel):
    enable: bool 
    storage_path: str 
    use_clip: bool
    max_size_mb: int 
    auto_describe: bool 

class KBConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

class UsersConfig(BaseModel):
    default_user_id: str 
    enable_multi_user: bool 

# --- 根配置 ---

class AppConfig(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig
    storage: StorageConfig
    search: SearchConfig
    entity_extraction: EntityExtractionConfig
    scheduler: SchedulerConfig
    image: ImageConfig
    kb: KBConfig
    users: UsersConfig

    @classmethod
    def load_config(cls, yaml_path: Optional[str] = None):
        """
        加载配置文件，支持显式传入路径或自动定位。
        """
        if yaml_path is None:
            # 自动定位：当前文件是 memos/api/load_config.py
            # 向上三级到达 Sakura/ 根目录
            base_dir = Path(__file__).parent.parent.parent
            yaml_path = base_dir / "config" / "settings.yaml"
        else:
            yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 如果顶层有 'memos' 键，提取出来（
        if 'memos' in data:
            data = data['memos']
        
        return cls(**data)

# --- 全局单例 ---

_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig.load_config()
    return _config