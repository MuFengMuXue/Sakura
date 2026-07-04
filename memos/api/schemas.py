# api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# ----- 记忆相关 -----
class AddMemoryRequest(BaseModel):
    messages: List[Dict[str, str]]
    user_id: Optional[str] = None  # 默认在路由中处理

class RawMemoryMessage(BaseModel):
    content: str
    role: Optional[str] = "user"
    importance: Optional[float] = 0.8
    memory_type: Optional[str] = "general"
    tags: Optional[List[str]] = None

class AddRawMemoryRequest(BaseModel):
    messages: List[RawMemoryMessage]
    user_id: Optional[str] = None
    extract_entities: Optional[bool] = None

class SearchMemoryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    user_id: Optional[str] = None
    similarity_threshold: Optional[float] = None
    use_graph: Optional[bool] = None
    use_bm25: Optional[bool] = None
    tags: Optional[List[str]] = None
    memory_types: Optional[List[str]] = None

# ----- 偏好相关 -----
class AddPreferenceRequest(BaseModel):
    item: str
    category: str = "other"
    preference_type: str = "like"
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    reason: Optional[str] = None
    user_id: Optional[str] = None

class ExtractPreferencesRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

# ----- 工具记忆 -----
class RecordToolUsageRequest(BaseModel):
    tool_name: str
    tool_category: str = "other"
    parameters: Optional[Dict[str, Any]] = None
    success: bool = True
    result_summary: Optional[str] = None
    context: Optional[str] = None
    user_intent: Optional[str] = None
    user_id: Optional[str] = None

# ----- 反馈 -----
class MemoryFeedbackRequest(BaseModel):
    memory_id: str
    feedback_type: str  # correct, supplement, delete, merge
    correction: Optional[str] = None
    reason: Optional[str] = None
    user_id: Optional[str] = None

# ----- 知识库导入 -----
class ImportDocumentRequest(BaseModel):
    source: str
    tags: Optional[List[str]] = None
    extract_entities: bool = False
    user_id: Optional[str] = None

class ImportBatchRequest(BaseModel):
    sources: List[str]
    tags: Optional[List[str]] = None
    extract_entities: bool = False
    user_id: Optional[str] = None

# ----- 图像 -----
class UploadImageRequest(BaseModel):
    image_base64: str = Field(..., alias="image_base64")
    filename: Optional[str] = "image.jpg"
    image_type: Optional[str] = "other"
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    user_id: Optional[str] = None
    auto_describe: bool = None

    class Config:
        populate_by_name = True

# ----- 调度器 -----
class SubmitTaskRequest(BaseModel):
    task_type: str  # add_memory, process_image, extract_entities
    payload: Dict[str, Any]
    priority: Optional[int] = 1
    user_id: Optional[str] = None
    timeout: Optional[int] = 60

# ----- 实体提取 -----
class ExtractEntitiesRequest(BaseModel):
    text: str
    context: Optional[str] = None
    store_to_graph: bool = None
    link_to_memory_id: Optional[str] = None
    user_id: Optional[str] = None

# ----- 图谱 -----
class AddEntityRequest(BaseModel):
    entity_id: Optional[str] = None
    entity_type: str
    name: str
    properties: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None

class AddRelationRequest(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    properties: Optional[Dict[str, Any]] = None