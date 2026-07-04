# api/utils/__init__.py

# ---------- 常量 ----------
MEMORY_TYPE_WEIGHTS = {
    'preference': 1.5,
    'fact': 1.3,
    'semantic': 1.2,
    'episodic': 1.0,
    'procedural': 1.1,
    'event': 1.0,
    'tool': 0.9,
    'general': 1.0,
}

# ---------- 导出工具函数 ----------
from .text_encoding import encode_text, encode_texts
from .bm25_index import update_bm25_index, rebuild_bm25_index
from .memories_extraction import extract_memories
from .memories_merge import merge_memories

__all__ = [
    'MEMORY_TYPE_WEIGHTS',
    'encode_text',
    'encode_texts',
    'update_bm25_index',
    'rebuild_bm25_index',
    'extract_memories',
    'merge_memories',
]