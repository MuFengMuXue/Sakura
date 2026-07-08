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

MEMORY_LAYERS = ["WorkingMemory", "LongTermMemory", "UserMemory"]

# ---------- 日期工具 ----------
from .time_utils import (
    parse_iso_datetime,
    safe_float,
    safe_int,
    recency_boost_for,
    frequency_boost_for,
)

# ---------- 分层工具 ----------
from .layer_utils import (
    DEFAULT_LAYER_WEIGHTS,
    LAYER_RANK,
    normalize_layer,
    infer_memory_layer,
)

# ---------- Payload 工具 ----------
from .payload_utils import (
    payload_of,
    ensure_list,
    merge_unique_list,
    deduplicate_keeper_score,
    choose_deduplicate_keeper,
    normalize_duplicate_action,
)

# ---------- 文本工具 ----------
from .text_utils import (
    MAX_CONTEXT_SUMMARY_CHARS,
    normalize_context_summary,
    truncate_text,
    clean_whitespace,
    extract_role_label,
)

# ---------- JSON 工具 ----------
from .json_utils import (
    balance_truncated_json,
    parse_memories_json,
    build_chat_completion_payload,
)

# ---------- 编码工具 ----------
from .encoding_utils import (
    encode_text,
    encode_texts,
    encode_text_with_registry,
    encode_texts_with_registry,
)

# ---------- BM25 工具 ----------
from .bm25_utils import (
    update_bm25_index,
    rebuild_bm25_index,
    remove_bm25_document,
)

# ---------- 记忆提取 ----------
from .memories_extraction import extract_memories

# ---------- 记忆合并 ----------
from .memories_merge import merge_memories

# ---------- 记忆操作 ----------
from .memory_utils import (
    get_memory,
    flatten_memory,
    update_memory_usage,
    dispose_merged_duplicate,
)

# ---------- 实体工具 ----------
from .entity_utils import store_entities_for_memory

__all__ = [
    # constants
    'MEMORY_TYPE_WEIGHTS',
    'MEMORY_LAYERS',
    # date_utils
    'parse_iso_datetime',
    'safe_float',
    'safe_int',
    'recency_boost_for',
    'frequency_boost_for',
    # layer_utils
    'DEFAULT_LAYER_WEIGHTS',
    'LAYER_RANK',
    'normalize_layer',
    'infer_memory_layer',
    # payload_utils
    'payload_of',
    'ensure_list',
    'merge_unique_list',
    'deduplicate_keeper_score',
    'choose_deduplicate_keeper',
    'normalize_duplicate_action',
    # text_utils
    'MAX_CONTEXT_SUMMARY_CHARS',
    'normalize_context_summary',
    'truncate_text',
    'clean_whitespace',
    'extract_role_label',
    # json_utils
    'balance_truncated_json',
    'parse_memories_json',
    'build_chat_completion_payload',
    # encoding_utils
    'encode_text',
    'encode_texts',
    'encode_text_with_registry',
    'encode_texts_with_registry',
    # bm25_index
    'update_bm25_index',
    'rebuild_bm25_index',
    'remove_bm25_document',
    # memories_extraction
    'extract_memories',
    # memories_merge
    'merge_memories',
    # memory_utils
    'get_memory',
    'flatten_memory',
    'update_memory_usage',
    'dispose_merged_duplicate',
    # entity_utils
    'store_entities_for_memory',
]