"""
实体存储工具函数
提供从文本中提取实体和关系，并存储到知识图谱的功能
"""
from typing import Dict, Any, List, Optional
import uuid
import logging

logger = logging.getLogger(__name__)


async def store_entities_for_memory(
    text: str,
    memory_id: str,
    user_id: str,
    registry,  # 由调用方传入，不使用 Depends
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从文本中提取实体和关系，并存入知识图谱，同时与记忆关联。

    Args:
        text: 要提取实体的文本内容
        memory_id: 关联的记忆 ID
        user_id: 用户 ID
        registry: 服务注册表（由调用方传入）
        context: 可选的上下文信息（如知识库标题）

    Returns:
        {
            'entity_ids': [ent_id1, ent_id2, ...],
            'entities': [{'id': ent_id, 'name': '实体名', 'entity_type': '类型'}, ...],
            'relations_created': 3
        }
    """
    # 检查是否具备提取条件
    if not registry.entity_extractor or not registry.graph or not registry.graph.is_available():
        logger.debug("实体提取器或图数据库未启用，跳过")
        return {'entity_ids': [], 'entities': [], 'relations_created': 0}

    try:
        # 调用实体提取器
        entities, relations = await registry.entity_extractor.extract(text, context)
    except Exception as e:
        logger.warning(f"实体提取失败: {e}")
        return {'entity_ids': [], 'entities': [], 'relations_created': 0}

    entity_ids = []
    stored_entities = []
    entity_name_to_id = {}

    # 存储实体
    for entity in entities:
        try:
            entity_name = entity.name if hasattr(entity, 'name') else str(entity)
            entity_type = entity.entity_type.value if hasattr(entity, 'entity_type') else 'unknown'
            description = entity.description if hasattr(entity, 'description') else ''
            confidence = entity.confidence if hasattr(entity, 'confidence') else 0.8

            # 检查是否已存在
            existing = registry.graph.find_entity_by_name(
                entity_name, user_id, entity_type=entity_type
            )

            if existing:
                ent_id = existing['id']
                # 如果存在，将记忆关联到已有实体
                if hasattr(registry.graph, 'link_entity_to_memory'):
                    registry.graph.link_entity_to_memory(ent_id, memory_id)
            else:
                # 创建新实体
                ent_id = f"ent_{uuid.uuid4().hex[:12]}"
                registry.graph.create_entity(
                    entity_id=ent_id,
                    name=entity_name,
                    entity_type=entity_type,
                    user_id=user_id,
                    properties={
                        'description': description,
                        'confidence': confidence,
                        'source_memory_ids': [memory_id]
                    }
                )

            entity_name_to_id[entity_name] = ent_id
            entity_ids.append(ent_id)
            stored_entities.append({
                'id': ent_id,
                'name': entity_name,
                'entity_type': entity_type
            })

        except Exception as e:
            logger.warning(f"存储实体失败: {e}")

    # 存储关系
    relations_created = 0
    for relation in relations or []:
        try:
            source_name = getattr(relation, 'source_name', '')
            target_name = getattr(relation, 'target_name', '')
            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)

            if source_id and target_id:
                relation_type = relation.relation_type.value if hasattr(
                    relation, 'relation_type'
                ) else 'related_to'
                registry.graph.create_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    properties={
                        'description': getattr(relation, 'description', ''),
                        'confidence': getattr(relation, 'confidence', 0.8),
                        'source_memory_id': memory_id
                    }
                )
                relations_created += 1
        except Exception as e:
            logger.warning(f"存储关系失败: {e}")

    return {
        'entity_ids': list(dict.fromkeys(entity_ids)),  # 去重
        'entities': stored_entities,
        'relations_created': relations_created
    }