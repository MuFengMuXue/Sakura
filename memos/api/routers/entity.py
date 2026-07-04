# api/routes/entities.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import uuid
import logging

from ..schemas import ExtractEntitiesRequest
from ..service_registry import get_service_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Entities"])

@router.post("/entities/extract")
async def extract_entities(request: ExtractEntitiesRequest):
    """从文本中提取实体和关系
    
    使用 LLM 自动识别文本中的实体和它们之间的关系。
    可选择是否存储到知识图谱。
    """
    registry = get_service_registry()
    entity_extractor = registry.entity_extractor
    graph = registry.graph
    config = registry.config   
    if not entity_extractor:
        raise HTTPException(status_code=503, detail="实体提取器未启用（需要配置 entity_extraction.enabled=true）")
    
    try:
        # 提取实体和关系
        entities, relations = await entity_extractor.extract(
            request.text,
            request.context
        )
        
        result = {
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type.value,
                    "description": e.description,
                    "confidence": e.confidence
                }
                for e in entities
            ],
            "relations": [
                {
                    "source": r.source_name,
                    "target": r.target_name,
                    "type": r.relation_type.value,
                    "description": r.description,
                    "confidence": r.confidence
                }
                for r in relations
            ],
            "entity_count": len(entities),
            "relation_count": len(relations)
        }
        
        # 存储到图谱（如果启用）
        if request.store_to_graph and graph and graph.is_available():
            user_id = request.user_id if request.user_id is not None else config.users.default_user_id
            stored_entities = []
            entity_name_to_id = {}
            
            for entity in entities:
                ent_id = f"ent_{uuid.uuid4().hex[:12]}"
                
                # 检查是否已存在
                existing = graph.find_entity_by_name(entity.name, user_id)
                if existing:
                    ent_id = existing['id']
                    if request.link_to_memory_id and hasattr(graph, 'link_entity_to_memory'):
                        graph.link_entity_to_memory(ent_id, request.link_to_memory_id)
                else:
                    props = {
                        'description': entity.description,
                        'confidence': entity.confidence
                    }
                    if request.link_to_memory_id:
                        props['source_memory_ids'] = [request.link_to_memory_id]
                    
                    graph.add_entity(
                        entity_id=ent_id,
                        entity_type=entity.entity_type.value,
                        name=entity.name,
                        properties=props,
                        user_id=user_id
                    )
                
                entity_name_to_id[entity.name] = ent_id
                stored_entities.append({"id": ent_id, "name": entity.name})
            
            # 存储关系
            stored_relations = 0
            for relation in relations:
                src_id = entity_name_to_id.get(relation.source_name)
                tgt_id = entity_name_to_id.get(relation.target_name)
                if src_id and tgt_id:
                    graph.add_relation(
                        source_id=src_id,
                        target_id=tgt_id,
                        relation_type=relation.relation_type.value,
                        properties={
                            'description': relation.description,
                            'confidence': relation.confidence,
                            'source_memory_id': request.link_to_memory_id
                        }
                    )
                    stored_relations += 1
            
            result["stored_to_graph"] = True
            result["stored_entities"] = stored_entities
            result["stored_relations"] = stored_relations
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-all-entities")
async def extract_entities_from_all_memories(
    dry_run: bool = False,
    limit: int = 10000
):
    """从所有记忆中批量提取实体和关系，丰富知识图谱
    
    Args:
        dry_run: 是否只预览不执行（True 时只返回预览结果，不修改数据）
        limit: 处理的最大记忆数量
    """
    registry = get_service_registry()
    qdrant = registry.qdrant
    entity_extractor = registry.entity_extractor
    graph = registry.graph
    config = registry.config
    
    if not entity_extractor:
        raise HTTPException(status_code=503, detail="实体提取器未启用（需要配置 entity_extraction.enabled=true）")
    
    if not graph or not graph.is_available():
        raise HTTPException(status_code=503, detail="知识图谱不可用")
    
    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=500, detail="存储不可用")
    
    try:
        # 1. 获取所有记忆
        all_memories = qdrant.get_all_memories(limit=limit)
        total_memories = len(all_memories)
        
        if total_memories == 0:
            return {"status": "success", "message": "没有记忆需要处理"}
        
        print(f"开始从 {total_memories} 条记忆中提取实体...")
        
        entities_created = 0
        entities_skipped = 0
        relations_created = 0
        failed_count = 0
        entity_types_count = {}
        
        preview_entities = []
        preview_relations = []
        
        user_id = request.user_id if request.user_id is not None else config.users.default_user_id
        
        # 2. 逐条处理
        for idx, mem in enumerate(all_memories):
            content = mem.get('content', '')
            
            if not content or len(content) < 5:
                continue
            
            print(f"\n[{idx+1}/{total_memories}] 处理: {content[:50]}...")
            
            try:
                entities, relations = await entity_extractor.extract(content)
                
                if not entities:
                    print(f"未发现实体")
                    continue
                
                print(f"发现 {len(entities)} 个实体, {len(relations) if relations else 0} 个关系")
                
                for entity in entities:
                    try:
                        entity_name = entity.name if hasattr(entity, 'name') else str(entity)
                        entity_type = entity.entity_type.value if hasattr(entity, 'entity_type') else 'unknown'
                        entity_desc = entity.description if hasattr(entity, 'description') else ''
                        
                        entity_types_count[entity_type] = entity_types_count.get(entity_type, 0) + 1
                        
                        if dry_run:
                            preview_entities.append({
                                "name": entity_name,
                                "type": entity_type,
                                "description": entity_desc,
                                "source": content[:50]
                            })
                            continue
                        
                        existing = graph.find_entity_by_name(entity_name, user_id)
                        if existing:
                            entities_skipped += 1
                            print(f"实体已存在: {entity_name}")
                            continue
                        
                        new_entity_id = str(uuid.uuid4())
                        success = graph.create_entity(
                            entity_id=new_entity_id,
                            name=entity_name,
                            entity_type=entity_type,
                            user_id=user_id,
                            properties={'description': entity_desc} if entity_desc else {}
                        )
                        
                        if success:
                            entities_created += 1
                            print(f"创建实体: {entity_name} [{entity_type}]")
                        
                    except Exception as ee:
                        logger.warning(f"保存实体失败: {ee}")
                
                if relations:
                    for rel in relations:
                        try:
                            source_name = rel.source_name if hasattr(rel, 'source_name') else ''
                            target_name = rel.target_name if hasattr(rel, 'target_name') else ''
                            relation_type = rel.relation_type.value if hasattr(rel, 'relation_type') else 'related_to'
                            rel_desc = rel.description if hasattr(rel, 'description') else ''
                            
                            if dry_run:
                                preview_relations.append({
                                    "source": source_name,
                                    "target": target_name,
                                    "type": relation_type,
                                    "description": rel_desc
                                })
                                continue
                            
                            source_entity = graph.find_entity_by_name(source_name, user_id)
                            target_entity = graph.find_entity_by_name(target_name, user_id)
                            
                            if source_entity and target_entity:
                                graph.create_relation(
                                    source_id=source_entity['id'],
                                    target_id=target_entity['id'],
                                    relation_type=relation_type,
                                    properties={'description': rel_desc} if rel_desc else {}
                                )
                                relations_created += 1
                                print(f"创建关系: {source_name} --[{relation_type}]--> {target_name}")
                                
                        except Exception as re:
                            logger.warning(f"保存关系失败: {re}")
                
            except Exception as e:
                print(f"提取失败: {e}")
                failed_count += 1
                continue
        
        # 3. 返回结果
        if dry_run:
            print(f"不执行实际修改")
            
            return {
                "status": "preview",
                "dry_run": True,
                "memories_processed": total_memories,
                "entities_found": len(preview_entities),
                "relations_found": len(preview_relations),
                "entity_types": entity_types_count,
                "sample_entities": preview_entities[:20],
                "sample_relations": preview_relations[:10]
            }
        print(f"实体提取完成！")
        print(f"处理记忆: {total_memories} 条")
        print(f"创建实体: {entities_created} 个")
        print(f"跳过已存在: {entities_skipped} 个")
        print(f"创建关系: {relations_created} 个")
        print(f"失败: {failed_count} 条")
        
        return {
            "status": "success",
            "memories_processed": total_memories,
            "entities_created": entities_created,
            "entities_skipped": entities_skipped,
            "relations_created": relations_created,
            "failed": failed_count,
            "entity_types": entity_types_count
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"实体提取失败: {str(e)}")

@router.get("/entities/stats")
async def get_entity_stats():
    """获取实体统计"""
    registry = get_service_registry()
    entity_extractor = registry.entity_extractor
    graph = registry.graph
    config = registry.config
    
    if not entity_extractor:
        return {"status": "disabled", "message": "实体提取器未启用"}
    
    stats = {"status": "enabled"}
    
    if graph and graph.is_available():
        user_id = request.user_id if request.user_id is not None else config.users.default_user_id
        graph_stats = graph.get_stats(user_id)
        stats.update({
            "entity_count": graph_stats.get('entity_count', 0),
            "relation_count": graph_stats.get('relation_count', 0),
            "entity_types": graph_stats.get('entity_types', {}),
            "relation_types": graph_stats.get('relation_types', {})
        })
    
    return stats