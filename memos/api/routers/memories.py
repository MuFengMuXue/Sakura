# api/routes/memories.py
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
import uuid
from datetime import datetime

from ..schemas import (
    AddMemoryRequest, AddRawMemoryRequest, SearchMemoryRequest,
    RawMemoryMessage
)

from ..utils import (
    encode_text, extract_memories, update_bm25_index,
    MEMORY_TYPE_WEIGHTS
)

from ..dependencies import get_registry
from ..service_registry import ServiceRegistry


router = APIRouter(tags=["Memories"])


@router.post("/add")
async def add_memory(
    request: AddMemoryRequest,
    registry: ServiceRegistry = Depends(get_registry)
):
    
    """添加记忆（LLM 加工版）"""
    embedder = registry.embedder
    qdrant = registry.qdrant
    preference_memory = registry.preference_memory
    entity_extractor = registry.entity_extractor
    graph = registry.graph
    config = registry.config
    user_id = request.user_id if request.user_id is not None else config.users.default_user_id

    if embedder is None:
        raise HTTPException(status_code=500, detail="Embedding 模型未加载")

    
    added_count = 0
    merged_count = 0
    skipped_count = 0
    preference_count = 0
    entity_count = 0

    # 合并对话
    conversation_text = []
    for msg in request.messages:
        content = msg.get('content', '')
        role = msg.get('role', 'user')
        if content and len(content.strip()) > 0:
            role_label = "主人" if role == 'user' else "沐樱"
            conversation_text.append(f"【{role_label}】{content}")
    if not conversation_text:
        return {"status": "success", "message": "无有效对话", "added": 0}
    full_conversation = "\n".join(conversation_text)

    print(f"正在处理 {len(request.messages)} 条对话消息...")

    # ========== 1. LLM 提取记忆 ==========
    processed_result = await extract_memories(full_conversation,registry)
    if processed_result.get("memories"):
        for mem_item in processed_result["memories"]:
            content = mem_item.get("content", "").strip()
            importance = mem_item.get("importance", 0.5)
            memory_type = mem_item.get("memory_type", "general")
            tags = mem_item.get("tags", [])

            valid_types = ['preference', 'fact', 'episodic', 'semantic', 'procedural', 'general']
            if memory_type not in valid_types:
                memory_type = 'general'
            if not isinstance(tags, list):
                tags = []

            if not content or len(content) < 5:
                continue
            if importance < 0.3:
                skipped_count += 1
                continue

            vector = await encode_text(content.registry)
            # 去重检查
            if qdrant and qdrant.is_available():
                similar = qdrant.find_similar(vector, threshold=0.95, user_id=user_id)
                if similar:
                    qdrant.update_memory(
                        similar['id'],
                        {
                            'merge_count': similar.get('payload', {}).get('merge_count', 0) + 1,
                            'importance': max(importance, similar.get('importance', 0.5))
                        }
                    )
                    merged_count += 1
                    continue

            memory_id = str(uuid.uuid4())
            payload = {
                'content': content,
                'user_id': user_id,
                'importance': importance,
                'memory_type': memory_type,
                'tags': tags,
                'created_at': datetime.now().isoformat(),
                'merge_count': 0,
                'processed': True
            }
            if qdrant and qdrant.is_available():
                qdrant.add_memory(memory_id, vector, payload)
                await update_bm25_index(memory_id, content,registry)
            added_count += 1
            type_label = {'preference': '偏好', 'fact': '事实', 'episodic': '情景',
                          'semantic': '语义', 'procedural': '程序性', 'general': '通用'}.get(memory_type, memory_type)
            tags_str = f" 标签:{tags}" if tags else ""
            print(f"新增记忆: {content[:60]}{'...' if len(content) > 60 else ''}")
            print(f"└─ 类型:{type_label} | 重要度:{importance:.0%}{tags_str}")

    # ========== 2. 自动提取实体 ==========
    if entity_extractor and graph:
        try:
            print(f"\n正在分析知识图谱实体...")
            entities, relations = await entity_extractor.extract(full_conversation)
            if entities:
                print(f"发现 {len(entities)} 个实体, {len(relations) if relations else 0} 个关系:")
                for entity in entities:
                    try:
                        entity_name = entity.name if hasattr(entity, 'name') else str(entity)
                        entity_type = entity.entity_type.value if hasattr(entity, 'entity_type') else 'unknown'
                        existing = graph.find_entity_by_name(entity_name, user_id)
                        if not existing:
                            new_entity_id = str(uuid.uuid4())
                            success = graph.create_entity(
                                entity_id=new_entity_id,
                                name=entity_name,
                                entity_type=entity_type,
                                user_id=user_id,
                                properties={'description': entity.description} if hasattr(entity, 'description') and entity.description else {}
                            )
                            if success:
                                entity_count += 1
                                print(f"实体: {entity_name} [{entity_type}]")
                    except Exception as ee:
                        print(f"保存实体失败: {ee}")
                if entity_count > 0:
                    print(f"成功保存 {entity_count} 个实体")
            else:
                print(f"未发现新实体")

            # 保存关系
            if relations:
                for rel in relations:
                    try:
                        source_name = rel.source_name if hasattr(rel, 'source_name') else ''
                        target_name = rel.target_name if hasattr(rel, 'target_name') else ''
                        relation_type = rel.relation_type.value if hasattr(rel, 'relation_type') else 'related_to'
                        source_entity = graph.find_entity_by_name(source_name, user_id)
                        target_entity = graph.find_entity_by_name(target_name, user_id)
                        if source_entity and target_entity:
                            graph.create_relation(
                                source_id=source_entity['id'],
                                target_id=target_entity['id'],
                                relation_type=relation_type,
                                properties={'description': rel.description} if hasattr(rel, 'description') and rel.description else {}
                            )
                    except Exception as re:
                        print(f"保存关系失败: {re}")
        except Exception as e:
            print(f"实体提取失败: {e}")

    # 构建返回结果
    result_parts = []
    if added_count > 0:
        result_parts.append(f"新增记忆 {added_count} 条")
    if merged_count > 0:
        result_parts.append(f"合并 {merged_count} 条")
    if skipped_count > 0:
        result_parts.append(f"跳过 {skipped_count} 条")
    if preference_count > 0:
        result_parts.append(f"提取偏好 {preference_count} 条")
    if entity_count > 0:
        result_parts.append(f"提取实体 {entity_count} 个")

    message = "、".join(result_parts) if result_parts else "无有效记忆"

    print(f"[记忆总结完成]")
    print(f"新增记忆: {added_count} 条")
    print(f"合并记忆: {merged_count} 条")
    print(f"跳过低重要度: {skipped_count} 条")
    print(f"提取偏好: {preference_count} 条")
    print(f"提取实体: {entity_count} 个")

    return {
        "status": "success",
        "message": message,
        "added": added_count,
        "merged": merged_count,
        "skipped": skipped_count,
        "preferences_extracted": preference_count,
        "entities_extracted": entity_count
    }

@router.post("/add_raw")
async def add_memory_raw(
    request: AddRawMemoryRequest,
    registry: ServiceRegistry = Depends(get_registry)
    ):
    """直接添加记忆（不经过 LLM 加工）"""
    embedder = registry.embedder
    qdrant = registry.qdrant
    graph = registry.graph
    entity_extractor = registry.config.entity_extraction.auto_extract_on_add
    config = registry.config
    if embedder is None:
        raise HTTPException(status_code=500, detail="Embedding 模型未加载")

    user_id = request.user_id if request.user_id is not None else config.users.default_user_id
    added_count = 0
    type_counts = {}
    extracted_entities = []

    for msg in request.messages:
        content = msg.content
        importance = msg.importance if msg.importance is not None else 0.8
        memory_type = msg.memory_type or "general"
        tags = msg.tags or []

        if memory_type not in MEMORY_TYPE_WEIGHTS:
            memory_type = "general"

        if content and len(content) > 5:
            vector = await encode_text(content,registry)
            if qdrant and qdrant.is_available():
                similar = qdrant.find_similar(vector, threshold=0.95, user_id=user_id)
                if similar:
                    continue

            memory_id = str(uuid.uuid4())
            entity_ids = []

            # 实体提取（如果启用）
            if request.extract_entities and entity_extractor and graph:
                try:
                    entities, relations = await entity_extractor.extract(content)
                    entity_name_to_id = {}
                    for entity in entities:
                        ent_id = f"ent_{uuid.uuid4().hex[:12]}"
                        existing = graph.find_entity_by_name(entity.name, user_id)
                        if existing:
                            ent_id = existing['id']
                            if hasattr(graph, 'link_entity_to_memory'):
                                graph.link_entity_to_memory(ent_id, memory_id)
                        else:
                            graph.add_entity(
                                entity_id=ent_id,
                                entity_type=entity.entity_type.value,
                                name=entity.name,
                                properties={
                                    'description': entity.description,
                                    'confidence': entity.confidence,
                                    'source_memory_ids': [memory_id]
                                },
                                user_id=user_id
                            )
                        entity_name_to_id[entity.name] = ent_id
                        entity_ids.append(ent_id)
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
                                    'source_memory_id': memory_id
                                }
                            )
                    extracted_entities.extend([{'id': eid, 'name': name} for name, eid in entity_name_to_id.items()])
                except Exception as e:
                    print(f"实体提取失败: {e}")

            payload = {
                'content': content,
                'user_id': user_id,
                'importance': importance,
                'memory_type': memory_type,
                'tags': tags,
                'entity_ids': entity_ids,
                'created_at': datetime.now().isoformat(),
                'processed': False
            }
            if qdrant and qdrant.is_available():
                qdrant.add_memory(memory_id, vector, payload)
                await update_bm25_index(memory_id, content,registry)
            added_count += 1
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1

    result = {
        "status": "success",
        "message": f"直接添加 {added_count} 条记忆",
        "added": added_count,
        "type_breakdown": type_counts
    }
    if extracted_entities:
        result["extracted_entities"] = extracted_entities
        result["entity_count"] = len(extracted_entities)
    return result

@router.post("/search")
async def search_memory(
    request: SearchMemoryRequest,
    registry: ServiceRegistry = Depends(get_registry)
    ):
    """搜索记忆"""
    embedder = registry.embedder
    qdrant = registry.qdrant
    bm25 = registry.bm25
    graph = registry.graph
    config = registry.config

    if embedder is None:
        raise HTTPException(status_code=500, detail="Embedding 模型未加载")

    user_id = request.user_id if request.user_id is not None else config.users.default_user_id

    # 确定是否启用 BM25 和图增强
    top_k = request.top_k if request.top_k is not None else config.search.top_k
    threshold = request.similarity_threshold if request.similarity_threshold is not None else config.search.similarity_threshold
    enable_bm25 = request.use_bm25 if request.use_bm25 is not None else config.search.enable_bm25
    enable_graph = request.use_graph if request.use_graph is not None else config.search.enable_graph_query

    print(f"{request.query[:80]}{'...' if len(request.query) > 80 else ''}")
    query_vector = await encode_text(request.query,registry)

    results_map = {}

    # 1. Qdrant 向量搜索
    if qdrant and qdrant.is_available():
        vector_results = qdrant.search(
            query_vector=query_vector,
            top_k= top_k * 3,
            score_threshold=threshold,
            user_id=user_id
        )
        for r in vector_results:
            r_id = r.get('id')
            if r_id:
                results_map[r_id] = {
                    'data': r,
                    'scores': {'vector': r.get('similarity', 0)}
                }

    # 2. BM25 关键词搜索
    if enable_bm25 and bm25:
        try:
            bm25_results = bm25.search(request.query, top_k=top_k * 3)
            if bm25_results:
                max_bm25_score = max(score for _, score in bm25_results) or 1
                for doc_id, score in bm25_results:
                    normalized_score = score / max_bm25_score
                    if doc_id in results_map:
                        results_map[doc_id]['scores']['bm25'] = normalized_score
                    else:
                        # 仅 BM25 找到的，需从 Qdrant 获取完整数据
                        memory = qdrant.get_memory(doc_id) if qdrant else None
                        if memory and memory.get('payload', {}).get('user_id') == user_id:
                            memory_data = {
                                'id': doc_id,
                                'content': memory.get('payload', {}).get('content', memory.get('content', '')),
                                'similarity': 0,
                                'importance': memory.get('payload', {}).get('importance', memory.get('importance', 0.5)),
                                'memory_type': memory.get('payload', {}).get('memory_type', memory.get('memory_type', 'general')),
                                'tags': memory.get('payload', {}).get('tags', memory.get('tags', [])),
                                'created_at': memory.get('payload', {}).get('created_at', memory.get('created_at')),
                                'updated_at': memory.get('payload', {}).get('updated_at', memory.get('updated_at')),
                                'entity_ids': memory.get('payload', {}).get('entity_ids', memory.get('entity_ids', [])),
                                'bm25_only': True
                            }
                            results_map[doc_id] = {
                                'data': memory_data,
                                'scores': {'bm25': normalized_score}
                            }
            print(f"BM25 找到 {len(bm25_results)} 条候选记忆")
        except Exception as e:
            print(f"BM25 搜索失败: {e}")

    # 3. 合并结果并计算混合得分
    bm25_weight = config.search.bm25_weight if config else 0.3
    results = []
    for r_id, r_data in results_map.items():
        result = r_data['data'].copy()
        scores = r_data['scores']
        if enable_bm25:
            vector_score = scores.get('vector', 0)
            bm25_score = scores.get('bm25', 0)
            if vector_score > 0 and bm25_score > 0:
                mixed_similarity = vector_score + bm25_weight * bm25_score
            elif vector_score > 0:
                mixed_similarity = vector_score
            else:
                mixed_similarity = bm25_weight * bm25_score
            result['similarity'] = mixed_similarity
            result['vector_score'] = vector_score
            result['bm25_score'] = bm25_score
        results.append(result)

    # 标签过滤
    if request.tags:
        results = [r for r in results if any(tag in r.get('tags', []) for tag in request.tags)]

    # 记忆类型过滤
    if request.memory_types:
        results = [r for r in results if r.get('memory_type', 'general') in request.memory_types]

    # 图增强搜索
    if enable_graph and graph and graph.is_available():
        try:
            import re
            potential_entities = []
            potential_entities.extend(re.findall(r'[\u4e00-\u9fff]{2,4}', request.query))
            potential_entities.extend(re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', request.query))
            print(f"图谱搜索: 提取到 {len(potential_entities)} 个候选实体")
            matched_entity_ids = []
            for name in potential_entities[:10]:
                entity = graph.find_entity_by_name(name, user_id)
                if entity:
                    matched_entity_ids.append(entity['id'])
                    related = graph.find_related_entities(entity['id'], max_depth=2)
                    for rel in related:
                        if rel['id'] not in matched_entity_ids:
                            matched_entity_ids.append(rel['id'])
            if matched_entity_ids:
                print(f"图谱搜索: 匹配到 {len(matched_entity_ids)} 个实体")
                graph_memory_ids = []
                if hasattr(graph, 'get_memories_by_entities'):
                    graph_memory_ids = graph.get_memories_by_entities(matched_entity_ids)
                else:
                    for eid in matched_entity_ids:
                        mids = graph.get_entity_memories(eid) if hasattr(graph, 'get_entity_memories') else []
                        graph_memory_ids.extend(mids)
                    graph_memory_ids = list(set(graph_memory_ids))
                result_ids = {r.get('id') for r in results}
                graph_boost_count = 0
                for r in results:
                    r_id = r.get('id')
                    r_entities = r.get('entity_ids', [])
                    if r_id in graph_memory_ids:
                        r['graph_boost'] = 0.15
                        graph_boost_count += 1
                    elif any(eid in r_entities for eid in matched_entity_ids):
                        r['graph_boost'] = 0.1
                        graph_boost_count += 1
                graph_only_count = 0
                for mem_id in graph_memory_ids[:5]:
                    if mem_id not in result_ids:
                        memory = qdrant.get_memory(mem_id)
                        if memory and memory.get('payload', {}).get('user_id') == user_id:
                            memory['graph_boost'] = 0.2
                            memory['similarity'] = 0.5
                            memory['graph_only'] = True
                            results.append(memory)
                            graph_only_count += 1
                if graph_boost_count > 0 or graph_only_count > 0:
                    print(f"图谱增强: {graph_boost_count} 条加分, {graph_only_count} 条仅图谱")
            else:
                print(f"图谱搜索: 未匹配到任何实体")
        except Exception as e:
            print(f"图增强搜索失败: {e}")

    # 应用多维加权
    importance_weight = config.search.importance_weight if config else 0.3
    type_weight_factor = config.search.type_weight_factor if config else 0.2
    for result in results:
        similarity = result.get('similarity', 0)
        importance = result.get('importance', 0.5)
        memory_type = result.get('memory_type', 'general')
        graph_boost = result.get('graph_boost', 0)
        type_weight = MEMORY_TYPE_WEIGHTS.get(memory_type, 1.0)
        result['final_score'] = similarity * (1 + importance * importance_weight + (type_weight - 1) * type_weight_factor) + graph_boost
        result['type_weight'] = type_weight
        result['graph_boost'] = graph_boost

    # 排序和过滤
    results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    results = [r for r in results if r.get('similarity', 0) >= threshold]
    results = results[:top_k]

    # 格式化返回
    formatted_results = []
    for r in results:
        item = {
            "content": r.get('content', ''),
            "similarity": round(r.get('similarity', 0), 4),
            "importance": r.get('importance', 0.5),
            "memory_type": r.get('memory_type', 'general'),
            "tags": r.get('tags', []),
            "type_weight": r.get('type_weight', 1.0),
            "graph_boost": round(r.get('graph_boost', 0), 4),
            "final_score": round(r.get('final_score', 0), 4),
            "timestamp": r.get('created_at'),
            "created_at": r.get('created_at'),
            "updated_at": r.get('updated_at')
        }
        if enable_bm25:
            item["vector_score"] = round(r.get('vector_score', 0), 4)
            item["bm25_score"] = round(r.get('bm25_score', 0), 4)
            if r.get('bm25_only'):
                item["bm25_only"] = True
        formatted_results.append(item)

    if formatted_results:
        print(f"找到 {len(formatted_results)} 条相关记忆:")
        for i, mem in enumerate(formatted_results[:5]):
            content_preview = mem['content'][:50].replace('\n', ' ')
            type_label = {'preference': '偏好', 'fact': '事实', 'episodic': '情景',
                          'semantic': '语义', 'procedural': '程序性', 'general': '通用'}.get(mem['memory_type'], mem['memory_type'])
            graph_info = f"|图谱:{mem.get('graph_boost', 0):.2f}" if mem.get('graph_boost', 0) > 0 else ""
            bm25_info = ""
            if enable_bm25:
                bm25_info = f"|向量:{mem.get('vector_score', 0):.2f}|BM25:{mem.get('bm25_score', 0):.2f}"
                if mem.get('bm25_only'):
                    bm25_info += "(仅BM25)"
            print(f"   {i+1}. [{type_label}] {content_preview}...")
            print(f"      └─ 相似度:{mem['similarity']:.2f}{bm25_info} | 类型权重:{mem['type_weight']:.1f}x | 最终得分:{mem['final_score']:.2f}{graph_info}")
    else:
        print(f"未找到相关记忆")

    return {
        "query": request.query,
        "memories": formatted_results,
        "count": len(formatted_results)
    }

@router.get("/list")
async def list_memories(
    user_id: Optional[str] = None,
    limit: int = 100,
    registry: ServiceRegistry = Depends(get_registry)
    ):
    """列出记忆"""
    qdrant = registry.qdrant
    config = registry.config
    user_id = user_id if user_id is not None else config.users.default_user_id
    memories = []
    if qdrant and qdrant.is_available():
        memories = qdrant.get_all_memories(user_id=user_id, limit=limit)
    results = [
        {
            "id": mem.get('id', ''),
            "content": mem.get('content', ''),
            "timestamp": mem.get('created_at'),
            "created_at": mem.get('created_at'),
            "updated_at": mem.get('updated_at'),
            "importance": mem.get('importance', 0.5),
            "merge_count": mem.get('merge_count', 0),
            "memory_type": mem.get('memory_type', 'general'),
            "tags": mem.get('tags', [])
        }
        for mem in memories
    ]
    return {"user_id": user_id, "count": len(results), "memories": results}

@router.delete("/delete/{memory_id}")
async def delete_memory(
    memory_id: str,
    registry: ServiceRegistry = Depends(get_registry)
    ):
    """删除记忆"""
    qdrant = registry.qdrant
    if qdrant and qdrant.is_available():
        success = qdrant.delete_memory(memory_id)
        if success:
            return {"status": "success", "message": f"记忆 {memory_id} 已删除"}
    raise HTTPException(status_code=404, detail=f"记忆 {memory_id} 不存在")

@router.get("/memory-types")
async def get_memory_types():
    """获取记忆类型及其权重"""
    return {
        "types": MEMORY_TYPE_WEIGHTS,
        "description": {
            "episodic": "情景记忆 - 具体事件、对话、经历（有时间地点）",
            "semantic": "语义记忆 - 抽象知识、概念、事实（无具体时间）",
            "procedural": "程序记忆 - 技能、习惯、操作方式",
            "preference": "偏好记忆 - 用户喜好、厌恶",
            "fact": "事实记忆 - 客观事实信息",
            "tool": "工具记忆 - 工具使用记录",
            "event": "事件记忆 - 重要事件",
            "general": "通用记忆 - 未分类"
        }
    }

@router.get("/memories/by-type/{memory_type}")
async def get_memories_by_type(
    memory_type: str,
    user_id: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    registry: ServiceRegistry = Depends(get_registry)
):
    """按类型获取记忆"""
    qdrant = registry.qdrant
    config = registry.config
    user_id = user_id if user_id is not None else config.users.default_user_id
    if not qdrant or not qdrant.is_available():
        return {"memories": [], "message": "存储不可用"}
    valid_types = list(MEMORY_TYPE_WEIGHTS.keys())
    if memory_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"无效的记忆类型。有效类型: {valid_types}")
    all_memories = qdrant.get_all_memories(user_id=user_id, limit=limit * 3)
    filtered = [m for m in all_memories if m.get('memory_type') == memory_type][:limit]
    return {"memory_type": memory_type, "memories": filtered, "count": len(filtered)}

@router.post("/memories/classify")
async def classify_memory(
    content: str,
    registry: ServiceRegistry = Depends(get_registry)                
    ):
    """使用 LLM 对记忆内容进行类型分类"""
    llm_cfg = registry.config.llm.config
    if not llm_cfg.api_key or not llm_cfg.model or not llm_cfg.base_url:
        raise HTTPException(status_code=503, detail="LLM 未配置")
    import httpx
    prompt = f"""请对以下记忆内容进行分类，返回最合适的记忆类型。

记忆内容：{content}

可选类型：
- episodic: 情景记忆（具体事件、对话、经历，有时间地点）
- semantic: 语义记忆（抽象知识、用户属性，如"用户是医生"）
- procedural: 程序记忆（习惯、操作方式，如"用户习惯晚睡"）
- preference: 偏好记忆（喜好、厌恶，如"用户喜欢火锅"）
- fact: 事实记忆（客观事实）
- event: 事件记忆（重要事件）
- general: 通用记忆（无法分类）

请只返回一个类型名称（英文），不要其他内容。"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{llm_cfg.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {llm_cfg.api_key}"},
                json={
                    "model": llm_cfg.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.1
                }
            )
            if response.status_code == 200:
                result = response.json()
                classified_type = result['choices'][0]['message']['content'].strip().lower()
                if classified_type not in MEMORY_TYPE_WEIGHTS:
                    classified_type = "general"
                return {"content": content, "classified_type": classified_type, "type_weight": MEMORY_TYPE_WEIGHTS.get(classified_type, 1.0)}
            else:
                return {"content": content, "classified_type": "general", "error": "LLM 调用失败"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))