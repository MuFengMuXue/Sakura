# api/routes/memories.py
import asyncio
import re
import uuid
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Depends

from ..schemas import (
    AddMemoryRequest,
    AddRawMemoryRequest,
    SearchMemoryRequest,
)
from ..service_registry import ServiceRegistry
from ..dependencies import get_registry
from ..utils import (
    encode_text,
    extract_memories,
    update_bm25_index,
    remove_bm25_document,
    MEMORY_TYPE_WEIGHTS,
    MEMORY_LAYERS,
    normalize_context_summary,
    infer_memory_layer,
    normalize_layer,
    flatten_memory,
    update_memory_usage,
    payload_of,
    safe_float,
    recency_boost_for,   # 新增
    frequency_boost_for,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Memories"])


# ==================== 添加记忆（LLM 加工） ====================
@router.post("/add")
async def add_memory(
    request: AddMemoryRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    添加记忆（LLM 加工版）
    支持上下文摘要、分层存储、自动实体提取
    """
    embedder = registry.embedder
    qdrant = registry.qdrant
    entity_extractor = registry.entity_extractor
    graph = registry.graph
    config = registry.config

    if embedder is None:
        raise HTTPException(500, "Embedding 模型未加载")

    user_id = request.user_id or config.users.default_user_id

    added_count = 0
    merged_count = 0
    skipped_count = 0
    entity_count = 0
    added_memory_ids = []

    # ---------- 1. 合并对话 ----------
    conversation_parts = []
    for msg in request.messages:
        content = msg.get("content", "").strip()
        if not content:
            continue
        role = msg.get("role", "user")
        label = "主人" if role == "user" else "沐樱"
        conversation_parts.append(f"【{label}】{content}")

    if not conversation_parts:
        return {"status": "success", "message": "无有效对话", "added": 0}

    full_conversation = "\n".join(conversation_parts)

    # ---------- 2. 处理上下文摘要 ----------
    context_summary = normalize_context_summary(
        request.context_summary,
        request.history_summary,
        request.conversation_summary,
        request.compressed_context,
    )

    logger.info(f"处理对话（{len(request.messages)} 条消息）")
    if context_summary:
        logger.info(f"附带历史摘要（{len(context_summary)} 字符）")

    # ---------- 3. LLM 提取记忆 ----------
    extracted = await extract_memories(
        conversation=full_conversation,
        registry=registry,
        context_summary=context_summary,
    )

    memories = extracted.get("memories", [])
    if not memories:
        logger.info("未提取到有效记忆")
        return {
            "status": "success",
            "message": "未提取到有效记忆",
            "added": 0,
            "merged": 0,
            "skipped": 0,
            "preferences_extracted": 0,
            "entities_extracted": 0,
        }

    # ---------- 4. 处理每条提取的记忆 ----------
    for mem_item in memories:
        content = mem_item.get("content", "").strip()
        if len(content) < 5:
            continue

        importance = safe_float(mem_item.get("importance"), 0.5)
        if importance < 0.3:
            skipped_count += 1
            logger.info(f"跳过低重要度记忆: {content[:30]}... (重要度:{importance:.0%})")
            continue

        memory_type = mem_item.get("memory_type", "general")
        if memory_type not in MEMORY_TYPE_WEIGHTS:
            memory_type = "general"

        tags = mem_item.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        # 生成向量
        vector = await encode_text(content, embedder)

        # 去重（相似度 >= 0.95 视为重复，合并）
        if qdrant and qdrant.is_available():
            similar = await qdrant.find_similar(
                vector, threshold=0.95, user_id=user_id
            )
            if similar:
                await qdrant.update_memory(
                    similar["id"],
                    {
                        "merge_count": payload_of(similar).get("merge_count", 0) + 1,
                        "importance": max(importance, similar.get("importance", 0.5)),
                    },
                )
                merged_count += 1
                logger.info(f"合并重复记忆: {content[:30]}...")
                continue

        # 推断生命周期层
        layer = infer_memory_layer(
            memory_type=memory_type,
            explicit_layer=mem_item.get("layer"),
        )

        # 构造 payload
        memory_id = str(uuid.uuid4())
        payload = {
            "content": content,
            "user_id": user_id,
            "importance": importance,
            "memory_type": memory_type,
            "layer": layer,
            "tags": tags,
            "created_at": datetime.now().isoformat(),
            "merge_count": 0,
            "processed": True,
        }

        # 写入 Qdrant
        if qdrant and qdrant.is_available():
            await qdrant.add_memory(memory_id, vector, payload)
            await update_bm25_index(memory_id, content, registry)

        added_count += 1
        added_memory_ids.append(memory_id)

        type_label = {
            'preference': '偏好',
            'fact': '事实',
            'episodic': '情景',
            'semantic': '语义',
            'procedural': '程序性',
            'general': '通用'
        }.get(memory_type, memory_type)
        tags_str = f" 标签:{tags}" if tags else ""
        logger.info(
            f"新增记忆: {content[:60]}{'...' if len(content) > 60 else ''} "
            f"(类型:{type_label}, 层:{layer}, 重要度:{importance:.0%}{tags_str})"
        )

    # ---------- 5. 自动提取实体 ----------
    if entity_extractor and graph and graph.is_available() and added_memory_ids:
        try:
            logger.info("正在从对话中提取知识图谱实体...")
            entities, relations = await entity_extractor.extract(full_conversation)

            if entities:
                logger.info(f"发现 {len(entities)} 个实体, {len(relations) if relations else 0} 个关系")
                for ent in entities:
                    try:
                        ent_name = ent.name if hasattr(ent, "name") else str(ent)
                        ent_type = ent.entity_type.value if hasattr(ent, "entity_type") else "unknown"
                        desc = ent.description if hasattr(ent, "description") else ""

                        existing = graph.find_entity_by_name(ent_name, user_id)
                        if existing:
                            for mid in added_memory_ids:
                                if hasattr(graph, "link_entity_to_memory"):
                                    graph.link_entity_to_memory(existing["id"], mid)
                            logger.info(f"关联已有实体: {ent_name} [{ent_type}]")
                        else:
                            new_id = str(uuid.uuid4())
                            success = graph.create_entity(
                                entity_id=new_id,
                                name=ent_name,
                                entity_type=ent_type,
                                user_id=user_id,
                                properties={
                                    "description": desc,
                                    "source_memory_ids": added_memory_ids,
                                },
                            )
                            if success:
                                entity_count += 1
                                logger.info(f"创建实体: {ent_name} [{ent_type}]")
                    except Exception as e:
                        logger.warning(f"保存实体失败: {e}")

                if entity_count > 0:
                    logger.info(f"成功保存 {entity_count} 个实体")
                else:
                    logger.info("未发现新实体")
            else:
                logger.info("未发现新实体")

            # 保存关系
            if relations:
                for rel in relations:
                    try:
                        src_name = getattr(rel, "source_name", "")
                        tgt_name = getattr(rel, "target_name", "")
                        rel_type = (
                            rel.relation_type.value
                            if hasattr(rel, "relation_type")
                            else "related_to"
                        )
                        desc = getattr(rel, "description", "")

                        src_ent = graph.find_entity_by_name(src_name, user_id)
                        tgt_ent = graph.find_entity_by_name(tgt_name, user_id)
                        if src_ent and tgt_ent:
                            graph.create_relation(
                                source_id=src_ent["id"],
                                target_id=tgt_ent["id"],
                                relation_type=rel_type,
                                properties={
                                    "description": desc,
                                    "source_memory_id": added_memory_ids[0],
                                },
                            )
                            logger.info(f"创建关系: {src_name} --[{rel_type}]--> {tgt_name}")
                    except Exception as e:
                        logger.warning(f"保存关系失败: {e}")
        except Exception as e:
            logger.warning(f"实体提取失败: {e}")

    # ---------- 6. 返回结果 ----------
    parts = []
    if added_count:
        parts.append(f"新增记忆 {added_count} 条")
    if merged_count:
        parts.append(f"合并 {merged_count} 条")
    if skipped_count:
        parts.append(f"跳过 {skipped_count} 条")
    if entity_count:
        parts.append(f"提取实体 {entity_count} 个")

    message = "、".join(parts) if parts else "无有效记忆"

    logger.info(f"记忆总结完成: {message}")

    return {
        "status": "success",
        "message": message,
        "added": added_count,
        "merged": merged_count,
        "skipped": skipped_count,
        "preferences_extracted": 0,
        "entities_extracted": entity_count,
    }


# ==================== 直接添加记忆（不加工） ====================
@router.post("/add_raw")
async def add_memory_raw(
    request: AddRawMemoryRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """直接添加记忆（不经过 LLM 加工），支持指定类型、标签、生命周期层"""
    embedder = registry.embedder
    qdrant = registry.qdrant
    graph = registry.graph
    entity_extractor = registry.entity_extractor
    config = registry.config

    if embedder is None:
        raise HTTPException(500, "Embedding 模型未加载")

    user_id = request.user_id or config.users.default_user_id
    added_count = 0
    type_counts = {}
    extracted_entities = []

    for msg in request.messages:
        content = msg.content.strip()
        if len(content) < 5:
            continue

        importance = msg.importance if msg.importance is not None else 0.8
        memory_type = msg.memory_type or "general"
        if memory_type not in MEMORY_TYPE_WEIGHTS:
            memory_type = "general"
        tags = msg.tags or []

        # 推断生命周期层
        layer = infer_memory_layer(
            memory_type=memory_type,
            explicit_layer=msg.layer,
        )

        vector = await encode_text(content, registry.embedder)

        # 去重
        if qdrant and qdrant.is_available():
            similar = await qdrant.find_similar(vector, threshold=0.95, user_id=user_id)
            if similar:
                # 重复则跳过（可改为合并，此处保留简单跳过）
                continue

        memory_id = str(uuid.uuid4())
        payload = {
            "content": content,
            "user_id": user_id,
            "importance": importance,
            "memory_type": memory_type,
            "layer": layer,
            "tags": tags,
            "created_at": datetime.now().isoformat(),
            "merge_count": 0,
            "processed": False,
        }

        # 实体提取（若启用）
        entity_ids = []
        if request.extract_entities and entity_extractor and graph and graph.is_available():
            try:
                entities, relations = await entity_extractor.extract(content)
                name_to_id = {}
                for ent in entities:
                    ent_name = ent.name if hasattr(ent, "name") else str(ent)
                    ent_type = ent.entity_type.value if hasattr(ent, "entity_type") else "unknown"
                    desc = ent.description if hasattr(ent, "description") else ""
                    conf = ent.confidence if hasattr(ent, "confidence") else 0.8

                    existing = graph.find_entity_by_name(ent_name, user_id)
                    if existing:
                        eid = existing["id"]
                        if hasattr(graph, "link_entity_to_memory"):
                            graph.link_entity_to_memory(eid, memory_id)
                    else:
                        eid = str(uuid.uuid4())
                        graph.add_entity(
                            entity_id=eid,
                            entity_type=ent_type,
                            name=ent_name,
                            properties={
                                "description": desc,
                                "confidence": conf,
                                "source_memory_ids": [memory_id],
                            },
                            user_id=user_id,
                        )
                    name_to_id[ent_name] = eid
                    entity_ids.append(eid)

                for rel in relations or []:
                    src = getattr(rel, "source_name", "")
                    tgt = getattr(rel, "target_name", "")
                    rtype = (
                        rel.relation_type.value
                        if hasattr(rel, "relation_type")
                        else "related_to"
                    )
                    desc = getattr(rel, "description", "")
                    conf = getattr(rel, "confidence", 0.8)
                    sid = name_to_id.get(src)
                    tid = name_to_id.get(tgt)
                    if sid and tid:
                        graph.add_relation(
                            source_id=sid,
                            target_id=tid,
                            relation_type=rtype,
                            properties={
                                "description": desc,
                                "confidence": conf,
                                "source_memory_id": memory_id,
                            },
                        )

                extracted_entities.extend(
                    {"id": eid, "name": name} for name, eid in name_to_id.items()
                )
            except Exception as e:
                logger.warning(f"实体提取失败: {e}")

        if entity_ids:
            payload["entity_ids"] = entity_ids

        if qdrant and qdrant.is_available():
            await qdrant.add_memory(memory_id, vector, payload)
            await update_bm25_index(memory_id, content, registry)

        added_count += 1
        type_counts[memory_type] = type_counts.get(memory_type, 0) + 1

    result = {
        "status": "success",
        "message": f"直接添加 {added_count} 条记忆",
        "added": added_count,
        "type_breakdown": type_counts,
    }
    if extracted_entities:
        result["extracted_entities"] = extracted_entities
        result["entity_count"] = len(extracted_entities)
    return result


# ==================== 搜索记忆 ====================
@router.post("/search")
async def search_memory(
    request: SearchMemoryRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    混合搜索（向量 + BM25 + 图增强），支持分层过滤、访问计数自动更新、CrossEncoder 精排
    返回完整评分详情，便于前端展示和调试
    """
    embedder = registry.embedder
    qdrant = registry.qdrant
    bm25 = registry.bm25
    graph = registry.graph
    reranker = registry.reranker
    config = registry.config

    if embedder is None:
        raise HTTPException(500, "Embedding 模型未加载")

    user_id = request.user_id or config.users.default_user_id

    # 读取请求参数
    top_k = request.top_k or config.search.top_k
    threshold = request.similarity_threshold or config.search.similarity_threshold
    enable_bm25 = (
        request.use_bm25
        if request.use_bm25 is not None
        else config.search.enable_bm25
    )
    enable_graph = (
        request.use_graph
        if request.use_graph is not None
        else config.search.enable_graph_query
    )

    # 解析生命周期层过滤
    requested_layers = request.layers or MEMORY_LAYERS
    requested_layers = [
        normalize_layer(layer, default="LongTermMemory")
        for layer in requested_layers
        if layer in MEMORY_LAYERS
    ] or MEMORY_LAYERS

    logger.info(f"搜索: {request.query[:80]}{'…' if len(request.query) > 80 else ''}")
    logger.info(f"  层过滤: {requested_layers}, BM25={enable_bm25}, 图={enable_graph}")

    query_vector = await encode_text(request.query, embedder)

    # ---------- 1. 向量搜索 ----------
    results_map = {}
    if qdrant and qdrant.is_available():
        recall_k = max(top_k * 3, 8)
        for layer in requested_layers:
            layer_results = await qdrant.search(
                query_vector=query_vector,
                top_k=recall_k,
                score_threshold=threshold,
                user_id=user_id,
                memory_type=request.memory_types[0] if request.memory_types and len(request.memory_types) == 1 else None,
                memory_types=request.memory_types if request.memory_types and len(request.memory_types) > 1 else None,
                tags=request.tags,
                layer=layer,
            )
            for r in layer_results:
                rid = r.get("id")
                if rid:
                    if rid not in results_map:
                        results_map[rid] = {"data": r, "scores": {"vector": 0, "bm25": 0}}
                    vec_score = r.get("similarity", 0)
                    if vec_score > results_map[rid]["scores"]["vector"]:
                        results_map[rid]["scores"]["vector"] = vec_score
                    results_map[rid]["data"].update(r)

        # 兼容缺 layer 的旧数据
        if 'LongTermMemory' in requested_layers:
            legacy_results = await qdrant.search(
                query_vector=query_vector,
                top_k=recall_k,
                score_threshold=threshold,
                user_id=user_id,
                memory_type=request.memory_types[0] if request.memory_types and len(request.memory_types) == 1 else None,
                memory_types=request.memory_types if request.memory_types and len(request.memory_types) > 1 else None,
                tags=request.tags,
            )
            for r in legacy_results:
                payload = payload_of(r)
                if payload.get('layer'):
                    continue
                rid = r.get("id")
                if rid:
                    r['layer'] = 'LongTermMemory'
                    if rid not in results_map:
                        results_map[rid] = {"data": r, "scores": {"vector": 0, "bm25": 0}}
                    vec_score = r.get("similarity", 0)
                    if vec_score > results_map[rid]["scores"]["vector"]:
                        results_map[rid]["scores"]["vector"] = vec_score
                    results_map[rid]["data"].update(r)

    # ---------- 2. BM25 ----------
    if enable_bm25 and bm25:
        try:
            bm25_hits = bm25.search(request.query, top_k=top_k * 3)
            if bm25_hits:
                max_score = max(score for _, score in bm25_hits) or 1.0
                for doc_id, score in bm25_hits:
                    norm_score = score / max_score
                    if doc_id in results_map:
                        results_map[doc_id]["scores"]["bm25"] = norm_score
                    else:
                        memory = await qdrant.get_memory(doc_id)
                        if memory:
                            payload = payload_of(memory)
                            if payload.get("user_id") == user_id:
                                mem_layer = normalize_layer(
                                    payload.get("layer"), default="LongTermMemory"
                                )
                                if mem_layer not in requested_layers:
                                    continue
                                if request.tags and not any(t in payload.get('tags', []) for t in request.tags):
                                    continue
                                if request.memory_types and payload.get('memory_type', 'general') not in request.memory_types:
                                    continue
                                data = {
                                    "id": doc_id,
                                    "content": payload.get("content", memory.get("content", "")),
                                    "similarity": 0,
                                    "importance": payload.get("importance", memory.get("importance", 0.5)),
                                    "memory_type": payload.get("memory_type", memory.get("memory_type", "general")),
                                    "layer": mem_layer,
                                    "status": payload.get('status', 'active'),
                                    "access_count": payload.get('access_count', 0),
                                    "last_accessed_at": payload.get('last_accessed_at'),
                                    "tags": payload.get("tags", memory.get("tags", [])),
                                    "created_at": payload.get("created_at", memory.get("created_at")),
                                    "updated_at": payload.get("updated_at", memory.get("updated_at")),
                                    "entity_ids": payload.get("entity_ids", memory.get("entity_ids", [])),
                                    "payload": payload,
                                    "bm25_only": True,
                                }
                                results_map[doc_id] = {
                                    "data": data,
                                    "scores": {"vector": 0, "bm25": norm_score},
                                }
            logger.info(f"BM25 找到 {len(bm25_hits)} 条候选")
        except Exception as e:
            logger.warning(f"BM25 搜索失败: {e}")

    # ---------- 3. 合并结果 ----------
    bm25_weight = config.search.bm25_weight
    results = []
    for rid, item in results_map.items():
        data = item["data"].copy()
        vec_score = item["scores"]["vector"]
        bm25_score = item["scores"]["bm25"]

        if enable_bm25:
            if vec_score > 0 and bm25_score > 0:
                mixed = vec_score + bm25_weight * bm25_score
            elif vec_score > 0:
                mixed = vec_score
            else:
                mixed = bm25_weight * bm25_score
            data["similarity"] = mixed
            data["vector_score"] = vec_score
            data["bm25_score"] = bm25_score
        else:
            data["similarity"] = vec_score

        if request.tags and not any(t in data.get("tags", []) for t in request.tags):
            continue
        if request.memory_types and data.get("memory_type") not in request.memory_types:
            continue
        mem_layer = normalize_layer(
            data.get("layer") or payload_of(data).get("layer"),
            default="LongTermMemory",
        )
        if mem_layer not in requested_layers:
            continue
        results.append(data)

    # ---------- 4. 图增强 ----------
    matched_entities_info = []
    graph_paths = []
    if enable_graph and graph and graph.is_available() and results:
        try:
            import re
            potential_names = re.findall(r"[\u4e00-\u9fff]{2,4}", request.query)
            potential_names += re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", request.query)
            logger.info(f"图增强: 提取到 {len(potential_names)} 个候选实体")

            matched_ids = set()
            for name in potential_names[:10]:
                ent = graph.find_entity_by_name(name, user_id)
                if ent:
                    matched_ids.add(ent["id"])
                    matched_entities_info.append({
                        'id': ent["id"],
                        'name': ent.get('name'),
                        'entity_type': ent.get('entity_type'),
                        'matched_text': name,
                    })
                    related = graph.find_related_entities(ent["id"], max_depth=2)
                    for rel in related:
                        if rel['id'] not in matched_ids:
                            matched_ids.add(rel['id'])
                            matched_entities_info.append({
                                'id': rel['id'],
                                'name': rel.get('name'),
                                'entity_type': rel.get('entity_type'),
                                'matched_text': f"related_to_{name}",
                            })

            if matched_ids:
                graph_mem_ids = set()
                for eid in matched_ids:
                    mids = graph.get_entity_memories(eid) if hasattr(graph, "get_entity_memories") else []
                    graph_mem_ids.update(mids)

                logger.info(f"图增强: 匹配到 {len(matched_ids)} 个实体, {len(graph_mem_ids)} 条关联记忆")

                # 对已有结果加 graph_boost
                for r in results:
                    r_entities = r.get("entity_ids", [])
                    r_id = r.get("id")
                    if r_id in graph_mem_ids:
                        r["graph_boost"] = 0.15
                        r["graph_boost_reason"] = "matched_entity_memory"
                    elif any(eid in r_entities for eid in matched_ids):
                        r["graph_boost"] = 0.1
                        r["graph_boost_reason"] = "result_entity_overlap"
                    else:
                        r.setdefault("graph_boost", 0)

                # 补充图关联但向量未命中的记忆
                existing_ids = {r.get("id") for r in results if r.get("id")}
                graph_only_count = 0
                for mid in list(graph_mem_ids)[:5]:
                    if mid not in existing_ids:
                        mem = await qdrant.get_memory(mid)
                        if mem and payload_of(mem).get("user_id") == user_id:
                            flat = flatten_memory(mem)
                            flat["similarity"] = 0.5
                            flat["graph_boost"] = 0.2
                            flat["graph_boost_reason"] = "graph_only_entity_memory"
                            flat["matched_entities"] = matched_entities_info
                            flat["graph_only"] = True
                            results.append(flat)
                            graph_only_count += 1

                # 查找实体间路径
                if len(matched_ids) >= 2 and hasattr(graph, 'find_path'):
                    for idx, source_id in enumerate(list(matched_ids)[:3]):
                        for target_id in list(matched_ids)[idx+1:4]:
                            path = graph.find_path(source_id, target_id, max_length=3)
                            if path:
                                graph_paths.append(path)

                logger.info(f"图增强: {len(matched_ids)} 个实体, {graph_only_count} 条仅图谱记忆")
            else:
                logger.info("图增强: 未匹配到实体")
        except Exception as e:
            logger.warning(f"图增强失败: {e}")

    # ---------- 5. 多维加权 ----------
    importance_weight = config.search.importance_weight
    type_weight_factor = config.search.type_weight_factor
    layer_weights = getattr(config.search, "layer_weights", {})
    recency_weight = getattr(config.search, "recency_weight", 0)
    frequency_weight = getattr(config.search, "frequency_weight", 0)

    for r in results:
        pl = payload_of(r)
        similarity = r.get("similarity", 0)
        importance = safe_float(r.get("importance", pl.get("importance", 0.5)), 0.5)
        memory_type = r.get("memory_type", pl.get("memory_type", "general"))
        layer = normalize_layer(r.get("layer") or pl.get("layer"), default="LongTermMemory")
        graph_boost = safe_float(r.get("graph_boost", 0), 0)

        type_weight = MEMORY_TYPE_WEIGHTS.get(memory_type, 1.0)
        layer_boost = safe_float(layer_weights.get(layer, 0), 0)
        recency_boost = recency_boost_for(pl, recency_weight)
        freq_boost = frequency_boost_for(pl, frequency_weight)

        r["memory_type"] = memory_type
        r["layer"] = layer
        r["type_weight"] = type_weight
        r["layer_boost"] = layer_boost
        r["recency_boost"] = recency_boost
        r["frequency_boost"] = freq_boost
        r["graph_boost"] = graph_boost
        if matched_entities_info:
            r["matched_entities"] = matched_entities_info
        if graph_paths:
            r["graph_paths"] = graph_paths

        r["final_score"] = similarity * (
            1
            + importance * importance_weight
            + (type_weight - 1) * type_weight_factor
            + layer_boost
            + recency_boost
            + freq_boost
        ) + graph_boost

        r["coarse_score"] = r["final_score"]
        r["score_breakdown"] = {
            "mixed_similarity": round(similarity, 4),
            "importance": round(importance * importance_weight, 4),
            "type": round((type_weight - 1) * type_weight_factor, 4),
            "layer": round(layer_boost, 4),
            "recency": round(recency_boost, 4),
            "frequency": round(freq_boost, 4),
            "graph": round(graph_boost, 4),
        }

    # ---------- 6. 过滤与排序 ----------
    results = [r for r in results if r.get("similarity", 0) >= threshold]
    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    results = results[:top_k * 2]

    # ---------- 7. CrossEncoder 精排 ----------
    enable_reranker = getattr(config.search, 'enable_reranker', False)
    reranker_used = False

    if enable_reranker and reranker and results:
        try:
            rerank_top_n = getattr(config.search, 'rerank_top_n', 20)
            candidates = results[:max(rerank_top_n, top_k)]
            docs = [r.get('content', '') for r in candidates]

            rerank_results = await reranker.rerank(
                query=request.query,
                documents=docs,
                top_n=top_k,
            )

            reranked = []
            for item in rerank_results:
                idx = item.get('index')
                score = item.get('relevance_score', 0.0)
                if idx < len(candidates):
                    candidate = candidates[idx].copy()
                    candidate['rerank_score'] = score
                    candidate['final_score'] = score
                    candidate['coarse_score'] = candidate.get('coarse_score', candidate.get('final_score', 0))
                    reranked.append(candidate)

            if reranked:
                results = reranked
                reranker_used = True
                logger.info(f"重排序完成，返回 {len(results)} 条结果")
            else:
                results = results[:top_k]
        except Exception as e:
            logger.warning(f"重排序失败，回退粗排: {e}")
            results = results[:top_k]
    else:
        results = results[:top_k]
        if enable_reranker and not reranker:
            logger.debug("reranker 未初始化或不可用，跳过精排")

    # ---------- 8. 格式化输出 ----------
    formatted_results = []
    for r in results:
        pl = payload_of(r)

        created_at = (
            r.get('created_at')
            or r.get('timestamp')
            or pl.get('created_at')
            or pl.get('timestamp')
        )
        updated_at = r.get('updated_at') or pl.get('updated_at')

        _tags = r.get('tags')
        if not isinstance(_tags, list):
            _tags = pl.get('tags', [])

        item = {
            "id": r.get('id'),
            "content": r.get('content') or pl.get('content', ''),
            "similarity": round(r.get('similarity', 0), 4),
            "importance": r.get('importance', pl.get('importance', 0.5)),
            "memory_type": r.get('memory_type', pl.get('memory_type', 'general')),
            "layer": r.get('layer', pl.get('layer', 'LongTermMemory')),
            "status": r.get('status', pl.get('status', 'active')),
            "access_count": r.get('access_count', pl.get('access_count', 0)),
            "last_accessed_at": r.get('last_accessed_at', pl.get('last_accessed_at')),
            "tags": _tags,
            "type_weight": r.get('type_weight', 1.0),
            "layer_boost": round(r.get('layer_boost', 0), 4),
            "recency_boost": round(r.get('recency_boost', 0), 4),
            "frequency_boost": round(r.get('frequency_boost', 0), 4),
            "graph_boost": round(r.get('graph_boost', 0), 4),
            "graph_boost_reason": r.get('graph_boost_reason'),
            "matched_entities": r.get('matched_entities', []),
            "graph_paths": r.get('graph_paths', []),
            "final_score": round(r.get('final_score', 0), 4),
            "coarse_score": round(r.get('coarse_score', r.get('final_score', 0)), 4),
            "rerank_score": round(r.get('rerank_score'), 4) if r.get('rerank_score') is not None else None,
            "score_breakdown": r.get('score_breakdown', {}),
            "source_type": r.get('source_type') or pl.get('source_type'),
            "source": r.get('source') or pl.get('source'),
            "timestamp": created_at,
            "created_at": created_at,
            "updated_at": updated_at,
        }

        if enable_bm25:
            item["vector_score"] = round(r.get('vector_score', 0), 4)
            item["bm25_score"] = round(r.get('bm25_score', 0), 4)
            if r.get('bm25_only'):
                item["bm25_only"] = True

        formatted_results.append(item)

    # 异步更新访问计数
    if formatted_results:
        asyncio.create_task(
            update_memory_usage([m.get("id") for m in formatted_results if m.get("id")], registry)
        )

    logger.info(f"返回 {len(formatted_results)} 条记忆")
    return {
        "query": request.query,
        "memories": formatted_results,
        "count": len(formatted_results),
        "layers": requested_layers,
        "reranker_used": reranker_used,
    }
# ==================== 列出记忆 ====================
@router.get("/list")
async def list_memories(
    user_id: Optional[str] = None,
    limit: int = Query(100, ge=0, description="返回数量，0 表示不限制"),
    status: Optional[str] = Query(None, description="状态过滤: active/archived/deleted"),
    layer: Optional[str] = Query(None, description="生命周期层过滤"),
    include_deleted: bool = Query(False),
    registry: ServiceRegistry = Depends(get_registry),
):
    """列出记忆，支持状态和层过滤，结果自动展平"""
    qdrant = registry.qdrant
    config = registry.config
    if not qdrant or not qdrant.is_available():
        return {"memories": [], "count": 0}

    user_id = user_id or config.users.default_user_id
    normalized_layer = normalize_layer(layer, default="LongTermMemory") if layer else None
    include_archived = status == "archived"
    include_deleted_effective = include_deleted or status == "deleted"

    fetch_limit = None if limit == 0 else limit
    raw_memories = await qdrant.get_all_memories(
        user_id=user_id,
        limit=fetch_limit,
        include_deleted=include_deleted_effective,
        include_archived=include_archived,
        status=status,
        layer=normalized_layer,
    )

    flat_memories = [flatten_memory(m) for m in raw_memories]
    return {
        "user_id": user_id,
        "count": len(flat_memories),
        "memories": flat_memories,
    }


# ==================== 删除记忆 ====================
@router.delete("/delete/{memory_id}")
async def delete_memory(
    memory_id: str,
    hard: bool = Query(False, description="物理删除（不可恢复）"),
    reason: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """删除记忆，默认软删除，同时从 BM25 索引移除"""
    qdrant = registry.qdrant
    if not qdrant or not qdrant.is_available():
        raise HTTPException(503, "存储不可用")

    if hard:
        success = await qdrant.delete_memory(memory_id)
    elif hasattr(qdrant, "soft_delete_memory"):
        success = await qdrant.soft_delete_memory(memory_id, reason=reason)
    else:
        success = await qdrant.delete_memory(memory_id)

    if success:
        await remove_bm25_document(memory_id, registry)
        return {
            "status": "success",
            "message": f"记忆 {memory_id} 已删除",
            "hard": hard,
        }
    raise HTTPException(404, f"记忆 {memory_id} 不存在")


# ==================== 记忆类型信息 ====================
@router.get("/memory-types")
async def get_memory_types():
    """获取所有记忆类型及其权重"""
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
            "general": "通用记忆 - 未分类",
        },
    }


@router.get("/memories/by-type/{memory_type}")
async def get_memories_by_type(
    memory_type: str,
    user_id: Optional[str] = None,
    limit: int = Query(50, le=200),
    registry: ServiceRegistry = Depends(get_registry),
):
    """按类型获取记忆"""
    qdrant = registry.qdrant
    config = registry.config
    if not qdrant or not qdrant.is_available():
        return {"memories": [], "message": "存储不可用"}

    if memory_type not in MEMORY_TYPE_WEIGHTS:
        raise HTTPException(
            400, f"无效的记忆类型。有效类型: {list(MEMORY_TYPE_WEIGHTS.keys())}"
        )

    user_id = user_id or config.users.default_user_id
    raw = await qdrant.get_all_memories(
        user_id=user_id, memory_type=memory_type, limit=limit * 3
    )
    # 二次过滤确保类型准确（因为后端可能不支持类型过滤）
    filtered = [m for m in raw if m.get("memory_type") == memory_type][:limit]
    return {"memory_type": memory_type, "memories": filtered, "count": len(filtered)}


@router.post("/memories/classify")
async def classify_memory(
    content: str,
    registry: ServiceRegistry = Depends(get_registry),
):
    """使用 LLM 对单条记忆内容进行类型分类"""
    llm_cfg = registry.config.llm
    if not llm_cfg.api_key or not llm_cfg.model or not llm_cfg.base_url:
        raise HTTPException(503, "LLM 未配置")

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
            resp = await client.post(
                f"{llm_cfg.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {llm_cfg.api_key}"},
                json={
                    "model": llm_cfg.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.1,
                },
            )
            if resp.status_code == 200:
                result = resp.json()
                classified = result["choices"][0]["message"]["content"].strip().lower()
                if classified not in MEMORY_TYPE_WEIGHTS:
                    classified = "general"
                return {
                    "content": content,
                    "classified_type": classified,
                    "type_weight": MEMORY_TYPE_WEIGHTS.get(classified, 1.0),
                }
            return {"content": content, "classified_type": "general", "error": "LLM 调用失败"}
    except Exception as e:
        raise HTTPException(500, str(e))