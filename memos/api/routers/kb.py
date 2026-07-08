# api/routes/kb.py
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
import uuid
from datetime import datetime
import hashlib

from ..schemas import ImportDocumentRequest, ImportBatchRequest
from ..service_registry import ServiceRegistry
from ..utils import encode_text, update_bm25_index
from ..dependencies import get_registry

import logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["KnowledgeBase"])


def _document_title(source: str) -> str:
    """从来源路径提取标题"""
    if source.startswith(('http://', 'https://')):
        return source.rstrip('/').split('/')[-1] or source
    return source.split('/')[-1] or source


@router.post("/kb/import")
async def import_document(
    request: ImportDocumentRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    导入文档到知识库

    支持：
    - 文本文件 (.txt)
    - PDF 文件 (.pdf)
    - Markdown 文件 (.md)
    - 网页 URL (http/https)
    """
    document_loader = registry.document_loader
    qdrant = registry.qdrant
    config = registry.config

    if not document_loader:
        raise HTTPException(status_code=503, detail="文档加载器未初始化")

    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")

    user_id = request.user_id if request.user_id is not None else config.users.default_user_id
    tags = request.tags or []
    kb_id = request.kb_id or "default"
    doc_id = request.doc_id or f"doc_{uuid.uuid4().hex[:12]}"

    try:
        # 加载文档
        chunks = document_loader.load(request.source)

        if not chunks:
            return {
                "status": "failed",
                "message": f"无法加载文档: {request.source}",
                "chunks_count": 0
            }

        # 计算校验和
        hasher = hashlib.sha256()
        for chunk in chunks:
            hasher.update((chunk.content or '').encode('utf-8', errors='ignore'))
        checksum = hasher.hexdigest()

        title = request.title or _document_title(request.source)
        imported_at = datetime.now().isoformat()

        imported_count = 0
        memory_ids = []
        extracted_entity_count = 0

        for chunk in chunks:
            content = chunk.content
            if not content or len(content) < 10:
                continue

            vector = await encode_text(content, registry.embedder)
            memory_id = str(uuid.uuid4())

            payload = {
                'content': content,
                'user_id': user_id,
                'importance': 0.6,
                'memory_type': 'document',
                'scope': 'kb',
                'kb_id': kb_id,
                'doc_id': doc_id,
                'source_uri': request.source,
                'title': title,
                'checksum': checksum,
                'chunk_count': len(chunks),
                'imported_at': imported_at,
                'tags': tags + [chunk.metadata.get('type', 'document')],
                'source': chunk.source,
                'source_type': chunk.metadata.get('type'),
                'chunk_index': chunk.chunk_index,
                'created_at': datetime.now().isoformat()
            }

            # 实体提取（如果启用）
            if request.extract_entities:
                try:
                    from ..utils import store_entities_for_memory
                    graph_result = await store_entities_for_memory(
                        text=content,
                        memory_id=memory_id,
                        user_id=user_id,
                        context=f"KB:{title}",
                        registry=registry
                    )
                    payload['entity_ids'] = graph_result.get('entity_ids', [])
                    extracted_entity_count += len(graph_result.get('entity_ids', []))
                except Exception as e:
                    logger.warning(f"KB 实体提取失败: {e}")

            await qdrant.add_memory(memory_id, vector, payload)
            # 更新 BM25 索引
            await update_bm25_index(memory_id, content, registry)

            memory_ids.append(memory_id)
            imported_count += 1

        return {
            "status": "success",
            "source": request.source,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "title": title,
            "checksum": checksum,
            "chunks_count": len(chunks),
            "imported_count": imported_count,
            "entities_extracted": extracted_entity_count,
            "memory_ids": memory_ids[:10],
            "total_memory_ids": len(memory_ids)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kb/import/batch")
async def import_documents_batch(
    request: ImportBatchRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """批量导入文档"""
    results = []
    total_imported = 0
    total_failed = 0

    for source in request.sources:
        try:
            single_request = ImportDocumentRequest(
                source=source,
                tags=request.tags,
                extract_entities=request.extract_entities,
                user_id=request.user_id,
                kb_id=getattr(request, 'kb_id', "default"),
            )
            result = await import_document(single_request, registry)  # 传递 registry
            results.append(result)

            if result.get('status') == 'success':
                total_imported += result.get('imported_count', 0)
            else:
                total_failed += 1
        except Exception as e:
            results.append({
                "source": source,
                "status": "failed",
                "error": str(e)
            })
            total_failed += 1

    return {
        "total_sources": len(request.sources),
        "total_imported": total_imported,
        "total_failed": total_failed,
        "details": results
    }


@router.post("/kb/import/url")
async def import_from_url(
    url: str,
    tags: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """从 URL 导入网页内容"""
    request = ImportDocumentRequest(
        source=url,
        tags=tags or ['web'],
        user_id=user_id,
        extract_entities=False,
    )
    return await import_document(request, registry)  # 传递 registry