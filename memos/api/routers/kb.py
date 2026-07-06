# api/routes/kb.py
from fastapi import APIRouter, HTTPException, Query,Depends
from typing import Optional, List
import uuid
from datetime import datetime

from ..schemas import ImportDocumentRequest, ImportBatchRequest
from ..service_registry import ServiceRegistry
from ..utils import encode_text
from ..dependencies import get_registry

router = APIRouter(tags=["KnowledgeBase"])

@router.post("/kb/import")
async def import_document(request: ImportDocumentRequest,registry: ServiceRegistry = Depends(get_registry),):
    """导入文档到知识库
    
    支持：
    - 文本文件 (.txt)
    - PDF 文件 (.pdf)
    - Markdown 文件 (.md)
    - 网页 URL (http/https)
    """
    document_loader = registry.document_loader
    qdrant = registry.qdrant
    
    if not document_loader:
        raise HTTPException(status_code=503, detail="文档加载器未初始化")
    
    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")
    
    user_id = request.user_id or registry.config.users.default_user_id
    tags = request.tags or []
    
    try:
        # 加载文档
        chunks = document_loader.load(request.source)
        
        if not chunks:
            return {
                "status": "failed",
                "message": f"无法加载文档: {request.source}",
                "chunks_count": 0
            }
        
        imported_count = 0
        memory_ids = []
        
        for chunk in chunks:
            content = chunk.content
            if not content or len(content) < 10:
                continue
            
            vector = await encode_text(content,registry)
            memory_id = str(uuid.uuid4())
            payload = {
                'content': content,
                'user_id': user_id,
                'importance': 0.6,
                'memory_type': 'document',
                'tags': tags + [chunk.metadata.get('type', 'document')],
                'source': chunk.source,
                'source_type': chunk.metadata.get('type'),
                'chunk_index': chunk.chunk_index,
                'created_at': datetime.now().isoformat()
            }
            
            await qdrant.add_memory(memory_id, vector, payload)
            memory_ids.append(memory_id)
            imported_count += 1
        
        return {
            "status": "success",
            "source": request.source,
            "chunks_count": len(chunks),
            "imported_count": imported_count,
            "memory_ids": memory_ids[:10],
            "total_memory_ids": len(memory_ids)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/kb/import/batch")
async def import_documents_batch(request: ImportBatchRequest):
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
                user_id=request.user_id
            )
            result = await import_document(single_request)
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
    user_id: Optional[str] = None
):
    """从 URL 导入网页内容"""
    request = ImportDocumentRequest(
        source=url,
        tags=tags or ['web'],
        user_id=user_id
    )
    return await import_document(request)