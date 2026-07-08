# api/routes/images.py
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
import logging

from ..schemas import UploadImageRequest
from ..service_registry import ServiceRegistry
from ..dependencies import get_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Images"])


@router.get("/images/stats")
async def get_image_stats(
    user_id: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """获取图像记忆统计"""
    image_memory = registry.image_memory
    config = registry.config

    if not image_memory:
        return {"status": "disabled", "message": "图像记忆未启用"}

    user_id = user_id if user_id is not None else config.users.default_user_id
    stats = await image_memory.get_stats(user_id)  # 异步，需要 await
    return {"status": "enabled", **stats}


@router.post("/images/upload")
async def upload_image(
    request: UploadImageRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """上传图像"""
    image_memory = registry.image_memory
    config = registry.config

    if not image_memory:
        raise HTTPException(status_code=503, detail="图像记忆未启用")

    user_id = request.user_id if request.user_id is not None else config.users.default_user_id

    try:
        result = await image_memory.save_image_from_base64(
            base64_data=request.image_base64,
            original_name=request.filename,
            image_type=request.image_type,
            description=request.description,
            tags=request.tags,
            user_id=user_id,
            auto_describe=config.image.auto_describe,  # 直接从配置读取
        )

        if result:
            return {
                "status": "success",
                "image_id": result.id,
                "filename": result.filename,
                "description": result.description,
                "size_bytes": result.size_bytes,
                "dimensions": f"{result.width}x{result.height}"
            }
        else:
            raise HTTPException(status_code=400, detail="图像保存失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/search")
async def search_images(
    query: str,
    top_k: int = 5,
    image_type: Optional[str] = None,
    user_id: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """搜索图像"""
    image_memory = registry.image_memory
    config = registry.config

    if not image_memory:
        return {"images": [], "message": "图像记忆未启用"}

    user_id = user_id if user_id is not None else config.users.default_user_id
    results = await image_memory.search(
        query=query,
        user_id=user_id,
        top_k=top_k,
        image_type=image_type
    )

    return {
        "query": query,
        "images": results,
        "count": len(results)
    }


@router.get("/images/{image_id}")
async def get_image_info(
    image_id: str,
    registry: ServiceRegistry = Depends(get_registry),
):
    """获取图像信息"""
    image_memory = registry.image_memory

    if not image_memory:
        raise HTTPException(status_code=503, detail="图像记忆未启用")

    metadata = await image_memory.get_image(image_id)
    if metadata:
        return metadata.to_dict()
    raise HTTPException(status_code=404, detail="图像不存在")


@router.get("/images/{image_id}/data")
async def get_image_data(
    image_id: str,
    thumbnail: bool = False,
    registry: ServiceRegistry = Depends(get_registry),
):
    """获取图像数据（Base64）"""
    image_memory = registry.image_memory

    if not image_memory:
        raise HTTPException(status_code=503, detail="图像记忆未启用")

    data = await image_memory.get_image_base64(image_id, thumbnail)
    if data:
        return {
            "image_id": image_id,
            "thumbnail": thumbnail,
            "data": data
        }
    raise HTTPException(status_code=404, detail="图像不存在")


@router.delete("/images/{image_id}")
async def delete_image(
    image_id: str,
    registry: ServiceRegistry = Depends(get_registry),
):
    """删除图像"""
    image_memory = registry.image_memory

    if not image_memory:
        raise HTTPException(status_code=503, detail="图像记忆未启用")

    success = await image_memory.delete_image(image_id)
    if success:
        return {"status": "success", "message": f"已删除图像 {image_id}"}
    raise HTTPException(status_code=404, detail="图像不存在或删除失败")


@router.get("/images")
async def list_images(
    image_type: Optional[str] = None,
    limit: int = 50,
    user_id: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """列出用户的图像"""
    image_memory = registry.image_memory
    config = registry.config

    if not image_memory:
        return {"images": [], "message": "图像记忆未启用"}

    user_id = user_id if user_id is not None else config.users.default_user_id
    images = await image_memory.list_images(user_id, image_type, limit)

    return {
        "images": [m.to_dict() for m in images],
        "count": len(images)
    }


@router.post("/images/regenerate-descriptions")
async def regenerate_image_descriptions(
    force: bool = Query(default=False, description="是否强制重新生成所有描述"),
    user_id: Optional[str] = None,
    registry: ServiceRegistry = Depends(get_registry),
):
    """为没有描述的图片重新生成描述"""
    image_memory = registry.image_memory
    config = registry.config

    if not image_memory:
        raise HTTPException(status_code=503, detail="图像记忆未启用")

    user_id = user_id if user_id is not None else config.users.default_user_id

    try:
        images = await image_memory.list_images(user_id, limit=500)
        updated_count = 0
        failed_count = 0

        for img_meta in images:
            if not force and img_meta.description and img_meta.description.strip():
                continue

            try:
                img_data = await image_memory.get_image_data(img_meta.id, thumbnail=False)
                if not img_data:
                    failed_count += 1
                    continue

                from PIL import Image
                from io import BytesIO
                image = Image.open(BytesIO(img_data))

                description = await image_memory._generate_description(image, img_meta.original_name)

                if description:
                    img_meta.description = description
                    image_memory.metadata_cache[img_meta.id] = img_meta
                    updated_count += 1
                    logger.info(f"已为图片 {img_meta.id} 生成描述: {description[:50]}...")
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"生成图片 {img_meta.id} 描述失败: {e}")
                failed_count += 1

        if updated_count > 0:
            image_memory._save_metadata_to_file()

        return {
            "status": "success",
            "message": f"已更新 {updated_count} 张图片的描述，{failed_count} 张失败",
            "updated_count": updated_count,
            "failed_count": failed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))