# api/main.py
"""
FastAPI 应用入口
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .lifecycle import lifespan

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
    handlers=[
        logging.StreamHandler()        # 输出到控制台
    ]
)

# 创建 FastAPI 应用实例
app = FastAPI(
    title="MemOS API ",
    version="2.0.0",
    lifespan=lifespan,  # 挂载生命周期管理
)

# ---------- CORS 配置 ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 注册路由 ----------

from .routers import (
    health_router,
    memories_router,
    preference_router,
    tools_router,
    graph_router,
    entity_router,
    images_router,
    feedback_router,
    deduplicate_router,
    reclassify_router,
    kb_router,
    evolution_router,
    memory_ops_router


)


app.include_router(health_router)
app.include_router(memories_router)
app.include_router(preference_router)
app.include_router(tools_router)
app.include_router(graph_router)
app.include_router(entity_router)
app.include_router(images_router)
app.include_router(feedback_router)
app.include_router(deduplicate_router)
app.include_router(reclassify_router)
app.include_router(kb_router)
app.include_router(evolution_router)
app.include_router(memory_ops_router)


# ---------- 启动入口（仅开发环境使用） ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)