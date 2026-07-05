# api/dependencies.py
from fastapi import Request
from .service_registry import ServiceRegistry

def get_registry(request: Request) -> ServiceRegistry:
    """
    依赖注入：从应用状态获取服务注册表
    """
    return request.app.state.registry