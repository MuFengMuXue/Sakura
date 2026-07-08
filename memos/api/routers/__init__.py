# api/routes/__init__.py
from .health import router as health_router
from .memories import router as memories_router
from .preference import router as preference_router
from .tools import router as tools_router
from .graph import router as graph_router
from .feedback import router as feedback_router
from .images import router as images_router
from .entity import router as entity_router
from .deduplicate import router as deduplicate_router
from .reclassify import router as reclassify_router
from .kb import router as kb_router
from .evolution import router as evolution_router
from .memory_ops import router as memory_ops_router