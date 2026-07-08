# api/routes/evolution.py
from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import os
import logging

from ..service_registry import ServiceRegistry
from ..dependencies import get_registry
from ..utils import parse_iso_datetime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Evolution"])

# ---------- 演化状态管理（与原 v2 一致） ----------
# 这些状态在原 v2 中是全局变量，这里放在路由模块中
# 实际生产环境应使用 registry 或 Redis 持久化

_evolution_state = {
    "last_completed_at": None,
    "inflight": False,
    "submission_pending": False,
    "schedule_anchor_at": None,
}

# 状态文件路径（与原 v2 保持一致）
EVOLUTION_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "evolution_state.json"
)


def _load_evolution_state() -> Dict[str, Any]:
    """加载演化调度状态。文件不存在时返回默认值。"""
    try:
        if os.path.exists(EVOLUTION_STATE_FILE):
            with open(EVOLUTION_STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        logger.warning(f"加载演化状态失败: {e}")
    return {}


def _save_evolution_state(state: Dict[str, Any]):
    """持久化演化调度状态。"""
    try:
        os.makedirs(os.path.dirname(EVOLUTION_STATE_FILE), exist_ok=True)
        with open(EVOLUTION_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存演化状态失败: {e}")


def _get_last_evolution_completed_at() -> Optional[datetime]:
    """获取上次演化完成时间。"""
    state = _load_evolution_state()
    return parse_iso_datetime(state.get('last_completed_at'))


def _record_evolution_completed(when: Optional[datetime] = None, result: Optional[Dict[str, Any]] = None):
    """记录演化完成时间。"""
    ts = when or datetime.now()
    state = {
        'last_completed_at': ts.isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    if result is not None:
        state['last_result'] = result
    _save_evolution_state(state)


def _seconds_until_next_evolution(interval: int, now: Optional[datetime] = None) -> int:
    """根据上次完成时间计算下一轮还需等待多久。到期则返回 0。"""
    now = now or datetime.now()
    last_completed = _get_last_evolution_completed_at()
    if not last_completed:
        return 0
    elapsed = max((now - last_completed).total_seconds(), 0)
    return max(int(interval - elapsed), 0)


def _get_evolution_status_snapshot(
    config: Any,
    now: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    返回当前演化调度状态，供 API/仪表盘展示。

    Args:
        config: AppConfig 实例
        now: 当前时间
    """
    now = now or datetime.now()
    evolution_config = config.evolution if hasattr(config, 'evolution') else {}
    enabled = bool(getattr(evolution_config, 'enabled', False))
    interval = max(int(getattr(evolution_config, 'evolve_interval', 86400)), 60)

    last_completed = _get_last_evolution_completed_at()
    seconds_remaining = _seconds_until_next_evolution(interval, now=now)
    next_due_at = (last_completed + timedelta(seconds=interval)) if last_completed else None

    # 确定阶段
    if not last_completed:
        phase = 'initial_catch_up'
    elif _evolution_state.get('inflight', False):
        phase = 'running'
    elif _evolution_state.get('submission_pending', False):
        phase = 'queued'
    elif seconds_remaining == 0:
        phase = 'due'
    else:
        phase = 'waiting'

    # 获取调度器运行状态（如果有）
    scheduler_running = False
    if hasattr(config, 'scheduler') and config.scheduler:
        scheduler_running = getattr(config.scheduler, '_running', False)

    return {
        'enabled': enabled,
        'interval_seconds': interval,
        'last_completed_at': last_completed.isoformat() if last_completed else None,
        'next_due_at': next_due_at.isoformat() if next_due_at else None,
        'seconds_until_next_run': seconds_remaining,
        'overdue': phase in {'initial_catch_up', 'due'},
        'phase': phase,
        'inflight': _evolution_state.get('inflight', False),
        'submission_pending': _evolution_state.get('submission_pending', False),
        'scheduler_running': scheduler_running,
        'state_file': EVOLUTION_STATE_FILE,
    }


# ==================== 端点 ====================

@router.get("/memory/evolution/status")
async def get_memory_evolution_status(
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    获取演化调度状态（下一轮剩余时间、是否到期、是否正在执行）。
    """
    return _get_evolution_status_snapshot(registry.config)


@router.post("/memory/evolve")
async def trigger_memory_evolution(
    user_id: Optional[str] = Query(None, description="用户ID（可选）"),
    limit: int = Query(10000, ge=1, le=100000, description="处理记忆数量上限"),
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    手动触发一轮记忆自演化。

    Args:
        user_id: 用户 ID，不指定则使用默认用户
        limit: 处理记忆数量上限
    """
    qdrant = registry.qdrant
    memory_evolution = getattr(registry, 'memory_evolution', None)
    config = registry.config

    if not memory_evolution:
        raise HTTPException(status_code=503, detail="记忆自演化引擎未启用或未初始化")

    if not qdrant or not qdrant.is_available():
        raise HTTPException(status_code=503, detail="存储不可用")

    # 防止并发执行
    if _evolution_state.get('inflight', False):
        raise HTTPException(status_code=409, detail="演化任务正在执行中，请稍后再试")

    try:
        # 标记正在执行
        _evolution_state['inflight'] = True

        user_id = user_id if user_id is not None else config.users.default_user_id

        # 执行演化
        result = await memory_evolution.evolve(user_id=user_id, limit=limit)

        # 重建 BM25 索引
        if registry.bm25 and qdrant:
            try:
                from ..utils import rebuild_bm25_index
                await rebuild_bm25_index(registry)
                logger.info("演化后 BM25 索引重建完成")
            except Exception as e:
                logger.warning(f"BM25 索引重建失败: {e}")

        # 记录完成状态
        if result.get('status') == 'success':
            completed_at = datetime.now()
            _evolution_state['schedule_anchor_at'] = completed_at
            _record_evolution_completed(completed_at, result)

        return result

    except Exception as e:
        logger.error(f"演化执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"演化执行失败: {str(e)}")

    finally:
        _evolution_state['inflight'] = False
        _evolution_state['submission_pending'] = False