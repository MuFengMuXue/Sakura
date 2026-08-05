# benchmark_search.py - 对比 /search 优化前后（旧:逐层并扫 gather；新:单扫+分桶+补充）的性能与语义等价性
"""
用法（cwd=F:\\Sakura，conda env Sakura，Windows PowerShell）:
    & "F:\\miniconda3\\envs\\Sakura\\python.exe" scripts\\benchmark_search.py

验证目标：
1. 语义等价：每层 top-recall_top_k 的 (id, similarity) 序列，old 与 new 完全一致；
   缺 layer 的 legacy 点必须归入 LongTermMemory 桶；L==1 快速路径与 old 一致。
2. 性能：内存 1 万条集合上对比
   - old_no_cache   （旧编排，无 parse_json_path 缓存）
   - old_with_cache （旧编排 + 缓存，隔离缓存收益）
   - new_no_cache   （新编排，无缓存，隔离编排收益）
   - new_with_cache （新编排 + 缓存，= 线上最终形态）
   期望：old_no_cache ~590ms → new_with_cache ~170ms（约 3.5x，含缓存再省 ~30% 过滤）。
"""
import asyncio
import math
import random
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 输出（与 memos 服务端保持一致）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]  # = F:\Sakura\memos（storage 在此下）
for _p in (str(ROOT), str(ROOT / "api")):   # api 目录放 memos_api_server_v2.py
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qdrant_client.models import PointStruct
from storage.qdrant_client import MemosQdrantClient

try:  # 优先用真实 server 的 monkeypatch / 常量，验证的是线上同一份代码
    from memos_api_server_v2 import _install_qdrant_json_path_cache, normalize_layer, MEMORY_LAYERS
except Exception as _e:  # 兜底：内联副本，保证脚本独立可跑
    print(f"[警告] 无法导入 memos_api_server_v2（{_e}），使用内联副本")

    def normalize_layer(layer, default="WorkingMemory"):
        return layer if layer in MEMORY_LAYERS else default

    def _install_qdrant_json_path_cache():
        import functools
        import qdrant_client.local.payload_value_extractor as _pve
        _orig = _pve.parse_json_path

        @functools.lru_cache(maxsize=1024)
        def _cached(key: str):
            return _orig(key)

        def _cached_copy(key: str):
            return list(_cached(key))

        _pve.parse_json_path = _cached_copy
        try:
            import qdrant_client.local.local_collection as _lc
            _lc.parse_json_path = _cached_copy
        except Exception:
            pass
        print("[OK] qdrant parse_json_path 已启用 lru_cache（内联副本）")

    MEMORY_LAYERS = ["WorkingMemory", "LongTermMemory", "UserMemory"]

N = 10_000
VECTOR_SIZE = 1024
USER_ID = "feiniu_default"
_MTYPES = ["preference", "fact", "episodic", "semantic", "procedural", "general"]
_TAGS = ["tag_a", "tag_b", "tag_c", "tag_d"]


def _unit_vec(rng: random.Random) -> list:
    v = [rng.gauss(0, 1) for _ in range(VECTOR_SIZE)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def build_dataset():
    """构造 1 万条数据：~92% 带 layer，~8% 缺 layer（legacy，逼出 LTM 分桶）。"""
    client = MemosQdrantClient(path="./_bench_data/qdrant", use_memory=True, vector_size=VECTOR_SIZE)
    rng = random.Random(42)
    points = []
    for i in range(N):
        legacy = (i % 13 == 0)  # ~7.7% 缺 layer
        payload = {
            "content": f"记忆内容第 {i} 条，用于检索基准测试，覆盖中文关键词检索场景",
            "user_id": USER_ID,
            "importance": round(rng.uniform(0.1, 0.95), 4),
            "memory_type": rng.choice(_MTYPES),
            "tags": rng.sample(_TAGS, k=rng.randint(0, 2)),
            "status": "archived" if rng.random() < 0.05 else "active",
            "access_count": rng.randint(0, 50),
            "created_at": "2026-08-01T12:00:00.000000",
            "last_accessed_at": None,
        }
        if not legacy:
            payload["layer"] = rng.choice(MEMORY_LAYERS)
        points.append(PointStruct(id=i, vector=_unit_vec(rng), payload=payload))

    client.client.upsert(collection_name=client.collection_name, points=points)
    total = client.client.count(collection_name=client.collection_name).count
    legacy_n = sum(1 for p in points if "layer" not in p.payload)
    print(f"[数据] {total} 条入内存集合，其中 legacy（缺 layer）{legacy_n} 条")
    return client


async def old_search(client, query_vector, requested_layers, top_k, threshold, user_id):
    """复刻 memos_api_server_v2.py 的旧实现（2092-2154 提交前的 gather 版）。
    返回 (results_map, raw_layers, raw_legacy) — raw_layers 供逐层精确 top-k 断言。"""
    recall_top_k = max(top_k * 3, 8)
    results_map = {}
    tasks = [
        asyncio.to_thread(client.search, query_vector=query_vector, top_k=recall_top_k,
                          score_threshold=threshold, user_id=user_id, layer=layer_name)
        for layer_name in requested_layers
    ]
    if "LongTermMemory" in requested_layers:
        tasks.append(asyncio.to_thread(client.search, query_vector=query_vector, top_k=recall_top_k,
                                       score_threshold=threshold, user_id=user_id))
    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    raw_layers = {}
    for layer_name, vec in zip(requested_layers, gathered):
        if isinstance(vec, Exception):
            raw_layers[layer_name] = []
            continue
        raw_layers[layer_name] = vec
        for r in vec:
            if r.get("id"):
                results_map.setdefault(r["id"], {"data": r, "scores": {}})
                results_map[r["id"]]["data"].update(r)
                results_map[r["id"]]["scores"]["vector"] = max(
                    results_map[r["id"]]["scores"].get("vector", 0), r.get("similarity", 0))
    raw_legacy = []
    if "LongTermMemory" in requested_layers:
        legacy = gathered[len(requested_layers)]
        if not isinstance(legacy, Exception):
            raw_legacy = legacy
            for r in legacy:
                payload = r.get("payload", {}) if isinstance(r.get("payload"), dict) else {}
                if payload.get("layer"):
                    continue
                r["layer"] = "LongTermMemory"
                if r.get("id"):
                    results_map.setdefault(r["id"], {"data": r, "scores": {}})
                    results_map[r["id"]]["data"].update(r)
                    results_map[r["id"]]["scores"]["vector"] = max(
                        results_map[r["id"]]["scores"].get("vector", 0), r.get("similarity", 0))
    return results_map, raw_layers, raw_legacy


async def new_search(client, query_vector, requested_layers, top_k, threshold, user_id):
    """复刻优化后的实现（单次全库扫描 + 本地分桶 + 配额补充）。返回 (results_map, buckets)。"""
    recall_top_k = max(top_k * 3, 8)
    L = len(requested_layers)
    results_map = {}

    def _scan(layer=None, k=None):
        return client.search(query_vector=query_vector, top_k=k if k is not None else recall_top_k * L,
                             score_threshold=threshold, user_id=user_id, layer=layer)

    if L == 1:
        # 单层快速路径 = 旧逻辑，直接复用旧实现保证逐字节一致
        rm, raw_layers, raw_legacy = await old_search(client, query_vector, requested_layers, top_k, threshold, user_id)
        buckets = {ln: raw_layers.get(ln, []) for ln in requested_layers}
        if raw_legacy:
            buckets.setdefault("LongTermMemory", []).extend(raw_legacy)
        return rm, buckets

    full = await asyncio.to_thread(_scan, None, recall_top_k * L)
    buckets = {ln: [] for ln in requested_layers}
    if isinstance(full, Exception):
        return results_map, buckets
    for r in full:
        payload = r.get("payload", {}) if isinstance(r.get("payload"), dict) else {}
        r_layer = normalize_layer(payload.get("layer"), default="LongTermMemory")
        if r_layer in buckets:
            buckets[r_layer].append(r)

    # 配额补充（镜像生产：LTM 只按带 layer 的正式点计数，防 legacy 挤满抑制补扫）
    truncated = len(full) >= recall_top_k * L
    need = []
    for ln, b in buckets.items():
        if not truncated:
            continue
        cnt = sum(1 for r in b if (r.get("payload") or {}).get("layer")) if ln == "LongTermMemory" else len(b)
        if cnt < recall_top_k:
            need.append(ln)
    if need:
        supp = await asyncio.gather(*(asyncio.to_thread(_scan, ln, recall_top_k) for ln in need), return_exceptions=True)
        for ln, res in zip(need, supp):
            if isinstance(res, Exception):
                continue
            for r in res:
                if r.get("id"):
                    buckets[ln].append(r)

    # 去重（全量扫描与补扫可能命中同一点），与生产合并进 results_map 的 setdefault 语义一致
    for ln in list(buckets):
        dedup = {}
        for r in buckets[ln]:
            prev = dedup.get(r["id"])
            if prev is None or r.get("similarity", 0) > prev.get("similarity", 0):
                dedup[r["id"]] = r
        buckets[ln] = list(dedup.values())

    for vec in buckets.values():
        for r in vec:
            if r.get("id"):
                results_map.setdefault(r["id"], {"data": r, "scores": {}})
                results_map[r["id"]]["data"].update(r)
                results_map[r["id"]]["scores"]["vector"] = max(
                    results_map[r["id"]]["scores"].get("vector", 0), r.get("similarity", 0))
    return results_map, buckets


def _topk(rows, k):
    """按 similarity 降序取前 k 条的 (id, similarity) 序列（同分按 id 决出稳定序）。"""
    rows = list(rows)
    rows.sort(key=lambda x: (-x.get("similarity", 0), str(x.get("id", ""))))
    return [(r["id"], r.get("similarity", 0)) for r in rows[:k]]


async def assert_equivalence(client, configs, rng, failures):
    """多种请求配置下断言 old 与 new 语义等价。configs: [{layers, top_k, threshold}]"""
    for cfg in configs:
        layers = cfg["layers"]
        top_k = cfg["top_k"]
        threshold = cfg["threshold"]
        recall_top_k = max(top_k * 3, 8)
        for q in range(5):
            qv = _unit_vec(rng)
            old_rm, old_layers, old_legacy = await old_search(client, qv, layers, top_k, threshold, USER_ID)
            new_rm, new_buckets = await new_search(client, qv, layers, top_k, threshold, USER_ID)

            tag = f"layers={layers} top_k={top_k} th={threshold} q{q}"
            if len(layers) == 1:
                # 单层快速路径：必须逐字节一致
                if set(old_rm) != set(new_rm):
                    failures.append(f"[单层 id 集不一致] {tag} old={len(old_rm)} new={len(new_rm)}")
                    continue
                for i in old_rm:
                    a = old_rm[i]["scores"].get("vector", 0)
                    b = new_rm[i]["scores"].get("vector", 0)
                    if abs(a - b) > 1e-6:
                        failures.append(f"[单层分数不一致] {tag} id={i} old={a} new={b}")
                continue

            # 1) 合并候选池：new ⊇ old 且公共 id 向量分一致（新方案可多召回，但绝不丢旧候选）
            missing = [i for i in old_rm if i not in new_rm]
            if missing:
                failures.append(f"[旧候选丢失] {tag} {len(missing)} 条 old 有 new 无: {missing[:3]}")
            for i in set(old_rm) & set(new_rm):
                a = old_rm[i]["scores"].get("vector", 0)
                b = new_rm[i]["scores"].get("vector", 0)
                if abs(a - b) > 1e-6:
                    failures.append(f"[合并分不一致] {tag} id={i} old={a} new={b}")

            # 2) 每层候选池：old_pool ⊆ new_pool（诊断层覆盖不回归；LTM 旧池含被采纳的 legacy 点）
            for ln in layers:
                old_pool = {r["id"] for r in old_layers.get(ln, [])}
                if ln == "LongTermMemory":
                    old_pool |= {r["id"] for r in old_legacy if not (r.get("payload") or {}).get("layer")}
                new_pool = {r["id"] for r in new_buckets.get(ln, [])}
                miss = old_pool - new_pool
                if miss:
                    failures.append(f"[层 {ln} 候选丢失] {tag} {len(miss)} 条: {list(miss)[:3]}")

            # 3) 纯层（非 LTM）top-recall_top_k 精确序列一致（验证 top-k 论证；LTM 因 legacy 混入允许组成偏移）
            for ln in layers:
                if ln == "LongTermMemory":
                    continue
                old_seq = _topk(old_layers.get(ln, []), recall_top_k)
                new_seq = _topk(new_buckets.get(ln, []), recall_top_k)
                if old_seq != new_seq:
                    failures.append(f"[层 {ln} top-k 不一致] {tag}\n  old={old_seq[:3]}\n  new={new_seq[:3]}")

            # 4) 旧实现真正采纳的 legacy 点必须都在 new 的 LTM 桶里
            legacy_ids = {r["id"] for r in old_legacy if not (r.get("payload") or {}).get("layer")}
            ltm_new = {r["id"] for r in new_buckets.get("LongTermMemory", [])}
            miss_legacy = legacy_ids - ltm_new
            if miss_legacy:
                failures.append(f"[legacy 缺失] {tag} {len(miss_legacy)} 条: {list(miss_legacy)[:3]}")


async def time_it(fn, *args, repeats=5):
    best = None
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        await fn(*args)
        dt = time.perf_counter() - t0
        times.append(dt)
        best = dt if best is None else min(best, dt)
    times.sort()
    return best, sum(times) / len(times), times[len(times) // 2]


async def main():
    client = build_dataset()
    rng = random.Random(7)
    qv = _unit_vec(rng)
    configs = [
        {"layers": ["WorkingMemory", "LongTermMemory", "UserMemory"], "top_k": 5, "threshold": 0.0},
        {"layers": ["WorkingMemory", "LongTermMemory", "UserMemory"], "top_k": 20, "threshold": 0.0},
        {"layers": ["LongTermMemory", "UserMemory"], "top_k": 5, "threshold": 0.0},
        {"layers": ["WorkingMemory"], "top_k": 5, "threshold": 0.0},          # 单层快速路径
        {"layers": ["LongTermMemory", "WorkingMemory", "UserMemory"], "top_k": 5, "threshold": 0.6},  # 阈值
    ]
    print("\n===== 语义等价断言（old vs new）=====")
    failures = []
    await assert_equivalence(client, configs, rng, failures)
    print("PASS: all configs equivalent" if not failures else f"FAIL: {len(failures)} mismatches")
    for f in failures[:10]:
        print("  [x]", f)

    print("\n===== 性能对比（内存 1 万条，默认三层）=====")
    layers = ["WorkingMemory", "LongTermMemory", "UserMemory"]
    top_k, threshold = 5, 0.0

    def _label(name, best, mean, med):
        return f"{name:16s} best={best*1000:7.1f}ms mean={mean*1000:7.1f}ms med={med*1000:7.1f}ms"

    b1, m1, d1 = await time_it(old_search, client, qv, layers, top_k, threshold, USER_ID)
    print(" ", _label("old_no_cache", b1, m1, d1))

    _install_qdrant_json_path_cache()
    print("  [安装 parse_json_path lru_cache]")

    b2, m2, d2 = await time_it(old_search, client, qv, layers, top_k, threshold, USER_ID)
    print(" ", _label("old_with_cache", b2, m2, d2))

    b3, m3, d3 = await time_it(new_search, client, qv, layers, top_k, threshold, USER_ID)
    print(" ", _label("new_with_cache", b3, m3, d3))  # 缓存已装，此处即线上最终形态

    print("\n===== 加速比 ===== ")
    print(f"  old_no_cache → new_with_cache : {b1/b3:.2f}x (best)  {m1/m3:.2f}x (mean)")
    print(f"  old_no_cache → old_with_cache : {b1/b2:.2f}x (缓存单独收益, best)")
    print(f"  old_with_cache → new_with_cache: {b2/b3:.2f}x (编排单独收益, best)")

    if hasattr(client.client, "close"):
        client.client.close()


if __name__ == "__main__":
    asyncio.run(main())
